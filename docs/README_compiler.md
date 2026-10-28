# Emmy PTX Compiler — `compile.py`

This document describes **`emmy-hackathon-2026/compile.py`** — a from-scratch compiler that
translates [Emmy](https://github.com/cloudrift-ai/emmy) Kernel IR directly to **PTX assembly**,
skipping Emmy's usual CUDA C++ + `nvcc` pipeline entirely.

## Contents

- [Why a PTX compiler?](#why-a-ptx-compiler)
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [How it was built (design)](#how-it-was-built-design)
- [How it was optimised](#how-it-was-optimised)
- [Precision strategy](#precision-strategy)
- [Validation](#validation)
- [Usage](#usage)
- [Scope and limitations](#scope-and-limitations)
- [Repo map](#repo-map)

---

## Why a PTX compiler?

Emmy compiles PyTorch to CUDA C++ and builds each variant with the full `nvcc` toolchain. That
middle step is expensive, and Emmy's autotuner compiles *thousands* of kernel variants — every
extra second per compilation multiplies.

Emmy's own `ptxas` turns PTX into a GPU binary roughly **19× faster** than the full `nvcc` path.
So instead of emitting CUDA C++ and letting `nvcc` do the rest, this compiler emits **PTX
directly**: `python compile.py kernel.ir.json > out.ptx`, and only `ptxas` runs afterwards.

The problems are scheduled for `sm_89`; the hosted evaluation runner is an RTX 5090 (`sm_120`),
where `sm_89` PTX runs unchanged.

## What it does

```
kernel.ir.json ──► (Python IR classes) ──► PTX text ──► ptxas ──► GPU
                     ^                       ^
                     |                       |
                de-serialized          compiled by us,
                by Emmy only          never nvcc
```

Given a `*.ir.json` file the compiler:

1. Deserializes the graph with Emmy's own classes (`emmy.compiler.graph.Graph`) — the only use of
   the Emmy package. No torch, no CUDA, no GPU are needed to *compile*.
2. Finds the `KernelOp` node in the graph.
3. Reads the launch contract (`grid`, `block`, shared-memory bytes, scalar args, argument order)
   from the sibling `meta.json`.
4. Walks the kernel body's typed IR statement tree and emits PTX for it.
5. Prints the PTX module on stdout (diagnostics go to stderr only, so stdout stays pipe-clean).

### IR node coverage

| IR node                | Emitted as                                                        |
|------------------------|-------------------------------------------------------------------|
| `Tile`                 | thread-id decode of the axes, tail guard `bra`, optional RASTER decode |
| `StridedLoop`          | counted loop / guarded branch / generic `setp+bra` loop           |
| `Loop`                 | counted loop (seed accumulators first)                            |
| `Cond`                 | predicated branch or **compile-time decided** (no branch at all)  |
| `Load` / `Write`       | vectorized `ld`/`st` (v2/v4), scalar + `cvt` for f16/bf16/f64/i64 |
| `Assign`               | one f32 lowering per elementwise op                               |
| `Accum`                | in-place `add.f32` / `mul.f32` / `max.f32` / `min.f32` or fma chain |
| `Init` / `Let`         | `mov.f32` of the identity / folded literal or index binding       |
| `Select`               | chained `selp.f32`                                                |
| `Smem`                 | allocated up-front in `plan_buffers`                              |
| `Sync`                 | `bar.sync` / `bar.warp.sync` / count-aware barriers               |
| `WarpShuffle`          | XOR butterfly `shfl.sync.bfly`                                    |
| `TreeHalve`            | shared-memory reduction tree with barriers                        |
| `CpAsyncCopy`          | `cp.async.cg/ca.shared.global`, swizzled destination              |
| `CpAsyncCommit` / `CpAsyncWait` | `cp.async.commit_group` / `cp.async.wait_group`           |

Supported buffer element types: `f32`, `f16`, `bf16`, `f64`, `i32`, `i64` (see the `DTYPES`
table in the source). Shared-memory C types `float`, `__half`, `half`, `__nv_bfloat16`,
`double`, `int`, `long long` map onto the same set.

## Architecture

The file is organised into five sections that mirror the stages of the lowering pipeline:

```
┌───────────────────────────  kernel.ir.json ──────────────────────────────┐
│                    Graph.from_dict (Emmy classes)                        │
│                    + meta.json launch contract                           │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │    KernelCompiler   │  §4  one pass, statement by statement
                         │  ─────────────────  │
                         │ plan_buffers  smem layout, written-set analysis │
                         │ prologue      ptrs, cvta.to.global, scalars     │
                         │ body          IR walker → PTX lines             │
                         └───────┬─────────────┘
                                 │  uses three cooperating engines
              ┌──────────────────┼───────────────────────┐
              ▼                  ▼                       ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │  IndexAlgebra    │ │    FloatMath     │ │    Registers     │
   │  §2  Lin exprs   │ │  §3  f32 ops     │ │  §4  typed virt  │
   │  value ranges    │ │  const folding   │ │  regs + scope    │
   │  div/mod rules   │ │  polynomial fits │ │  depth tracking  │
   │  strength reduce │ │  fma fusion      │ │  (CSE ordering)  │
   └──────────────────┘ └──────────────────┘ └──────────────────┘
              │                  │
              └──────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  CSE caches     │  scopes stack: Lin / atom / pred /
            │  (§4)           │  address / load keys, per block
            └─────────────────┘
                     ▼
              ┌─────────────┐
              │   module()  │  §4  .version 8.0, .target sm_89,
              │             │  .entry, .maxntid, .reg declarations
              └─────────────┘
                     ▼
                   out.ptx  ──► ptxas ──► designed for sm_89 / sm_120 runner
```

The one-pass design is deliberate: every piece of state the walker needs (register ranges,
scope depth, shared-memory layout, the "written" set for cache hints) is computed **before** or
**on first use**, so there is no separate lowering IR or fixpoint pass — the IR is lowered in a
single forward walk that interleaves simplification, code motion and emission.

### The three engines

**`IndexAlgebra` (Lin).** Integer expressions are kept in a canonical form
`const + Σ cᵢ·atomᵢ` where every register (thread id, block id, loop counter…) has a **known
value range** because all shapes are static. With ranges, algebraic rules turn expensive
divisions into shifts and collapses:

```
floor((d·q + r) / d) = q + floor(r / d)        (d·q + r) mod d = r mod d
floor(r / d) = 0,  r mod d = r                 when 0 ≤ r < d
```

A Tile decode such as `a1 = (gid / 32) % 16` with `gid = 512·ctaid + tid` therefore collapses to
`tid >> 5`. Conditions the ranges prove true/false become plain booleans at compile time, so
whole branches disappear. Equal expressions compare and hash equal, which is exactly what the
common-subexpression caches key on.

**`FloatMath`.** One lowering per elementwise op on f32 values (register or exact-bit-pattern
immediate). Ops are aliased onto a canonical set (`true_divide → divide`, `power → pow`,
`amax → maximum`, …), constants are folded with IEEE-f32 `numpy`, and every transcendental has
an explicit, precision-budgeted lowering (see [Precision strategy](#precision-strategy)).

**`Registers` + CSE scopes.** Fresh virtual registers per PTX type (`%p/%h/%r/%rd/%f/%fd`),
declared once at the top of the module. A stack of CSE caches is pushed per block statement
(`scope()`), so a value computed inside a branch or loop body is **never reused** where it may
not have been computed; `atom_depth` / `reg_depth` additionally order sub-expressions so
loop-invariant parts are materialised outside the loop.

## How it was built (design)

The compiler was grown node-by-node against the 15 open problems, in the same order the task
defines them — which is also the order the hidden evaluation set is split into:

| Category | Open problems | First IR nodes needed |
|---|---|---|
| 1. Pointwise | `p00`–`p03` | `Tile`, `Load`, `Assign`, `Write`, `Smem` |
| 2. Reductions | `r01`–`r04` | `StridedLoop`, `Loop`, `Accum`, `TreeHalve`, `WarpShuffle`, atomics |
| 3. Matmul | `m01`–`m02` | `CpAsyncCopy`, tiled `Tile` + RASTER decode, `Sync` |
| 4. Fused | `f01`–`f03` | `Cond`, `Select`, `Init`, `Let`, softmax/softplus/silu |
| 5. Attention-like | `a01`–`a02` | 2-D `Tile` decomposition, exp in dot-product, multi-barrier `Sync` |

Key early decisions that shaped everything after:

- **Static shapes only.** Emmy's IR carries `is_static` extents; the compiler refuses symbolic
  ones. This is what makes the whole index algebra sound — every loop trip count and every
  register range is known, so the folding and strength-reduction rules are always valid.
- **One walk, no IR-to-IR passes.** Simplification happens *while* emitting, by constructing
  canonical `Lin` expressions and caching their materialisations. There is no separate
  optimiser; optimisation *is* the construction.
- **Exact bit patterns for f32 immediates.** `fimm()` prints `0f%08X` from a `numpy` bit
  reinterpretation — `1.0` is always `0f3F800000`. This makes constant folding, folding checks
  and splat-loads exact and free of decimal-rounding surprises.
- **Every statement is commented with the IR it came from.** `stmt()` prefixes the lowering
  with `// <ir pretty>`, so the PTX is self-documenting and easy to audit against the input —
  the same style the repo's hand-written `examples/*.ptx` use. It also became an essential
  debugging aid while the walker was being brought up.
- **C semantics, not Python semantics, for integer math.** Emmy's CUDA render uses C truncating
  `/` and `%`, so `c_div` / `c_rem` mirror that exactly, and the algebra tracks whether an
  expression is provably non-negative (floor `div`/`mod`) before choosing the cheaper unsigned
  instructions.
- **Shared memory mirrors Emmy's own scheme.** Static buffers up to 48 KiB are `.shared`
  symbols aligned to 16 bytes; larger totals fall back to a dynamic `smem_pool[]` with per-buffer
  offsets — exactly the static/dynamic split Emmy performs, so launch descriptors match.

## How it was optimised

The optimisations fall into six buckets. Performance claims marked *measured* come from the
author's own runner runs and are quoted from the source comments; they were not re-verified in
this offline bundle (see [Validation](#validation)).

### 1. Register-range-driven index algebra (strength reduction)

- Every register carries `(lo, hi)`; division/remainder rules flip expressions into shifts and
  masks.
- A constant power-of-two divisor becomes `shr.u32` / `and.b32`; a power-of-two coefficient
  becomes `shl.b32`.
- Non-power-of-two constant division stays a real `div.u32` / `rem.u32`, but `_gcd_split` first
  factors a gcd out of the remainder so `r / d` reduces to `r₁ / (d/g)` — often a shift.
- Multiplications become `mad.lo.s32` fused multiply-adds during address assembly, so
  `base + idx·elemsize` is one instruction, not two.
- Loop-invariant pieces of an index are hoisted: `ival()` splits terms by scope depth, emits the
  invariant partial sum once, and reuses it on every iteration.

### 2. Aggressive common-subexpression elimination (CSE)

Caches are keyed on the *canonical forms*, so the same expression reached through different
syntax still hits:

- `("lin", Lin)`, `("atom", atom)`, `("pred", cond)` — computed integers, sub-atoms, predicates;
- `("saddr", buf, var)` / `("gaddr", buf, var)` — shared/global *address registers*, so a loop
  body that touches several buffers at the same index computes the address once;
- `("bytes", var, size)` — one `mul.wide` per (index shape, element size), shared across all
  buffers with the same strides;
- `("ld", buf, idx, width)` — loaded *values*, so a buffer read twice in the same block reads
  the hardware once.

The scope stack guarantees all of this is *sound*: caches are pushed per `Cond`/loop body and
invalidated on boundaries (branches, barriers, shared-memory writes), so nothing stale is ever
reused.

### 3. Loop codegen that ptxas likes

- **Full unrolling up to 1024 IR statements** for constant-trip innermost loops
  (`FULL_UNROLL = 1024`). Full unrolls turn every address into `base + constant-offset`, which
  lets ptxas keep many loads in flight and eliminates all loop overhead; the loop-index algebra
  consequently folds to constants. *Measured: the attention and long-sum kernels run 25–30%
  faster than nvcc's build of Emmy's reference CUDA.*
- **Partial unrolling 16/8/4/2** (largest divisor of the trip count) when the unrolled body
  stays under a 64-statement budget (`PARTIAL_UNROLL = 64`).
- **Canonical counted-loop shape:** the counter counts *up from 0* to `trips·step` and the loop
  variable is `start + counter`, instead of a counter compared against a per-thread bound.
  *Measured: 1.25× on a 65 536-element sum purely from this shape.*
- `StridedLoop` with static bounds is analysed: if every thread provably runs the same trip
  count it becomes a counted loop (e.g. `for v = tid; v < 1024; v += 128` runs exactly 8 times);
  a loop that runs 0-or-1 times becomes a single guarded branch. No loop is emitted that doesn't
  need to be.
- Bodies containing loops are never unrolled (`has_loops` check); nested loops own their own
  accumulators (`seed_accums` deliberately does not descend into them).

### 4. Memory: vectorisation, cache hints, swizzle

- `ld`/`st` of 2 or 4 **consecutive aligned f32** emit single `v2`/`v4` vectorised instructions
  (alignment checked algebraically via `divisible_by`).
- Read-only global buffers load through **`ld.global.nc`** (non-coherent, the read-only data
  cache), discovered by a pre-pass over the body for `Write` targets.
- f16/bf16 make two trips (load `b16`, `cvt` to f32; bf16 is the **top half of an f32**, so
  `cvt.u32.u16` then `shl 16`); stores convert back with `cvt.rn.f16.f32` /
  `cvt.rn.bf16.f32`.
- The constant part of an index becomes the immediate offset in `[reg+imm]`, so the common
  `gid·4 + round` addressing sits in one operand.
- **Shared memory** is laid out up front with 16-byte alignment padding. Static allocations up
  to 48 KiB become `.shared` arrays; anything bigger moves to the **dynamic pool**
  (`smem_pool[]`, per-buffer offsets, checked against `meta.json`'s `smem_bytes`).
- `cp.async` copies honour Emmy's swizzle codec (`B128/B64/B32 @ shift` ↔ XOR of bits 3.. of the
  index) and use `cg` for 16-byte copies to bypass L1, mirroring Emmy's helper; commit/wait
  groups give the double-buffering pipeline its structure.

### 5. Cross-thread combines tuned for arbitrary shapes

- **`tree_halve_stages`** produces a reduction schedule that covers *any* positive length, not
  just powers of two: at each stage the lower ⌈n/2⌉ lanes survive while the lower ⌊n/2⌋ combine
  with a partner ⌈n/2⌉ away, and an unpaired middle slot is carried forward. Nothing is ever
  dropped or duplicated. (The offline unit tests verify this for every length 1..512.)
- Warp shuffles are XOR butterflies limited to power-of-two widths in [1, 32], with
  `activemask.b32` when the tile's tail makes the warp partial.
- `TreeHalve` loads both operands, combines, stores, **restores the caller's IR values** after
  the tree, and broadcasts slot 0 — so the walker's SSA state stays correct across the barrier.

### 6. Precision-aware float lowering

Several ops deliberately *avoid* the hardware approximate units (which are fast but imprecise)
in favour of explicit polynomial or identity-based paths, while still emitting only a handful of
instructions each:

- `exp` = `2ⁿ·2ᶠ` via the 1.5·2²³ rounding magic + the hardware `ex2.approx` unit, with `log₂e`
  split hi+lo for full precision — **9 instructions, ~2 ulp**.
- `log` extracts the exponent and mantissa with bit ops, evaluates `log m = 2·atanh(s)` as an
  odd series (Horner, fma-fused), and *predicates between* that fast path and the hardware
  `lg2.approx` path for zero/negative/∞/NaN/subnormals.
- `tanh`: an offline least-squares odd polynomial `x + x³P(x²)` on `|x| < 0.55`
  (**6e-8 relative**) beats `tanh.approx` (~5e-4) while staying cheap; the large-argument path
  is `1 − 2/(e^{2|x|}+1)` with `copysign` restoring the sign.
- `erf`: `x·P(x²)` on `|x| < 1` (**1.7e-7 relative**), Abramowitz–Stegun 7.1.26 above.
- `pow` special-cases `b ∈ {2, 1, ½, 0}` to exact results and handles negative bases for
  integral exponents; `sin`/`cos`/`tan` reduce by 2π split hi+lo before the hardware
  `sin.approx` / `cos.approx` units (which are only accurate near 0).
- Constants fold: `max(x, 1)`, `x · 1.0`, and pure-literal expressions collapse before any PTX
  is emitted; reduction identities encode `±inf` correctly so `min`/`max` reductions seed with
  the right extreme value.

## Precision strategy

The target is **CUDA-compatible numerics with a few-ulp budget for transcendentals**, since
ptxas may fuse `add`/`sub`/`mul` into `fma` exactly like nvcc does (so the compiler doesn't
fight it), division and sqrt stay IEEE round-to-nearest (`div.rn` / `sqrt.rn`), and every
non-arithmetic function is lowered explicitly:

| Op | Lowering | Accuracy |
|----|----------|----------|
| `exp` | magic-add rounding + `ex2.approx`, hi/lo `log₂e` | ~2 ulp, 9 instr |
| `log` | bit-trick exponent, `atanh` odd series, `lg2` fallback | few ulp |
| `tanh` | least-squares poly + exp form, `copysign` | ~6e-8 rel |
| `erf` | least-squares poly + A&S 7.1.26 | ~1.7e-7 rel |
| `sin`/`cos` | 2π hi/lo reduction + hardware approx | good near 0 |
| `pow` | exp(b·log|a|), exact `b ∈ {2,1,½,0}`, negative-base handling | a few ulp |
| `div` / `sqrt` | `div.rn` / `sqrt.rn` | IEEE round-to-nearest |
| fma-eligible | plain `add/sub/mul`, ptxas fuses | same as nvcc builds |

The polynomial coefficients (`TANH_POLY`, `ERF_POLY`) are least-squares fits that were checked
offline in f32; the error numbers in the table are the *offline measured* bounds recorded in the
source comments.

## Validation

The bundle around this compiler contains an offline validation harness (no GPU required):

- **`test_compile_candidate.py`** — 7 unit tests that import the compiler without Emmy or a GPU:
  `tree_halve_stages` covers every reduction length 1..512 and preserves the sum and final
  length; `minimum`/`maximum` identities encode ±∞ through `fimm`/`imm_value` round-trips;
  integer lexical bindings restore/remove shadowed names; and accumulator seeding walks into
  conditionals but leaves nested loops to their own initialisers.
- **`benchmark_emmy.py`** — runs a baseline and a candidate compiler through the official
  `submit.py` on all 15 open problems, parsing PASS/FAIL and kernel timings, and writes
  `benchmark_emmy_results.json` / `.csv`. It needs a checkout with a GPU or the hosted runner.
- **`README_validation.txt`** — documents what was verified in this environment (syntax, the 7
  unit tests) and what was *not* (the 15 official problems, GPU timings).

Honesty note: this bundle's environment cannot reach the runner (no network / no `ptxas` /
`nvcc`), so the **measured** performance numbers quoted above come from the compiler author's
own runner runs (recorded in `compile.py`'s comments) and were **not re-verified here**. Treat
them as the compiler's self-reported figures until the official run passes.

## Usage

```bash
# compile one kernel IR to PTX
python compile.py path/to/kernel.ir.json > out.ptx

# full end-to-end check against a problem's expected outputs (uses the hosted runner)
python submit.py p00_scalar_add --compiler compile.py
```

CLI contract (matches the hackathon's evaluation contract):

- exactly one argument: the path to a `*.ir.json` file;
- PTX goes to **stdout only** — any error or warning is printed to stderr and the exit code is
  non-zero, so piping is safe;
- the output is runnable by `ptxas` as-is: it declares `.version 8.0`, `.target sm_89`,
  `.address_size 64`, a `.visible .entry` with `.maxntid` from the launch block, and typed
  `.reg` declarations at the top.

## Scope and limitations

- **Static shapes only** — symbolic extents raise `NotImplementedError` by design.
- **f32 is the arithmetic currency.** Other element types are converted at memory boundaries
  (`f16`, `bf16`, `f64`, `i32`, `i64`); scalar math runs in f32 like the reference CUDA.
- **A fixed set of ops** — anything not in `FloatMath.ALIASES` / the `op_*` handlers raises
  `NotImplementedError` with the op name, so new problems fail loudly rather than silently.
- **One `KernelOp` per IR file** is compiled (the first one found), which matches the problem
  format.
- A partial `Tile` that uses a CTA/warp collective is rejected (tail threads would skip the
  collective), and `WarpShuffle` is limited to power-of-two widths in [1, 32].

## Repo map

```
emmy_compiler_validation_bundle/
├── README_compiler.md          ← this document
├── README_validation.txt       what this bundle verified (and what it did not)
├── test_compile_candidate.py   offline unit tests (no GPU / Emmy package)
├── benchmark_emmy.py           baseline-vs-candidate harness across all 15 problems
└── emmy-hackathon-2026/
    ├── compile.py              ← the compiler this document describes
    ├── submit.py               official compile + run + check driver
    ├── problems/               the 15 open problems (IR, meta.json, references, expected.npz)
    ├── examples/               hand-written PTX annotated back to the IR
    ├── tools/                  how the problems were generated
    ├── TASK.md / MVP.md        the task and the minimum bar
    └── docs/API.md             hosted-runner API
```