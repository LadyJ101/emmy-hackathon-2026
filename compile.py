#!/usr/bin/env python3
"""Emmy Kernel IR -> PTX compiler.

    python compile.py path/to/kernel.ir.json > out.ptx

The file reads top to bottom in the order the compiler works:

  1. PTX helpers      f32 immediates and a virtual-register allocator.
  2. Index algebra    integer index math kept symbolic as  const + sum(coef * atom)  with a value
                      range for every atom, so thread-id decodes, div/mod by constants and address
                      offsets simplify at compile time instead of costing instructions at run time.
  3. Float math       one lowering per elementwise op (add, exp, tanh, ...), all computed in f32.
  4. KernelCompiler   walks the KernelOp statement tree in order and emits PTX for each statement
                      (Tile, loops, loads/stores, reductions, shuffles, cp.async, ...).
  5. Driver           loads the IR with Emmy's Graph, reads launch info from meta.json, prints PTX.

Launch geometry and parameter order come from the meta.json next to the IR (the task's contract).
We emit plain virtual-register code; ptxas does register allocation and instruction scheduling.
"""

from __future__ import annotations

import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PTX_VERSION = "8.0"              # sm_89 needs PTX ISA 7.8 or newer
PTX_TARGET = "sm_89"             # problems are scheduled for sm_89; newer GPUs run it unchanged
STATIC_SMEM_LIMIT = 48 * 1024    # bigger shared arrays go to the dynamic pool, exactly like Emmy
INT_MIN, INT_MAX = -(2**31), 2**31 - 1

# Unrolling budgets, counted in IR statements. Full unrolling turns every address into
# base + constant offset and lets ptxas keep many loads in flight; on the runner it makes the
# attention and long-sum kernels 25-30% faster than nvcc's build of Emmy's reference CUDA.
FULL_UNROLL = 1024               # an innermost constant-trip loop is fully unrolled up to this size
PARTIAL_UNROLL = 64              # otherwise unroll by 16/8/4/2 while the copies stay this small


# =============================================================================================
# 1. PTX helpers
# =============================================================================================

def fimm(x: float) -> str:
    """An f32 PTX immediate with the exact bit pattern, e.g. 1.0 -> 0f3F800000."""
    with np.errstate(over="ignore"):
        return "0f%08X" % int(np.array(x, dtype=np.float32).view(np.uint32))


def is_imm(v: str) -> bool:
    return v.startswith("0f")


def imm_value(v: str) -> np.float32:
    return np.array(int(v[2:], 16), dtype=np.uint32).view(np.float32)[()]


def is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def tree_halve_stages(length: int) -> list[tuple[int, int]]:
    """Return (active_count, partner_offset) stages covering every reduction input once.

    The schedule supports arbitrary positive lengths. At each stage, the lower
    ceil(n/2) slots survive; the lower floor(n/2) slots combine with a partner
    offset by ceil(n/2). An unpaired middle slot is carried to the next stage.
    """
    if length < 1:
        raise ValueError(f"reduction length must be positive, got {length}")
    stages = []
    remaining = length
    while remaining > 1:
        partner_offset = (remaining + 1) // 2
        active_count = remaining - partner_offset
        stages.append((active_count, partner_offset))
        remaining = partner_offset
    return stages


def c_div(a: int, b: int) -> int:
    """C integer division (truncates toward zero), which is what Emmy's CUDA renders."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def c_rem(a: int, b: int) -> int:
    return a - b * c_div(a, b)


class Registers:
    """Hands out fresh virtual registers per PTX type and declares them at the top."""

    PREFIX = {"pred": "%p", "b16": "%h", "b32": "%r", "b64": "%rd", "f32": "%f", "f64": "%fd"}

    def __init__(self):
        self.count = dict.fromkeys(self.PREFIX, 0)

    def new(self, kind: str) -> str:
        self.count[kind] += 1
        return f"{self.PREFIX[kind]}{self.count[kind]}"

    def declarations(self) -> list[str]:
        return [f"\t.reg .{kind} {self.PREFIX[kind]}<{n + 1}>;" for kind, n in self.count.items() if n]


# Buffer element types: PTX type used for loads/stores, and the byte size.
DTYPES = {"f32": ("f32", 4), "f16": ("b16", 2), "bf16": ("b16", 2), "f64": ("f64", 8),
          "i32": ("s32", 4), "i64": ("s64", 8)}
SMEM_CTYPES = {"float": "f32", "__half": "f16", "half": "f16", "__nv_bfloat16": "bf16",
               "double": "f64", "int": "i32", "long long": "i64"}


# =============================================================================================
# 2. Index algebra
# =============================================================================================

class Lin:
    """An integer expression in canonical form: const + sum(coef * atom).

    Atoms are hashable tuples for the parts the affine algebra cannot see through:
      ("reg", r)                  an integer register (thread id, block id, loop counter)
      ("div", L, d) ("mod", L, d) floor division / remainder of a non-negative L by a constant
      ("sdiv", L, M) ("srem", L, M)  C truncating division / remainder (sign unknown or M not const)
      ("mul", L, M)               product of two non-constant expressions
      ("min", L, M) ("max", L, M) minimum / maximum
      ("sel", cond, L, M)         cond ? L : M
      ("bool", cond)              1 if cond else 0
      ("swz", L, shift, mask)     Emmy's XOR shared-memory swizzle of L
    Equal expressions compare and hash equal, which is what the CSE caches key on.
    """

    __slots__ = ("terms", "const", "_hash", "_repr")

    def __init__(self, terms=(), const: int = 0):
        merged: dict = {}
        for atom, coef in (terms.items() if isinstance(terms, dict) else terms):
            merged[atom] = merged.get(atom, 0) + coef
        self.terms = tuple(sorted(((a, c) for a, c in merged.items() if c), key=lambda t: repr(t[0])))
        self.const = int(const)
        self._hash = hash((self.terms, self.const))
        self._repr = None

    @staticmethod
    def k(n: int) -> "Lin":
        return Lin((), n)

    @staticmethod
    def atom(a: tuple, coef: int = 1) -> "Lin":
        return Lin(((a, coef),), 0)

    def is_const(self) -> bool:
        return not self.terms

    def __add__(self, other) -> "Lin":
        if isinstance(other, int):
            return Lin(self.terms, self.const + other)
        return Lin(self.terms + other.terms, self.const + other.const)

    def __sub__(self, other) -> "Lin":
        return self + (-other if isinstance(other, int) else other.scale(-1))

    def scale(self, k: int) -> "Lin":
        return Lin(tuple((a, c * k) for a, c in self.terms), self.const * k)

    def split_const(self) -> tuple["Lin", int]:
        return Lin(self.terms, 0), self.const

    def divisible_by(self, w: int) -> bool:
        return self.const % w == 0 and all(c % w == 0 for _, c in self.terms)

    def __eq__(self, other) -> bool:
        return isinstance(other, Lin) and self.terms == other.terms and self.const == other.const

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        if self._repr is None:
            parts = [f"{c}*{a!r}" for a, c in self.terms]
            if self.const or not parts:
                parts.append(str(self.const))
            self._repr = "L(" + " + ".join(parts) + ")"
        return self._repr


class IndexAlgebra:
    """Builds Lin expressions and simplifies them with value ranges.

    Every shape is static, so every register has a known range (thread id in [0, block), block id
    in [0, grid), loop counter in [start, end)). With ranges, rules such as
        floor((d*q + r) / d) = q + floor(r / d)          and   (d*q + r) mod d = r mod d
        floor(r / d) = 0, r mod d = r                     when 0 <= r < d
    turn a Tile decode like  a1 = (gid / 32) % 16  with gid = 512*ctaid + tid  into  tid >> 5.
    Conditions are tuples too: ("cmp", op, L, M), ("fcmp", op, fa, fb), ("and"|"or", c1, c2),
    ("not", c); a condition the ranges decide becomes a plain Python bool.
    """

    def __init__(self):
        self.reg_range: dict[str, tuple[int, int]] = {}
        self._ranges: dict[tuple, tuple[int, int]] = {}

    # ---- ranges ------------------------------------------------------------------------------
    def range(self, lin: Lin) -> tuple[int, int]:
        lo = hi = lin.const
        for atom, c in lin.terms:
            alo, ahi = self.atom_range(atom)
            if c >= 0:
                lo, hi = lo + c * alo, hi + c * ahi
            else:
                lo, hi = lo + c * ahi, hi + c * alo
        return lo, hi

    def atom_range(self, atom: tuple) -> tuple[int, int]:
        r = self._ranges.get(atom)
        if r is None:
            r = self._ranges[atom] = self._atom_range(atom)
        return r

    def _atom_range(self, atom: tuple) -> tuple[int, int]:
        kind = atom[0]
        if kind == "reg":
            return self.reg_range.get(atom[1], (INT_MIN, INT_MAX))
        if kind == "div":
            lo, hi = self.range(atom[1])
            return lo // atom[2], hi // atom[2]
        if kind == "mod":
            return 0, min(atom[2] - 1, self.range(atom[1])[1])
        if kind in ("sdiv", "srem"):
            (alo, ahi), (blo, bhi) = self.range(atom[1]), self.range(atom[2])
            if blo <= 0:
                return INT_MIN, INT_MAX
            if kind == "sdiv":
                qs = [c_div(x, y) for x in (alo, ahi) for y in (blo, bhi)]
                return min(qs), max(qs)
            return (-(bhi - 1) if alo < 0 else 0), ((bhi - 1) if ahi > 0 else 0)
        if kind == "mul":
            (alo, ahi), (blo, bhi) = self.range(atom[1]), self.range(atom[2])
            ps = [alo * blo, alo * bhi, ahi * blo, ahi * bhi]
            return min(ps), max(ps)
        if kind in ("min", "max"):
            (alo, ahi), (blo, bhi) = self.range(atom[1]), self.range(atom[2])
            f = min if kind == "min" else max
            return f(alo, blo), f(ahi, bhi)
        if kind == "sel":
            (alo, ahi), (blo, bhi) = self.range(atom[2]), self.range(atom[3])
            return min(alo, blo), max(ahi, bhi)
        if kind == "bool":
            return 0, 1
        if kind == "swz":                       # the XOR only flips bits 3.. of the index
            lo, hi = self.range(atom[1])
            flip = atom[3] << 3
            return max(0, lo - flip), hi + flip
        raise ValueError(f"unknown atom {kind}")

    # ---- arithmetic --------------------------------------------------------------------------
    def mul(self, a: Lin, b: Lin) -> Lin:
        if a.is_const():
            return b.scale(a.const)
        if b.is_const():
            return a.scale(b.const)
        x, y = sorted((a, b), key=repr)
        return Lin.atom(("mul", x, y))

    def _pull(self, a: Lin, d: int) -> tuple[Lin, Lin]:
        """Split a = d*q + r, moving every whole multiple of d into q."""
        qt = {at: c // d for at, c in a.terms if c % d == 0}
        rt = {at: c for at, c in a.terms if c % d}
        return Lin(qt, a.const // d), Lin(rt, a.const % d)

    def _gcd_split(self, r: Lin, d: int):
        """Find g | d with r = g*r1 + s, 0 <= s < g; then r / d == r1 / (d/g) (floor division)."""
        cands = {math.gcd(c, d) for _, c in r.terms} | {math.gcd(r.const, d)}
        for g in sorted(cands - {1, d}, reverse=True):
            r1 = Lin({at: c // g for at, c in r.terms if c % g == 0}, r.const // g)
            s = Lin({at: c for at, c in r.terms if c % g}, r.const % g)
            slo, shi = self.range(s)
            if slo >= 0 and shi < g and self.range(r1)[0] >= 0:
                return g, r1, s
        return None

    def div(self, a: Lin, b: Lin) -> Lin:
        if a.is_const() and b.is_const() and b.const:
            return Lin.k(c_div(a.const, b.const))
        if not b.is_const() or b.const <= 0:
            return Lin.atom(("sdiv", a, b))
        d = b.const
        if d == 1:
            return a
        lo, hi = self.range(a)
        if lo < 0:
            return Lin.atom(("sdiv", a, b))
        if hi < d:
            return Lin.k(0)
        q, r = self._pull(a, d)
        rlo, rhi = self.range(r)
        if rlo < 0:
            return Lin.atom(("div", a, d))
        if rhi < d:
            return q
        split = self._gcd_split(r, d)
        if split:
            g, r1, _ = split
            return q + self.div(r1, Lin.k(d // g))
        return q + Lin.atom(("div", r, d))

    def mod(self, a: Lin, b: Lin) -> Lin:
        if a.is_const() and b.is_const() and b.const:
            return Lin.k(c_rem(a.const, b.const))
        if not b.is_const() or b.const <= 0:
            return Lin.atom(("srem", a, b))
        d = b.const
        if d == 1:
            return Lin.k(0)
        lo, hi = self.range(a)
        if lo < 0:
            return Lin.atom(("srem", a, b))
        if hi < d:
            return a
        _, r = self._pull(a, d)
        rlo, rhi = self.range(r)
        if rlo < 0:
            return Lin.atom(("mod", a, d))
        if rhi < d:
            return r
        split = self._gcd_split(r, d)
        if split:
            g, r1, s = split
            return self.mod(r1, Lin.k(d // g)).scale(g) + s
        return Lin.atom(("mod", r, d))

    def minmax(self, kind: str, a: Lin, b: Lin) -> Lin:
        (alo, ahi), (blo, bhi) = self.range(a), self.range(b)
        if kind == "min":
            if ahi <= blo:
                return a
            if bhi <= alo:
                return b
        else:
            if alo >= bhi:
                return a
            if blo >= ahi:
                return b
        x, y = sorted((a, b), key=repr)
        return Lin.atom((kind, x, y))

    def select(self, cond, a: Lin, b: Lin) -> Lin:
        if isinstance(cond, bool):
            return a if cond else b
        if a == b:
            return a
        return Lin.atom(("sel", cond, a, b))

    def boolean(self, cond) -> Lin:
        return Lin.k(int(cond)) if isinstance(cond, bool) else Lin.atom(("bool", cond))

    def swizzle(self, a: Lin, shift: int, mask: int) -> Lin:
        if a.is_const():
            return Lin.k(a.const ^ (((a.const >> shift) & mask) << 3))
        return Lin.atom(("swz", a, shift, mask))

    # ---- conditions --------------------------------------------------------------------------
    def cmp(self, op: str, a: Lin, b: Lin):
        lo, hi = self.range(a - b)
        always, never = {
            "<": (hi < 0, lo >= 0), "<=": (hi <= 0, lo > 0),
            ">": (lo > 0, hi <= 0), ">=": (lo >= 0, hi < 0),
            "==": (lo == hi == 0, lo > 0 or hi < 0), "!=": (lo > 0 or hi < 0, lo == hi == 0),
        }[op]
        if always:
            return True
        if never:
            return False
        return ("cmp", op, a, b)

    @staticmethod
    def conj(c1, c2):
        if c1 is False or c2 is False:
            return False
        if c1 is True:
            return c2
        return c1 if c2 is True else ("and", c1, c2)

    @staticmethod
    def disj(c1, c2):
        if c1 is True or c2 is True:
            return True
        if c1 is False:
            return c2
        return c1 if c2 is False else ("or", c1, c2)


# =============================================================================================
# 3. Float math: one lowering per elementwise op
# =============================================================================================

def _split(x: float) -> tuple[float, float]:
    """x as an f32 pair hi + lo, so fma(a, hi, fma(a, lo, b)) keeps ~48 bits of x."""
    hi = float(np.float32(x))
    return hi, float(np.float32(x - hi))


LOG2E_HI, LOG2E_LO = _split(1.4426950408889634)
LN2 = 0.6931471805599453
LN2_HI, LN2_LO = _split(LN2)
TWO_PI_HI, TWO_PI_LO = _split(2 * math.pi)
FLT_MIN = 1.1754943508222875e-38
MAGIC = 12582912.0               # 1.5 * 2^23: adding it rounds an f32 to an integer
ONE, ZERO = fimm(1.0), fimm(0.0)
# Least-squares fits checked offline in f32: tanh(x) = x + x^3*P(x^2) on |x| < 0.55 (6e-8 rel),
# erf(x) = x*P(x^2) on |x| < 1 (1.7e-7 rel). Coefficients run from the constant term upward.
TANH_POLY = (-0.3333333432674408, 0.13333363831043243, -0.05397460237145424,
             0.021907363086938858, -0.008854896761476994, 0.0029535582289099693)
ERF_POLY = (1.128379225730896, -0.37612637877464294, 0.11283782869577408, -0.02686544507741928,
            0.005221074912697077, -0.00084847112884745, 0.00011270544928265736,
            -9.680137736722827e-06)
# Abramowitz & Stegun 7.1.26 for |x| >= 1: erf = 1 - t*P(t)*exp(-x^2), t = 1/(1 + p*x).
ERF_AS_P = 0.3275911
ERF_AS = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)

IDENTITY = {"add": 0.0, "multiply": 1.0, "maximum": -math.inf, "minimum": math.inf}


class FloatMath:
    """Lowers elementwise ops on f32 values. A value is a register name or an f32 immediate.

    Precision follows CUDA's defaults: add/sub/mul may be fused into fma by ptxas (nvcc does the
    same), division and sqrt are IEEE round-to-nearest, and transcendentals are accurate to a few
    ulp rather than the raw hardware approximations (tanh.approx alone is only good to ~5e-4).
    """

    ALIASES = {"true_divide": "divide", "sub": "subtract", "mul": "multiply", "div": "divide",
               "power": "pow", "absolute": "abs", "fabs": "abs", "neg": "negative",
               "max": "maximum", "amax": "maximum", "fmax": "maximum", "min": "minimum",
               "amin": "minimum", "fmin": "minimum", "sum": "add", "prod": "multiply",
               "rint": "round", "fmod": "mod", "logical_not": "bitwise_not",
               "logical_and": "bitwise_and", "logical_or": "bitwise_or", "pad": "copy"}
    # Ops on constants are evaluated here instead (f32 arithmetic, or f64 math rounded to f32).
    FOLD = {"add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b,
            "maximum": max, "minimum": min, "negative": lambda a: -a, "abs": abs,
            "square": lambda a: a * a, "reciprocal": lambda a: np.float32(1) / a,
            "relu": lambda a: max(a, np.float32(0)), "copy": lambda a: a,
            "exp": lambda a: np.exp(np.float64(a)), "log": lambda a: np.log(np.float64(a)),
            "sqrt": lambda a: np.sqrt(np.float64(a)), "rsqrt": lambda a: 1 / np.sqrt(np.float64(a)),
            "tanh": lambda a: np.tanh(np.float64(a)), "sigmoid": lambda a: 1 / (1 + np.exp(-np.float64(a))),
            "pow": lambda a, b: np.power(np.float64(a), np.float64(b))}
    SET_CMP = {"equal": "eq", "not_equal": "neu", "greater": "gt", "less": "lt",
               "greater_equal": "ge", "less_equal": "le"}

    def __init__(self, kc: "KernelCompiler"):
        self.kc = kc

    @classmethod
    def canonical(cls, op: str) -> str:
        return cls.ALIASES.get(op, op)

    def apply(self, op: str, args: list[str]) -> str:
        op = self.canonical(op)
        if op in self.FOLD and all(is_imm(a) for a in args):     # constant folding, IEEE f32
            with np.errstate(all="ignore"):
                return fimm(self.FOLD[op](*[imm_value(a) for a in args]))
        if op == "multiply" and ONE in args:                     # x * 1.0 == x exactly
            return args[1] if args[0] == ONE else args[0]
        if op in self.SET_CMP:
            return self.f(f"set.{self.SET_CMP[op]}.f32.f32", *args)   # 1.0 or 0.0
        handler = getattr(self, "op_" + op, None)
        if handler is None:
            raise NotImplementedError(f"elementwise op {op!r} is not supported")
        return handler(*args)

    # ---- emit helpers ------------------------------------------------------------------------
    def f(self, instr: str, *srcs) -> str:
        dst = self.kc.regs.new("f32")
        self.kc.emit(f"{instr} {dst}, {', '.join(str(s) for s in srcs)}")
        return dst

    def setp(self, cmp: str, a: str, b: str) -> str:
        p = self.kc.regs.new("pred")
        self.kc.emit(f"setp.{cmp}.f32 {p}, {a}, {b}")
        return p

    def selp(self, a: str, b: str, p: str) -> str:
        return self.f("selp.f32", a, b, p)

    def reg(self, v: str) -> str:
        """Some instructions want a register operand: move an immediate into one."""
        return self.f("mov.f32", v) if is_imm(v) else v

    def poly(self, z: str, coeffs) -> str:
        """Horner evaluation of coeffs[0] + coeffs[1]*z + ... with fma."""
        acc = self.f("fma.rn.f32", z, fimm(coeffs[-1]), fimm(coeffs[-2]))
        for c in reversed(coeffs[:-2]):
            acc = self.f("fma.rn.f32", acc, z, fimm(c))
        return acc

    # ---- arithmetic --------------------------------------------------------------------------
    def op_add(self, a, b): return self.f("add.f32", a, b)
    def op_subtract(self, a, b): return self.f("sub.f32", a, b)
    def op_multiply(self, a, b): return self.f("mul.f32", a, b)
    def op_divide(self, a, b): return self.f("div.rn.f32", a, b)
    def op_maximum(self, a, b): return self.f("max.f32", a, b)
    def op_minimum(self, a, b): return self.f("min.f32", a, b)
    def op_negative(self, a): return self.f("neg.f32", a)
    def op_abs(self, a): return self.f("abs.f32", a)
    def op_square(self, a): return self.f("mul.f32", a, a)
    def op_reciprocal(self, a): return self.f("rcp.rn.f32", a)
    def op_relu(self, a): return self.f("max.f32", a, ZERO)
    def op_copy(self, a): return a
    def op_sqrt(self, a): return self.f("sqrt.rn.f32", a)
    def op_rsqrt(self, a): return self.f("rsqrt.approx.f32", a)
    def op_floor(self, a): return self.f("cvt.rmi.f32.f32", a)
    def op_ceil(self, a): return self.f("cvt.rpi.f32.f32", a)
    def op_trunc(self, a): return self.f("cvt.rzi.f32.f32", a)
    def op_round(self, a): return self.f("cvt.rni.f32.f32", a)

    def op_sign(self, a):
        return self.f("sub.f32", self.f("set.gt.f32.f32", a, ZERO), self.f("set.lt.f32.f32", a, ZERO))

    def op_mod(self, a, b):                       # C fmod: a - trunc(a/b)*b
        t = self.f("cvt.rzi.f32.f32", self.f("div.rn.f32", a, b))
        return self.f("fma.rn.f32", self.f("neg.f32", t), b, a)

    def op_remainder(self, a, b):                 # Python/torch remainder: result has b's sign
        r = self.op_mod(a, b)
        differ = self.setp("lt", self.f("mul.f32", r, b), ZERO)
        return self.selp(self.f("add.f32", r, b), r, differ)

    # ---- comparisons and masks (float 1.0 / 0.0, like the traced mask convention) -----------
    def op_where(self, c, a, b):
        return self.selp(a, b, self.setp("neu", c, ZERO))

    def _logical(self, a, b, combine: str):
        p = self.kc.regs.new("pred")
        self.kc.emit(f"{combine}.pred {p}, {self.setp('neu', a, ZERO)}, {self.setp('neu', b, ZERO)}")
        return self.selp(ONE, ZERO, p)

    def op_bitwise_and(self, a, b): return self._logical(a, b, "and")
    def op_bitwise_or(self, a, b): return self._logical(a, b, "or")
    def op_bitwise_not(self, a): return self.f("set.eq.f32.f32", a, ZERO)

    # ---- exponentials and logarithms ---------------------------------------------------------
    def _split_pow2(self, j: str):
        """j holds MAGIC + 127 + n for an integer n. Clamp n to [-126, 127] so 2^n is a normal
        float, and return (-n as f32, 2^n): the low mantissa bits of j are n's biased exponent."""
        j = self.f("min.f32", self.f("max.f32", j, fimm(MAGIC + 1)), fimm(MAGIC + 254))
        neg_n = self.f("sub.f32", fimm(MAGIC + 127), j)                       # exact
        scale = self.kc.iop("shl.b32", self.kc.iop("mov.b32", j), 23)         # bits of 2^n
        return neg_n, scale

    def op_exp(self, x):
        # exp(x) = 2^n * 2^f, n = round(x*log2(e)). Adding 1.5*2^23 rounds to an integer in the
        # low mantissa bits; f = x*log2(e) - n uses log2(e) split hi+lo so it keeps full
        # precision, and 2^f comes from the hardware ex2 unit, which also produces inf / 0 when
        # the clamped n is not enough. 9 instructions, ~2 ulp.
        neg_n, scale = self._split_pow2(self.f("fma.rn.f32", x, fimm(LOG2E_HI), fimm(MAGIC + 127)))
        fr = self.f("fma.rn.f32", x, fimm(LOG2E_LO), self.f("fma.rn.f32", x, fimm(LOG2E_HI), neg_n))
        return self.f("mul.f32", self.f("ex2.approx.ftz.f32", fr), scale)

    def op_exp2(self, x):
        neg_n, scale = self._split_pow2(self.f("add.f32", x, fimm(MAGIC + 127)))
        return self.f("mul.f32", self.f("ex2.approx.ftz.f32", self.f("add.f32", x, neg_n)), scale)

    def op_exp_fast(self, x):                     # Emmy's FAST_EXP spelling: plain hardware ex2
        return self.f("ex2.approx.ftz.f32", self.f("mul.f32", x, fimm(LOG2E_HI)))

    def op_expm1(self, x):
        return self.f("sub.f32", self.op_exp(x), ONE)

    def op_log(self, x):
        # x = m * 2^e with m in [sqrt(1/2), sqrt(2)); log(m) = 2*atanh(s), s = (m-1)/(m+1),
        # |s| < 0.172, as an odd series; log(x) = e*ln2 + log(m). Zero, negative, inf, NaN and
        # subnormal inputs take the hardware lg2 path instead (it handles them correctly).
        kc = self.kc
        x = self.reg(x)
        bits = kc.iop("mov.b32", x)
        e = kc.iop("shr.s32", kc.iop("sub.s32", bits, 0x3F3504F3), 23)
        m = self.f("mov.b32", kc.iop("sub.s32", bits, kc.iop("shl.b32", e, 23)))
        s = self.f("div.full.f32", self.f("sub.f32", m, ONE), self.f("add.f32", m, ONE))
        z = self.f("mul.f32", s, s)
        series = self.poly(z, (2 / 3, 2 / 5, 2 / 7, 2 / 9))
        log_m = self.f("fma.rn.f32", self.f("mul.f32", s, z), series, self.f("add.f32", s, s))
        ef = self.f("cvt.rn.f32.s32", e)
        r = self.f("fma.rn.f32", ef, fimm(LN2_HI), self.f("fma.rn.f32", ef, fimm(LN2_LO), log_m))
        hw = self.f("mul.f32", self.f("lg2.approx.f32", x), fimm(LN2))
        ok = kc.regs.new("pred")
        kc.emit(f"and.pred {ok}, {self.setp('ge', x, fimm(FLT_MIN))}, {self.setp('lt', x, fimm(math.inf))}")
        return self.selp(r, hw, ok)

    def op_log2(self, x): return self.f("mul.f32", self.op_log(x), fimm(1 / LN2))
    def op_log10(self, x): return self.f("mul.f32", self.op_log(x), fimm(1 / math.log(10)))
    def op_log1p(self, x): return self.op_log(self.f("add.f32", x, ONE))

    def op_pow(self, a, b):
        # |a|^b = exp(b*log|a|); a < 0 is defined for integral b (odd b flips the sign).
        # b in {0, 0.5, 1, 2} gets the exact result numpy also returns.
        a, b = self.reg(a), self.reg(b)
        g = self.op_exp(self.f("mul.f32", b, self.op_log(self.f("abs.f32", a))))
        is_int = self.setp("eq", self.f("cvt.rzi.f32.f32", b), b)
        half = self.f("mul.f32", b, fimm(0.5))
        odd = self.setp("neu", self.f("cvt.rzi.f32.f32", half), half)
        neg_base = self.selp(self.selp(self.f("neg.f32", g), g, odd), fimm(math.nan), is_int)
        r = self.selp(neg_base, g, self.setp("lt", a, ZERO))
        for exp_val, exact in ((2.0, lambda: self.f("mul.f32", a, a)), (1.0, lambda: a),
                               (0.5, lambda: self.f("sqrt.rn.f32", a)), (0.0, lambda: ONE)):
            r = self.selp(exact(), r, self.setp("eq", b, fimm(exp_val)))
        return r

    # ---- activations -------------------------------------------------------------------------
    def op_sigmoid(self, x):
        return self.f("rcp.rn.f32", self.f("add.f32", self.op_exp(self.f("neg.f32", x)), ONE))

    def op_silu(self, x):
        return self.f("div.rn.f32", x, self.f("add.f32", self.op_exp(self.f("neg.f32", x)), ONE))

    def op_softplus(self, x):                     # max(x, 0) + log(1 + exp(-|x|)), never overflows
        tail = self.op_log(self.f("add.f32", self.op_exp(self.f("neg.f32", self.f("abs.f32", x))), ONE))
        return self.f("add.f32", self.f("max.f32", x, ZERO), tail)

    def op_tanh(self, x):
        # |x| < 0.55: x + x^3*P(x^2);  otherwise 1 - 2/(exp(2|x|) + 1) with the sign put back.
        x = self.reg(x)
        ax = self.f("abs.f32", x)
        e = self.op_exp(self.f("add.f32", ax, ax))
        big = self.f("sub.f32", ONE, self.f("div.full.f32", fimm(2.0), self.f("add.f32", e, ONE)))
        z = self.f("mul.f32", x, x)
        small = self.f("fma.rn.f32", self.f("mul.f32", x, z), self.poly(z, TANH_POLY), x)
        return self.selp(small, self.f("copysign.f32", x, big), self.setp("lt", ax, fimm(0.55)))

    def op_erf(self, x):
        x = self.reg(x)
        ax = self.f("abs.f32", x)
        z = self.f("mul.f32", x, x)
        small = self.f("mul.f32", x, self.poly(z, ERF_POLY))
        t = self.f("div.full.f32", ONE, self.f("fma.rn.f32", ax, fimm(ERF_AS_P), ONE))
        tail = self.f("mul.f32", self.f("mul.f32", t, self.poly(t, ERF_AS)), self.op_exp(self.f("neg.f32", z)))
        big = self.f("copysign.f32", x, self.f("sub.f32", ONE, tail))
        return self.selp(small, big, self.setp("lt", ax, ONE))

    def op_gelu(self, x):                         # 0.5*x*(1 + erf(x/sqrt(2)))
        e = self.op_erf(self.f("mul.f32", x, fimm(1 / math.sqrt(2))))
        return self.f("mul.f32", self.f("mul.f32", x, fimm(0.5)), self.f("add.f32", e, ONE))

    def op_gelu_tanh(self, x):                    # 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715 x^3)))
        x3 = self.f("mul.f32", self.f("mul.f32", x, x), x)
        inner = self.f("mul.f32", self.f("fma.rn.f32", x3, fimm(0.044715), x), fimm(math.sqrt(2 / math.pi)))
        th = self.op_tanh(inner)
        return self.f("mul.f32", self.f("mul.f32", x, fimm(0.5)), self.f("add.f32", th, ONE))

    # ---- trigonometry ------------------------------------------------------------------------
    def _reduce_2pi(self, x):                     # x - k*2pi with 2pi split hi+lo
        neg_k = self.f("neg.f32", self.f("cvt.rni.f32.f32", self.f("mul.f32", x, fimm(1 / (2 * math.pi)))))
        r = self.f("fma.rn.f32", neg_k, fimm(TWO_PI_HI), x)
        return self.f("fma.rn.f32", neg_k, fimm(TWO_PI_LO), r)

    def op_sin(self, x): return self.f("sin.approx.f32", self._reduce_2pi(x))
    def op_cos(self, x): return self.f("cos.approx.f32", self._reduce_2pi(x))
    def op_tan(self, x):
        r = self._reduce_2pi(x)
        return self.f("div.rn.f32", self.f("sin.approx.f32", r), self.f("cos.approx.f32", r))


# =============================================================================================
# 4. The kernel compiler
# =============================================================================================

@dataclass
class Launch:
    entry: str
    arg_order: list[str]
    grid: list[int]
    block: list[int]
    smem_bytes: int = 0
    scalars: dict[str, str] = field(default_factory=dict)    # by-value args: name -> dtype


@dataclass
class SmemBuf:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int          # byte offset inside the dynamic pool (dynamic layout only)
    symbol: str          # PTX symbol to address it through


def nested_bodies(s) -> tuple:
    return s.nested() if hasattr(s, "nested") else ()


def walk(stmts):
    for s in stmts:
        yield s
        for b in nested_bodies(s):
            yield from walk(b)


def body_size(stmts) -> int:
    return sum(1 + sum(body_size(b) for b in nested_bodies(s)) for s in stmts)


def has_loops(stmts) -> bool:
    return any(type(s).__name__ in ("Loop", "StridedLoop") for s in walk(stmts))


def static_int(dim) -> int:
    if not dim.is_static:
        raise NotImplementedError(f"symbolic extent {dim} is not supported (static shapes only)")
    return int(dim.as_static())


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


class KernelCompiler:
    """Walks one KernelOp body and emits the PTX for it.

    State while walking:
      ints   integer names (Tile axes, loop variables, lane/warp, integer Lets) -> Lin
      vals   float SSA names -> f32 register or immediate
      scopes stack of CSE caches; a block statement pushes one, so a value computed inside a
             branch or loop body is never reused where it may not have been computed.
    """

    def __init__(self, kernel, tensors: dict, launch: Launch):
        self.kernel = kernel
        self.tensors = tensors            # buffer name -> (shape tuple, dtype name)
        self.launch = launch
        self.regs = Registers()
        self.alg = IndexAlgebra()
        self.math = FloatMath(self)
        self.lines: list[str] = []
        self.ints: dict[str, Lin] = {}
        self.vals: dict[str, str] = {}
        self.scopes: list[dict] = [{}]
        self.reg_depth: dict[str, int] = {}
        self.n_labels = 0
        self.ptr: dict[str, str] = {}             # global buffer -> global-space pointer
        self.smem: dict[str, SmemBuf] = {}
        self.smem_base: dict[str, str] = {}       # smem buffer -> 32-bit base address register
        self.smem_dynamic = False
        self.smem_decls: list[str] = []
        self.written: set[str] = set()            # global buffers this kernel stores to
        self.smem_epoch = 0                       # bumped whenever shared memory may change
        self.tid: Lin | None = None
        self.full_warps = True
        self.exit_label: str | None = None

    # ---- emission ----------------------------------------------------------------------------
    def emit(self, text: str):
        self.lines.append(f"\t{text};")

    def comment(self, text: str):
        self.lines.append("\t// " + text.encode("ascii", "replace").decode())

    def label(self, name: str):
        self.lines.append(f"{name}:")

    def new_label(self, hint: str) -> str:
        self.n_labels += 1
        return f"$L_{hint}_{self.n_labels}"

    def iop(self, op: str, *srcs) -> str:
        """Emit one 32-bit integer instruction into a fresh register."""
        r = self.regs.new("b32")
        self.reg_depth[r] = len(self.scopes)
        self.emit(" ".join([op, ", ".join([r, *(str(s) for s in srcs)])]))
        return r

    def int_reg(self, lo: int, hi: int, depth: int | None = None) -> str:
        r = self.regs.new("b32")
        self.alg.reg_range[r] = (lo, hi)
        self.reg_depth[r] = len(self.scopes) if depth is None else depth
        return r

    # ---- CSE scopes --------------------------------------------------------------------------
    @contextmanager
    def scope(self):
        self.scopes.append({})
        try:
            yield
        finally:
            self.scopes.pop()

    @contextmanager
    def bind_int(self, name: str, value: Lin):
        """Bind a loop/let integer lexically and restore any shadowed outer value."""
        existed = name in self.ints
        previous = self.ints.get(name)
        self.ints[name] = value
        try:
            yield
        finally:
            if existed:
                self.ints[name] = previous
            else:
                self.ints.pop(name, None)

    def lookup(self, key):
        for cache in reversed(self.scopes):
            if key in cache:
                return cache[key]
        return None

    def remember(self, key, value):
        self.scopes[-1][key] = value
        return value

    # ---- integer values ----------------------------------------------------------------------
    def atom_depth(self, atom) -> int:
        """Innermost scope depth of any register the atom reads (orders loop-invariant terms first)."""
        if atom[0] == "reg":
            return self.reg_depth.get(atom[1], 0)
        depth = 0
        for part in atom[1:]:
            if isinstance(part, Lin):
                depth = max([depth] + [self.atom_depth(a) for a, _ in part.terms])
            elif isinstance(part, tuple):
                depth = max(depth, self.atom_depth(part))
        return depth

    def ival(self, lin: Lin) -> str:
        """Materialize an integer expression into a register (or an immediate if constant)."""
        if lin.is_const():
            return str(lin.const)
        hit = self.lookup(("lin", lin))
        if hit:
            return hit
        if lin.const:                                       # x + c reuses x
            return self.remember(("lin", lin), self.iop("add.s32", self.ival(Lin(lin.terms)), lin.const))
        depths = [self.atom_depth(a) for a, _ in lin.terms]
        deepest = max(depths)
        outer = [t for t, d in zip(lin.terms, depths) if d < deepest]
        inner = [t for t, d in zip(lin.terms, depths) if d == deepest]
        acc = self.ival(Lin(outer)) if outer else None     # invariant partial sum: its own CSE entry
        for atom, coef in inner:
            x = self.atom_val(atom)
            if acc is None:
                acc = x if coef == 1 else (self.iop("shl.b32", x, coef.bit_length() - 1) if is_pow2(coef)
                                           else self.iop("mul.lo.s32", x, coef))
            elif coef == 1:
                acc = self.iop("add.s32", acc, x)
            elif coef == -1:
                acc = self.iop("sub.s32", acc, x)
            else:
                acc = self.iop("mad.lo.s32", x, coef, acc)
        return self.remember(("lin", lin), acc)

    def atom_val(self, atom) -> str:
        kind = atom[0]
        if kind == "reg":
            return atom[1]
        hit = self.lookup(("atom", atom))
        if hit:
            return hit
        if kind in ("div", "mod"):
            x, d = self.ival(atom[1]), atom[2]
            if is_pow2(d):
                r = self.iop("shr.u32", x, d.bit_length() - 1) if kind == "div" else self.iop("and.b32", x, d - 1)
            else:
                r = self.iop("div.u32" if kind == "div" else "rem.u32", x, d)
        elif kind in ("sdiv", "srem", "mul", "min", "max"):
            op = {"sdiv": "div.s32", "srem": "rem.s32", "mul": "mul.lo.s32", "min": "min.s32", "max": "max.s32"}[kind]
            r = self.iop(op, self.ival(atom[1]), self.ival(atom[2]))
        elif kind == "sel":
            r = self.iop("selp.b32", self.ival(atom[2]), self.ival(atom[3]), self.pred(atom[1]))
        elif kind == "bool":
            r = self.iop("selp.u32", 1, 0, self.pred(atom[1]))
        elif kind == "swz":                       # e ^ (((e >> shift) & mask) << 3)
            x, shift, mask = self.ival(atom[1]), atom[2], atom[3]
            t = self.iop("shl.b32", self.iop("and.b32", self.iop("shr.u32", x, shift), mask), 3)
            r = self.iop("xor.b32", x, t)
        else:
            raise ValueError(f"unknown atom {kind}")
        return self.remember(("atom", atom), r)

    def pred(self, c):
        """Materialize a condition into a predicate register (or return a decided bool)."""
        if isinstance(c, bool):
            return c
        hit = self.lookup(("pred", c))
        if hit:
            return hit
        kind = c[0]
        p = self.regs.new("pred")
        if kind == "cmp":
            ops = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
            self.emit(f"setp.{ops[c[1]]}.s32 {p}, {self.ival(c[2])}, {self.ival(c[3])}")
        elif kind == "fcmp":
            ops = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "neu"}
            self.emit(f"setp.{ops[c[1]]}.f32 {p}, {c[2]}, {c[3]}")
        elif kind in ("and", "or"):
            self.emit(f"{kind}.pred {p}, {self.pred(c[1])}, {self.pred(c[2])}")
        elif kind == "not":
            self.emit(f"not.pred {p}, {self.pred(c[1])}")
        else:
            raise ValueError(f"unknown condition {kind}")
        return self.remember(("pred", c), p)

    # ---- IR expressions ----------------------------------------------------------------------
    def int_var(self, name: str) -> Lin:
        if name in self.ints:
            return self.ints[name]
        if name in ("lane", "warp") and self.tid is not None:     # Emmy's helpers: tid & 31, tid >> 5
            v = self.alg.mod(self.tid, Lin.k(32)) if name == "lane" else self.alg.div(self.tid, Lin.k(32))
            self.ints[name] = v
            return v
        raise KeyError(f"unbound integer name {name!r}")

    def is_float(self, e) -> bool:
        t = type(e).__name__
        if t == "Var":
            return e.name in self.vals and e.name not in self.ints
        if t == "Literal":
            return isinstance(e.value, float) and not float(e.value).is_integer()
        if t == "BinaryExpr":
            return self.is_float(e.left) or self.is_float(e.right)
        if t == "TernaryExpr":
            return self.is_float(e.if_true) or self.is_float(e.if_false)
        return t == "FuncCallExpr"

    def lin_of(self, e) -> Lin:
        """Translate an integer IR expression (index, bound, predicate operand) to a Lin."""
        t = type(e).__name__
        if t == "Literal":
            if isinstance(e.value, float) and not e.value.is_integer():
                raise NotImplementedError(f"non-integral literal {e.value} in an index expression")
            return Lin.k(int(e.value))
        if t == "Var":
            return self.int_var(e.name)
        if t == "BinaryExpr":
            op = e.op
            if op in ("+", "-", "*", "/", "//", "%"):
                a, b = self.lin_of(e.left), self.lin_of(e.right)
                if op == "+":
                    return a + b
                if op == "-":
                    return a - b
                if op == "*":
                    return self.alg.mul(a, b)
                return self.alg.div(a, b) if op in ("/", "//") else self.alg.mod(a, b)
            if op in ("min", "max"):
                return self.alg.minmax(op, self.lin_of(e.left), self.lin_of(e.right))
            return self.alg.boolean(self.cond_of(e))
        if t == "TernaryExpr":
            return self.alg.select(self.cond_of(e.cond), self.lin_of(e.if_true), self.lin_of(e.if_false))
        if t == "FlatIndex":
            return self.flat_index(e.buffer, e.index)
        if t == "CastExpr":
            return self.lin_of(e.expr)
        raise NotImplementedError(f"integer expression {t} is not supported")

    def cond_of(self, e):
        t = type(e).__name__
        if t == "BinaryExpr" and e.op in ("<", "<=", ">", ">=", "==", "!="):
            if self.is_float(e.left) or self.is_float(e.right):
                return ("fcmp", e.op, self.fexpr(e.left), self.fexpr(e.right))
            return self.alg.cmp(e.op, self.lin_of(e.left), self.lin_of(e.right))
        if t == "BinaryExpr" and e.op in ("&&", "||"):
            combine = self.alg.conj if e.op == "&&" else self.alg.disj
            return combine(self.cond_of(e.left), self.cond_of(e.right))
        if t == "Literal":
            return bool(e.value)
        if self.is_float(e):
            return ("fcmp", "!=", self.fexpr(e), ZERO)
        return self.alg.cmp("!=", self.lin_of(e), Lin.k(0))

    def fexpr(self, e) -> str:
        """Translate a float-valued IR expression (rare: Let values, float compares)."""
        t = type(e).__name__
        if t == "Literal":
            return fimm(float(e.value))
        if t == "Var":
            return self.fval(e.name)
        if t == "BinaryExpr" and e.op in ("+", "-", "*", "/"):
            op = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}[e.op]
            return self.math.apply(op, [self.fexpr(e.left), self.fexpr(e.right)])
        if t == "TernaryExpr":
            p = self.pred(self.cond_of(e.cond))
            a, b = self.fexpr(e.if_true), self.fexpr(e.if_false)
            return a if p is True else b if p is False else self.math.selp(a, b, p)
        if t == "FuncCallExpr":
            return self.math.apply(e.name, [self.fexpr(a) for a in e.args])
        if t == "CastExpr":
            return self.fexpr(e.expr) if self.is_float(e.expr) else self.int_to_float(self.lin_of(e.expr))
        raise NotImplementedError(f"float expression {t} is not supported")

    def int_to_float(self, lin: Lin) -> str:
        if lin.is_const():
            return fimm(float(lin.const))
        return self.math.f("cvt.rn.f32.s32", self.ival(lin))

    def fval(self, name: str) -> str:
        if name in self.vals:
            return self.vals[name]
        if name in self.ints:
            return self.int_to_float(self.ints[name])
        raise KeyError(f"unbound value {name!r}")

    def set_inplace(self, name: str, value: str):
        """Write into the register that already holds `name` (loop-carried state must stay put)."""
        dst = self.vals.get(name)
        if dst is None or is_imm(dst):
            dst = self.vals[name] = self.regs.new("f32")
        if dst != value:
            self.emit(f"mov.f32 {dst}, {value}")

    # ---- buffers and addresses ---------------------------------------------------------------
    def shape_of(self, buf: str):
        if buf in self.smem:
            return self.smem[buf].shape
        return self.tensors.get(buf, (None, "f32"))[0]

    def dtype_of(self, buf: str) -> str:
        if buf in self.smem:
            return self.smem[buf].dtype
        return self.tensors.get(buf, (None, "f32"))[1]

    def flat_index(self, buf: str, index) -> Lin:
        """Row-major flattening of a multi-dimensional index against the buffer's shape."""
        idx = [self.lin_of(e) for e in index]
        if not idx:
            return Lin.k(0)
        shape = self.shape_of(buf)
        if len(idx) == 1 or shape is None or len(shape) != len(idx):
            total = idx[0]
            for x in idx[1:]:
                total = total + x
            return total
        flat, stride = Lin.k(0), 1
        for d in range(len(idx) - 1, -1, -1):
            flat = flat + idx[d].scale(stride)
            stride *= shape[d]
        return flat

    def address(self, buf: str, idx: Lin) -> str:
        """[base+offset] operand for element idx; the constant part of idx becomes the offset."""
        size = DTYPES[self.dtype_of(buf)][1]
        var, c = idx.split_const()
        if buf in self.smem:
            sb = self.smem[buf]
            offset = c * size + (sb.offset if self.smem_dynamic else 0)
            if var.is_const():
                return f"[{sb.symbol}+{offset}]" if offset else f"[{sb.symbol}]"
            reg = self.lookup(("saddr", buf, var))
            if reg is None:
                if len(var.terms) == 1:                       # one mad: base + atom*coef*size
                    atom, coef = var.terms[0]
                    reg = self.iop("mad.lo.s32", self.atom_val(atom), coef * size, self.smem_base[buf])
                else:
                    reg = self.iop("mad.lo.s32", self.ival(var), size, self.smem_base[buf])
                self.remember(("saddr", buf, var), reg)
            return f"[{reg}+{c * size}]" if c else f"[{reg}]"
        base = self.ptr[buf]
        if var.is_const():
            return f"[{base}+{c * size}]" if c else f"[{base}]"
        reg = self.lookup(("gaddr", buf, var))
        if reg is None:
            reg = self.regs.new("b64")
            self.emit(f"add.s64 {reg}, {base}, {self.byte_offset(var, size)}")
            self.remember(("gaddr", buf, var), reg)
        return f"[{reg}+{c * size}]" if c else f"[{reg}]"

    def byte_offset(self, var: Lin, size: int) -> str:
        """64-bit byte offset var*size via mul.wide (shared between buffers with the same index)."""
        key = ("bytes", var, size)
        hit = self.lookup(key)
        if hit:
            return hit
        signed = self.alg.range(var)[0] < 0
        if len(var.terms) == 1:
            atom, coef = var.terms[0]
            src, scale = self.atom_val(atom), coef * size
            signed = signed or self.alg.atom_range(atom)[0] < 0 or scale < 0
        else:
            src, scale = self.ival(var), size
        wide = self.regs.new("b64")
        self.emit(f"mul.wide.{'s32' if signed else 'u32'} {wide}, {src}, {scale}")
        return self.remember(key, wide)

    def load(self, buf: str, idx: Lin, width: int) -> list[str]:
        """Load `width` consecutive elements as f32 values, vectorized when aligned."""
        in_smem = buf in self.smem
        cacheable = in_smem or buf not in self.written
        key = ("ld", buf, idx, width)
        if cacheable:
            hit = self.lookup(key)
            if hit and (not in_smem or hit[0] == self.smem_epoch):
                return hit[1]
        dtype = self.dtype_of(buf)
        space = "shared" if in_smem else ("global.nc" if buf not in self.written else "global")
        if width in (2, 4) and dtype == "f32" and idx.divisible_by(width):
            regs = [self.regs.new("f32") for _ in range(width)]
            self.emit(f"ld.{space}.v{width}.f32 {{{', '.join(regs)}}}, {self.address(buf, idx)}")
        else:
            regs = [self.load_scalar(space, dtype, self.address(buf, idx + k)) for k in range(width)]
        if cacheable:
            self.remember(key, (self.smem_epoch, regs))
            for k, r in enumerate(regs):
                self.remember(("ld", buf, idx + k, 1), (self.smem_epoch, [r]))
        return regs

    def load_scalar(self, space: str, dtype: str, addr: str) -> str:
        ptx_type = DTYPES[dtype][0]
        if dtype == "f32":
            r = self.regs.new("f32")
            self.emit(f"ld.{space}.f32 {r}, {addr}")
            return r
        raw = self.regs.new({"b16": "b16", "f64": "f64", "s32": "b32", "s64": "b64"}[ptx_type])
        self.emit(f"ld.{space}.{ptx_type} {raw}, {addr}")
        conv = {"f16": "cvt.f32.f16", "f64": "cvt.rn.f32.f64", "i32": "cvt.rn.f32.s32",
                "i64": "cvt.rn.f32.s64"}.get(dtype)
        if dtype == "bf16":                           # bf16 is the top half of an f32
            wide = self.iop("cvt.u32.u16", raw)
            return self.math.f("mov.b32", self.iop("shl.b32", wide, 16))
        return self.math.f(conv, raw)

    def store(self, buf: str, idx: Lin, values: list[str], atomic: bool = False):
        values = [self.math.reg(v) for v in values]
        dtype = self.dtype_of(buf)
        in_smem = buf in self.smem
        space = "shared" if in_smem else "global"
        if in_smem:
            self.smem_epoch += 1
        if atomic:
            for k, v in enumerate(values):
                self.emit(f"red.{space}.add.f32 {self.address(buf, idx + k)}, {v}")
            return
        if len(values) in (2, 4) and dtype == "f32" and idx.divisible_by(len(values)):
            self.emit(f"st.{space}.v{len(values)}.f32 {self.address(buf, idx)}, {{{', '.join(values)}}}")
            return
        for k, v in enumerate(values):
            self.store_scalar(space, dtype, self.address(buf, idx + k), v)

    def store_scalar(self, space: str, dtype: str, addr: str, v: str):
        if dtype == "f32":
            self.emit(f"st.{space}.f32 {addr}, {v}")
            return
        ptx_type = DTYPES[dtype][0]
        kind = {"b16": "b16", "f64": "f64", "s32": "b32", "s64": "b64"}[ptx_type]
        conv = {"f16": "cvt.rn.f16.f32", "bf16": "cvt.rn.bf16.f32", "f64": "cvt.f64.f32",
                "i32": "cvt.rzi.s32.f32", "i64": "cvt.rzi.s64.f32"}[dtype]
        raw = self.regs.new(kind)
        self.emit(f"{conv} {raw}, {v}")
        self.emit(f"st.{space}.{ptx_type} {addr}, {raw}")

    # ---- driver ------------------------------------------------------------------------------
    def compile(self) -> str:
        body = list(self.kernel.body)
        self.plan_buffers(body)
        self.prologue()
        self.body(body)
        if self.exit_label:
            self.label(self.exit_label)
        self.emit("ret")
        return self.module()

    def plan_buffers(self, body):
        """Pre-pass: lay out shared memory and find which global buffers are stored to."""
        decls = {}
        for s in walk(body):
            kind = type(s).__name__
            if kind == "Smem" and s.name not in decls:
                decls[s.name] = s
            elif kind == "Write":
                self.written.add(s.output)
        offset, sizes = 0, {}
        for name, s in decls.items():
            dtype = SMEM_CTYPES.get(s.dtype)
            if dtype is None:
                raise NotImplementedError(f"shared memory dtype {s.dtype!r} is not supported")
            shape = tuple(int(e) for e in s.extents)
            align = max(16, int(s.align or 0))
            offset = -(-offset // align) * align
            sizes[name] = (math.prod(shape) * DTYPES[dtype][1], align)
            self.smem[name] = SmemBuf(name, shape, dtype, offset, f"smem_{sanitize(name)}")
            offset += sizes[name][0]
        self.written -= set(decls)
        self.smem_dynamic = offset > STATIC_SMEM_LIMIT
        if self.smem_dynamic:
            if self.launch.smem_bytes < offset:
                raise ValueError(f"kernel needs {offset} B of dynamic shared memory, launch gives {self.launch.smem_bytes}")
            self.smem_decls.append(".extern .shared .align 16 .b8 smem_pool[];")
            for sb in self.smem.values():
                sb.symbol = "smem_pool"
        else:
            for name, (nbytes, align) in sizes.items():
                self.smem_decls.append(f".shared .align {align} .b8 {self.smem[name].symbol}[{max(nbytes, 1)}];")

    def prologue(self):
        """Load every buffer pointer, convert it to the global window, and set up smem bases."""
        used = {s.input for s in walk(self.kernel.body) if type(s).__name__ == "Load"}
        used |= {s.src for s in walk(self.kernel.body) if type(s).__name__ == "CpAsyncCopy"}
        used |= self.written
        for i, name in enumerate(self.launch.arg_order):
            param = f"{self.launch.entry}_param_{i}"
            if name in self.launch.scalars:
                r = self.int_reg(INT_MIN, INT_MAX, depth=0)
                self.emit(f"ld.param.s32 {r}, [{param}]")
                self.ints[name] = Lin.atom(("reg", r))
            elif name in used and name not in self.smem:
                raw, ptr = self.regs.new("b64"), self.regs.new("b64")
                self.emit(f"ld.param.u64 {raw}, [{param}]")
                self.emit(f"cvta.to.global.u64 {ptr}, {raw}")
                self.ptr[name] = ptr
        missing = (used - set(self.smem)) - set(self.ptr)
        if missing:
            raise ValueError(f"buffers {sorted(missing)} are not in meta.json's arg_order")
        for name, sb in self.smem.items():
            base = self.int_reg(0, 2**24, depth=0)
            self.emit(f"mov.u32 {base}, {sb.symbol}")
            if self.smem_dynamic and sb.offset:
                base = self.iop("add.s32", base, sb.offset)
            self.smem_base[name] = base

    def module(self) -> str:
        L = self.launch
        head = [
            "//",
            f"// {L.entry}: generated by compile.py from Emmy Kernel IR",
            f"// launch: grid ({', '.join(map(str, L.grid))}) block ({', '.join(map(str, L.block))})"
            f" dynamic smem {L.smem_bytes} B",
            "//",
            f".version {PTX_VERSION}",
            f".target {PTX_TARGET}",
            ".address_size 64",
            "",
            *self.smem_decls,
            "",
            f".visible .entry {L.entry}(",
        ]
        params = []
        for i, name in enumerate(L.arg_order):
            ptype = ".s32" if name in L.scalars else ".u64"
            sep = "," if i + 1 < len(L.arg_order) else ""
            params.append(f"\t.param {ptype} {L.entry}_param_{i}{sep}\t// {name}")
        tail = [")", f".maxntid {L.block[0]}, {L.block[1]}, {L.block[2]}", "{",
                *self.regs.declarations(), "", *self.lines, "}", ""]
        return "\n".join(head + params + tail)

    # ---- statements --------------------------------------------------------------------------
    def body(self, stmts):
        for s in stmts:
            self.stmt(s)

    def stmt(self, s):
        kind = type(s).__name__
        handler = getattr(self, "s_" + kind, None)
        if handler is None:
            raise NotImplementedError(f"IR statement {kind} is not supported")
        try:
            self.comment(s.pretty("")[0].strip())
        except Exception:
            self.comment(kind)
        handler(s)

    def s_Tile(self, t):
        """One thread per cell: decode the axes from the global thread id, guard the tail."""
        block, grid = self.launch.block[0], self.launch.grid[0]
        tid = self.int_reg(0, block - 1, depth=0)
        self.emit(f"mov.u32 {tid}, %tid.x")
        self.tid = Lin.atom(("reg", tid))
        cta = Lin.k(0)                                       # a one-block grid has ctaid 0
        if grid > 1:
            reg = self.int_reg(0, grid - 1, depth=0)
            self.emit(f"mov.u32 {reg}, %ctaid.x")
            cta = Lin.atom(("reg", reg))
        cells = t.block_threads or block
        if t.aux_threads:          # producer warps past the cells wrap onto this CTA's first cells
            gid = cta.scale(cells) + self.alg.mod(self.tid, Lin.k(cells))
        else:
            gid = cta.scale(block) + self.tid
        extents = [static_int(a.extent) for a in t.axes]
        n = math.prod(extents)
        if self.alg.range(gid)[1] >= n:                      # more threads than cells
            collectives = {type(stmt).__name__ for stmt in walk(t.body)}
            unsafe = collectives & {"Sync", "TreeHalve", "WarpShuffle"}
            if unsafe:
                raise NotImplementedError(
                    "partial Tile with CTA/warp collective is unsafe because tail threads would "
                    f"skip collective participation: {', '.join(sorted(unsafe))}"
                )
            self.exit_label = self.new_label("exit")
            self.emit(f"@{self.pred(self.alg.cmp('>=', gid, Lin.k(n)))} bra {self.exit_label}")
            self.full_warps = n % 32 == 0
        self.full_warps = self.full_warps and block % 32 == 0
        strides = [math.prod(extents[i + 1:]) for i in range(len(extents))]
        raster = self.raster_decode(t, extents, strides, gid)
        for i, a in enumerate(t.axes):
            if raster and a.name in raster:
                self.ints[a.name] = raster[a.name]
                continue
            v = self.alg.div(gid, Lin.k(strides[i]))
            self.ints[a.name] = v if i == 0 else self.alg.mod(v, Lin.k(extents[i]))
        self.body(t.body)

    def raster_decode(self, t, extents, strides, gid):
        """Grouped CTA rasterization (Emmy's RASTER codec) for 2-D tiled contractions."""
        if not (t.raster_group and t.raster_axes):
            return None
        names = [a.name for a in t.axes]
        m_name, n_name = t.raster_axes
        if m_name not in names or n_name not in names:
            return None
        mi, ni = names.index(m_name), names.index(n_name)
        if ni != mi + 1:
            return None
        gi, oi = (mi, ni) if t.raster_orient == "m" else (ni, mi)
        g = min(t.raster_group, extents[gi])
        if extents[gi] <= 1 or g <= 1:
            return None
        A = self.alg
        ea, eb = extents[gi], extents[oi]
        sub = A.div(gid, Lin.k(strides[ni]))
        if mi != 0:
            sub = A.mod(sub, Lin.k(ea * eb))
        within = A.mod(sub, Lin.k(g * eb))
        if ea % g == 0:
            a_val = A.div(sub, Lin.k(g * eb)).scale(g) + A.mod(sub, Lin.k(g))
            b_val = A.div(within, Lin.k(g))
        else:
            grp = A.div(sub, Lin.k(g * eb))
            size = A.minmax("min", Lin.k(g), Lin.k(ea) - grp.scale(g))
            a_val = grp.scale(g) + A.mod(within, size)
            b_val = A.div(within, size)
        return {names[gi]: a_val, names[oi]: b_val}

    # ---- loops -------------------------------------------------------------------------------
    def seed_accums(self, body, seed: bool):
        """Seed loop-carried accumulators, including those nested in conditionals.

        Nested loops own their accumulators and are intentionally not traversed here.
        """
        seen = set()

        def collect(stmts):
            for stmt in stmts:
                kind = type(stmt).__name__
                if kind == "Accum":
                    yield stmt
                elif kind in ("Loop", "StridedLoop"):
                    continue
                else:
                    for nested in nested_bodies(stmt):
                        yield from collect(nested)

        for stmt in collect(body):
            if stmt.name in seen:
                continue
            seen.add(stmt.name)
            if seed or stmt.name not in self.vals or is_imm(self.vals[stmt.name]):
                r = self.vals[stmt.name] = self.regs.new("f32")
                identity = IDENTITY.get(FloatMath.canonical(stmt.op.name), 0.0)
                self.emit(f"mov.f32 {r}, {fimm(identity)}")

    def s_Loop(self, s):
        self.seed_accums(s.body, s.seed)
        self.counted_loop(s.axis.name, Lin.k(0), 1, static_int(s.axis.extent), s.body)

    def s_StridedLoop(self, s):
        """for (v = start; v < end; v += step). With static bounds most become counted loops:
        e.g. start = tid in [0,128), step 128, end 1024 runs exactly 8 times for every thread."""
        self.seed_accums(s.body, s.seed)
        start = self.lin_of(s.start)
        step = self.lin_of(s.step) if hasattr(s.step, "pretty") else Lin.k(int(s.step))
        if step.is_const() and step.const <= 0:
            raise NotImplementedError(f"StridedLoop requires a positive step, got {step.const}")
        end = self.lin_of(s.end) if s.end is not None else Lin.k(static_int(s.axis.extent))
        if step.is_const() and step.const > 0 and end.is_const():
            lo, hi = self.alg.range(start)
            trips_max = max(0, -((lo - end.const) // step.const))
            trips_min = max(0, -((hi - end.const) // step.const))
            if trips_max == trips_min:
                return self.counted_loop(s.axis.name, start, step.const, trips_max, s.body)
            if trips_max == 1:                                  # zero or one trip: just a branch
                return self.guarded_once(s.axis.name, start, end, s.body)
        self.generic_loop(s.axis.name, start, step, end, s.body)

    def counted_loop(self, var: str, start: Lin, step: int, trips: int, body):
        """A loop whose trip count is the same constant for every thread."""
        if trips <= 0:
            return
        size, nested = body_size(body), has_loops(body)
        if trips == 1 or (not nested and trips * size <= FULL_UNROLL):
            with self.scope():                                  # fully unrolled: constants fold
                for k in range(trips):
                    with self.bind_int(var, start + k * step):
                        self.body(body)
            return
        unroll = 1
        if not nested:
            unroll = next((u for u in (16, 8, 4, 2) if trips % u == 0 and u * size <= PARTIAL_UNROLL), 1)
        # Canonical form: the counter is the offset 0, step, ... below the constant trips*step and
        # the loop variable is start + offset. ptxas pipelines this shape far better than a
        # counter compared against a per-thread bound (1.25x on a 65536-element sum).
        counter = self.int_reg(0, (trips - unroll) * step, depth=len(self.scopes) + 1)
        self.emit(f"mov.u32 {counter}, 0")
        top = self.new_label("loop")
        self.smem_epoch += 1
        self.label(top)
        with self.scope():
            for k in range(unroll):
                with self.bind_int(var, start + Lin.atom(("reg", counter)) + k * step):
                    self.body(body)
        self.emit(f"add.s32 {counter}, {counter}, {unroll * step}")
        p = self.regs.new("pred")
        self.emit(f"setp.lt.s32 {p}, {counter}, {trips * step}")
        self.emit(f"@{p} bra {top}")
        self.smem_epoch += 1

    def guarded_once(self, var: str, start: Lin, end: Lin, body):
        skip = self.new_label("skip")
        self.emit(f"@!{self.pred(self.alg.cmp('<', start, end))} bra {skip}")
        with self.scope(), self.bind_int(var, start):
            self.body(body)
        self.label(skip)
        self.smem_epoch += 1

    def generic_loop(self, var: str, start: Lin, step: Lin, end: Lin, body):
        lo, _ = self.alg.range(start)
        _, end_hi = self.alg.range(end)
        counter = self.int_reg(lo, max(lo, end_hi - 1), depth=len(self.scopes) + 1)
        self.emit(f"mov.u32 {counter}, {self.ival(start)}")
        end_v, step_v = self.ival(end), self.ival(step)
        top, done = self.new_label("loop"), self.new_label("done")
        first = self.pred(self.alg.cmp(">=", start, end))
        if first is not False:
            self.emit(f"@{first} bra {done}" if first is not True else f"bra {done}")
        self.smem_epoch += 1
        self.label(top)
        with self.scope(), self.bind_int(var, Lin.atom(("reg", counter))):
            self.body(body)
        self.emit(f"add.s32 {counter}, {counter}, {step_v}")
        p = self.regs.new("pred")
        self.emit(f"setp.lt.s32 {p}, {counter}, {end_v}")
        self.emit(f"@{p} bra {top}")
        self.label(done)
        self.smem_epoch += 1

    def s_Cond(self, c):
        p = self.pred(self.cond_of(c.cond))
        if isinstance(p, bool):                                 # decided at compile time
            with self.scope():
                self.body(c.body if p else c.else_body)
            self.smem_epoch += 1
            return
        else_label, end_label = self.new_label("else"), self.new_label("endif")
        self.emit(f"@!{p} bra {else_label}")
        with self.scope():
            self.body(c.body)
        if c.else_body:
            self.emit(f"bra.uni {end_label}")
            self.label(else_label)
            with self.scope():
                self.body(c.else_body)
            self.label(end_label)
        else:
            self.label(else_label)
        self.smem_epoch += 1

    # ---- scalar compute ----------------------------------------------------------------------
    def s_Load(self, s):
        idx = self.flat_index(s.input, s.index)
        for name, r in zip(s.names, self.load(s.input, idx, len(s.names))):
            self.vals[name] = r

    def s_Assign(self, s):
        self.vals[s.name] = self.math.apply(s.op.name, [self.fval(a) for a in s.args])

    def s_Accum(self, s):
        """name = op(base or name, value), updated in place in the accumulator's register."""
        if s.name not in self.vals or is_imm(self.vals[s.name]):
            self.seed_accums([s], True)
        dst = self.vals[s.name]
        left = self.fval(s.base) if s.base is not None else dst
        value = self.fval(s.value)
        op = FloatMath.canonical(s.op.name)
        simple = {"add": "add.f32", "multiply": "mul.f32", "maximum": "max.f32", "minimum": "min.f32"}
        if op in simple:
            self.emit(f"{simple[op]} {dst}, {left}, {value}")
        else:
            self.set_inplace(s.name, self.math.apply(op, [left, value]))

    def s_Init(self, s):
        r = self.vals[s.name] = self.regs.new("f32")
        self.emit(f"mov.f32 {r}, {fimm(float(s.identity))}")

    def s_Let(self, s):
        """A float literal (online softmax's `one = 1.0`) or an integer index bound once."""
        v = s.value
        literal = type(v).__name__ == "Literal"
        if s.dtype is not None:
            is_float = s.dtype.name in ("f32", "f16", "bf16", "f64")
        else:
            is_float = literal and getattr(v, "dtype", "float") == "float"
        if is_float:
            self.vals[s.name] = fimm(float(v.value)) if literal else self.fexpr(v)
        else:
            self.ints[s.name] = self.lin_of(v)

    def s_Select(self, s):
        branches = list(s.branches)
        result = self.fval(branches[-1].value)
        for b in reversed(branches[:-1]):
            p = self.pred(self.cond_of(b.select))
            v = self.fval(b.value)
            result = v if p is True else result if p is False else self.math.selp(v, result, p)
        self.vals[s.name] = result

    def s_Write(self, s):
        idx = self.flat_index(s.output, s.index)
        if s.output in self.smem:
            idx = self.swizzled(s.swizzle, idx)
        self.store(s.output, idx, [self.fval(v) for v in s.values], atomic=s.atomic)

    def s_Smem(self, s):
        pass                                       # laid out up front by plan_buffers

    def s_Sync(self, s):
        if s.warp:
            self.emit("bar.warp.sync 0xffffffff")
        elif s.barrier_id == 0:
            self.emit("bar.sync 0")
        else:
            self.emit(f"bar.sync {s.barrier_id}, {s.count}")
        self.smem_epoch += 1

    # ---- cross-thread combines ---------------------------------------------------------------
    def run_program(self, program, state):
        """A combine program: Assigns whose targets in `state` overwrite the carried registers."""
        state = set(state)
        for a in program:
            result = self.math.apply(a.op.name, [self.fval(x) for x in a.args])
            if a.name in state:
                self.set_inplace(a.name, result)
            else:
                self.vals[a.name] = result

    def s_WarpShuffle(self, s):
        """XOR butterfly over a power-of-two number of lanes."""
        if not is_pow2(s.length) or s.length > 32:
            raise NotImplementedError(
                f"WarpShuffle requires a power-of-two length in [1, 32], got {s.length}"
            )
        mask = "0xffffffff"
        if not self.full_warps:                    # partial warp: name only the live lanes
            mask = self.iop("activemask.b32")
        lane_mask = s.length // 2
        while lane_mask >= 1:
            for st, sb in zip(s.state, s.state_b):
                other = self.regs.new("f32")
                self.emit(f"shfl.sync.bfly.b32 {other}, {self.math.reg(self.fval(st))}, {lane_mask}, 0x1f, {mask}")
                self.vals[sb] = other
            self.run_program(s.combine_states, s.state)
            lane_mask //= 2

    def s_TreeHalve(self, s):
        """Shared-memory tree over `length` partials (one per warp), then broadcast slot 0."""
        tid = self.int_var(s.tid_var)
        if s.inner is not None:
            inner, scale = s.inner
            iv = self.int_var(inner)

            def slot(x):
                return x.scale(scale) + iv
            root = iv
        else:
            def slot(x):
                return x
            root = Lin.k(0)
        barrier = "bar.sync 0" if s.barrier_id == 0 else f"bar.sync {s.barrier_id}, {s.barrier_count}"
        remaining = s.length
        for active_count, half in tree_halve_stages(s.length):
            p = self.pred(self.alg.cmp("<", tid, Lin.k(active_count)))
            if p is not False:
                skip = self.new_label("tree")
                if p is not True:
                    self.emit(f"@!{p} bra {skip}")
                with self.scope():
                    saved = {n: self.vals.get(n) for n in (*s.state, *s.state_b)}
                    for buf, st, sb in zip(s.bufs, s.state, s.state_b):
                        mine = self.load(buf, slot(tid), 1)[0]
                        self.vals[st] = self.math.f("mov.f32", mine)    # private copy, updated in place
                        self.vals[sb] = self.load(buf, slot(tid + half), 1)[0]
                    self.run_program(s.combine_states, s.state)
                    for buf, st in zip(s.bufs, s.state):
                        self.store(buf, slot(tid), [self.vals[st]])
                    for n, v in saved.items():
                        if v is None:
                            self.vals.pop(n, None)
                        else:
                            self.vals[n] = v
                if p is not True:
                    self.label(skip)
            self.emit(barrier)
            self.smem_epoch += 1
        for buf, st in zip(s.bufs, s.state):
            self.set_inplace(st, self.load(buf, root, 1)[0])

    # ---- asynchronous copies into shared memory ----------------------------------------------
    def swizzled(self, mode: str, idx: Lin) -> Lin:
        if not mode or mode == "NONE":
            return idx
        base, _, shift = mode.partition("@")
        table = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}
        if base not in table:
            raise NotImplementedError(f"shared memory layout {mode!r} is not supported")
        sh, mask = table[base]
        return self.alg.swizzle(idx, int(shift) if shift else sh, mask)

    def s_CpAsyncCopy(self, s):
        sidx = self.flat_index(s.smem, s.smem_index)
        if s.lane_index is not None:               # stripe base + this lane's (row, col)
            cols = self.smem[s.smem].shape[-1]
            sidx = sidx + self.lin_of(s.lane_index[0]).scale(cols) + self.lin_of(s.lane_index[1])
        dst = self.address(s.smem, self.swizzled(s.swizzle, sidx))
        src = self.address(s.src, self.flat_index(s.src, s.src_index))
        cache = "cg" if s.nbytes == 16 else "ca"     # 16-byte copies bypass L1, like Emmy's helper
        self.emit(f"cp.async.{cache}.shared.global {dst}, {src}, {s.nbytes}")

    def s_CpAsyncCommit(self, s):
        self.emit("cp.async.commit_group")

    def s_CpAsyncWait(self, s):
        self.emit(f"cp.async.wait_group {s.group}")
        self.smem_epoch += 1


# =============================================================================================
# 5. Driver
# =============================================================================================

def graph_tensors(graph) -> dict:
    """Every tensor the graph names -> (static shape, dtype name)."""
    out = {}
    for node in graph.nodes.values():
        for t in getattr(node, "outputs", ()) or ():
            try:
                shape = tuple(static_int(d) for d in t.shape)
            except Exception:
                shape = None
            out[t.name] = (shape, t.dtype.name)
    return out


def find_launch(ir_path: Path, kernel, node, tensors: dict) -> Launch:
    """Launch info from meta.json (the contract); a best-effort default if it is missing."""
    meta_path = ir_path.with_name("meta.json")
    kernels = json.loads(meta_path.read_text())["kernels"] if meta_path.exists() else []
    match = [k for k in kernels if k.get("entry") == kernel.name]
    match = match or [k for k in kernels if ir_path.name == f"{k.get('entry')}.ir.json"]
    if match:
        k = match[0]
        for a in k.get("args", []):                # fill shapes/dtypes the graph does not carry
            tensors.setdefault(a["name"], (tuple(a["shape"]), a["dtype"]))
        scalars = {a["name"]: a["dtype"] for a in k.get("args", []) if a.get("scalar")}
        return Launch(kernel.name, list(k["arg_order"]), list(k["grid"]), list(k["block"]),
                      int(k.get("smem_bytes", 0)), scalars)
    print(f"compile.py: no meta.json entry for {kernel.name}; using a default launch", file=sys.stderr)
    tile = next(s for s in walk(kernel.body) if type(s).__name__ == "Tile")
    cells = tile.block_threads or 256
    n = math.prod(static_int(a.extent) for a in tile.axes)
    args = list(node.inputs) + [t.name for t in node.outputs]
    return Launch(kernel.name, args, [-(-n // cells), 1, 1], [cells, 1, 1], 0)


def compile_ir(ir_path: Path) -> str:
    from emmy.compiler.graph import Graph       # Emmy is only used to load the IR

    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    nodes = [n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if not nodes:
        raise ValueError("no KernelOp in this IR")
    node = nodes[0]
    tensors = graph_tensors(graph)
    launch = find_launch(ir_path, node.op, node, tensors)
    return KernelCompiler(node.op, tensors, launch).compile()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        ptx = compile_ir(Path(argv[1]))
    except Exception as exc:                       # report on stderr; stdout must stay PTX-only
        print(f"compile.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
