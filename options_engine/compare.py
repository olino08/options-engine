"""Run all three pricing methods side by side."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

from .binomial import binomial_greeks, binomial_price
from .black_scholes import bs_greeks, bs_price
from .greeks import numerical_greeks
from .models import Greeks, Option
from .monte_carlo import mc_price, strike_shift


@dataclass(frozen=True)
class MethodResult:
    method: str
    price: float
    error: float  # vs Black-Scholes
    millis: float
    std_error: Optional[float] = None


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000


def compare_methods(
    opt: Option, mc_paths: int = 200_000, tree_steps: int = 1000, seed: int = 42, importance_sampling: bool = True
) -> List[MethodResult]:
    bs, bs_ms = _timed(lambda: bs_price(opt))
    rows = [MethodResult("Black-Scholes", bs, 0.0, bs_ms)]

    use_is = importance_sampling and strike_shift(opt) != 0.0
    if use_is:
        # Show the plain estimate too, so the variance reduction is visible.
        plain, plain_ms = _timed(lambda: mc_price(opt, n_paths=mc_paths, seed=seed, importance_sampling=False))
        rows.append(MethodResult(f"Monte Carlo ({mc_paths:,} paths)", plain.price, plain.price - bs, plain_ms, plain.std_error))
    mc, mc_ms = _timed(lambda: mc_price(opt, n_paths=mc_paths, seed=seed, importance_sampling=importance_sampling))
    label = "MC + importance sampling" if use_is else f"Monte Carlo ({mc_paths:,} paths)"
    rows.append(MethodResult(label, mc.price, mc.price - bs, mc_ms, mc.std_error))

    tree, tree_ms = _timed(lambda: binomial_price(opt, steps=tree_steps))
    rows.append(MethodResult(f"Binomial CRR ({tree_steps:,} steps)", tree, tree - bs, tree_ms))
    return rows


def compare_greeks(
    opt: Option, mc_paths: int = 200_000, tree_steps: int = 500, seed: int = 42, importance_sampling: bool = True
):
    """Analytic BS Greeks vs bump-and-reprice Greeks from the numerical engines."""
    shift = strike_shift(opt) if importance_sampling else 0.0
    return {
        "Black-Scholes (analytic)": bs_greeks(opt),
        "Binomial (tree nodes)": binomial_greeks(opt, steps=tree_steps),
        # Common random numbers (fixed seed) and one IS shift for every bump
        # keep the bumped prices comparable.
        "Monte Carlo (finite diff.)": numerical_greeks(
            opt, lambda o: mc_price(o, n_paths=mc_paths, seed=seed, shift=shift).price
        ),
    }


def put_call_parity_gap(opt: Option) -> float:
    """C - P - (S e^{-qT} - K e^{-rT}); zero for a consistent model."""
    call = bs_price(opt.with_(option_type="call"))
    put = bs_price(opt.with_(option_type="put"))
    return call - put - (opt.S * math.exp(-opt.q * opt.T) - opt.K * math.exp(-opt.r * opt.T))


def fmt(x: float, sign: bool = False) -> str:
    """4 decimals normally; scientific notation for small non-zero values that would print as 0.0000."""
    if x != 0 and abs(x) < 0.01:
        return f"{x:+.2e}" if sign else f"{x:.2e}"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def format_greeks_table(rows: dict) -> str:
    header = f"{'Method':<28}{'Delta':>10}{'Gamma':>10}{'Vega':>10}{'Theta':>10}"
    lines = [header, "-" * len(header)]
    for name, g in rows.items():
        lines.append(f"{name:<28}" + "".join(f"{fmt(v):>10}" for v in (g.delta, g.gamma, g.vega, g.theta)))
    lines.append("Vega per 1 vol point; theta per calendar day.")
    return "\n".join(lines)
