"""Model-agnostic Greeks via central finite differences (bump and reprice).

Works with any pricer ``f(Option) -> float``, so the numerical Greeks of the
binomial and Monte Carlo engines can be compared with the analytic
Black-Scholes values. Monte Carlo pricers should use a fixed seed (common
random numbers), otherwise sampling noise swamps the bump.
"""

from __future__ import annotations

from typing import Callable

from .black_scholes import DAYS_PER_YEAR
from .models import Greeks, Option

Pricer = Callable[[Option], float]


def numerical_greeks(
    opt: Option,
    pricer: Pricer,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    time_bump: float = 1.0 / DAYS_PER_YEAR,
) -> Greeks:
    """``spot_bump`` is relative (1% of S); ``vol_bump`` absolute; ``time_bump`` in years."""
    h = opt.S * spot_bump
    v0 = pricer(opt)
    up = pricer(opt.with_(S=opt.S + h))
    down = pricer(opt.with_(S=opt.S - h))

    delta = (up - down) / (2 * h)
    gamma = (up - 2 * v0 + down) / h**2

    vega_raw = (pricer(opt.with_(sigma=opt.sigma + vol_bump)) - pricer(opt.with_(sigma=opt.sigma - vol_bump))) / (2 * vol_bump)

    dt = min(time_bump, opt.T / 2)
    # Theta is the change as calendar time passes, i.e. as T shrinks.
    theta_raw = (pricer(opt.with_(T=opt.T - dt)) - pricer(opt.with_(T=opt.T + dt))) / (2 * dt)

    return Greeks(delta=delta, gamma=gamma, vega=vega_raw / 100.0, theta=theta_raw / DAYS_PER_YEAR)
