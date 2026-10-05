"""Cox-Ross-Rubinstein binomial tree."""

from __future__ import annotations

import math

import numpy as np

from .models import Greeks, Option

DAYS_PER_YEAR = 365.0


def _roll_back(opt: Option, steps: int, american: bool, stop_at: int = 0):
    """Backward induction down to time-step ``stop_at``; returns (node values, u, dt)."""
    if steps < max(1, stop_at):
        raise ValueError(f"steps must be >= {max(1, stop_at)}")

    dt = opt.T / steps
    u = math.exp(opt.sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((opt.r - opt.q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0, 1); increase steps")
    disc = math.exp(-opt.r * dt)

    # Terminal spot prices, from all-down (j=0) to all-up (j=steps).
    j = np.arange(steps + 1)
    values = opt.payoff(opt.S * u**j * d ** (steps - j))

    for n in range(steps - 1, stop_at - 1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(n + 1)
            values = np.maximum(values, opt.payoff(opt.S * u**j * d ** (n - j)))

    return values, u, dt


def binomial_price(opt: Option, steps: int = 500, american: bool = False) -> float:
    """Price on an N-step CRR tree. ``american=True`` adds early-exercise checks."""
    values, _, _ = _roll_back(opt, steps, american)
    return float(values[0])


def binomial_greeks(opt: Option, steps: int = 500, american: bool = False, vol_bump: float = 0.01) -> Greeks:
    """Delta, gamma and theta read off the tree's own nodes; vega by repricing.

    Bumping S and repricing on a CRR tree is unreliable for gamma because the
    strike moves relative to the node grid. Using the nodes at step 2
    (spots S*d^2, S, S*u^2) avoids that.
    """
    v2, u, dt = _roll_back(opt, steps, american, stop_at=2)
    v0 = binomial_price(opt, steps, american)
    s_dd, s_mid, s_uu = opt.S / u**2, opt.S, opt.S * u**2

    delta_up = (v2[2] - v2[1]) / (s_uu - s_mid)
    delta_dn = (v2[1] - v2[0]) / (s_mid - s_dd)
    gamma = (delta_up - delta_dn) / (0.5 * (s_uu - s_dd))

    # Delta from the two step-1 nodes is first order; the step-2 average is centred on S.
    delta = (v2[2] - v2[0]) / (s_uu - s_dd)
    theta = (v2[1] - v0) / (2 * dt)  # middle node at step 2 has the same spot, 2*dt later

    vega = (binomial_price(opt.with_(sigma=opt.sigma + vol_bump), steps, american)
            - binomial_price(opt.with_(sigma=opt.sigma - vol_bump), steps, american)) / (2 * vol_bump)

    return Greeks(delta=delta, gamma=gamma, vega=vega / 100.0, theta=theta / DAYS_PER_YEAR)
