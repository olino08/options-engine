"""Monte Carlo pricing under geometric Brownian motion.

European payoffs depend only on S_T, so we sample the terminal price exactly
(no time-stepping). Three variance-reduction techniques are available:

* antithetic variates  - pair every normal draw z with -z
* control variate      - use the discounted terminal spot, whose expectation
                         S*exp(-qT) is known, to cancel sampling noise
* importance sampling  - for out-of-the-money options, shift the draws so the
                         median path finishes at the strike, then reweight each
                         path by the likelihood ratio so the estimate stays
                         unbiased. Otherwise most paths pay nothing.
                         Replaces the control variate when active.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import Option


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int
    shift: float = 0.0  # importance-sampling mean shift applied to the normal draws

    @property
    def ci95(self):
        half = 1.96 * self.std_error
        return self.price - half, self.price + half


def terminal_prices(opt: Option, z: np.ndarray) -> np.ndarray:
    drift = (opt.r - opt.q - 0.5 * opt.sigma**2) * opt.T
    return opt.S * np.exp(drift + opt.sigma * math.sqrt(opt.T) * z)


# Below this many standard deviations out of the money, the control variate
# beats importance sampling (measured break-even is ~0.9-1.0 across strikes,
# expiries and volatilities).
IS_MIN_SHIFT = 1.0


def strike_shift(opt: Option) -> float:
    """Mean shift for the normal draws that puts the median terminal price at K.

    Zero unless the option is at least IS_MIN_SHIFT standard deviations out of
    the money: closer in, most paths already pay out and the control variate
    does better.
    """
    drift = (opt.r - opt.q - 0.5 * opt.sigma**2) * opt.T
    z_strike = (math.log(opt.K / opt.S) - drift) / (opt.sigma * math.sqrt(opt.T))
    out_of_the_money = z_strike > 0 if opt.is_call else z_strike < 0
    return z_strike if out_of_the_money and abs(z_strike) >= IS_MIN_SHIFT else 0.0


def mc_price(
    opt: Option,
    n_paths: int = 100_000,
    seed: int | None = 42,
    antithetic: bool = True,
    control_variate: bool = True,
    importance_sampling: bool = True,
    shift: float | None = None,
) -> MCResult:
    """``shift`` overrides the automatic importance-sampling shift; pin it when
    bump-and-reprice Greeks must not switch IS on or off between bumps."""
    rng = np.random.default_rng(seed)
    disc = math.exp(-opt.r * opt.T)
    if shift is not None:
        mu = shift
    else:
        mu = strike_shift(opt) if importance_sampling else 0.0

    if antithetic:
        half = rng.standard_normal((n_paths + 1) // 2)
        eps_pairs = (half, -half)
    else:
        eps_pairs = (rng.standard_normal(n_paths),)

    z_pairs = [mu + eps for eps in eps_pairs]
    # Likelihood ratio N(0,1)/N(mu,1); identically 1 when mu = 0.
    weights = [np.exp(-mu * z + 0.5 * mu * mu) for z in z_pairs]
    st = [terminal_prices(opt, z) for z in z_pairs]
    # Antithetic pairs are correlated, so the independent samples are the pair averages.
    samples = sum(disc * opt.payoff(s) * w for s, w in zip(st, weights)) / len(st)

    # With a shift the control variate stops reducing variance, and estimating
    # beta from the same weighted samples biases the price, so skip it then.
    if control_variate and mu == 0.0:
        control = sum(disc * s * w for s, w in zip(st, weights)) / len(st)
        expected = opt.S * math.exp(-opt.q * opt.T)
        var_x = control.var(ddof=1)
        beta = np.cov(samples, control)[0, 1] / var_x if var_x > 0 else 0.0
        samples = samples - beta * (control - expected)

    price = float(samples.mean())
    std_error = float(samples.std(ddof=1) / math.sqrt(samples.size))
    return MCResult(price=price, std_error=std_error, n_paths=samples.size * len(st), shift=mu)
