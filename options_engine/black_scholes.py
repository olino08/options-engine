"""Closed-form Black-Scholes-Merton pricing and analytic Greeks."""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

from .models import Greeks, Option

DAYS_PER_YEAR = 365.0


def _d1_d2(opt: Option):
    vol_sqrt_t = opt.sigma * math.sqrt(opt.T)
    d1 = (math.log(opt.S / opt.K) + (opt.r - opt.q + 0.5 * opt.sigma**2) * opt.T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(opt: Option) -> float:
    d1, d2 = _d1_d2(opt)
    df_r = math.exp(-opt.r * opt.T)
    df_q = math.exp(-opt.q * opt.T)
    if opt.is_call:
        return opt.S * df_q * norm.cdf(d1) - opt.K * df_r * norm.cdf(d2)
    return opt.K * df_r * norm.cdf(-d2) - opt.S * df_q * norm.cdf(-d1)


def bs_greeks(opt: Option) -> Greeks:
    d1, d2 = _d1_d2(opt)
    sqrt_t = math.sqrt(opt.T)
    df_r = math.exp(-opt.r * opt.T)
    df_q = math.exp(-opt.q * opt.T)
    pdf_d1 = norm.pdf(d1)

    gamma = df_q * pdf_d1 / (opt.S * opt.sigma * sqrt_t)
    vega = opt.S * df_q * pdf_d1 * sqrt_t  # per 1.00 of vol
    decay = -opt.S * df_q * pdf_d1 * opt.sigma / (2 * sqrt_t)

    if opt.is_call:
        delta = df_q * norm.cdf(d1)
        theta = decay - opt.r * opt.K * df_r * norm.cdf(d2) + opt.q * opt.S * df_q * norm.cdf(d1)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = decay + opt.r * opt.K * df_r * norm.cdf(-d2) - opt.q * opt.S * df_q * norm.cdf(-d1)

    return Greeks(delta=delta, gamma=gamma, vega=vega / 100.0, theta=theta / DAYS_PER_YEAR)


def implied_volatility(opt: Option, market_price: float, lo: float = 1e-4, hi: float = 5.0) -> float:
    """Volatility that makes the Black-Scholes price equal ``market_price``."""
    return brentq(lambda s: bs_price(opt.with_(sigma=s)) - market_price, lo, hi, xtol=1e-10)
