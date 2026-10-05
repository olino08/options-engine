"""Multi-leg positions for payoff diagrams (straddles, spreads, ...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .black_scholes import bs_price
from .models import Option, OptionType


@dataclass(frozen=True)
class Leg:
    option_type: OptionType
    strike: float
    quantity: float = 1.0  # positive = long, negative = short


@dataclass(frozen=True)
class Strategy:
    name: str
    legs: List[Leg]

    def options(self, S: float, T: float, r: float, sigma: float, q: float = 0.0):
        return [(Option(S, leg.strike, T, r, sigma, leg.option_type, q), leg.quantity) for leg in self.legs]

    def premium(self, S, T, r, sigma, q=0.0) -> float:
        """Net cost to open (positive = debit paid)."""
        return sum(qty * bs_price(o) for o, qty in self.options(S, T, r, sigma, q))

    def payoff(self, spots) -> np.ndarray:
        spots = np.asarray(spots, dtype=float)
        total = np.zeros_like(spots)
        for leg in self.legs:
            intrinsic = np.maximum(spots - leg.strike, 0) if leg.option_type is OptionType.CALL else np.maximum(leg.strike - spots, 0)
            total += leg.quantity * intrinsic
        return total

    def value_today(self, spots, T, r, sigma, q=0.0) -> np.ndarray:
        """Black-Scholes mark-to-market of the position as a function of today's spot."""
        return np.array([sum(qty * bs_price(o) for o, qty in self.options(s, T, r, sigma, q)) for s in spots])

    def breakevens(self, premium: float, lo: float, hi: float, n: int = 20001):
        spots = np.linspace(lo, hi, n)
        pnl = self.payoff(spots) - premium
        crossings = np.where(np.diff(np.sign(pnl)) != 0)[0]
        # Linear interpolation between grid points on either side of each sign change.
        return [float(spots[i] - pnl[i] * (spots[i + 1] - spots[i]) / (pnl[i + 1] - pnl[i])) for i in crossings]


C, P = OptionType.CALL, OptionType.PUT


def build_strategy(name: str, K: float, width: float) -> Strategy:
    """Preset strategies centred on strike K; ``width`` spaces the outer strikes."""
    presets = {
        "long-call": [Leg(C, K, 1)],
        "long-put": [Leg(P, K, 1)],
        "short-call": [Leg(C, K, -1)],
        "short-put": [Leg(P, K, -1)],
        "straddle": [Leg(C, K, 1), Leg(P, K, 1)],
        "strangle": [Leg(P, K - width, 1), Leg(C, K + width, 1)],
        "bull-call-spread": [Leg(C, K, 1), Leg(C, K + width, -1)],
        "bear-put-spread": [Leg(P, K, 1), Leg(P, K - width, -1)],
        "butterfly": [Leg(C, K - width, 1), Leg(C, K, -2), Leg(C, K + width, 1)],
        "iron-condor": [
            Leg(P, K - 2 * width, 1), Leg(P, K - width, -1),
            Leg(C, K + width, -1), Leg(C, K + 2 * width, 1),
        ],
    }
    if name not in presets:
        raise ValueError(f"unknown strategy {name!r}; choose from {', '.join(presets)}")
    return Strategy(name, presets[name])


STRATEGY_NAMES = [
    "long-call", "long-put", "short-call", "short-put", "straddle", "strangle",
    "bull-call-spread", "bear-put-spread", "butterfly", "iron-condor",
]
