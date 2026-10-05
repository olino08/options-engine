"""Core data types."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Option:
    """A European option on a (possibly dividend-paying) underlying.

    S      spot price of the underlying
    K      strike
    T      time to expiry in years
    r      continuously compounded risk-free rate (0.05 = 5%)
    sigma  annualised volatility (0.2 = 20%)
    q      continuous dividend yield
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType = OptionType.CALL
    q: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        if self.S <= 0 or self.K <= 0:
            raise ValueError("S and K must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL

    def payoff(self, spot):
        """Intrinsic value at expiry for a scalar or array of terminal spot prices."""
        spot = np.asarray(spot, dtype=float)
        if self.is_call:
            return np.maximum(spot - self.K, 0.0)
        return np.maximum(self.K - spot, 0.0)

    def with_(self, **changes) -> "Option":
        """Return a copy with some fields changed (used for bump-and-reprice)."""
        return replace(self, **changes)


@dataclass(frozen=True)
class Greeks:
    """Option sensitivities in trader conventions.

    delta  dV/dS
    gamma  d2V/dS2
    vega   change in V for a 1 percentage-point rise in volatility
    theta  change in V for one calendar day passing (negative = time decay)
    """

    delta: float
    gamma: float
    vega: float
    theta: float

    def as_dict(self) -> dict:
        return {"delta": self.delta, "gamma": self.gamma, "vega": self.vega, "theta": self.theta}
