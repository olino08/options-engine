"""Matplotlib charts: payoff diagrams, Greeks profiles and method convergence."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .binomial import binomial_price  # noqa: E402
from .black_scholes import bs_greeks, bs_price  # noqa: E402
from .models import Option  # noqa: E402
from .monte_carlo import mc_price  # noqa: E402
from .strategies import Strategy  # noqa: E402

# Validated categorical palette (fixed order) plus neutral ink for text and axes.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, **kw)
    return leg


def _figure(nrows=1, ncols=1, size=(9, 5.5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE)
    return fig, axes


def plot_payoff(strategy: Strategy, S, T, r, sigma, q=0.0, path="payoff.png"):
    strikes = [leg.strike for leg in strategy.legs]
    lo, hi = 0.5 * min(strikes + [S]), 1.5 * max(strikes + [S])
    spots = np.linspace(lo, hi, 400)

    premium = strategy.premium(S, T, r, sigma, q)
    payoff = strategy.payoff(spots)
    pnl_expiry = payoff - premium
    pnl_today = strategy.value_today(spots, T, r, sigma, q) - premium

    fig, ax = _figure()
    _style(ax, "", "Underlying price", "Profit / loss per unit")
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.fill_between(spots, pnl_expiry, 0, where=pnl_expiry >= 0, color=SERIES[0], alpha=0.08, linewidth=0)
    ax.fill_between(spots, pnl_expiry, 0, where=pnl_expiry < 0, color=SERIES[1], alpha=0.08, linewidth=0)
    ax.plot(spots, pnl_expiry, color=SERIES[0], linewidth=2, label="P&L at expiry")
    ax.plot(spots, pnl_today, color=SERIES[1], linewidth=2, linestyle="--", label=f"P&L today ({T:.2f}y left)")
    ax.plot(spots, payoff, color=SERIES[2], linewidth=1.5, linestyle=":", label="Payoff (excl. premium)")

    for k in sorted(set(strikes)):
        ax.axvline(k, color=GRID, linewidth=1, zorder=0)
        ax.annotate(f"K={k:g}", (k, ax.get_ylim()[0]), xytext=(3, 4), textcoords="offset points", fontsize=8, color=INK_MUTED)
    for be in strategy.breakevens(premium, lo, hi):
        ax.plot([be], [0], "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(f"BE {be:.2f}", (be, 0), xytext=(4, -14), textcoords="offset points", fontsize=8, color=INK)
    ax.axvline(S, color=INK_MUTED, linewidth=1, linestyle=(0, (2, 3)))
    ax.annotate(f"spot {S:g}", (S, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK_MUTED)

    cost = f"net debit {premium:.2f}" if premium >= 0 else f"net credit {-premium:.2f}"
    ax.set_title(f"{strategy.name}: profit & loss  ·  {cost}", loc="left", color=INK, fontsize=11, fontweight="semibold", pad=28)
    _legend(ax, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_greeks(K, T, r, sigma, q=0.0, path="greeks.png"):
    spots = np.linspace(0.5 * K, 1.5 * K, 200)
    data = {}
    for kind in ("call", "put"):
        gs = [bs_greeks(Option(s, K, T, r, sigma, kind, q)) for s in spots]
        data[kind] = {name: np.array([getattr(g, name) for g in gs]) for name in ("delta", "gamma", "vega", "theta")}

    fig, axes = _figure(2, 2, size=(10, 7))
    titles = {
        "delta": ("Delta", "dV/dS"),
        "gamma": ("Gamma", "d²V/dS²"),
        "vega": ("Vega", "per 1 vol point"),
        "theta": ("Theta", "per calendar day"),
    }
    for ax, (name, (title, unit)) in zip(axes.flat, titles.items()):
        _style(ax, title, "Spot", unit)
        ax.axvline(K, color=GRID, linewidth=1, zorder=0)
        ax.plot(spots, data["call"][name], color=SERIES[0], linewidth=2, label="Call")
        ax.plot(spots, data["put"][name], color=SERIES[1], linewidth=2, linestyle="--", label="Put")
        _legend(ax, loc="best")
    axes.flat[1].annotate("call & put gamma are identical", (0.98, 0.95), xycoords="axes fraction", ha="right", fontsize=8, color=INK_MUTED)
    axes.flat[2].annotate("call & put vega are identical", (0.98, 0.95), xycoords="axes fraction", ha="right", fontsize=8, color=INK_MUTED)
    fig.suptitle(f"Black-Scholes Greeks  (K={K:g}, T={T:g}y, r={r:.1%}, σ={sigma:.0%})", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_convergence(opt: Option, path="convergence.png", seed: int = 42):
    """How the binomial and Monte Carlo prices converge to the Black-Scholes value."""
    exact = bs_price(opt)
    fig, (ax_tree, ax_mc) = _figure(1, 2, size=(11, 4.8))

    steps = np.arange(10, 301)
    tree = np.array([binomial_price(opt, steps=int(n)) for n in steps])
    _style(ax_tree, "Binomial tree vs steps", "Tree steps", "Option price")
    ax_tree.axhline(exact, color=INK_MUTED, linewidth=1, linestyle="--", label=f"Black-Scholes {exact:.4f}")
    ax_tree.plot(steps, tree, color=SERIES[0], linewidth=1, label="CRR binomial")
    ax_tree.annotate("odd/even oscillation shrinks as O(1/N)", (0.98, 0.04), xycoords="axes fraction", ha="right", fontsize=8, color=INK_MUTED)
    _legend(ax_tree, loc="upper right")

    paths = np.unique(np.logspace(2, 6, 25).astype(int))
    # A different seed per point keeps the estimates independent of each other.
    results = [mc_price(opt, n_paths=int(n), seed=seed + i, control_variate=False, importance_sampling=False) for i, n in enumerate(paths)]
    prices = np.array([res.price for res in results])
    half = 1.96 * np.array([res.std_error for res in results])
    _style(ax_mc, "Monte Carlo vs paths (antithetic only)", "Paths (log scale)", "Option price")
    ax_mc.set_xscale("log")
    ax_mc.fill_between(paths, prices - half, prices + half, color=SERIES[1], alpha=0.15, linewidth=0, label="95% confidence band")
    ax_mc.axhline(exact, color=INK_MUTED, linewidth=1, linestyle="--", label=f"Black-Scholes {exact:.4f}")
    ax_mc.plot(paths, prices, color=SERIES[1], linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE, label="MC estimate")
    _legend(ax_mc, loc="lower right")

    fig.suptitle(f"Convergence to Black-Scholes  ({opt.option_type.value}, S={opt.S:g}, K={opt.K:g}, T={opt.T:g}y)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
