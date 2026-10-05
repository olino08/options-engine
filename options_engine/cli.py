"""Command-line interface.

    python -m options_engine price   --type put
    python -m options_engine compare
    python -m options_engine greeks  --type call
    python -m options_engine plot    payoff --strategy straddle
    python -m options_engine plot    greeks
    python -m options_engine plot    convergence
"""

from __future__ import annotations

import argparse

from .binomial import binomial_price
from .black_scholes import bs_greeks, bs_price
from .compare import compare_greeks, compare_methods, fmt, format_greeks_table, put_call_parity_gap
from .models import Option
from .monte_carlo import mc_price
from .strategies import STRATEGY_NAMES, build_strategy


def _add_market_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-S", "--spot", type=float, default=100.0, help="spot price (default 100)")
    p.add_argument("-K", "--strike", type=float, default=100.0, help="strike (default 100)")
    p.add_argument("-T", "--expiry", type=float, default=1.0, help="years to expiry (default 1)")
    p.add_argument("-r", "--rate", type=float, default=0.05, help="risk-free rate (default 0.05)")
    p.add_argument("-v", "--vol", type=float, default=0.20, help="volatility (default 0.20)")
    p.add_argument("-q", "--div", type=float, default=0.0, help="dividend yield (default 0)")
    p.add_argument("--type", choices=["call", "put"], default="call")


def _option(a) -> Option:
    return Option(a.spot, a.strike, a.expiry, a.rate, a.vol, a.type, a.div)


def _describe(opt: Option) -> str:
    return (f"European {opt.option_type.value}: S={opt.S:g} K={opt.K:g} T={opt.T:g}y "
            f"r={opt.r:.2%} σ={opt.sigma:.2%} q={opt.q:.2%}")


def cmd_price(a):
    opt = _option(a)
    print(_describe(opt))
    if a.method in ("bs", "all"):
        print(f"  Black-Scholes : {fmt(bs_price(opt))}")
    if a.method in ("mc", "all"):
        res = mc_price(opt, n_paths=a.paths, seed=a.seed, importance_sampling=not a.no_is)
        lo, hi = res.ci95
        print(f"  Monte Carlo   : {fmt(res.price)}  ± {fmt(res.std_error)} s.e.  (95% CI {fmt(lo)}–{fmt(hi)})")
        if res.shift:
            print(f"                  importance sampling: draws shifted by {res.shift:+.2f} std devs")
    if a.method in ("tree", "all"):
        print(f"  Binomial      : {fmt(binomial_price(opt, steps=a.steps))}  ({a.steps} steps)")


def cmd_compare(a):
    opt = _option(a)
    print(_describe(opt) + "\n")
    header = f"{'Method':<32}{'Price':>10}{'vs BS':>11}{'Std err':>10}{'Time (ms)':>11}"
    print(header)
    print("-" * len(header))
    for row in compare_methods(opt, mc_paths=a.paths, tree_steps=a.steps, seed=a.seed, importance_sampling=not a.no_is):
        se = fmt(row.std_error) if row.std_error is not None else "—"
        print(f"{row.method:<32}{fmt(row.price):>10}{fmt(row.error, sign=True):>11}{se:>10}{row.millis:>11.2f}")
    print(f"\nPut-call parity gap (BS): {put_call_parity_gap(opt):.2e}")


def cmd_greeks(a):
    opt = _option(a)
    print(_describe(opt) + "\n")
    if a.analytic_only:
        print(format_greeks_table({"Black-Scholes (analytic)": bs_greeks(opt)}))
    else:
        rows = compare_greeks(opt, mc_paths=a.paths, tree_steps=a.steps, seed=a.seed, importance_sampling=not a.no_is)
        print(format_greeks_table(rows))


def cmd_plot(a):
    from . import plotting

    if a.chart == "payoff":
        strategy = build_strategy(a.strategy, a.strike, a.width)
        out = plotting.plot_payoff(strategy, a.spot, a.expiry, a.rate, a.vol, a.div, path=a.out or f"payoff_{a.strategy}.png")
    elif a.chart == "greeks":
        out = plotting.plot_greeks(a.strike, a.expiry, a.rate, a.vol, a.div, path=a.out or "greeks.png")
    else:
        out = plotting.plot_convergence(_option(a), path=a.out or f"convergence_{a.type}.png", seed=a.seed)
    print(f"Saved {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="options_engine", description="European option pricing engine")
    sub = parser.add_subparsers(dest="command", required=True)

    def numeric(p):
        p.add_argument("--paths", type=int, default=200_000, help="Monte Carlo paths")
        p.add_argument("--steps", type=int, default=1000, help="binomial tree steps")
        p.add_argument("--seed", type=int, default=42, help="Monte Carlo RNG seed")
        p.add_argument("--no-is", action="store_true",
                       help="disable importance sampling for deep out-of-the-money options")

    p = sub.add_parser("price", help="price one option")
    _add_market_args(p)
    numeric(p)
    p.add_argument("--method", choices=["bs", "mc", "tree", "all"], default="all")
    p.set_defaults(func=cmd_price)

    p = sub.add_parser("compare", help="Black-Scholes vs Monte Carlo vs binomial")
    _add_market_args(p)
    numeric(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("greeks", help="delta, gamma, vega, theta")
    _add_market_args(p)
    numeric(p)
    p.add_argument("--analytic-only", action="store_true", help="skip the finite-difference engines")
    p.set_defaults(func=cmd_greeks, steps=500)

    p = sub.add_parser("plot", help="save a chart as PNG")
    p.add_argument("chart", choices=["payoff", "greeks", "convergence"])
    _add_market_args(p)
    p.add_argument("--strategy", choices=STRATEGY_NAMES, default="long-call", help="payoff preset")
    p.add_argument("--width", type=float, default=10.0, help="strike spacing for spreads/strangles")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--out", help="output PNG path")
    p.set_defaults(func=cmd_plot)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
