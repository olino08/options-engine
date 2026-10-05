import math

import numpy as np
import pytest

from options_engine import (
    Option, binomial_greeks, binomial_price, bs_greeks, bs_price,
    implied_volatility, mc_price, numerical_greeks,
)
from options_engine.cli import main
from options_engine.compare import compare_greeks, compare_methods, fmt, put_call_parity_gap
from options_engine.monte_carlo import IS_MIN_SHIFT, strike_shift
from options_engine.strategies import STRATEGY_NAMES, build_strategy

ATM = Option(100, 100, 1.0, 0.05, 0.2)
CASES = [
    Option(100, 100, 1.0, 0.05, 0.2, "call"),
    Option(100, 100, 1.0, 0.05, 0.2, "put"),
    Option(100, 120, 0.25, 0.03, 0.35, "call", q=0.02),
    Option(100, 80, 2.0, 0.01, 0.15, "put", q=0.04),
]

DEEP_OTM = [
    Option(100, 150, 0.5, 0.05, 0.2, "call"),
    Option(100, 200, 1.0, 0.05, 0.25, "call"),
    Option(100, 70, 0.5, 0.05, 0.2, "put"),
]


def test_black_scholes_textbook_values():
    # Hull, Options, Futures & Other Derivatives, Example 15.6
    call = Option(42, 40, 0.5, 0.10, 0.20, "call")
    assert bs_price(call) == pytest.approx(4.76, abs=0.005)
    assert bs_price(call.with_(option_type="put")) == pytest.approx(0.81, abs=0.005)
    assert bs_price(ATM) == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("opt", CASES)
def test_put_call_parity(opt):
    assert abs(put_call_parity_gap(opt)) < 1e-10


@pytest.mark.parametrize("opt", CASES)
def test_binomial_converges_to_black_scholes(opt):
    assert binomial_price(opt, steps=2000) == pytest.approx(bs_price(opt), abs=5e-3)


@pytest.mark.parametrize("opt", CASES + DEEP_OTM)
@pytest.mark.parametrize("antithetic,cv", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("use_is", [False, True])
def test_monte_carlo_within_confidence(opt, antithetic, cv, use_is):
    res = mc_price(opt, n_paths=200_000, seed=7, antithetic=antithetic, control_variate=cv, importance_sampling=use_is)
    assert abs(res.price - bs_price(opt)) < 4 * res.std_error


def test_monte_carlo_error_bars_are_calibrated():
    z = [(r.price - bs_price(ATM)) / r.std_error for r in (mc_price(ATM, 20_000, seed=s) for s in range(100))]
    assert abs(np.mean(z)) < 0.4
    assert 0.75 < np.std(z) < 1.3


@pytest.mark.parametrize("opt", DEEP_OTM)
def test_importance_sampling_is_unbiased_and_calibrated(opt):
    results = [mc_price(opt, 20_000, seed=s) for s in range(200)]
    assert all(r.shift != 0 for r in results)
    z = np.array([(r.price - bs_price(opt)) / r.std_error for r in results])
    assert abs(z.mean()) < 0.25
    assert 0.85 < z.std() < 1.15


@pytest.mark.parametrize("opt", DEEP_OTM)
def test_importance_sampling_cuts_variance_for_deep_otm(opt):
    plain = mc_price(opt, 100_000, importance_sampling=False)
    shifted = mc_price(opt, 100_000)
    assert (plain.std_error / shifted.std_error) ** 2 > 100


@pytest.mark.parametrize("opt", [
    Option(100, 100, 1.0, 0.05, 0.2, "call"),  # at the money
    Option(100, 80, 1.0, 0.05, 0.2, "call"),   # in the money
    Option(100, 120, 1.0, 0.05, 0.2, "put"),   # in the money
    Option(100, 105, 1.0, 0.05, 0.2, "call"),  # out of the money, but under the threshold
])
def test_importance_sampling_only_kicks_in_beyond_threshold(opt):
    assert strike_shift(opt) == 0.0
    assert mc_price(opt, 10_000).price == mc_price(opt, 10_000, importance_sampling=False).price


def test_shift_points_toward_strike():
    assert strike_shift(Option(100, 150, 0.5, 0.05, 0.2, "call")) > IS_MIN_SHIFT
    assert strike_shift(Option(100, 70, 0.5, 0.05, 0.2, "put")) < -IS_MIN_SHIFT


def test_mc_greeks_with_importance_sampling():
    opt = DEEP_OTM[0]
    rows = compare_greeks(opt, mc_paths=100_000, tree_steps=200)
    mc, bs = rows["Monte Carlo (finite diff.)"], bs_greeks(opt)
    for name in ("delta", "gamma", "vega", "theta"):
        assert getattr(mc, name) == pytest.approx(getattr(bs, name), rel=0.1), name


def test_variance_reduction_helps():
    plain = mc_price(ATM, 100_000, antithetic=False, control_variate=False)
    both = mc_price(ATM, 100_000)
    assert both.std_error < plain.std_error / 3


@pytest.mark.parametrize("opt", CASES)
def test_analytic_greeks_match_finite_differences(opt):
    analytic = bs_greeks(opt)
    fd = numerical_greeks(opt, bs_price, spot_bump=1e-3, vol_bump=1e-4, time_bump=1e-5)
    for name in ("delta", "gamma", "vega", "theta"):
        assert getattr(fd, name) == pytest.approx(getattr(analytic, name), rel=1e-3, abs=1e-6), name


@pytest.mark.parametrize("opt", CASES)
def test_binomial_greeks_match_analytic(opt):
    tree, bs = binomial_greeks(opt, steps=1000), bs_greeks(opt)
    assert tree.delta == pytest.approx(bs.delta, abs=2e-3)
    assert tree.gamma == pytest.approx(bs.gamma, rel=0.02)
    assert tree.vega == pytest.approx(bs.vega, rel=0.01)
    assert tree.theta == pytest.approx(bs.theta, rel=0.03, abs=2e-4)


def test_greek_signs_and_relationships():
    call, put = bs_greeks(ATM), bs_greeks(ATM.with_(option_type="put"))
    assert 0 < call.delta < 1 and -1 < put.delta < 0
    assert call.delta - put.delta == pytest.approx(1.0)  # q = 0
    assert call.gamma == pytest.approx(put.gamma)
    assert call.vega == pytest.approx(put.vega)
    assert call.theta < 0


def test_implied_vol_round_trip():
    opt = Option(100, 110, 0.5, 0.03, 0.27, "put")
    assert implied_volatility(opt.with_(sigma=0.5), bs_price(opt)) == pytest.approx(0.27, abs=1e-8)


def test_american_put_worth_at_least_european():
    put = ATM.with_(option_type="put")
    assert binomial_price(put, 500, american=True) > binomial_price(put, 500)
    # No early exercise premium for a call without dividends.
    assert binomial_price(ATM, 500, american=True) == pytest.approx(binomial_price(ATM, 500))


@pytest.mark.parametrize("bad", [dict(S=-1), dict(T=0), dict(sigma=0)])
def test_invalid_inputs_rejected(bad):
    args = {**dict(S=100, K=100, T=1, r=0.05, sigma=0.2), **bad}
    with pytest.raises(ValueError):
        Option(**args)


def test_strategy_breakevens():
    straddle = build_strategy("straddle", 100, 10)
    prem = straddle.premium(100, 1, 0.05, 0.2)
    lo, hi = straddle.breakevens(prem, 50, 150)
    assert lo == pytest.approx(100 - prem, abs=1e-3)
    assert hi == pytest.approx(100 + prem, abs=1e-3)


def test_butterfly_payoff_shape():
    fly = build_strategy("butterfly", 100, 10)
    assert fly.payoff([80, 90, 100, 110, 120]).tolist() == [0, 0, 10, 0, 0]


def test_compare_methods_agree():
    rows = compare_methods(ATM, mc_paths=100_000, tree_steps=500)
    assert [r.method.split(" ")[0] for r in rows] == ["Black-Scholes", "Monte", "Binomial"]
    assert all(abs(r.error) < 0.05 for r in rows)


def test_compare_shows_both_mc_rows_when_importance_sampling_applies():
    rows = compare_methods(DEEP_OTM[0], mc_paths=100_000, tree_steps=500)
    assert [r.method.split(" ")[0] for r in rows] == ["Black-Scholes", "Monte", "MC", "Binomial"]
    assert rows[2].std_error < rows[1].std_error / 10
    no_is = compare_methods(DEEP_OTM[0], mc_paths=100_000, tree_steps=500, importance_sampling=False)
    assert len(no_is) == 3


def test_fmt_keeps_small_values_visible():
    assert fmt(10.45058) == "10.4506"
    assert fmt(3.15e-05) == "3.15e-05"
    assert fmt(-5.89e-05, sign=True) == "-5.89e-05"
    assert fmt(0.0) == "0.0000"


@pytest.mark.parametrize("argv", [
    ["price"], ["compare", "--paths", "10000"], ["greeks", "--paths", "10000", "--steps", "100"],
    ["greeks", "--analytic-only", "--type", "put"],
    ["compare", "-K", "150", "-T", "0.5", "--paths", "10000"],
    ["compare", "-K", "150", "-T", "0.5", "--paths", "10000", "--no-is"],
    ["price", "-K", "70", "--type", "put", "--method", "mc"],
])
def test_cli_text_commands(argv, capsys):
    main(argv)
    assert "European" in capsys.readouterr().out


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_plot_payoff(strategy, tmp_path):
    out = tmp_path / f"{strategy}.png"
    main(["plot", "payoff", "--strategy", strategy, "-o", str(out)])
    assert out.stat().st_size > 10_000


@pytest.mark.parametrize("chart", ["greeks", "convergence"])
def test_plot_other_charts(chart, tmp_path):
    out = tmp_path / f"{chart}.png"
    main(["plot", chart, "-o", str(out)])
    assert out.stat().st_size > 10_000
