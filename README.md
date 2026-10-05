# Options pricing engine

Prices European calls and puts three ways, computes Greeks, and draws payoff diagrams.

| Method | Module | Notes |
|---|---|---|
| Black-Scholes-Merton | `black_scholes.py` | Closed form with dividend yield `q`, analytic Greeks, implied vol |
| Monte Carlo | `monte_carlo.py` | Exact GBM terminal sampling, antithetic variates, control variate, importance sampling for deep out-of-the-money options, standard error & 95% CI |
| Binomial (CRR) | `binomial.py` | Vectorised backward induction; Greeks read from tree nodes; optional American exercise |

Greeks follow trader conventions: **vega per 1 vol point**, **theta per calendar day**.
`greeks.py` provides bump-and-reprice Greeks for any pricer, so Monte Carlo Greeks can be checked against the analytic values.

## Importance sampling

When an option is at least one standard deviation out of the money, most simulated paths would expire worthless. The Monte Carlo pricer then shifts its normal draws so the median path finishes at the strike, and reweights each path by the likelihood ratio so the price stays unbiased. In that mode it replaces the control variate, because combining the two adds a small bias without lowering the error.

| Option | Variance reduction |
|---|---|
| Call, K=120, 3 months, σ=35% | ~2× |
| Put, K=70, 6 months, σ=20% | ~650× |
| Call, K=150, 6 months, σ=20% | ~1,150× |

Closer to the money, the control variate does better, so importance sampling switches off automatically (`IS_MIN_SHIFT` in `monte_carlo.py`). `compare` shows the plain and importance-sampled Monte Carlo rows side by side whenever it applies; pass `--no-is` to turn it off.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Usage

Defaults: S=100, K=100, T=1y, r=5%, σ=20%, q=0, call. Override them with `-S -K -T -r -v -q --type`.

```bash
python -m options_engine price --type put
python -m options_engine compare -K 110 -q 0.02 --type put
python -m options_engine compare -K 150 -T 0.5          # deep OTM: importance sampling kicks in
python -m options_engine greeks
python -m options_engine plot payoff --strategy iron-condor --width 10
python -m options_engine plot greeks
python -m options_engine plot convergence
```

Payoff presets: long/short call/put, straddle, strangle, bull-call-spread, bear-put-spread, butterfly, iron-condor.

As a library:

```python
from options_engine import Option, bs_price, bs_greeks, mc_price, binomial_price

opt = Option(S=100, K=105, T=0.5, r=0.04, sigma=0.25, option_type="put")
bs_price(opt), mc_price(opt).price, binomial_price(opt, steps=1000), bs_greeks(opt)
```

## Tests

```bash
.venv/bin/python -m pytest
```

The tests check textbook values (Hull), put-call parity, that the binomial and Monte Carlo prices converge to Black-Scholes, that MC error bars are calibrated, that analytic Greeks match finite differences, and that the CLI and plots run end to end.
