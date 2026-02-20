# Monte Carlo Option Pricing

Production-grade Monte Carlo option pricing library and research repo built for clarity, correctness, and extensibility. The codebase implements European and American pricing under multiple stochastic models, variance reduction, calibration with robust validation, and reproducible experiments.

## Overview

**Key features**
- Models: Black–Scholes–Merton (GBM), Heston stochastic volatility, and Merton jump-diffusion scaffold.
- Payoffs: European vanilla and digital, Asian arithmetic average, and up-and-out barrier.
- Pricers: standard MC, antithetic variates, control variates, QMC (Sobol) for GBM.
- American options via Longstaff–Schwartz (LSM) with out-of-sample policy evaluation.
- Calibration engine with train/validation splits by maturity buckets.
- Deterministic RNG, explicit configs, and repeatable experiments.

**Design goals**
- Numerically stable implementations with explicit assumptions.
- Modular API for easy extension to new models and payoffs.
- Professional testing and CI with linting and type-checking.

## Installation

```bash
python -m pip install -e .[dev]
```

## Quickstart

```python
from mc_option_pricing.config import MarketParams, SimulationParams
from mc_option_pricing.models.gbm import GBMModel, GBMParams
from mc_option_pricing.payoffs.vanilla import VanillaPayoff
from mc_option_pricing.pricers.european_mc import EuropeanMCPricer

market = MarketParams(spot=100.0, rate=0.05, dividend=0.0)
model = GBMModel(GBMParams(sigma=0.2))
payoff = VanillaPayoff(strike=100.0, option_type="call")
sim = SimulationParams(n_paths=200000, n_steps=1, maturity=1.0, seed=7)

res = EuropeanMCPricer().price(model, market, payoff, sim)
print(res.price, res.stderr)
```

## Math Summary

### Risk-neutral dynamics
Let \( r \) be the risk-free rate and \( q \) the dividend yield. Under \( \mathbb{Q} \):

**GBM (Black–Scholes–Merton)**
\[
 dS_t = (r - q) S_t dt + \sigma S_t dW_t
\]
Exact discretization:
\[
 S_{t+\Delta t} = S_t \exp\left((r-q-\tfrac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z\right)
\]

**Heston (stochastic volatility)**
\[
 dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_t^S
\]
\[
 dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v
\]
with \( \mathrm{corr}(dW_t^S, dW_t^v)=\rho \).

Full truncation Euler for variance:
\[
 v_{t+\Delta t} = v_t + \kappa(\theta - \max(v_t, 0))\Delta t + \xi \sqrt{\max(v_t,0)}\sqrt{\Delta t} Z_2
\]

**Merton jump-diffusion (scaffold)**
\[
 dS_t = (r-q)S_t dt + \sigma S_t dW_t + (J-1) S_t dN_t
\]
with Poisson jumps and lognormal jump sizes.

### Discounting
All payoffs are discounted by \( e^{-rT} \). Dividend yield is included in the drift under \( \mathbb{Q} \).

### Variance reduction
- **Antithetic variates:** averages payoffs from \(Z\) and \(-Z\) to reduce variance when payoff is monotone.
- **Control variates:** uses a related payoff with known expectation (GBM European with BS closed-form) and applies
  \( \hat{X} = X - \beta (Y - \mathbb{E}[Y]) \) with \( \beta = \mathrm{Cov}(X,Y) / \mathrm{Var}(Y) \).
- **QMC (Sobol):** low-discrepancy sequences to reduce integration error for smooth payoffs.

### Longstaff–Schwartz for American options
The price satisfies a dynamic programming principle. LSM approximates the continuation value by regression:
\[
 C_t(S_t) \approx \sum_k \beta_k \phi_k(S_t)
\]
Exercise when immediate payoff exceeds estimated continuation value. This implementation:
- uses polynomial basis \( [1, S, S^2] \)
- includes ridge regularization
- **splits paths** into train/eval to avoid optimistic bias in exercise policy

## Calibration and Robust Validation

Calibration fits model parameters to a surface of option prices or implied volatilities. This repo provides:
- objectives: price SSE and implied-vol SSE (preferred)
- parameter bounds and feasibility checks
- multiple random starts to mitigate local minima
- maturity-based train/validation splits

Example workflow:
1. Generate or load market prices on a grid of strikes and maturities.
2. Split maturities: train on short/medium, validate on longer.
3. Calibrate parameters to the training surface.
4. Report in-sample vs out-of-sample errors.

## Reproducibility
- Deterministic RNG (`PCG64`), explicit `SimulationParams`.
- All examples accept seeds and run deterministically.
- Use `examples/` for fully-scripted experiments.

## Examples

```bash
python examples/01_black_scholes_european.py
python examples/02_variance_reduction_demo.py
python examples/03_heston_pricing_and_calibration.py
python examples/04_lsm_american.py
```

## Project Structure

```
mc_option_pricing/
  config.py
  rng.py
  models/
  payoffs/
  pricers/
  variance_reduction/
  calibration/
  diagnostics/
  utils/
examples/
tests/
```

## Limitations
- Heston discretization uses full truncation Euler (bias possible for coarse steps).
- Barrier monitoring is discrete and may underprice due to missed barrier crossings.
- QMC currently implemented for GBM European terminal-only pricing.
- Merton jump-diffusion is scaffolded but not fully integrated into pricers.

## Roadmap
- Andersen QE for Heston variance simulation
- Brownian bridge correction for barrier options
- Numba acceleration for path generation
- Pathwise/likelihood-ratio Greeks

## References
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation.
- Andersen, L., & Piterbarg, V. (2010). *Interest Rate Modeling* (volatility simulation discussion).

