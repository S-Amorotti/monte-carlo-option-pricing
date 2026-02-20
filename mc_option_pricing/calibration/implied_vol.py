from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..utils.black_scholes import bs_price


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    option_type: str,
) -> float:
    if price <= 0.0:
        return 0.0

    def objective(sigma: float) -> float:
        return bs_price(spot, strike, maturity, rate, dividend, sigma, option_type) - price

    return float(brentq(objective, 1e-6, 5.0, maxiter=200))


def implied_vol_surface(
    prices: np.ndarray,
    spots: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    rate: float,
    dividend: float,
    option_type: str,
) -> np.ndarray:
    vols = np.zeros_like(prices)
    for i, t in enumerate(maturities):
        for j, k in enumerate(strikes):
            vols[i, j] = implied_volatility(
                price=float(prices[i, j]),
                spot=spots,
                strike=float(k),
                maturity=float(t),
                rate=rate,
                dividend=dividend,
                option_type=option_type,
            )
    return vols
