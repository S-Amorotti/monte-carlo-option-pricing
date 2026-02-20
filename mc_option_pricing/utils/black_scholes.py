from __future__ import annotations

import math

from scipy.stats import norm


def bs_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    sigma: float,
    option_type: str,
) -> float:
    if maturity <= 0.0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    if sigma <= 0.0:
        forward = spot * math.exp((rate - dividend) * maturity)
        discount = math.exp(-rate * maturity)
        if option_type == "call":
            return discount * max(forward - strike, 0.0)
        return discount * max(strike - forward, 0.0)

    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * sigma**2) * maturity) / (
        sigma * math.sqrt(maturity)
    )
    d2 = d1 - sigma * math.sqrt(maturity)

    if option_type == "call":
        return spot * math.exp(-dividend * maturity) * norm.cdf(d1) - strike * math.exp(
            -rate * maturity
        ) * norm.cdf(d2)
    if option_type == "put":
        return strike * math.exp(-rate * maturity) * norm.cdf(-d2) - spot * math.exp(
            -dividend * maturity
        ) * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")
