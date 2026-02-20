from __future__ import annotations

import math
from typing import cast

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
    cdf_d1 = float(cast(float, norm.cdf(d1)))
    cdf_d2 = float(cast(float, norm.cdf(d2)))
    cdf_neg_d1 = float(cast(float, norm.cdf(-d1)))
    cdf_neg_d2 = float(cast(float, norm.cdf(-d2)))

    if option_type == "call":
        return spot * math.exp(-dividend * maturity) * cdf_d1 - strike * math.exp(
            -rate * maturity
        ) * cdf_d2
    if option_type == "put":
        return strike * math.exp(-rate * maturity) * cdf_neg_d2 - spot * math.exp(
            -dividend * maturity
        ) * cdf_neg_d1
    raise ValueError("option_type must be 'call' or 'put'")
