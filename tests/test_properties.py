from hypothesis import given, settings
from hypothesis import strategies as st

from mc_option_pricing.utils.black_scholes import bs_price


@given(
    spot=st.floats(min_value=50, max_value=150),
    strike_low=st.floats(min_value=50, max_value=150),
    strike_high=st.floats(min_value=50, max_value=150),
    maturity=st.floats(min_value=0.1, max_value=2.0),
    rate=st.floats(min_value=0.0, max_value=0.1),
    dividend=st.floats(min_value=0.0, max_value=0.05),
    sigma=st.floats(min_value=0.05, max_value=0.6),
)
@settings(max_examples=50)
def test_bs_call_monotone_in_strike(spot, strike_low, strike_high, maturity, rate, dividend, sigma):
    k1 = min(strike_low, strike_high)
    k2 = max(strike_low, strike_high)

    c1 = bs_price(spot, k1, maturity, rate, dividend, sigma, "call")
    c2 = bs_price(spot, k2, maturity, rate, dividend, sigma, "call")
    assert c1 >= c2
