import pandas as pd
from market_ec.data.returns import compute_log_returns


def test_compute_log_returns_basic():
    df = pd.DataFrame({
        "ticker": ["X", "X", "X", "Y", "Y"],
        "date": pd.to_datetime([
            "2020-01-01", "2020-01-02", "2020-01-03",
            "2020-01-01", "2020-01-02"
        ]),
        "open": [1, 1, 1, 1, 1],
        "high": [1, 1, 1, 1, 1],
        "low": [1, 1, 1, 1, 1],
        "close": [100, 110, 121, 200, 220],
        "adj_close": [100, 110, 121, 200, 220],
        "volume": [1000, 1000, 1000, 1000, 1000],
    })
    rets = compute_log_returns(df)
    # For X: log(110/100), log(121/110) ; For Y: log(220/200)
    assert len(rets) == 3
    assert set(rets["ticker"]) == {"X", "Y"}