import os
import pytest
from market_ec.data.price_loader import PriceRequest, download_prices
from market_ec.data.validators import validate_prices

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# Run only when explicitly allowed (so CI doesn't depend on the internet)
integration = pytest.mark.integration

@integration
@pytest.mark.skipif(os.getenv("ALLOW_NET_TESTS") != "1", reason="Net tests disabled by default")
def test_download_and_validate_integration():
    req = PriceRequest(tickers=["SPY"], start="2020-01-01", end="2020-03-01")
    df = download_prices(req)
    validate_prices(df)
    assert set(df["ticker"].unique()) == {"SPY"}
    assert len(df) > 0
