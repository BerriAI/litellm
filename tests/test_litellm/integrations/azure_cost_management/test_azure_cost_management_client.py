from datetime import date
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.integrations.azure_cost_management.azure_cost_management_client import (
    AzureCostManagementClient,
    AzureCostManagementConfig,
    AzureCostManagementError,
)


def _config() -> AzureCostManagementConfig:
    return AzureCostManagementConfig(
        subscription_id="00000000-0000-0000-0000-000000000000",
        tenant_id="tenant",
        client_id="client",
        client_secret="secret",
    )


def _http_ok(payload: dict) -> MagicMock:
    http = MagicMock()
    response = MagicMock()
    response.json.return_value = payload
    http.post = AsyncMock(return_value=response)
    return http


@pytest.mark.asyncio
async def test_get_daily_cost_returns_sum_of_rows_when_usd():
    payload = {
        "properties": {
            "columns": [{"name": "Cost"}, {"name": "UsageDate"}, {"name": "Currency"}],
            "rows": [
                [120.0, 20260715, "USD"],
                [30.0, 20260715, "USD"],
            ],
        }
    }
    http = _http_ok(payload)

    client = AzureCostManagementClient(
        config=_config(),
        http_handler=http,
        token_provider=lambda: "fake-token",
    )

    result = await client.get_daily_cost("/subs/x/deploy/y", date(2026, 7, 15))

    assert result == pytest.approx(150.0)
    assert client.last_currency == "USD"


@pytest.mark.asyncio
async def test_get_daily_cost_returns_zero_when_no_rows():
    """Azure returns 200 with empty rows when reporting lag hasn't populated yet."""
    payload = {"properties": {"columns": [{"name": "Cost"}], "rows": []}}
    http = _http_ok(payload)
    client = AzureCostManagementClient(
        config=_config(),
        http_handler=http,
        token_provider=lambda: "fake-token",
    )

    result = await client.get_daily_cost("/subs/x/deploy/y", date(2026, 7, 15))

    assert result == 0.0
    assert client.last_currency is None


@pytest.mark.asyncio
async def test_get_daily_cost_records_non_usd_currency():
    payload = {
        "properties": {
            "columns": [{"name": "Cost"}, {"name": "Currency"}],
            "rows": [[100.0, "EUR"]],
        }
    }
    http = _http_ok(payload)
    client = AzureCostManagementClient(
        config=_config(),
        http_handler=http,
        token_provider=lambda: "fake-token",
    )

    result = await client.get_daily_cost("/subs/x/deploy/y", date(2026, 7, 15))

    assert result == 100.0
    assert client.last_currency == "EUR"


@pytest.mark.asyncio
async def test_get_daily_cost_raises_on_http_error():
    http = MagicMock()
    response = MagicMock()
    response.status_code = 403
    response.text = "forbidden"
    http.post = AsyncMock(side_effect=httpx.HTTPStatusError("forbidden", request=MagicMock(), response=response))

    client = AzureCostManagementClient(
        config=_config(),
        http_handler=http,
        token_provider=lambda: "fake-token",
    )

    with pytest.raises(AzureCostManagementError) as exc:
        await client.get_daily_cost("/subs/x/deploy/y", date(2026, 7, 15))
    assert "403" in str(exc.value)


@pytest.mark.asyncio
async def test_get_daily_cost_posts_expected_body_shape():
    payload = {"properties": {"columns": [{"name": "Cost"}], "rows": []}}
    http = _http_ok(payload)
    client = AzureCostManagementClient(
        config=_config(),
        http_handler=http,
        token_provider=lambda: "fake-token",
    )

    await client.get_daily_cost("/subs/x/deploy/y", date(2026, 7, 15))

    http.post.assert_awaited_once()
    kwargs = http.post.await_args.kwargs
    assert kwargs["url"].endswith("/providers/Microsoft.CostManagement/query")
    assert kwargs["params"] == {"api-version": "2023-11-01"}
    assert kwargs["headers"]["Authorization"] == "Bearer fake-token"
    body = kwargs["json"]
    assert body["type"] == "ActualCost"
    assert body["timeframe"] == "Custom"
    assert body["timePeriod"]["from"] == "2026-07-15T00:00:00Z"
    assert body["dataset"]["filter"]["dimensions"]["values"] == ["/subs/x/deploy/y"]


def test_config_from_env_raises_when_creds_missing(monkeypatch):
    for k in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)

    with pytest.raises(AzureCostManagementError) as exc:
        AzureCostManagementConfig.from_env(subscription_id="sub-x")
    assert "AZURE_TENANT_ID" in str(exc.value)


def test_client_secret_not_in_repr():
    """Regression: client_secret must not surface in repr(config) — the field is repr=False."""
    cfg = AzureCostManagementConfig(
        subscription_id="sub-x",
        tenant_id="tenant",
        client_id="client",
        client_secret="super-secret-value",
    )
    assert "super-secret-value" not in repr(cfg)
    assert "client_secret" not in repr(cfg)


@pytest.mark.asyncio
async def test_get_daily_cost_wraps_network_errors():
    """httpx.RequestError family (ConnectError, ReadTimeout, RemoteProtocolError) must surface as AzureCostManagementError."""
    http = MagicMock()
    http.post = AsyncMock(side_effect=httpx.ReadTimeout("upstream timeout"))
    client = AzureCostManagementClient(
        config=_config(),
        http_handler=http,
        token_provider=lambda: "fake-token",
    )

    with pytest.raises(AzureCostManagementError) as exc:
        await client.get_daily_cost("/subs/x/deploy/y", date(2026, 7, 15))
    assert "network error" in str(exc.value).lower()
