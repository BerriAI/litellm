"""Tests for vantage export endpoint."""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


@pytest.fixture
def admin_auth():
    return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)


def _make_mock_engine():
    engine = MagicMock()
    engine.export_window = AsyncMock()
    engine.export_all = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_sku_export_only_start_time_defaults_end_to_now(admin_auth, monkeypatch):
    """sku_breakdown with only start_time_utc should default end to now, not crash."""
    from litellm.proxy.spend_tracking.vantage_endpoints import vantage_export_endpoint
    from litellm.types.proxy.vantage_endpoints import VantageExportRequest

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request = VantageExportRequest(
        sku_breakdown=True,
        start_time_utc=start,
        end_time_utc=None,
    )

    mock_engine = _make_mock_engine()
    mock_settings = {"api_key": "key", "integration_token": "tok", "base_url": "http://x"}

    import litellm.integrations.focus.export_engine as ee

    monkeypatch.setattr(ee, "FocusExportEngine", MagicMock(return_value=mock_engine))

    with patch(
        "litellm.proxy.spend_tracking.vantage_endpoints._get_vantage_settings",
        new=AsyncMock(return_value=mock_settings),
    ):
        await vantage_export_endpoint(request=request, user_api_key_dict=admin_auth)

    mock_engine.export_window.assert_awaited_once()
    window = mock_engine.export_window.call_args.kwargs["window"]
    assert window.start_time == start
    assert window.end_time > start  # defaulted to now


@pytest.mark.asyncio
async def test_sku_export_only_end_time_defaults_start_to_epoch(admin_auth, monkeypatch):
    """sku_breakdown with only end_time_utc should default start to epoch, not crash."""
    from litellm.proxy.spend_tracking.vantage_endpoints import vantage_export_endpoint
    from litellm.types.proxy.vantage_endpoints import VantageExportRequest

    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    request = VantageExportRequest(
        sku_breakdown=True,
        start_time_utc=None,
        end_time_utc=end,
    )

    mock_engine = _make_mock_engine()
    mock_settings = {"api_key": "key", "integration_token": "tok", "base_url": "http://x"}

    import litellm.integrations.focus.export_engine as ee

    monkeypatch.setattr(ee, "FocusExportEngine", MagicMock(return_value=mock_engine))

    with patch(
        "litellm.proxy.spend_tracking.vantage_endpoints._get_vantage_settings",
        new=AsyncMock(return_value=mock_settings),
    ):
        await vantage_export_endpoint(request=request, user_api_key_dict=admin_auth)

    mock_engine.export_window.assert_awaited_once()
    window = mock_engine.export_window.call_args.kwargs["window"]
    assert window.end_time == end
    assert window.start_time == datetime(1970, 1, 1, tzinfo=timezone.utc)
