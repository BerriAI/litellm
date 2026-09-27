"""Tests for FocusGCSDestination."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.focus.destinations.base import FocusTimeWindow


def _make_window(frequency: str = "hourly") -> FocusTimeWindow:
    return FocusTimeWindow(
        start_time=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
        frequency=frequency,
    )


@pytest.mark.asyncio
async def test_deliver_posts_to_gcs_upload_endpoint():
    """deliver() must POST raw bytes to the GCS upload endpoint."""
    from litellm.integrations.focus.destinations.gcs_destination import (
        FocusGCSDestination,
    )

    dest = FocusGCSDestination(
        prefix="focus_exports",
        config={"bucket_name": "my-bucket", "service_account_json": None},
    )

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    dest.async_httpx_client = mock_client

    with patch.object(
        dest,
        "construct_request_headers",
        new=AsyncMock(return_value={"Authorization": "Bearer tok-123"}),
    ):
        await dest.deliver(
            content=b"col1,col2\nval1,val2\n",
            time_window=_make_window(),
            filename="usage_20260101T100000Z_20260101T110000Z.csv",
        )

    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    url = call_kwargs.kwargs.get("url") or call_kwargs.args[0]
    assert "my-bucket" in url
    assert "uploadType=media" in url
    headers = call_kwargs.kwargs["headers"]
    assert headers["Authorization"] == "Bearer tok-123"


@pytest.mark.asyncio
async def test_deliver_raises_on_gcs_error():
    """deliver() must raise RuntimeError when GCS returns non-200."""
    from litellm.integrations.focus.destinations.gcs_destination import (
        FocusGCSDestination,
    )

    dest = FocusGCSDestination(
        prefix="focus_exports",
        config={"bucket_name": "my-bucket"},
    )

    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Permission denied"

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    dest.async_httpx_client = mock_client

    with patch.object(
        dest,
        "construct_request_headers",
        new=AsyncMock(return_value={"Authorization": "Bearer tok-bad"}),
    ):
        with pytest.raises(RuntimeError, match="GCS upload failed"):
            await dest.deliver(
                content=b"data",
                time_window=_make_window(),
                filename="usage.csv",
            )


def test_build_object_key_hourly():
    """Hourly key must include date= and hour= components."""
    from litellm.integrations.focus.destinations.gcs_destination import (
        FocusGCSDestination,
    )

    dest = FocusGCSDestination(prefix="focus_exports", config={"bucket_name": "b"})
    key = dest._build_object_key(
        time_window=_make_window("hourly"), filename="usage.parquet"
    )

    assert key == "focus_exports/date=2026-01-01/hour=10/usage.parquet"


def test_build_object_key_daily():
    """Daily key must include date= but not hour=."""
    from litellm.integrations.focus.destinations.gcs_destination import (
        FocusGCSDestination,
    )

    dest = FocusGCSDestination(prefix="focus_exports", config={"bucket_name": "b"})
    window = FocusTimeWindow(
        start_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        frequency="daily",
    )
    key = dest._build_object_key(time_window=window, filename="usage.parquet")

    assert key == "focus_exports/date=2026-01-01/usage.parquet"


def test_missing_bucket_name_raises():
    """Constructing without bucket_name must raise ValueError."""
    from litellm.integrations.focus.destinations.gcs_destination import (
        FocusGCSDestination,
    )

    with pytest.raises(ValueError, match="bucket_name"):
        FocusGCSDestination(prefix="focus_exports", config={})


def test_global_gcs_service_account_not_overwritten_when_absent(monkeypatch):
    """service_account_json absent from config must not overwrite GCS_PATH_SERVICE_ACCOUNT.

    GCSBucketBase sets self.path_service_account_json from GCS_PATH_SERVICE_ACCOUNT.
    If config has no service_account_json key, we must leave the parent value intact
    so deployments using the global credential don't silently fall back to ADC.
    """
    monkeypatch.setenv("GCS_PATH_SERVICE_ACCOUNT", "/global/sa.json")

    from litellm.integrations.focus.destinations.gcs_destination import (
        FocusGCSDestination,
    )

    dest = FocusGCSDestination(prefix="focus_exports", config={"bucket_name": "b"})

    assert dest.path_service_account_json == "/global/sa.json"


def test_explicit_service_account_overrides_global(monkeypatch):
    """Explicit service_account_json in config must take precedence over GCS_PATH_SERVICE_ACCOUNT."""
    monkeypatch.setenv("GCS_PATH_SERVICE_ACCOUNT", "/global/sa.json")

    from litellm.integrations.focus.destinations.gcs_destination import (
        FocusGCSDestination,
    )

    dest = FocusGCSDestination(
        prefix="focus_exports",
        config={"bucket_name": "b", "service_account_json": "/focus/sa.json"},
    )

    assert dest.path_service_account_json == "/focus/sa.json"


def test_factory_creates_gcs_destination(monkeypatch):
    """FocusDestinationFactory.create(provider='gcs') must return FocusGCSDestination."""
    monkeypatch.setenv("FOCUS_GCS_BUCKET_NAME", "env-bucket")

    from litellm.integrations.focus.destinations.factory import FocusDestinationFactory
    from litellm.integrations.focus.destinations.gcs_destination import (
        FocusGCSDestination,
    )

    dest = FocusDestinationFactory.create(provider="gcs", prefix="focus_exports")

    assert isinstance(dest, FocusGCSDestination)
    assert dest.BUCKET_NAME == "env-bucket"
