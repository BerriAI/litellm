"""Tests for FocusMavvrikDestination."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.destinations.mavvrik_destination import (
    FocusMavvrikDestination,
    _validate_api_endpoint,
)

VALID_ENDPOINT = "https://api.mavvrik.ai/tenant123"


def _make_window() -> FocusTimeWindow:
    return FocusTimeWindow(
        start_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        frequency="daily",
    )


def _dest(**overrides) -> FocusMavvrikDestination:
    config = {
        "api_key": "test-key",
        "api_endpoint": VALID_ENDPOINT,
        "connection_id": "conn-123",
    }
    config.update(overrides)
    return FocusMavvrikDestination(prefix="mavvrik_focus_exports", config=config)


def _patch_resp(status: int = 204) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    return r


def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="MAVVRIK_API_KEY"):
        FocusMavvrikDestination(
            prefix="p",
            config={"api_endpoint": VALID_ENDPOINT, "connection_id": "c"},
        )


def test_missing_api_endpoint_raises():
    with pytest.raises(ValueError, match="MAVVRIK_API_ENDPOINT"):
        FocusMavvrikDestination(
            prefix="p",
            config={"api_key": "k", "connection_id": "c"},
        )


def test_missing_connection_id_raises():
    with pytest.raises(ValueError, match="MAVVRIK_CONNECTION_ID"):
        FocusMavvrikDestination(
            prefix="p",
            config={"api_key": "k", "api_endpoint": VALID_ENDPOINT},
        )


def test_non_https_endpoint_raises():
    with pytest.raises(ValueError, match="HTTPS"):
        _validate_api_endpoint("http://api.mavvrik.ai/tenant")


def test_non_mavvrik_domain_raises():
    with pytest.raises(ValueError, match="Mavvrik domain"):
        _validate_api_endpoint("https://evil.com/tenant")


def test_valid_mavvrik_domains_accepted():
    for domain in (
        "https://api.mavvrik.ai/tenant",
        "https://api.mavvrik.dev/tenant",
        "https://api.mavvrik.app/tenant",
    ):
        _validate_api_endpoint(domain)  # must not raise


def test_initializes_with_not_registered():
    dest = _dest()
    assert dest._registered is False


@pytest.mark.asyncio
async def test_deliver_skips_upload_but_advances_marker_for_empty_content():
    dest = _dest()

    register_resp = MagicMock()
    register_resp.status_code = 200
    register_resp.json.return_value = {"metricsMarker": 0}

    patch_resp = _patch_resp(204)

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(side_effect=[register_resp, patch_resp])
    dest._http = mock_http

    await dest.deliver(content=b"", time_window=_make_window(), filename="usage.csv")

    assert dest._registered is True
    assert mock_http.client.request.call_count == 2
    patch_call = mock_http.client.request.call_args_list[1]
    assert patch_call.kwargs["method"] == "PATCH"
    assert "metricsMarker" in patch_call.kwargs["json"]


@pytest.mark.asyncio
async def test_large_content_uploads_in_multiple_chunks():
    """Content larger than _GCS_CHUNK_SIZE must be uploaded in multiple chunks.

    GCS assembles intermediate chunks (308) + final chunk (200) into one object.
    The destination must send Content-Range headers for each chunk correctly.
    """
    from litellm.integrations.focus.destinations.mavvrik_destination import (
        FocusMavvrikDestination,
        _GCS_CHUNK_SIZE,
    )

    dest = FocusMavvrikDestination(
        prefix="p",
        config={"api_key": "k", "api_endpoint": VALID_ENDPOINT, "connection_id": "c"},
    )

    register_resp = MagicMock()
    register_resp.status_code = 200

    signed_url_resp = MagicMock()
    signed_url_resp.status_code = 200
    signed_url_resp.json.return_value = {"url": "https://storage.googleapis.com/upload?sig=x"}

    init_resp = MagicMock()
    init_resp.status_code = 200
    init_resp.headers = {"Location": "https://storage.googleapis.com/session"}

    # First chunk → 308, second (final) chunk → 200
    chunk1_resp = MagicMock()
    chunk1_resp.status_code = 308

    chunk2_resp = MagicMock()
    chunk2_resp.status_code = 200

    patch_resp = _patch_resp(204)

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(
        side_effect=[
            register_resp,
            signed_url_resp,
            init_resp,
            chunk1_resp,
            chunk2_resp,
            patch_resp,
        ]
    )
    dest._http = mock_http

    # Build content that when gzipped exceeds one chunk.
    # Use incompressible random-ish bytes to ensure gzip doesn't shrink it below the chunk size.
    import os as _os

    raw = b"col1,col2\n" + _os.urandom(_GCS_CHUNK_SIZE + 1024)

    await dest.deliver(
        content=raw,
        time_window=_make_window(),
        filename="usage.csv",
    )

    # register + get_signed_url + init + 2 chunk PUTs + PATCH = 6 calls
    assert mock_http.client.request.call_count == 6

    # Check Content-Range headers on the chunk PUTs (calls 3 and 4)
    put_calls = mock_http.client.request.call_args_list[3:5]
    assert "bytes" in put_calls[0].kwargs["headers"]["Content-Range"]
    assert "/*" in put_calls[0].kwargs["headers"]["Content-Range"]  # intermediate
    assert "/*" not in put_calls[1].kwargs["headers"]["Content-Range"]  # final

    # Verify the PATCH call advanced metricsMarker
    patch_call = mock_http.client.request.call_args_list[5]
    assert patch_call.kwargs["method"] == "PATCH"
    assert "metricsMarker" in patch_call.kwargs["json"]


@pytest.mark.asyncio
async def test_deliver_calls_register_get_url_and_upload():
    dest = _dest()

    register_resp = MagicMock()
    register_resp.status_code = 200

    signed_url_resp = MagicMock()
    signed_url_resp.status_code = 200
    signed_url_resp.json.return_value = {"url": "https://storage.googleapis.com/signed"}

    init_resp = MagicMock()
    init_resp.status_code = 200
    init_resp.headers = {"Location": "https://storage.googleapis.com/session-uri"}

    upload_resp = MagicMock()
    upload_resp.status_code = 200

    patch_resp = _patch_resp(204)

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    # All 5 calls go through self._http.client.request:
    # 1. register, 2. get_signed_url, 3. GCS session init POST, 4. GCS PUT, 5. PATCH marker
    mock_http.client.request = AsyncMock(
        side_effect=[register_resp, signed_url_resp, init_resp, upload_resp, patch_resp]
    )
    dest._http = mock_http

    await dest.deliver(
        content=b"header\nrow1\n",
        time_window=_make_window(),
        filename="usage.csv",
    )

    assert dest._registered is True
    assert mock_http.client.request.call_count == 5
    # Verify Content-Range header was set on the PUT
    put_call = mock_http.client.request.call_args_list[3]
    assert "Content-Range" in put_call.kwargs["headers"]
    # Verify PATCH was called last with metricsMarker
    patch_call = mock_http.client.request.call_args_list[4]
    assert patch_call.kwargs["method"] == "PATCH"
    assert "metricsMarker" in patch_call.kwargs["json"]


@pytest.mark.asyncio
async def test_register_called_only_once_across_multiple_deliveries():
    dest = _dest()

    register_resp = MagicMock()
    register_resp.status_code = 200

    def _signed_url_resp():
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = {"url": "https://storage.googleapis.com/signed"}
        return r

    init_resp = MagicMock()
    init_resp.status_code = 200
    init_resp.headers = {"Location": "https://storage.googleapis.com/session-uri"}

    upload_resp = MagicMock()
    upload_resp.status_code = 200

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    # First delivery:  register, get_signed_url, GCS init, GCS PUT, PATCH
    # Second delivery: get_signed_url, GCS init, GCS PUT, PATCH  (register skipped)
    mock_http.client.request = AsyncMock(
        side_effect=[
            register_resp,
            _signed_url_resp(),
            init_resp,
            upload_resp,
            _patch_resp(204),
            _signed_url_resp(),
            init_resp,
            upload_resp,
            _patch_resp(204),
        ]
    )
    dest._http = mock_http

    window = _make_window()
    await dest.deliver(content=b"header\nrow1\n", time_window=window, filename="1.csv")
    await dest.deliver(content=b"header\nrow2\n", time_window=window, filename="2.csv")

    # 9 total: register(1) + [get_url+init+put+patch](4) × 2 deliveries
    assert mock_http.client.request.call_count == 9
    # First call was register
    first_call = mock_http.client.request.call_args_list[0]
    assert first_call.kwargs["method"] == "POST"
    assert "/upload-url" not in first_call.kwargs["url"]


@pytest.mark.asyncio
async def test_deliver_raises_on_register_failure():
    dest = _dest()

    fail_resp = MagicMock()
    fail_resp.status_code = 403
    fail_resp.text = "Forbidden"

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(return_value=fail_resp)
    dest._http = mock_http

    with pytest.raises(RuntimeError, match="register failed"):
        await dest.deliver(
            content=b"data",
            time_window=_make_window(),
            filename="usage.csv",
        )


@pytest.mark.asyncio
async def test_deliver_raises_on_signed_url_api_error():
    """_get_signed_url must raise RuntimeError when the API returns a 4xx."""
    dest = _dest()

    register_resp = MagicMock()
    register_resp.status_code = 200

    fail_resp = MagicMock()
    fail_resp.status_code = 500
    fail_resp.text = "Internal Server Error"

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(side_effect=[register_resp, fail_resp])
    dest._http = mock_http

    with pytest.raises(RuntimeError, match="failed to get signed URL"):
        await dest.deliver(
            content=b"data",
            time_window=_make_window(),
            filename="usage.csv",
        )


@pytest.mark.asyncio
async def test_deliver_raises_on_missing_signed_url():
    dest = _dest()

    register_resp = MagicMock()
    register_resp.status_code = 200

    bad_url_resp = MagicMock()
    bad_url_resp.status_code = 200
    bad_url_resp.json.return_value = {}  # no 'url' field

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(side_effect=[register_resp, bad_url_resp])
    dest._http = mock_http

    with pytest.raises(RuntimeError, match="missing 'url' field"):
        await dest.deliver(
            content=b"data",
            time_window=_make_window(),
            filename="usage.csv",
        )


@pytest.mark.asyncio
async def test_deliver_raises_on_non_gcs_signed_url():
    """Signed URL pointing to a non-GCS host must be rejected before any upload."""
    from litellm.integrations.focus.destinations.mavvrik_destination import (
        _validate_gcs_url,
    )

    with pytest.raises(ValueError, match="GCS endpoint"):
        _validate_gcs_url("https://evil.com/upload?token=abc", "signed URL")


@pytest.mark.asyncio
async def test_deliver_raises_on_non_gcs_session_uri():
    """Session URI from Location header pointing to a non-GCS host must be rejected."""
    dest = _dest()

    register_resp = MagicMock()
    register_resp.status_code = 200

    signed_url_resp = MagicMock()
    signed_url_resp.status_code = 200
    # signed URL is valid GCS
    signed_url_resp.json.return_value = {"url": "https://storage.googleapis.com/upload?sig=abc"}

    # Location header points to a non-GCS host
    init_resp = MagicMock()
    init_resp.status_code = 200
    init_resp.headers = {"Location": "https://evil.com/session-uri"}

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    # register, get_signed_url, GCS session init (returns bad Location)
    mock_http.client.request = AsyncMock(side_effect=[register_resp, signed_url_resp, init_resp])
    dest._http = mock_http

    with pytest.raises(ValueError, match="GCS endpoint"):
        await dest.deliver(
            content=b"data",
            time_window=_make_window(),
            filename="usage.csv",
        )


def test_factory_creates_mavvrik_destination(monkeypatch):
    monkeypatch.setenv("MAVVRIK_API_KEY", "k")
    monkeypatch.setenv("MAVVRIK_API_ENDPOINT", VALID_ENDPOINT)
    monkeypatch.setenv("MAVVRIK_CONNECTION_ID", "c")

    from litellm.integrations.focus.destinations.factory import FocusDestinationFactory

    dest = FocusDestinationFactory.create(provider="mavvrik", prefix="p")

    assert isinstance(dest, FocusMavvrikDestination)
    assert dest.api_key == "k"
    assert dest.connection_id == "c"


def test_only_daily_frequency_is_supported():
    """MavvrikFocusLogger must raise ValueError for non-daily frequencies."""
    import importlib

    for freq in ("hourly", "interval"):

        def _make(f=freq, monkeypatch=None):
            import os

            old = os.environ.get("MAVVRIK_FOCUS_FREQUENCY")
            os.environ["MAVVRIK_FOCUS_FREQUENCY"] = f
            try:
                from litellm.integrations.mavvrik_focus import mavvrik_focus_logger

                importlib.reload(mavvrik_focus_logger)
                with pytest.raises(ValueError, match="Only 'daily' is allowed"):
                    mavvrik_focus_logger.MavvrikFocusLogger()
            finally:
                if old is None:
                    os.environ.pop("MAVVRIK_FOCUS_FREQUENCY", None)
                else:
                    os.environ["MAVVRIK_FOCUS_FREQUENCY"] = old

        _make()


def test_max_rows_defaults_to_500k():
    """MAVVRIK_FOCUS_MAX_ROWS defaults to 500_000 when not set."""
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        MavvrikFocusLogger,
    )

    logger = MavvrikFocusLogger()
    assert logger._max_rows == 500_000


def test_max_rows_reads_from_env(monkeypatch):
    """MAVVRIK_FOCUS_MAX_ROWS env var is respected."""
    monkeypatch.setenv("MAVVRIK_FOCUS_MAX_ROWS", "100000")

    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        MavvrikFocusLogger,
    )

    logger = MavvrikFocusLogger()
    assert logger._max_rows == 100_000


@pytest.mark.asyncio
async def test_export_window_passes_max_rows_as_limit(monkeypatch):
    """_export_window must pass _max_rows as limit to get_usage_data."""
    monkeypatch.setenv("MAVVRIK_FOCUS_MAX_ROWS", "1000")

    import polars as pl
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        MavvrikFocusLogger,
    )
    from litellm.integrations.focus.destinations.base import FocusTimeWindow
    from datetime import datetime, timezone

    logger = MavvrikFocusLogger()
    assert logger._max_rows == 1000

    # Mock the engine internals so _export_window runs through our new code path
    db_mock = MagicMock()
    db_mock.get_usage_data = AsyncMock(return_value=pl.DataFrame())  # empty deliver

    engine_mock = MagicMock()
    engine_mock._database = db_mock
    engine_mock._destination.deliver = AsyncMock()
    logger._engine = engine_mock

    window = FocusTimeWindow(
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
        frequency="daily",
    )
    await logger._export_window(window=window, limit=None)

    db_mock.get_usage_data.assert_called_once_with(
        limit=1000,
        start_time_utc=window.start_time,
        end_time_utc=window.end_time,
    )


@pytest.mark.asyncio
async def test_run_scheduled_export_catches_up_missed_dates():
    """If metricsMarker is 2 days behind, _run_scheduled_export exports missed dates first."""
    import polars as pl
    from datetime import datetime, timedelta, timezone
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        MavvrikFocusLogger,
    )
    from litellm.integrations.focus.destinations.mavvrik_destination import (
        FocusMavvrikDestination,
    )

    logger = MavvrikFocusLogger()

    # metricsMarker = 3 days ago → 2 missed dates (day-2 and day-1) + today's run
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = now - timedelta(days=1)
    two_days_ago = now - timedelta(days=2)
    three_days_ago = now - timedelta(days=3)

    marker_ts = int(three_days_ago.timestamp())

    # Mock destination
    dest_mock = MagicMock(spec=FocusMavvrikDestination)
    dest_mock.get_metrics_marker = AsyncMock(return_value=marker_ts)

    # Mock engine
    db_mock = MagicMock()
    db_mock.get_usage_data = AsyncMock(return_value=pl.DataFrame())
    engine_mock = MagicMock()
    engine_mock._database = db_mock
    engine_mock._destination = dest_mock
    logger._engine = engine_mock

    await logger._run_scheduled_export()

    # Should have queried DB 3 times: day-2, day-1 (yesterday), and the normal yesterday window
    # Actually: catch-up covers [three_days_ago+1 .. yesterday) = [two_days_ago, yesterday)
    # = two_days_ago only (1 missed), then normal yesterday = 2 total calls
    calls = db_mock.get_usage_data.call_args_list
    assert len(calls) == 2
    # First call is the catch-up (two_days_ago)
    assert calls[0].kwargs["start_time_utc"].date() == two_days_ago.date()
    # Second call is yesterday's normal daily run
    assert calls[1].kwargs["start_time_utc"].date() == yesterday.date()


@pytest.mark.asyncio
async def test_run_scheduled_export_no_catchup_when_marker_is_current():
    """If metricsMarker = yesterday, no catch-up needed — just export yesterday."""
    import polars as pl
    from datetime import datetime, timedelta, timezone
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        MavvrikFocusLogger,
    )
    from litellm.integrations.focus.destinations.mavvrik_destination import (
        FocusMavvrikDestination,
    )

    logger = MavvrikFocusLogger()

    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = now - timedelta(days=1)
    marker_ts = int(yesterday.timestamp())

    dest_mock = MagicMock(spec=FocusMavvrikDestination)
    dest_mock.get_metrics_marker = AsyncMock(return_value=marker_ts)

    db_mock = MagicMock()
    db_mock.get_usage_data = AsyncMock(return_value=pl.DataFrame())
    engine_mock = MagicMock()
    engine_mock._database = db_mock
    engine_mock._destination = dest_mock
    logger._engine = engine_mock

    await logger._run_scheduled_export()

    # Only one call — yesterday's normal run, no catch-up
    assert db_mock.get_usage_data.call_count == 1
    assert db_mock.get_usage_data.call_args.kwargs["start_time_utc"].date() == yesterday.date()


@pytest.mark.asyncio
async def test_run_scheduled_export_skips_catchup_when_marker_is_unparseable():
    import polars as pl
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        MavvrikFocusLogger,
    )
    from litellm.integrations.focus.destinations.mavvrik_destination import (
        FocusMavvrikDestination,
    )

    logger = MavvrikFocusLogger()
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = now - timedelta(days=1)

    dest_mock = MagicMock(spec=FocusMavvrikDestination)
    dest_mock.get_metrics_marker = AsyncMock(return_value="not-a-date")

    db_mock = MagicMock()
    db_mock.get_usage_data = AsyncMock(return_value=pl.DataFrame())
    engine_mock = MagicMock()
    engine_mock._database = db_mock
    engine_mock._destination = dest_mock
    logger._engine = engine_mock

    await logger._run_scheduled_export()

    assert db_mock.get_usage_data.call_count == 1
    assert (
        db_mock.get_usage_data.call_args.kwargs["start_time_utc"].date()
        == yesterday.date()
    )


@pytest.mark.asyncio
async def test_metrics_marker_always_calls_api():
    """get_metrics_marker must call the register API every time to get a fresh marker.

    This is the key difference from deliver() — catch-up requires the current
    metricsMarker on every scheduled run, not just the first one.
    """
    dest = _dest()

    register_resp = MagicMock()
    register_resp.status_code = 200
    register_resp.json.return_value = {
        "id": "litellm-conn-123",
        "metricsMarker": 1749340800,
    }

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(return_value=register_resp)
    dest._http = mock_http

    # First call
    marker = await dest.get_metrics_marker()
    assert marker == 1749340800
    assert dest._registered is True

    # Second call — must call API again to get fresh marker (not return None)
    marker2 = await dest.get_metrics_marker()
    assert marker2 == 1749340800
    assert mock_http.client.request.call_count == 2  # API called both times


def test_parse_metrics_marker_handles_unix_timestamp():
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        _parse_metrics_marker,
    )
    from datetime import datetime, timezone

    # Use a known date and compute its timestamp to avoid hardcoding
    known_date = datetime(2026, 6, 9, 0, 0, 0, tzinfo=timezone.utc)
    ts = int(known_date.timestamp())

    result = _parse_metrics_marker(ts)
    assert result is not None
    assert result.date().isoformat() == "2026-06-09"
    assert result.tzinfo == timezone.utc


def test_parse_metrics_marker_handles_iso_date_string():
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        _parse_metrics_marker,
    )

    result = _parse_metrics_marker("2026-06-09")
    assert result is not None
    assert result.date().isoformat() == "2026-06-09"


def test_parse_metrics_marker_handles_iso_datetime_string():
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        _parse_metrics_marker,
    )

    result = _parse_metrics_marker("2026-06-09T00:00:00Z")
    assert result is not None
    assert result.date().isoformat() == "2026-06-09"


def test_parse_metrics_marker_returns_none_for_zero():
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        _parse_metrics_marker,
    )

    assert _parse_metrics_marker(0) is None
    assert _parse_metrics_marker(None) is None
    assert _parse_metrics_marker("") is None


def test_parse_metrics_marker_returns_none_for_garbage():
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        _parse_metrics_marker,
    )

    # Should not raise — logs warning and returns None
    assert _parse_metrics_marker("not-a-date") is None


@pytest.mark.asyncio
async def test_catchup_capped_at_max_catchup_days():
    """Catch-up must not go further back than _MAX_CATCHUP_DAYS."""
    import polars as pl
    from datetime import datetime, timedelta, timezone
    from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (
        MavvrikFocusLogger,
    )
    from litellm.integrations.focus.destinations.mavvrik_destination import (
        FocusMavvrikDestination,
    )

    logger = MavvrikFocusLogger()
    max_days = MavvrikFocusLogger._MAX_CATCHUP_DAYS

    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = now - timedelta(days=1)
    # Marker is 30 days ago — well beyond the cap
    thirty_days_ago = now - timedelta(days=30)
    marker_ts = int(thirty_days_ago.timestamp())

    dest_mock = MagicMock(spec=FocusMavvrikDestination)
    dest_mock.get_metrics_marker = AsyncMock(return_value=marker_ts)

    db_mock = MagicMock()
    db_mock.get_usage_data = AsyncMock(return_value=pl.DataFrame())
    engine_mock = MagicMock()
    engine_mock._database = db_mock
    engine_mock._destination = dest_mock
    logger._engine = engine_mock

    await logger._run_scheduled_export()

    # Should have queried at most _MAX_CATCHUP_DAYS times
    # (max_days - 1 catch-up dates + 1 yesterday = max_days total)
    assert db_mock.get_usage_data.call_count <= max_days

    # First catch-up date must not be earlier than (yesterday - max_days + 1)
    earliest_allowed = yesterday - timedelta(days=max_days - 1)
    first_call_start = db_mock.get_usage_data.call_args_list[0].kwargs["start_time_utc"]
    assert first_call_start.date() >= earliest_allowed.date()


@pytest.mark.asyncio
async def test_register_resets_on_410():
    """_registered flag must be False after a 410 so next run re-registers."""
    dest = _dest()

    resp_410 = MagicMock()
    resp_410.status_code = 410
    resp_410.text = "Gone"

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(return_value=resp_410)
    dest._http = mock_http
    dest._registered = False  # not yet registered — trigger the call

    with pytest.raises(RuntimeError, match="disconnected"):
        await dest._ensure_registered()

    assert dest._registered is False


@pytest.mark.asyncio
async def test_gcs_session_cancelled_on_chunk_failure():
    """GCS session must be cancelled (DELETE) when a chunk PUT fails."""
    dest = _dest()

    register_resp = MagicMock()
    register_resp.status_code = 200

    signed_url_resp = MagicMock()
    signed_url_resp.status_code = 200
    signed_url_resp.json.return_value = {"url": "https://storage.googleapis.com/upload?sig=x"}

    init_resp = MagicMock()
    init_resp.status_code = 200
    init_resp.headers = {"Location": "https://storage.googleapis.com/session"}

    # Chunk PUT fails with 500
    fail_resp = MagicMock()
    fail_resp.status_code = 500
    fail_resp.text = "Internal Server Error"

    # DELETE (session cancel)
    delete_resp = MagicMock()
    delete_resp.status_code = 200

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(
        side_effect=[register_resp, signed_url_resp, init_resp, fail_resp, delete_resp]
    )
    dest._http = mock_http

    with pytest.raises(RuntimeError, match="GCS chunk upload failed"):
        await dest.deliver(
            content=b"header\nrow1\n",
            time_window=_make_window(),
            filename="usage.csv",
        )

    # Verify DELETE was called to cancel the session
    calls = mock_http.client.request.call_args_list
    delete_call = calls[4]
    assert delete_call.kwargs["method"] == "DELETE"
    assert "storage.googleapis.com/session" in delete_call.kwargs["url"]


@pytest.mark.asyncio
async def test_update_metrics_marker_raises_on_non_410_error():
    dest = _dest()

    fail_resp = MagicMock()
    fail_resp.status_code = 500
    fail_resp.text = "Internal Server Error"

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(return_value=fail_resp)
    dest._http = mock_http

    with pytest.raises(RuntimeError, match="failed to update metricsMarker"):
        await dest._update_metrics_marker(1234567890)
    assert mock_http.client.request.call_count == 1


@pytest.mark.asyncio
async def test_update_metrics_marker_raises_on_410():
    """_update_metrics_marker must raise RuntimeError and reset _registered on 410."""
    dest = _dest()
    dest._registered = True

    resp_410 = MagicMock()
    resp_410.status_code = 410
    resp_410.text = "Gone"

    mock_http = MagicMock()
    mock_http.client = MagicMock()
    mock_http.client.request = AsyncMock(return_value=resp_410)
    dest._http = mock_http

    with pytest.raises(RuntimeError, match="disconnected"):
        await dest._update_metrics_marker(1234567890)

    assert dest._registered is False
