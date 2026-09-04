"""Tests for TernaryLogger configuration and Tags token enrichment."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.focus_logger import FocusLogger
from litellm.integrations.ternary.ternary_logger import (
    TERNARY_USAGE_DATA_JOB_NAME,
    TernaryLogger,
    _drop_days_before,
    _merge_token_tags,
    _TernaryExportEngine,
)


def _logger() -> TernaryLogger:
    return TernaryLogger(api_key="k", connection_id="c", base_url="http://localhost:8080")


def _window() -> FocusTimeWindow:
    from datetime import datetime, timezone

    now = datetime(2026, 9, 4, tzinfo=timezone.utc)
    return FocusTimeWindow(start_time=now, end_time=now, frequency="daily")


def test_should_preset_focus_config_for_ternary():
    logger = TernaryLogger(api_key="k", connection_id="c", base_url="http://localhost:8080")
    assert isinstance(logger, FocusLogger)
    assert logger.provider == "ternary"
    assert logger.export_format == "csv"
    assert logger.frequency == "daily"
    assert logger.prefix == "ternary_exports"
    assert logger._destination_config == {
        "api_key": "k",
        "connection_id": "c",
        "base_url": "http://localhost:8080",
    }


def test_should_read_frequency_from_env(monkeypatch):
    monkeypatch.setenv("TERNARY_API_KEY", "k")
    monkeypatch.setenv("TERNARY_CONNECTION_ID", "c")
    monkeypatch.setenv("TERNARY_EXPORT_FREQUENCY", "daily")
    logger = TernaryLogger()
    assert logger.frequency == "daily"
    assert logger._destination_config["api_key"] == "k"
    assert logger._destination_config["connection_id"] == "c"


def test_should_default_to_daily():
    logger = TernaryLogger(api_key="k", connection_id="c", base_url="http://localhost:8080")
    assert logger.frequency == "daily"


def test_should_accept_interval_with_explicit_seconds():
    logger = TernaryLogger(
        api_key="k",
        connection_id="c",
        base_url="http://localhost:8080",
        frequency="interval",
        interval_seconds=30,
    )
    assert logger.frequency == "interval"
    assert logger.interval_seconds == 30


def test_should_reject_interval_without_seconds():
    with pytest.raises(ValueError, match="TERNARY_EXPORT_INTERVAL_SECONDS"):
        TernaryLogger(api_key="k", connection_id="c", base_url="http://localhost:8080", frequency="interval")


@pytest.mark.parametrize("bad_frequency", ["hourly", "weekly", "minutely"])
def test_should_reject_unsupported_frequency(bad_frequency):
    with pytest.raises(ValueError, match="TERNARY_EXPORT_FREQUENCY"):
        TernaryLogger(api_key="k", connection_id="c", base_url="http://localhost:8080", frequency=bad_frequency)


def test_compute_time_window_is_day_aligned_and_carries_whole_days():
    from datetime import datetime, timezone

    logger = TernaryLogger(api_key="k", connection_id="c", base_url="http://localhost:8080")
    now = datetime(2026, 9, 4, 13, 47, 5, tzinfo=timezone.utc)
    window = logger._compute_time_window(now)
    assert window.start_time == datetime(2026, 9, 3, 0, 0, 0, tzinfo=timezone.utc)
    assert window.end_time == now
    assert window.start_time.hour == 0 and window.start_time.minute == 0


def test_drop_days_before_removes_older_days():
    from datetime import datetime, timezone

    data = pl.DataFrame({"date": ["2026-09-02", "2026-09-03", "2026-09-04"], "spend": [1.0, 2.0, 3.0]})
    floor = datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc)
    out = _drop_days_before(data, floor)
    assert out["date"].to_list() == ["2026-09-03", "2026-09-04"]


def test_drop_days_before_noop_without_date_column():
    from datetime import datetime, timezone

    data = pl.DataFrame({"spend": [1.0]})
    out = _drop_days_before(data, datetime(2026, 9, 3, tzinfo=timezone.utc))
    assert out.equals(data)


@pytest.mark.asyncio
async def test_export_window_interposes_drop_days_before():
    from datetime import datetime, timezone

    multi_day = pl.DataFrame({"date": ["2026-09-01", "2026-09-03", "2026-09-04"], "spend": [1.0, 2.0, 3.0]})

    engine = MagicMock()

    async def fake_get_usage_data(**_kwargs):
        return multi_day

    engine._database.get_usage_data = fake_get_usage_data

    captured = {}

    async def fake_deliver(data, window):
        captured["data"] = data

    engine._deliver_enriched = fake_deliver

    window = FocusTimeWindow(
        start_time=datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
        frequency="daily",
    )
    await _TernaryExportEngine.export_window(engine, window=window, limit=None)

    assert captured["data"]["date"].to_list() == ["2026-09-03", "2026-09-04"]


def test_merge_token_tags_adds_raw_db_keys_and_preserves_existing():
    normalized = pl.DataFrame(
        {
            "Tags": [json.dumps({"team_id": "t1", "model": "gpt-4o"})],
            "BilledCost": [1.23],
        }
    )
    source = pl.DataFrame(
        {
            "prompt_tokens": [48],
            "completion_tokens": [274],
            "cache_read_input_tokens": [12],
            "cache_creation_input_tokens": [0],
        }
    )

    tags = json.loads(_merge_token_tags(normalized, source)["Tags"][0])

    assert tags["prompt_tokens"] == "48"
    assert tags["completion_tokens"] == "274"
    assert tags["cache_read_input_tokens"] == "12"
    assert tags["cache_creation_input_tokens"] == "0"
    assert not any(k.startswith("x_") for k in tags)
    assert tags["team_id"] == "t1"
    assert tags["model"] == "gpt-4o"


def test_merge_token_tags_aligns_per_row_through_the_real_transformer():
    from litellm.integrations.focus.transformer import FocusTransformer

    data = pl.DataFrame(
        {
            "date": ["2026-09-03", "2026-09-03", "2026-09-03"],
            "spend": [1.0, 2.0, 3.0],
            "api_key": ["k1", "k2", "k3"],
            "api_key_alias": ["a1", "a2", "a3"],
            "model": ["m1", "m2", "m3"],
            "model_group": ["g1", "g2", "g3"],
            "custom_llm_provider": ["openai", "openai", "anthropic"],
            "team_id": ["t1", "t2", "t3"],
            "team_alias": ["T1", "T2", "T3"],
            "api_requests": [1, 1, 1],
            "prompt_tokens": [10, 20, 30],
            "completion_tokens": [11, 22, 33],
            "cache_read_input_tokens": [0, 0, 0],
            "cache_creation_input_tokens": [0, 0, 0],
        }
    )

    transformed = FocusTransformer().transform(data)
    enriched = _merge_token_tags(transformed, data)

    assert enriched.height == 3
    for i, expected_tokens in enumerate(["10", "20", "30"]):
        tags = json.loads(enriched["Tags"][i])
        assert tags["prompt_tokens"] == expected_tokens
        assert tags["team_id"] == f"t{i + 1}"


def test_merge_token_tags_skips_none_values():
    normalized = pl.DataFrame({"Tags": [json.dumps({"team_id": "t1"})]})
    source = pl.DataFrame({"prompt_tokens": [None], "completion_tokens": [10]})
    tags = json.loads(_merge_token_tags(normalized, source)["Tags"][0])
    assert "prompt_tokens" not in tags
    assert tags["completion_tokens"] == "10"


def test_merge_token_tags_graceful_on_row_mismatch():
    normalized = pl.DataFrame({"Tags": [json.dumps({"team_id": "t1"})]})
    source = pl.DataFrame({"prompt_tokens": [1, 2]})
    out = _merge_token_tags(normalized, source)
    assert out["Tags"][0] == normalized["Tags"][0]


def test_merge_token_tags_non_object_tags_degrades_to_empty():
    normalized = pl.DataFrame({"Tags": ["[1, 2, 3]"]})
    source = pl.DataFrame({"prompt_tokens": [5]})
    tags = json.loads(_merge_token_tags(normalized, source)["Tags"][0])
    assert tags == {"prompt_tokens": "5"}


def test_drop_days_before_warns_when_everything_dropped():
    from datetime import datetime, timezone

    data = pl.DataFrame({"date": ["2026-09-01", "2026-09-02"], "spend": [1.0, 2.0]})
    out = _drop_days_before(data, datetime(2026, 9, 5, tzinfo=timezone.utc))
    assert out.height == 0


def test_merge_token_tags_graceful_on_malformed_tags():
    normalized = pl.DataFrame({"Tags": ["not-json"]})
    source = pl.DataFrame({"prompt_tokens": [5]})
    tags = json.loads(_merge_token_tags(normalized, source)["Tags"][0])
    assert tags == {"prompt_tokens": "5"}


def test_merge_token_tags_noop_without_tags_column():
    normalized = pl.DataFrame({"BilledCost": [1.0]})
    source = pl.DataFrame({"prompt_tokens": [5]})
    out = _merge_token_tags(normalized, source)
    assert "Tags" not in out.columns


def test_merge_token_tags_noop_without_token_columns():
    normalized = pl.DataFrame({"Tags": [json.dumps({"team_id": "t1"})]})
    source = pl.DataFrame({"spend": [1.0]})
    tags = json.loads(_merge_token_tags(normalized, source)["Tags"][0])
    assert tags == {"team_id": "t1"}


def _fake_engine(*, transformed: pl.DataFrame, payload: bytes) -> MagicMock:
    engine = MagicMock()
    engine._transformer.transform = MagicMock(return_value=transformed)
    engine._serializer.serialize = MagicMock(return_value=payload)
    engine._destination.deliver = AsyncMock()
    engine._build_filename = MagicMock(return_value="usage.csv")
    return engine


@pytest.mark.asyncio
async def test_transform_enrich_deliver_enriches_tags_then_delivers():
    data = pl.DataFrame({"Tags": [json.dumps({"team_id": "t1"})], "prompt_tokens": [7]})
    transformed = pl.DataFrame({"Tags": [json.dumps({"team_id": "t1"})]})
    engine = _fake_engine(transformed=transformed, payload=b"csv-bytes")

    await _TernaryExportEngine._deliver_enriched(engine, data, _window())

    serialized_frame = engine._serializer.serialize.call_args.args[0]
    assert json.loads(serialized_frame["Tags"][0])["prompt_tokens"] == "7"
    engine._destination.deliver.assert_awaited_once()
    kwargs = engine._destination.deliver.await_args.kwargs
    assert kwargs["content"] == b"csv-bytes"
    assert kwargs["filename"] == "usage.csv"


@pytest.mark.asyncio
async def test_transform_enrich_deliver_skips_empty_data():
    engine = _fake_engine(transformed=pl.DataFrame(), payload=b"")
    await _TernaryExportEngine._deliver_enriched(engine, pl.DataFrame(), _window())
    engine._destination.deliver.assert_not_awaited()


def test_pod_lock_key_is_ternary_specific():
    assert TERNARY_USAGE_DATA_JOB_NAME == "ternary_export_usage_data"


def test_should_reject_explicit_zero_interval_without_env_fallthrough(monkeypatch):
    monkeypatch.setenv("TERNARY_EXPORT_INTERVAL_SECONDS", "300")
    with pytest.raises(ValueError, match="TERNARY_EXPORT_INTERVAL_SECONDS"):
        TernaryLogger(
            api_key="k", connection_id="c", base_url="http://localhost:8080", frequency="interval", interval_seconds=0
        )


def test_should_reject_non_numeric_interval(monkeypatch):
    monkeypatch.setenv("TERNARY_EXPORT_INTERVAL_SECONDS", "5m")
    with pytest.raises(ValueError, match="TERNARY_EXPORT_INTERVAL_SECONDS"):
        TernaryLogger(api_key="k", connection_id="c", base_url="http://localhost:8080", frequency="interval")


@pytest.mark.asyncio
async def test_export_all_delivers_with_all_window():
    engine = MagicMock()
    frame = pl.DataFrame({"Tags": [json.dumps({"team_id": "t"})], "prompt_tokens": [5]})

    async def fake_get(**_kwargs):
        return frame

    engine._database.get_usage_data = fake_get

    captured = {}

    async def fake_deliver(data, window):
        captured["window"] = window
        captured["rows"] = data.height

    engine._deliver_enriched = fake_deliver
    await _TernaryExportEngine.export_all(engine, limit=None)
    assert captured["rows"] == 1
    assert captured["window"].frequency == "all"


@pytest.mark.asyncio
async def test_transform_enrich_deliver_skips_when_transform_empty():
    data = pl.DataFrame({"Tags": [json.dumps({"team_id": "t"})], "prompt_tokens": [1]})
    engine = _fake_engine(transformed=pl.DataFrame(), payload=b"")
    await _TernaryExportEngine._deliver_enriched(engine, data, _window())
    engine._destination.deliver.assert_not_awaited()


@pytest.mark.asyncio
async def test_transform_enrich_deliver_skips_when_payload_empty():
    data = pl.DataFrame({"Tags": [json.dumps({"team_id": "t"})]})
    transformed = pl.DataFrame({"Tags": [json.dumps({"team_id": "t"})]})
    engine = _fake_engine(transformed=transformed, payload=b"")
    await _TernaryExportEngine._deliver_enriched(engine, data, _window())
    engine._destination.deliver.assert_not_awaited()


def test_ensure_engine_builds_ternary_export_engine():
    logger = _logger()
    engine = logger._ensure_engine()
    assert isinstance(engine, _TernaryExportEngine)
    assert logger._ensure_engine() is engine


def test_init_custom_logger_compatible_class_returns_cached_ternary_logger(monkeypatch):
    monkeypatch.setenv("TERNARY_API_KEY", "k")
    monkeypatch.setenv("TERNARY_CONNECTION_ID", "c")
    monkeypatch.setenv("TERNARY_BASE_URL", "https://ternary.test")
    from litellm.litellm_core_utils.litellm_logging import _init_custom_logger_compatible_class

    first = _init_custom_logger_compatible_class("ternary", internal_usage_cache=None, llm_router=None)
    assert isinstance(first, TernaryLogger)
    second = _init_custom_logger_compatible_class("ternary", internal_usage_cache=None, llm_router=None)
    assert second is first
