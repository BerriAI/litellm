from typing import Final

import pytest

from litellm.constants import DEFAULT_AGENT_TRACING_RETENTION_DAYS, DEFAULT_CLICKHOUSE_DATABASE
from litellm.rust_bridge.clickhouse import spend_storage_config


def test_spend_storage_preserves_environment_settings_and_hides_credentials() -> None:
    config: Final = spend_storage_config(
        {
            "CLICKHOUSE_URL": "https://writer:private@clickhouse:8443",
            "CLICKHOUSE_DATABASE": "analytics",
            "AGENT_TRACING_RETENTION_DAYS": "7",
        }
    )
    assert (config.url, config.database, config.retention_days) == (
        "https://writer:private@clickhouse:8443",
        "analytics",
        7,
    )
    assert "private" not in repr(config)


def test_spend_storage_defaults_remain_gateway_defaults() -> None:
    config: Final = spend_storage_config({"CLICKHOUSE_URL": "http://clickhouse:8123"})
    assert (config.database, config.retention_days) == (
        DEFAULT_CLICKHOUSE_DATABASE,
        DEFAULT_AGENT_TRACING_RETENTION_DAYS,
    )


@pytest.mark.parametrize("retention", ("0", "-1", str(2**32), "1.5", "invalid"))
def test_spend_storage_rejects_invalid_retention(retention: str) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        spend_storage_config({"CLICKHOUSE_URL": "http://clickhouse:8123", "AGENT_TRACING_RETENTION_DAYS": retention})


def test_spend_storage_requires_clickhouse_without_lens_configuration() -> None:
    with pytest.raises(ValueError, match="CLICKHOUSE_URL is required"):
        spend_storage_config({"LITELLM_LENS_URL": "http://lens"})


def test_spend_rollup_request_projects_wire_names_and_epoch_millis() -> None:
    from datetime import datetime, timezone

    from litellm.rust_bridge.clickhouse import RollupDimension, SpendRollupRequest

    request: Final = SpendRollupRequest(
        start=datetime(2026, 10, 24, 21, 0, tzinfo=timezone.utc),
        end=datetime(2026, 10, 24, 22, 0, 0, 500_000, tzinfo=timezone.utc),
        bucket="15m",
        group_by=(RollupDimension.TAG, RollupDimension.PROVIDER),
        filters={RollupDimension.API_KEY: ("a", "b")},
        limit=25,
    )
    assert request.native() == {
        "start_ms": 1_792_875_600_000,
        "end_ms": 1_792_879_200_500,
        "bucket": "15m",
        "group_by": ["tag", "custom_llm_provider"],
        "filters": {"api_key": ["a", "b"]},
        "limit": 25,
    }


def test_spend_rollup_request_rejects_naive_timestamps() -> None:
    from datetime import datetime

    from litellm.rust_bridge.clickhouse import SpendRollupRequest

    request: Final = SpendRollupRequest(start=datetime(2026, 10, 24), end=datetime(2026, 10, 25), bucket="1h")
    with pytest.raises(ValueError, match="timezone-aware"):
        request.native()
