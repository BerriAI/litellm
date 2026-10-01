import pytest

from litellm.tracing.config import is_clickhouse_tracing_enabled, trace_storage_config


@pytest.mark.parametrize(
    ("settings", "enabled"),
    [
        ({"store": "clickhouse"}, True),
        ({"store": {"type": "clickhouse"}}, True),
        ({"store": {"type": "other"}}, False),
        (None, False),
    ],
)
def test_clickhouse_tracing_enablement(settings: object, enabled: bool) -> None:
    assert is_clickhouse_tracing_enabled(settings) is enabled


def test_yaml_values_override_environment_and_resolve_nested_references() -> None:
    config = trace_storage_config(
        {
            "store": {
                "type": "clickhouse",
                "url": "os.environ/TRACING_URL",
                "database": "analytics",
                "retention_days": 7,
            },
        },
        {
            "TRACING_URL": "https://writer:password@clickhouse.example:8443",
            "CLICKHOUSE_URL": "https://other.example:8443",
            "CLICKHOUSE_DATABASE": "other",
            "AGENT_TRACING_RETENTION_DAYS": "100",
        },
    )
    assert config.url == "https://writer:password@clickhouse.example:8443"
    assert config.database == "analytics"
    assert config.retention_days == 7
    assert "password" not in repr(config)


def test_legacy_environment_values_remain_supported() -> None:
    config = trace_storage_config(
        {},
        {
            "CLICKHOUSE_URL": "http://localhost:8123",
            "CLICKHOUSE_DATABASE": "legacy",
            "AGENT_TRACING_RETENTION_DAYS": "11",
            "AGENT_TRACING_SPEND_LOG_RETENTION_DAYS": "11",
        },
    )
    assert (config.database, config.retention_days) == ("legacy", 11)


def test_conflicting_legacy_retention_requires_one_explicit_value() -> None:
    environ = {
        "CLICKHOUSE_URL": "http://localhost:8123",
        "AGENT_TRACING_RETENTION_DAYS": "30",
        "AGENT_TRACING_SPEND_LOG_RETENTION_DAYS": "90",
    }
    with pytest.raises(ValueError, match="legacy tracing retention values differ"):
        trace_storage_config({}, environ)
    assert trace_storage_config({"store": {"type": "clickhouse", "retention_days": 14}}, environ).retention_days == 14


def test_retention_defaults_apply_when_unset() -> None:
    config = trace_storage_config({"store": {"type": "clickhouse", "url": "http://localhost:8123"}}, {})
    assert (config.database, config.retention_days) == ("litellm", 14)


@pytest.mark.parametrize("value", [0, -1, True, "not-a-number", 2**32])
def test_invalid_retention_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match=r"tracing.store.retention_days must be a positive integer"):
        trace_storage_config(
            {"store": {"type": "clickhouse", "url": "http://localhost:8123", "retention_days": value}}, {}
        )


def test_missing_url_is_rejected_without_echoing_secrets() -> None:
    with pytest.raises(ValueError, match=r"tracing.store.url or CLICKHOUSE_URL is required") as error:
        trace_storage_config({"store": {"type": "clickhouse", "url": "os.environ/MISSING"}}, {})
    assert "MISSING" not in str(error.value)


def test_legacy_reader_and_split_retention_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="reader_url, trace_retention_days"):
        trace_storage_config(
            {
                "store": {
                    "type": "clickhouse",
                    "url": "http://localhost:8123",
                    "reader_url": "http://localhost:8124",
                    "trace_retention_days": 30,
                }
            },
            {},
        )
