import pytest

from litellm import constants
from litellm.tracing import config as tracing_config
from litellm.tracing.config import is_clickhouse_tracing_enabled, trace_storage_config


@pytest.mark.parametrize(
    ("settings", "enabled"),
    [
        ({"store": "clickhouse"}, False),
        ({"store": {"type": "clickhouse"}}, True),
        ({"store": {"type": "other"}}, False),
        (None, False),
    ],
)
def test_clickhouse_tracing_enablement(settings: object, enabled: bool) -> None:
    assert is_clickhouse_tracing_enabled(settings) is enabled


def test_yaml_values_override_defaults_and_resolve_nested_references() -> None:
    config = trace_storage_config(
        {
            "store": {
                "type": "clickhouse",
                "url": "os.environ/TRACING_URL",
                "database": "os.environ/TRACING_DATABASE",
                "retention_days": "os.environ/TRACING_RETENTION_DAYS",
            },
        },
        {
            "TRACING_URL": "https://writer:password@clickhouse.example:8443",
            "TRACING_DATABASE": "analytics",
            "TRACING_RETENTION_DAYS": "7",
            "CLICKHOUSE_URL": "https://other.example:8443",
        },
    )
    assert config.url == "https://writer:password@clickhouse.example:8443"
    assert config.database == "analytics"
    assert config.retention_days == 7
    assert "password" not in repr(config)


def test_omitted_fields_use_environment_and_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing_config, "CLICKHOUSE_DATABASE", "env_database")
    monkeypatch.setattr(tracing_config, "AGENT_TRACING_RETENTION_DAYS", 11)
    config = trace_storage_config({}, {"CLICKHOUSE_URL": "http://localhost:8123"})
    assert (config.url, config.database, config.retention_days) == ("http://localhost:8123", "env_database", 11)


def test_constant_defaults_match_unified_store() -> None:
    assert (constants.CLICKHOUSE_DATABASE, constants.AGENT_TRACING_RETENTION_DAYS) == ("litellm", 14)


@pytest.mark.parametrize("field", ["url", "database", "retention_days"])
def test_unset_environment_reference_does_not_fall_back(field: str) -> None:
    store: dict[str, object] = {"type": "clickhouse", "url": "http://localhost:8123", field: "os.environ/MISSING"}
    with pytest.raises(ValueError, match=rf"tracing.store.{field} is set but resolved to no value") as error:
        trace_storage_config({"store": store}, {"CLICKHOUSE_URL": "http://fallback:8123"})
    assert "MISSING" not in str(error.value)


@pytest.mark.parametrize("store", ["clickhouse", {"type": "other"}])
def test_non_clickhouse_store_is_rejected(store: object) -> None:
    with pytest.raises(ValueError, match=r"tracing\.store\.type must be clickhouse"):
        trace_storage_config({"store": store}, {"CLICKHOUSE_URL": "http://localhost:8123"})


def test_non_string_database_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"tracing\.store\.database must be a string"):
        trace_storage_config({"store": {"type": "clickhouse", "url": "http://localhost:8123", "database": 1}}, {})


@pytest.mark.parametrize("value", [0, -1, True, "not-a-number", 2**32])
def test_invalid_retention_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match=r"tracing.store.retention_days must be a positive integer"):
        trace_storage_config(
            {"store": {"type": "clickhouse", "url": "http://localhost:8123", "retention_days": value}}, {}
        )


def test_missing_url_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"tracing.store.url or CLICKHOUSE_URL is required"):
        trace_storage_config({"store": {"type": "clickhouse"}}, {})


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
