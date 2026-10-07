from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.diagnostics import DiagnosticsConfig

_VALUE: Final = TypeAdapter(dict[str, JsonValue])


def test_python_configuration_matches_the_shared_rust_wire_contract() -> None:
    fixture: Final = Path(__file__).parents[2] / "litellm-rust/crates/tracing/tests/fixtures/diagnostics.json"
    value: Final = fixture.read_text()
    config: Final = DiagnosticsConfig.model_validate_json(value)
    assert _VALUE.validate_json(config.model_dump_json()) == _VALUE.validate_json(value)


def test_environment_replaces_the_gateway_yaml_block_without_merging_credentials() -> None:
    config: Final = DiagnosticsConfig.from_sources(
        {"enabled": True, "service_name": "yaml-service"},
        {"LITELLM_DIAGNOSTICS": '{"enabled": false, "service_name": "env-service"}'},
    )
    assert config.model_dump() == {
        "enabled": False,
        "payload_shapes": False,
        "service_name": "env-service",
        "policy": {"minimum_level": "INFO", "target_prefixes": (), "sample_rate": 1.0},
        "destinations": (),
    }


@pytest.mark.parametrize(
    "configuration",
    (
        '{"enabled": "false"}',
        '{"policy": {"sample_rate": -0.1}}',
        '{"policy": {"sample_rate": 1.1}}',
        '{"policy": {"sample_rate": "0.5"}}',
        '{"policy": {"sample_rate": NaN}}',
        '{"policy": {"minimum_level": "quiet"}}',
        '{"unexpected": true}',
        '{"enabled": true, "destinations": [{"transport": "otlp", "name": "same", "endpoint": "http://localhost/v1/logs"}, {"transport": "otlp", "name": "same", "endpoint": "http://localhost/v1/logs"}]}',
    ),
)
def test_malformed_diagnostic_policy_is_rejected(configuration: str) -> None:
    with pytest.raises(ValidationError):
        DiagnosticsConfig.model_validate_json(configuration)


def test_destination_credentials_are_excluded_from_configuration_repr() -> None:
    config: Final = DiagnosticsConfig.model_validate(
        {"destinations": [{"transport": "posthog", "name": "events", "api_key": "synthetic-secret"}]}
    )
    assert "synthetic-secret" not in repr(config)
    destination: Final = TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(
        _VALUE.validate_json(config.model_dump_json())["destinations"]
    )[0]
    assert destination["api_key"] == "synthetic-secret"
