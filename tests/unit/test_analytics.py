from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.analytics import AnalyticsHost, api_use, license_declared, track_async, track_sync


class RecordingEmitter:
    def __init__(self) -> None:
        self.configuration: object = None
        self.initializations: int = 0
        self.events: tuple[str, ...] = ()
        self.stops: int = 0

    def initialize_analytics(self, configuration: str) -> tuple[bool, bool]:
        self.initializations += 1
        self.configuration = json.loads(configuration)
        return True, False

    def emit_analytics(self, route: str) -> None:
        self.events += (route,)

    def shutdown_analytics(self) -> None:
        self.stops += 1


@pytest.mark.parametrize("environment", ({}, {"LITELLM_LICENSE": ""}, {"LITELLM_LICENSE": "invalid"}))
def test_sdk_freezes_inputs_and_owns_one_drain(environment: Mapping[str, str]) -> None:
    emitter: Final = RecordingEmitter()
    host: Final = AnalyticsHost(lambda: emitter, environment)
    host.api_used("completion")
    host.api_used("acompletion")
    assert emitter.initializations == 1
    assert isinstance(emitter.configuration, dict)
    assert emitter.configuration["inputs"] == {
        "do_not_track": None,
        "explicit": None,
        "license_configured": "LITELLM_LICENSE" in environment,
    }
    assert emitter.events == ("completion", "acompletion")
    host.shutdown()
    host.shutdown()
    host.api_used("completion")
    assert emitter.stops == 1
    assert emitter.initializations == 1
    assert emitter.events == ("completion", "acompletion")


def test_gateway_suppresses_sdk_until_final_inputs_and_remembers_unresolved_license() -> None:
    emitter: Final = RecordingEmitter()
    host: Final = AnalyticsHost(lambda: emitter, {"DO_NOT_TRACK": "1", "LITELLM_TELEMETRY": "true"})
    host.prepare_gateway()
    host.api_used("completion")
    assert emitter.initializations == 0
    host.declare_license()
    host.initialize("python_gateway")
    host.api_used("completion")
    assert emitter.initializations == 1
    assert isinstance(emitter.configuration, dict)
    assert emitter.configuration["inputs"] == {
        "do_not_track": "1",
        "explicit": "true",
        "license_configured": True,
    }
    assert emitter.configuration["surface"] == "python_gateway"
    assert emitter.events == ("completion",)
    host.shutdown()
    assert emitter.stops == 1


def test_missing_binding_is_optional_and_does_not_change_results() -> None:
    host: Final = AnalyticsHost(lambda: None, {})
    host.api_used("completion")
    host.shutdown()
    marker: Final = object()
    wrapped: Final = track_sync(lambda: marker, "completion")
    assert wrapped() is marker


@pytest.mark.asyncio
async def test_unstarted_calls_do_not_initialize_and_nested_calls_count_once() -> None:
    emitter: Final = RecordingEmitter()
    host: Final = AnalyticsHost(lambda: emitter, {})
    marker: Final = object()

    async def result() -> object:
        with api_use("nested", host):
            return marker

    wrapped: Final = track_async(result, "acompletion", host=host)
    unstarted: Final = wrapped()
    assert emitter.initializations == 0
    unstarted.close()
    assert emitter.events == ()
    assert await wrapped() is marker
    assert emitter.events == ("acompletion",)
    with pytest.raises(ValueError, match="failure"):
        with api_use("completion", host):
            raise ValueError("failure")
    assert await wrapped() is marker
    assert emitter.events == ("acompletion", "completion", "acompletion")
    host.shutdown()
    assert emitter.stops == 1


def test_inactive_sdk_session_is_cleared_before_gateway_configuration() -> None:
    class InactiveEmitter(RecordingEmitter):
        def initialize_analytics(self, configuration: str) -> tuple[bool, bool]:
            super().initialize_analytics(configuration)
            return False, False

    emitter: Final = InactiveEmitter()
    host: Final = AnalyticsHost(lambda: emitter, {})
    host.api_used("completion")
    host.prepare_gateway()
    assert emitter.stops == 1
    host.initialize("python_gateway", True)
    assert emitter.initializations == 2
    assert isinstance(emitter.configuration, dict)
    assert emitter.configuration["inputs"]["license_configured"] is True
    assert emitter.events == ()
    host.shutdown()
    assert emitter.stops == 2


@pytest.mark.parametrize(
    ("configuration", "expected"),
    (
        ({"environment_variables": {"LITELLM_LICENSE": ""}}, True),
        ({"environment_variables": {"LITELLM_LICENSE": "os.environ/MISSING"}}, True),
        ({"general_settings": {"litellm_license": None}}, True),
        ({"environment_variables": {"OTHER": "value"}}, False),
        ({}, False),
    ),
)
def test_yaml_license_declaration_survives_missing_or_unresolved_values(
    configuration: Mapping[str, object], expected: bool
) -> None:
    assert license_declared(configuration) is expected
