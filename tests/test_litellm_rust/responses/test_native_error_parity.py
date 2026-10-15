"""Native Responses failures must reach the caller and the failure callback exactly as Python's do."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import openai
import pytest
from pydantic import JsonValue, TypeAdapter

from tests.test_litellm_rust.responses.support import PUBLIC_ARESPONSES, PYTHON_ARESPONSES, FallbackProbe
from tests.test_litellm_rust.responses.test_native_parity import MODEL, WEATHER_TOOL
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import isolated_callback_registries
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

_LOGGED: Final = TypeAdapter(dict[str, object])
_COMPARED_LOG_FIELDS: Final = ("status", "error_str", "model", "custom_llm_provider", "call_type", "response_cost")
_COMPARED_ERROR_FIELDS: Final = ("error_code", "error_class", "llm_provider")


@dataclass(frozen=True, slots=True)
class Failure:
    exception: tuple[object, ...]
    logged: Mapping[str, object]
    fell_back: bool


@dataclass(frozen=True, slots=True)
class FailureCase:
    name: str
    kwargs: Mapping[str, object]
    upstream: ResponseSpec | None = None
    falls_back: bool = False


def _upstream(status: int, error: Mapping[str, JsonValue]) -> ResponseSpec:
    return ResponseSpec(body={"error": dict(error)}, status=status)


CASES: Final = (
    FailureCase(
        "bad_request",
        {"input": "hi"},
        _upstream(
            400,
            {"message": "Invalid value for 'input'.", "type": "invalid_request_error", "param": "input"},
        ),
    ),
    FailureCase(
        "invalid_api_key",
        {"input": "hi"},
        _upstream(401, {"message": "Incorrect API key provided", "type": "invalid_request_error"}),
    ),
    FailureCase(
        "rate_limited",
        {"input": "hi"},
        _upstream(429, {"message": "Rate limit reached", "type": "requests", "code": "rate_limit_exceeded"}),
    ),
    FailureCase("server_error", {"input": "hi"}, _upstream(500, {"message": "boom", "type": "server_error"})),
    FailureCase(
        "invalid_tool_parameters",
        {"input": "hi", "tools": [{**WEATHER_TOOL, "parameters": "{"}]},
        falls_back=True,
    ),
)


async def _failure(case: FailureCase, server: RecordingServer, probe: FallbackProbe | None) -> Failure:
    with isolated_callback_registries():
        recorder: Final = RecordingLogger()
        if case.upstream is not None:
            server.enqueue(case.upstream)
        fallbacks_before: Final = 0 if probe is None else probe.calls
        call: Final = PYTHON_ARESPONSES if probe is None else PUBLIC_ARESPONSES
        with pytest.raises(openai.APIError) as raised:
            await call(model=MODEL, api_key="sk-test", api_base=server.base_url, callbacks=[recorder], **case.kwargs)
        await drain_logging()
        failures: Final = await recorder.wait_for_async("async_log_failure_event")
    standard: Final = _LOGGED.validate_python(_LOGGED.validate_python(failures[0].kwargs)["standard_logging_object"])
    error_information: Final = _LOGGED.validate_python(standard["error_information"])
    error: Final = raised.value
    return Failure(
        exception=(
            type(error).__name__,
            getattr(error, "status_code", None),
            str(error),
            str(getattr(error, "llm_provider", None)),
            getattr(error, "model", None),
        ),
        logged=MappingProxyType(
            {
                **{name: standard.get(name) for name in _COMPARED_LOG_FIELDS},
                **{name: str(error_information.get(name)) for name in _COMPARED_ERROR_FIELDS},
            }
        ),
        fell_back=probe is not None and probe.calls > fallbacks_before,
    )


@pytest.fixture
def failure_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.expected_requests = None
    return recording_server


@pytest.mark.asyncio
@pytest.mark.parametrize("case", tuple(pytest.param(case, id=case.name) for case in CASES))
async def test_native_failures_match_python(
    case: FailureCase, failure_server: RecordingServer, native_responses: FallbackProbe
) -> None:
    python: Final = await _failure(case, failure_server, probe=None)
    native: Final = await _failure(case, failure_server, probe=native_responses)
    assert (native.exception, native.logged, native.fell_back) == (python.exception, python.logged, case.falls_back)
