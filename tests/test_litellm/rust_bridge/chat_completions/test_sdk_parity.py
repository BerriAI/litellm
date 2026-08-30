from __future__ import annotations

import asyncio
import sys
import traceback
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, cast

import pytest
from pydantic import BaseModel, JsonValue

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse, ModelResponseStream
from tests.route_parity.compare import (
    assert_request_parity,
    assert_response_parity,
    json_values_equal,
    validate_harness,
)
from tests.route_parity.fixture_models import JsonObject
from tests.route_parity.models import Execution, SDKCommand, SDKReport, WorkerFailure, WorkerResult, WorkerSuccess
from tests.route_parity.runner import (
    PythonScriptRunner,
    PythonScriptWorker,
    execution_worker_pair,
    parity_worker_main,
    run_execution,
)
from tests.test_litellm.rust_bridge.chat_completions.fixture_models import (
    AnthropicChatCompletionSdkInput,
    ChatCompletionParityCase,
)

API_KEY: Final = "test-key"
PYTHON_HTTP_SENTINEL: Final = "python-chat-completions-parity-fallback"


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING: Final = _Missing()
JsonPath = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class _ResponseGap:
    path: JsonPath
    python_value: JsonValue | _Missing
    rust_value: JsonValue | _Missing


STATIC_RESPONSE_GAPS: Final = (
    _ResponseGap(
        path=("choices", 0, "message", "provider_specific_fields"),
        python_value={"citations": None, "thinking_blocks": None},
        rust_value=None,
    ),
    _ResponseGap(
        path=("choices", 0, "provider_specific_fields"),
        python_value=_MISSING,
        rust_value={},
    ),
    _ResponseGap(
        path=("usage", "cache_creation_input_tokens"),
        python_value=0,
        rust_value=_MISSING,
    ),
    _ResponseGap(
        path=("usage", "cache_read_input_tokens"),
        python_value=0,
        rust_value=_MISSING,
    ),
    _ResponseGap(path=("usage", "inference_geo"), python_value=None, rust_value=_MISSING),
    _ResponseGap(path=("usage", "iterations"), python_value=None, rust_value=_MISSING),
    _ResponseGap(path=("usage", "service_tier"), python_value=None, rust_value=_MISSING),
    _ResponseGap(path=("usage", "speed"), python_value=None, rust_value=_MISSING),
)


class SDKRoute(str, Enum):
    COMPLETION = "completion"
    ACOMPLETION = "acompletion"


def _call_kwargs(sdk_input: AnthropicChatCompletionSdkInput, mock_url: str) -> dict[str, object]:
    return {
        **sdk_input.as_sdk_kwargs(),
        "api_base": mock_url,
        "api_key": API_KEY,
    }


def _normalized_response(response: ModelResponse | ModelResponseStream) -> JsonObject:
    payload: Final = cast(JsonObject, BaseModel.model_dump(response, mode="json"))
    response_id: Final = payload.get("id")
    created: Final = payload.get("created")
    assert isinstance(response_id, str) and response_id
    assert isinstance(created, int) and not isinstance(created, bool)
    return {**payload, "id": "<generated>", "created": 0}


def _nonstreaming_report(response: object) -> SDKReport:
    if not isinstance(response, ModelResponse):
        raise TypeError(f"expected ModelResponse, got {type(response).__name__}")
    return SDKReport(response=_normalized_response(response))


def _sync_report(response: object, stream: bool) -> SDKReport:
    if not stream:
        return _nonstreaming_report(response)
    if not isinstance(response, CustomStreamWrapper):
        raise TypeError(f"expected CustomStreamWrapper, got {type(response).__name__}")
    chunks: Final[list[JsonValue]] = [cast(JsonValue, _normalized_response(chunk)) for chunk in response]
    return SDKReport(response=chunks)


async def _async_report(response: object, stream: bool) -> SDKReport:
    if not stream:
        return _nonstreaming_report(response)
    if not isinstance(response, CustomStreamWrapper):
        raise TypeError(f"expected CustomStreamWrapper, got {type(response).__name__}")
    chunks: Final[list[JsonValue]] = [cast(JsonValue, _normalized_response(chunk)) async for chunk in response]
    return SDKReport(response=chunks)


def _execute_sdk_case(
    sdk_input: AnthropicChatCompletionSdkInput,
    route: SDKRoute,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> SDKReport:
    import litellm

    call_kwargs: Final = _call_kwargs(sdk_input, mock_url)
    if route is SDKRoute.COMPLETION:
        sync_route: Final = cast(Callable[..., object], litellm.completion)
        return _sync_report(sync_route(**call_kwargs), sdk_input.stream)
    async_route: Final = cast(Callable[..., Awaitable[object]], litellm.acompletion)
    response: Final = event_loop.run_until_complete(async_route(**call_kwargs))
    return event_loop.run_until_complete(_async_report(response, sdk_input.stream))


@pytest.fixture(scope="module")
def sdk_workers() -> Generator[tuple[PythonScriptWorker, PythonScriptWorker]]:
    runner: Final = PythonScriptRunner(
        entrypoint=Path(__file__),
        rust_env_var="LITELLM_RUST",
        python_user_agent=PYTHON_HTTP_SENTINEL,
        route_label="chat completions",
    )
    with execution_worker_pair(runner) as workers:
        yield workers


def _response_gap_values_equal(left: JsonValue | _Missing, right: JsonValue | _Missing) -> bool:
    if isinstance(left, _Missing) or isinstance(right, _Missing):
        return left is right
    return json_values_equal(left, right)


def _json_difference_paths(left: JsonValue, right: JsonValue, path: str = "$") -> tuple[str, ...]:
    if type(left) is not type(right):
        return (path,)
    if isinstance(left, dict) and isinstance(right, dict):
        keys: Final = frozenset(left) | frozenset(right)
        return tuple(
            difference
            for key in sorted(keys)
            for difference in (
                (f"{path}.{key}",)
                if key not in left or key not in right
                else _json_difference_paths(left[key], right[key], f"{path}.{key}")
            )
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return (path,)
        return tuple(
            difference
            for index in range(len(left))
            for difference in _json_difference_paths(left[index], right[index], f"{path}[{index}]")
        )
    return () if json_values_equal(left, right) else (path,)


def _json_value_at(value: JsonValue, path: JsonPath) -> JsonValue | _Missing:
    if not path:
        return value
    segment: Final = path[0]
    remaining: Final = path[1:]
    if isinstance(segment, str) and isinstance(value, dict):
        return _MISSING if segment not in value else _json_value_at(value[segment], remaining)
    if isinstance(segment, int) and isinstance(value, list) and 0 <= segment < len(value):
        return _json_value_at(value[segment], remaining)
    return _MISSING


def _format_json_path(path: JsonPath) -> str:
    return "$" + "".join(f"[{segment}]" if isinstance(segment, int) else f".{segment}" for segment in path)


def _assert_streaming_fallback(python: Execution, accelerated: Execution) -> None:
    assert python.request.user_agent == PYTHON_HTTP_SENTINEL
    assert accelerated.request.user_agent == PYTHON_HTTP_SENTINEL
    assert_request_parity(python, accelerated)
    assert_response_parity(python, accelerated)


def _assert_known_response_gaps(python: Execution, accelerated: Execution) -> None:
    validate_harness(python, accelerated, PYTHON_HTTP_SENTINEL)
    assert_request_parity(python, accelerated)
    completion_tokens: Final = _json_value_at(python.report.response, ("usage", "completion_tokens"))
    assert isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool)
    response_gaps: Final = (
        *STATIC_RESPONSE_GAPS,
        _ResponseGap(
            path=("usage", "completion_tokens_details"),
            python_value={
                "accepted_prediction_tokens": None,
                "audio_tokens": None,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": None,
                "text_tokens": completion_tokens,
                "image_tokens": None,
                "video_tokens": None,
            },
            rust_value=None,
        ),
    )
    differences: Final = frozenset(_json_difference_paths(python.report.response, accelerated.report.response))
    expected_differences: Final = frozenset(_format_json_path(gap.path) for gap in response_gaps)
    assert differences == expected_differences
    actual_values: Final = tuple(
        (
            _json_value_at(python.report.response, gap.path),
            _json_value_at(accelerated.report.response, gap.path),
        )
        for gap in response_gaps
    )
    expected_values: Final = tuple((gap.python_value, gap.rust_value) for gap in response_gaps)
    assert all(
        _response_gap_values_equal(actual_python, expected_python)
        and _response_gap_values_equal(actual_rust, expected_rust)
        for (actual_python, actual_rust), (expected_python, expected_rust) in zip(
            actual_values,
            expected_values,
            strict=True,
        )
    )


@pytest.mark.parametrize("route", tuple(SDKRoute), ids=tuple(route.value for route in SDKRoute))
def test_recorded_chat_completion_sdk_behavior(
    chat_completion_fixture: ChatCompletionParityCase,
    route: SDKRoute,
    tmp_path: Path,
    sdk_workers: tuple[PythonScriptWorker, PythonScriptWorker],
) -> None:
    case_file: Final = tmp_path / f"{route.value}-chat-completion-parity-case.json"
    case_file.write_text(chat_completion_fixture.model_dump_json(indent=2, exclude_unset=True), encoding="utf-8")
    response: Final = chat_completion_fixture.provider_response
    python_worker, rust_worker = sdk_workers
    python: Final = run_execution(python_worker, case_file, route.value, response)
    rust: Final = run_execution(rust_worker, case_file, route.value, response)

    if chat_completion_fixture.litellm_input.stream:
        _assert_streaming_fallback(python, rust)
        return
    _assert_known_response_gaps(python, rust)


def _execute_worker_command(
    command_json: str,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> WorkerResult:
    try:
        command: Final = SDKCommand.model_validate_json(command_json)
        case_file: Final = Path(command.case_file)
        route: Final = SDKRoute(command.route)
        case: Final = ChatCompletionParityCase.model_validate_json(case_file.read_text(encoding="utf-8"))
        return WorkerSuccess(report=_execute_sdk_case(case.litellm_input, route, mock_url, event_loop))
    except Exception:
        return WorkerFailure(error=traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--parity-worker":
        raise SystemExit("usage: test_sdk_parity.py --parity-worker MOCK_URL")
    parity_worker_main(_execute_worker_command, sys.argv[2])
