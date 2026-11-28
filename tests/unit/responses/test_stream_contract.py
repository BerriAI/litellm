import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Union, get_args

import httpx
import pytest
from openai.types.responses import ResponseStreamEvent
from prometheus_client import REGISTRY
from pydantic import Field, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.streaming_iterator import SyncResponsesAPIStreamingIterator
from litellm.types.utils import CallTypes, LlmProviders
from litellm.utils import ProviderConfigManager

_SDK_EVENT: Final = TypeAdapter(
    Annotated[Union[get_args(get_args(ResponseStreamEvent)[0])], Field(discriminator="type")]  # noqa: UP007  # a runtime tuple of members needs Union[...]
)
_RECORDED_STREAMS: Final = Path(__file__).parent / "stream_contract_fixtures"
_METRIC: Final = "litellm_responses_stream_contract_violations_total"

_GPT_OSS_SERVER_STREAM: Final = (
    {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-oss-120b", "status": "in_progress"}},
    {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1", "role": "assistant"}},
    {"type": "response.content_part.added", "item_id": "msg_1", "part": {"type": "output_text", "text": ""}},
    {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello"},
    {
        "type": "response.completed",
        "response": {"id": "resp_1", "model": "gpt-oss-120b", "status": "completed", "output": []},
    },
)
_NULL_ERROR_CODE_STREAM: Final = (
    {
        "type": "error",
        "sequence_number": 1,
        "error": {"type": "invalid_request_error", "code": None, "message": "Something went wrong", "param": None},
    },
)
_TYPELESS_ERROR_STREAM: Final = (
    {"error": {"type": "service_unavailable", "code": "overloaded", "message": "Model is overloaded"}},
)
_FIREWORKS_MCP_STREAM: Final = (
    {
        "type": "response.mcp_call.arguments.delta",
        "item_id": "mcp_1",
        "output_index": 1,
        "sequence_number": 4,
        "delta": {"repoName": "BerriAI/litellm"},
    },
    {
        "type": "response.mcp_call.arguments.done",
        "item_id": "mcp_1",
        "output_index": 1,
        "sequence_number": 5,
        "arguments": {"repoName": "BerriAI/litellm"},
    },
)
_REPORTED_STREAMS: Final = (
    pytest.param("openai", _GPT_OSS_SERVER_STREAM, id="issue_20570_openai_compatible_missing_fields"),
    pytest.param("hosted_vllm", _GPT_OSS_SERVER_STREAM, id="issue_20570_hosted_vllm_missing_fields"),
    pytest.param("openai", _NULL_ERROR_CODE_STREAM, id="issue_16759_null_error_code"),
    pytest.param("openai", _TYPELESS_ERROR_STREAM, id="pr_50466_typeless_error"),
    pytest.param("fireworks_ai", _FIREWORKS_MCP_STREAM, id="lit_10957_fireworks_mcp_names_and_object_arguments"),
)


def _iterator(provider: str, request: Mapping[str, object]) -> SyncResponsesAPIStreamingIterator:
    logging_obj: Final = LiteLLMLoggingObj(
        model="test-model",
        messages=[],
        stream=True,
        call_type="responses",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="test-call",
        function_id="test-function",
    )
    return SyncResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders(provider), model="test-model"
        ),
        logging_obj=logging_obj,
        litellm_metadata={},
        custom_llm_provider=provider,
        request_data=dict(request),
        call_type=CallTypes.responses.value,
    )


def _emitted(
    provider: str, chunks: Sequence[Mapping[str, object]], request: Mapping[str, object] | None = None
) -> tuple[dict[str, object], ...]:
    iterator: Final = _iterator(provider, request or {})
    events: Final = tuple(iterator._process_chunk(json.dumps(chunk)) for chunk in chunks)
    return tuple(json.loads(event.model_dump_json(exclude_none=True)) for event in events if event is not None)


def _violations(provider: str, event_type: str, reason: str) -> float:
    labels: Final = {"provider": provider, "event_type": event_type, "reason": reason}
    return REGISTRY.get_sample_value(_METRIC, labels) or 0.0


def _contract_warnings(caplog: pytest.LogCaptureFixture) -> tuple[str, ...]:
    return tuple(
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "does not match the OpenAI spec" in record.getMessage()
    )


@pytest.fixture
def contract_warnings(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.WARNING, logger=verbose_logger.name)
    return caplog


def _recorded_streams() -> tuple[object, ...]:
    return tuple(
        pytest.param(path.name.split("__")[0], path, id=path.stem) for path in sorted(_RECORDED_STREAMS.glob("*.jsonl"))
    )


_FIREWORKS_MCP_ITEM_EVENTS: Final = frozenset(
    {"response.output_item.added", "response.output_item.done", "response.mcp_call.completed"}
)
_UNREPAIRED_RECORDED_EVENTS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "fireworks_ai__mcp": _FIREWORKS_MCP_ITEM_EVENTS | {"response.completed"},
        "fireworks_ai__mcp_max_tool_calls": _FIREWORKS_MCP_ITEM_EVENTS | {"response.incomplete"},
    }
)


def _off_spec(event: Mapping[str, object]) -> bool:
    try:
        _SDK_EVENT.validate_python(event)
    except ValidationError:
        return True
    return False


@pytest.mark.parametrize(("provider", "path"), _recorded_streams())
def test_every_event_a_provider_really_streamed_reaches_the_client_in_openai_sdk_shape_or_is_reported(
    provider: str, path: Path, contract_warnings: pytest.LogCaptureFixture
):
    chunks: Final = tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    request: Final = json.loads(path.with_suffix(".meta.json").read_text())["request_body"]
    unrepaired: Final = _UNREPAIRED_RECORDED_EVENTS.get(path.stem, frozenset())

    emitted: Final = _emitted(provider, chunks, request)

    assert len(emitted) == len(chunks)
    assert {str(event["type"]) for event in emitted if _off_spec(event)} == unrepaired
    warnings: Final = _contract_warnings(contract_warnings)
    assert len(warnings) == len(unrepaired)
    assert all(any(f" {event_type}): " in warning for warning in warnings) for event_type in unrepaired)


@pytest.mark.parametrize(("provider", "chunks"), _REPORTED_STREAMS)
def test_every_event_from_a_reported_provider_deviation_reaches_the_client_in_openai_sdk_shape(
    provider: str, chunks: Sequence[Mapping[str, object]], contract_warnings: pytest.LogCaptureFixture
):
    emitted: Final = _emitted(provider, chunks)

    assert len(emitted) == len(chunks)
    for event in emitted:
        _SDK_EVENT.validate_python(event)
    assert _contract_warnings(contract_warnings) == ()


def test_events_the_provider_left_unnumbered_are_numbered_after_the_last_numbered_one():
    chunks: Final = (
        {"type": "response.output_text.delta", "item_id": "m", "output_index": 0, "content_index": 0, "delta": "a"},
        {
            "type": "response.output_text.delta",
            "item_id": "m",
            "output_index": 0,
            "content_index": 0,
            "delta": "b",
            "sequence_number": 7,
        },
        {"type": "response.output_text.delta", "item_id": "m", "output_index": 0, "content_index": 0, "delta": "c"},
    )

    assert [event["sequence_number"] for event in _emitted("openai", chunks)] == [0, 7, 8]


def test_values_the_provider_did_send_are_never_replaced_by_a_repair():
    chunk: Final = {
        "type": "response.output_text.delta",
        "item_id": "m",
        "output_index": 3,
        "content_index": 2,
        "delta": "a",
        "logprobs": [{"token": "a", "logprob": -0.5, "top_logprobs": []}],
        "sequence_number": 9,
    }

    assert _emitted("openai", (chunk,)) == (chunk,)


def test_a_typeless_error_keeps_its_nested_error_and_gains_the_top_level_fields_the_sdk_reads():
    (event,) = _emitted("openai", _TYPELESS_ERROR_STREAM)

    assert event["error"] == _TYPELESS_ERROR_STREAM[0]["error"]
    assert (event["message"], event["code"]) == ("Model is overloaded", "overloaded")


def test_a_null_error_code_becomes_unknown_error_in_both_places_the_client_reads_it():
    (event,) = _emitted("openai", _NULL_ERROR_CODE_STREAM)

    assert (event["code"], event["error"]["code"]) == ("unknown_error", "unknown_error")


_UNREPAIRABLE_DELTA: Final = {
    "type": "response.output_text.delta",
    "output_index": 0,
    "content_index": 0,
    "delta": "no item id",
    "sequence_number": 1,
}


def test_an_event_that_still_breaks_the_spec_reaches_the_client_as_received_and_is_reported_loudly(
    contract_warnings: pytest.LogCaptureFixture,
):
    before: Final = _violations("xai", "response.output_text.delta", "invalid")

    emitted: Final = _emitted("xai", (_UNREPAIRABLE_DELTA, {**_UNREPAIRABLE_DELTA, "sequence_number": 2}))

    assert emitted == (
        {**_UNREPAIRABLE_DELTA, "logprobs": []},
        {**_UNREPAIRABLE_DELTA, "logprobs": [], "sequence_number": 2},
    )
    assert _violations("xai", "response.output_text.delta", "invalid") == before + 2
    (warning,) = _contract_warnings(contract_warnings)
    assert "xai" in warning and "response.output_text.delta" in warning and "item_id" in warning


def test_each_stream_warns_once_per_off_spec_event_type(contract_warnings: pytest.LogCaptureFixture):
    _emitted("xai", (_UNREPAIRABLE_DELTA, _UNREPAIRABLE_DELTA))
    _emitted("xai", (_UNREPAIRABLE_DELTA,))

    assert len(_contract_warnings(contract_warnings)) == 2


def test_an_event_type_nobody_defines_reaches_the_client_as_received_and_is_reported_loudly(
    contract_warnings: pytest.LogCaptureFixture,
):
    chunk: Final = {"type": "response.provider_specific.delta", "delta": {"q": 1}, "sequence_number": 1}
    before: Final = _violations("openrouter", "response.provider_specific.delta", "unknown_type")

    assert _emitted("openrouter", (chunk,)) == (chunk,)
    assert _violations("openrouter", "response.provider_specific.delta", "unknown_type") == before + 1
    assert len(_contract_warnings(contract_warnings)) == 1


@pytest.mark.parametrize(
    ("chunk", "reported"),
    (
        pytest.param(
            {
                "type": "response.reasoning_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "think",
                "sequence_number": 1,
            },
            False,
            id="sdk_type_litellm_does_not_model_valid",
        ),
        pytest.param(
            {"type": "response.reasoning_text.delta", "output_index": 0, "delta": "think", "sequence_number": 1},
            True,
            id="sdk_type_litellm_does_not_model_missing_item_id",
        ),
        pytest.param(
            {"type": "response.shell_call.in_progress", "item_id": "sh_1", "sequence_number": 1},
            False,
            id="litellm_only_type",
        ),
    ),
)
def test_event_types_litellm_does_not_model_are_checked_against_the_sdk_when_it_knows_them(
    chunk: Mapping[str, object], reported: bool, contract_warnings: pytest.LogCaptureFixture
):
    (event,) = _emitted("openai", (chunk,))

    assert event.items() >= chunk.items()
    assert len(_contract_warnings(contract_warnings)) == int(reported)


def test_a_response_missing_its_tool_echo_gets_the_requested_tools_without_their_credentials():
    request: Final = {
        "tools": [
            {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}},
            {
                "type": "mcp",
                "server_label": "wiki",
                "server_url": "https://mcp.example.com",
                "headers": {"Authorization": "Bearer secret-token"},
                "authorization": "secret-token",
            },
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }

    (event,) = _emitted("openai", (_GPT_OSS_SERVER_STREAM[0],), request)

    assert event["response"]["tools"] == [
        request["tools"][0],
        {"type": "mcp", "server_label": "wiki", "server_url": "https://mcp.example.com"},
    ]
    assert (event["response"]["tool_choice"], event["response"]["parallel_tool_calls"]) == ("required", False)
    assert "secret-token" not in json.dumps(event)
