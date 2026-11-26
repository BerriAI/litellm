"""Reads of the dict-style accessors on litellm.types.utils models keep their runtime values."""

import pytest

from litellm.integrations.langfuse.langfuse import _chat_usage, _extract_cache_read_input_tokens
from litellm.integrations.newrelic.newrelic import NewRelicLogger
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    should_convert_tool_call_to_json_mode,
)
from litellm.proxy.guardrails.guardrail_hooks.singulr.singulr import SingulrGuardrail
from litellm.responses.litellm_completion_transformation.transformation import _tool_call_status
from litellm.types.utils import ChatCompletionMessageToolCall, Function, ModelResponse, Usage


@pytest.mark.parametrize(
    ("function", "expected"),
    (
        (Function(name="f", arguments="{}"), "completed"),
        ({"status": "in_progress"}, "in_progress"),
        ({"status": ""}, "completed"),
        ({"status": "not-a-status"}, "not-a-status"),
        ({"status": "incomplete"}, "incomplete"),
        ({}, "completed"),
    ),
)
def test_tool_call_status_passes_the_stored_value_through(function: object, expected: str) -> None:
    assert _tool_call_status(function) == expected


def test_tool_call_status_rejects_a_receiver_without_get() -> None:
    with pytest.raises(AttributeError):
        _tool_call_status(object())


@pytest.mark.parametrize(
    "tool_call",
    (
        ChatCompletionMessageToolCall(id="call_1", type="function", function=Function(name="f", arguments='{"a": 1}')),
        {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}, "index": 0},
    ),
)
def test_singulr_tool_call_reads_models_and_dicts_alike(tool_call) -> None:
    built = SingulrGuardrail._build_tool_call(tool_call)
    assert built is not None
    assert (built.id, built.type, built.function.name, built.function.arguments) == (
        "call_1",
        "function",
        "f",
        '{"a": 1}',
    )


def test_singulr_tool_call_without_id_is_skipped() -> None:
    assert SingulrGuardrail._build_tool_call({"id": None, "type": "function", "function": {}, "index": 0}) is None


def test_langfuse_chat_usage_keeps_cache_token_reads() -> None:
    usage = Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12, cache_read_input_tokens=4)
    chat_usage = _chat_usage(usage)
    assert chat_usage is usage
    assert _extract_cache_read_input_tokens(usage) == 4
    converted = _chat_usage({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
    assert isinstance(converted, Usage)
    assert converted.get("cache_creation_input_tokens") in (None, 0)


@pytest.mark.parametrize(
    ("response", "expected"),
    (
        (ModelResponse(id="chatcmpl-1", choices=[{"finish_reason": "length", "message": {"content": "hi"}}]), "length"),
        ({"choices": [{"finish_reason": "stop"}]}, "stop"),
        ({"choices": [{"finish_reason": None}]}, "unknown"),
        ({}, "unknown"),
    ),
)
def test_newrelic_finish_reason_reads_models_and_dicts_alike(response, expected: str) -> None:
    assert NewRelicLogger.__new__(NewRelicLogger)._get_finish_reason(response) == expected


def test_newrelic_completion_id_prefers_the_response_id() -> None:
    logger = NewRelicLogger.__new__(NewRelicLogger)
    assert logger._extract_completion_id({"litellm_call_id": "call"}, ModelResponse(id="chatcmpl-1")) == "chatcmpl-1"
    assert logger._extract_completion_id({"litellm_call_id": "call"}, {"id": None}) == "call"


@pytest.mark.parametrize(
    ("tool_calls", "expected"),
    (
        ([ChatCompletionMessageToolCall(function=Function(name="json_tool_call", arguments="{}"))], True),
        ([ChatCompletionMessageToolCall(function=Function(name="other", arguments="{}"))], False),
        ([{"function": {"name": "json_tool_call", "arguments": "{}"}}], True),
        ([{"id": "x"}], False),
    ),
)
def test_json_mode_tool_call_detection_reads_models_and_dicts_alike(tool_calls, expected: bool) -> None:
    assert should_convert_tool_call_to_json_mode(tool_calls, convert_tool_call_to_json_mode=True) is expected
