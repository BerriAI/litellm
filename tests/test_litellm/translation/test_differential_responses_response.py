"""Differential parity for the /v1/responses reverse (response) path.

v1's oracle is ``transform_chat_completion_response_to_responses_api_response``
which consumes an OpenAI ``ModelResponse`` and returns a ``ResponsesAPIResponse``.
Both sides start from the same IR ``ChatResponse`` (built by the anthropic
provider parser over a recorded Anthropic wire response): v2 reverse-serializes
it directly to a Responses body, while the oracle goes IR -> anthropic-dialect
chat body -> ``ModelResponse`` -> v1's reverse. The two Responses bodies must be
identical under canonical JSON.

Two faithfulness adjustments to the oracle bridge, neither masking a v2 bug:
- ``created_at`` is ambient envelope (the seam stamps it; the IR carries no
  timestamp), normalized on both sides;
- the usage on the bridged ``ModelResponse`` is rebuilt from the wire token
  counts (input/output/cache/reasoning) rather than the anthropic chat
  dialect's, because that dialect ESTIMATES reasoning tokens with the injected
  token counter (a chat-side concern the openai_chat/anthropic gates already
  pin); the Responses-reverse usage MAPPING (cache fold into input_tokens, the
  cached/text/reasoning detail shape) is what this gate isolates, so both sides
  read the same provider-style usage.

A mutation in the v2 serializer (wrong output-item order, a dropped reasoning/
message/function_call item, a mis-mapped status, wrong usage fold, a missing
``namespace``/``phase`` key) diverges from the v1 oracle.
"""

import json

import pytest
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig as V1,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)

from litellm.translation.inbound.openai_chat.response import (
    serialize_response as chat_serialize,
)
from litellm.translation.inbound.responses import parse_request
from litellm.translation.inbound.responses.response import (
    serialize_response as v2_serialize,
)
from litellm.translation.providers.anthropic.response import parse_response

from .conftest import build_real_deps

MODEL = "gpt-4o"

_REQUEST = {
    "model": MODEL,
    "input": "hi",
    "tools": [{"type": "function", "name": "get", "parameters": {"type": "object"}}],
}


def _wire(content: list, stop_reason: str, usage: dict) -> dict:
    return {
        "id": "chatcmpl-resp",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


_USAGE_PLAIN = {"input_tokens": 100, "output_tokens": 20}
_USAGE_CACHED = {
    "input_tokens": 100,
    "output_tokens": 20,
    "cache_creation_input_tokens": 5,
    "cache_read_input_tokens": 10,
}

# Each recorded Anthropic wire response, its stop_reason, its usage, and the
# wire reasoning-token count the provider would have reported (anthropic does
# not, so 0). The corpus exercises output-item ordering
# (reasoning->message->function_call), each block type, the status arms, and the
# cache usage fold.
_CASES: dict[str, dict] = {
    "text_stop": {
        "wire": _wire([{"type": "text", "text": "hello"}], "end_turn", _USAGE_PLAIN),
        "reasoning_tokens": 0,
    },
    "text_max_tokens_incomplete": {
        "wire": _wire(
            [{"type": "text", "text": "truncated"}], "max_tokens", _USAGE_PLAIN
        ),
        "reasoning_tokens": 0,
    },
    "thinking_then_text": {
        "wire": _wire(
            [
                {"type": "thinking", "thinking": "plan", "signature": "sg"},
                {"type": "text", "text": "answer"},
            ],
            "end_turn",
            _USAGE_PLAIN,
        ),
        "reasoning_tokens": 4,
    },
    "tool_use_only": {
        "wire": _wire(
            [{"type": "tool_use", "id": "toolu_1", "name": "get", "input": {"q": 1}}],
            "tool_use",
            _USAGE_PLAIN,
        ),
        "reasoning_tokens": 0,
    },
    "text_and_tool_use": {
        "wire": _wire(
            [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "toolu_1", "name": "get", "input": {"q": 1}},
            ],
            "tool_use",
            _USAGE_PLAIN,
        ),
        "reasoning_tokens": 0,
    },
    "thinking_text_tool_use_cached": {
        "wire": _wire(
            [
                {"type": "thinking", "thinking": "plan", "signature": "sg"},
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "toolu_2", "name": "get", "input": {}},
            ],
            "tool_use",
            _USAGE_CACHED,
        ),
        "reasoning_tokens": 6,
    },
    # Anthropic ``refusal`` -> IR ``content_filter``, which the responses reverse
    # status map has no explicit arm for and folds to its default ``incomplete``
    # (v1's same default). Pins the content_filter status arm the corpus
    # otherwise omitted (verifier-inbound GAP-2).
    "refusal_content_filter_incomplete": {
        "wire": _wire(
            [{"type": "text", "text": "i cannot help with that"}],
            "refusal",
            _USAGE_PLAIN,
        ),
        "reasoning_tokens": 0,
    },
}


def _bridge_usage(ir_usage, reasoning_tokens: int) -> Usage:
    """A provider-style Usage carrying the wire token counts the responses
    reverse maps (both sides read this same object)."""
    prompt = (
        ir_usage.input_tokens
        + ir_usage.cache_creation_input_tokens
        + ir_usage.cache_read_input_tokens
    )
    usage = Usage(
        prompt_tokens=prompt,
        completion_tokens=ir_usage.output_tokens,
        total_tokens=prompt + ir_usage.output_tokens,
    )
    usage.prompt_tokens_details = PromptTokensDetailsWrapper(
        cached_tokens=ir_usage.cache_read_input_tokens,
        text_tokens=ir_usage.input_tokens,
    )
    text_tokens = (
        ir_usage.output_tokens - reasoning_tokens
        if reasoning_tokens > 0
        else ir_usage.output_tokens
    )
    usage.completion_tokens_details = CompletionTokensDetailsWrapper(
        reasoning_tokens=reasoning_tokens, text_tokens=text_tokens
    )
    return usage


@pytest.mark.parametrize("name", sorted(_CASES))
def test_response_matches_v1_responses_reverse(name: str) -> None:
    case = _CASES[name]
    deps = build_real_deps()
    request = parse_request(_REQUEST)
    assert request.is_ok()
    ir_result = parse_response(case["wire"], request.ok)
    assert ir_result.is_ok(), ir_result.error.summary
    ir = ir_result.ok

    # Thread the wire reasoning tokens into the IR so both the v2 serializer and
    # the bridged v1 ModelResponse map the same usage (anthropic reports none).
    from dataclasses import replace
    from expression import Some

    threaded = (
        replace(
            ir, usage=replace(ir.usage, reasoning_tokens=Some(case["reasoning_tokens"]))
        )
        if case["reasoning_tokens"] > 0
        else ir
    )

    v2_body = dict(v2_serialize(threaded, deps))
    assert "summary" not in v2_body, v2_body

    chat_body = chat_serialize(ir, deps, "anthropic")
    assert isinstance(chat_body, dict)
    model_response = ModelResponse(
        **{"id": ir.id, "created": 0, "model": ir.model, **chat_body}
    )
    model_response.usage = _bridge_usage(ir.usage, case["reasoning_tokens"])
    v1_body = V1.transform_chat_completion_response_to_responses_api_response(
        request_input="hi",
        responses_api_request={},
        chat_completion_response=model_response,
    ).model_dump()

    v2_body["created_at"] = model_response.created

    assert json.dumps(v2_body, sort_keys=True, default=str) == json.dumps(
        v1_body, sort_keys=True, default=str
    ), name


def test_function_call_arguments_ride_verbatim_not_reserialized() -> None:
    """The IR ToolUse.arguments_raw carries the wire bytes verbatim; the reverse
    must emit them unchanged. The corpus tool args ``{"q":1}`` happen to equal
    ``json.dumps`` of their parsed value, so a reserialize regression hides
    there. A compact-spaced ``{"a":1,"b":2}`` does NOT survive json.dumps
    (which inserts ``", "``/``": "``), so this pins the raw path. v1 copies the
    ModelResponse tool_call arguments string verbatim (transformation.py:1515),
    so the verbatim emission is also what v1 does (verifier-inbound GAP-4)."""
    from dataclasses import replace

    from expression import Nothing, Some
    from expression.collections import Block as IRBlock

    from litellm.translation.ir import ContentBlock, JsonBlob, ToolUse

    deps = build_real_deps()
    base = parse_response(
        _wire(
            [{"type": "tool_use", "id": "toolu_x", "name": "get", "input": {"a": 1}}],
            "tool_use",
            _USAGE_PLAIN,
        ),
        parse_request(_REQUEST).ok,
    ).ok
    compact = '{"a":1,"b":2}'
    raw_block = ContentBlock.of_tool_use(
        ToolUse(
            id="toolu_x",
            name="get",
            arguments=JsonBlob(value={"a": 1, "b": 2}),
            cache=Nothing,
            arguments_raw=Some(compact),
        )
    )
    ir = replace(base, content=IRBlock.of_seq([raw_block]))

    body = dict(v2_serialize(ir, deps))
    call = next(item for item in body["output"] if item["type"] == "function_call")
    assert call["arguments"] == compact
    assert call["arguments"] != json.dumps({"a": 1, "b": 2})


def test_reasoning_item_precedes_message_and_function_call() -> None:
    """Pin v1's output-item ORDER (reasoning -> message -> function_call): a
    mutation that reorders the items (or drops the reasoning item) would still
    pass a set comparison but breaks this sequence assertion."""
    deps = build_real_deps()
    request = parse_request(_REQUEST).ok
    wire = _wire(
        [
            {"type": "thinking", "thinking": "why", "signature": "s"},
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "toolu_9", "name": "get", "input": {}},
        ],
        "tool_use",
        _USAGE_PLAIN,
    )
    ir = parse_response(wire, request).ok
    body = dict(v2_serialize(ir, deps))
    kinds = [item["type"] for item in body["output"]]
    assert kinds == ["reasoning", "message", "function_call"]
