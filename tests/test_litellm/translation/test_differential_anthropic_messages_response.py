"""Differential parity for the /v1/messages reverse (response) path.

v1's oracle is ``translate_openai_response_to_anthropic`` (transformation.py)
which consumes an OpenAI ``ModelResponse``. Both sides start from the same IR
``ChatResponse`` (built by the anthropic provider parser over a recorded
Anthropic wire response): v2 reverse-serializes it directly to an Anthropic
Messages body, while the oracle goes IR -> anthropic-dialect chat body ->
``ModelResponse`` -> v1's adapter. The two Anthropic bodies must be identical.

A mutation in the v2 serializer (wrong content order, a dropped block, a
mis-mapped stop_reason, wrong usage math, a missing
``provider_specific_fields`` on tool_use) diverges from the v1 oracle.
"""

import json

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.utils import ModelResponse

from litellm.translation.inbound.anthropic_messages import parse_request
from litellm.translation.inbound.anthropic_messages.response import (
    serialize_response as v2_serialize,
)
from litellm.translation.inbound.openai_chat.response import (
    serialize_response as chat_serialize,
)
from litellm.translation.providers.anthropic.response import parse_response

from .conftest import build_real_deps

MODEL = "claude-sonnet-4-5"

_REQUEST = {
    "model": MODEL,
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [{"name": "get", "input_schema": {"type": "object"}}],
}


def _wire(content: list, stop_reason: str, usage: dict) -> dict:
    return {
        "id": "msg_resp",
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

# Each recorded Anthropic wire response, its stop_reason, and its usage. The
# corpus exercises content ordering (thinking->text->tool_use), each block
# type, the finish-reason arms, and the cache usage math.
_CASES: dict[str, dict] = {
    "text_end_turn": _wire(
        [{"type": "text", "text": "hello"}], "end_turn", _USAGE_PLAIN
    ),
    "text_max_tokens": _wire(
        [{"type": "text", "text": "truncated"}], "max_tokens", _USAGE_PLAIN
    ),
    "thinking_then_text": _wire(
        [
            {"type": "thinking", "thinking": "hmm", "signature": "sg"},
            {"type": "text", "text": "answer"},
        ],
        "end_turn",
        _USAGE_PLAIN,
    ),
    "redacted_thinking_then_text": _wire(
        [
            {"type": "redacted_thinking", "data": "ENC"},
            {"type": "text", "text": "answer"},
        ],
        "end_turn",
        _USAGE_PLAIN,
    ),
    "tool_use_only": _wire(
        [{"type": "tool_use", "id": "toolu_1", "name": "get", "input": {"q": 1}}],
        "tool_use",
        _USAGE_PLAIN,
    ),
    "text_and_tool_use": _wire(
        [
            {"type": "text", "text": "calling"},
            {"type": "tool_use", "id": "toolu_1", "name": "get", "input": {"q": 1}},
        ],
        "tool_use",
        _USAGE_PLAIN,
    ),
    "thinking_text_tool_use_with_cache": _wire(
        [
            {"type": "thinking", "thinking": "plan", "signature": "sg"},
            {"type": "text", "text": "calling"},
            {"type": "tool_use", "id": "toolu_2", "name": "get", "input": {}},
        ],
        "tool_use",
        _USAGE_CACHED,
    ),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_response_matches_v1_translate_openai_response_to_anthropic(name: str) -> None:
    deps = build_real_deps()
    request = parse_request(_REQUEST)
    assert request.is_ok()
    ir_result = parse_response(_CASES[name], request.ok)
    assert ir_result.is_ok(), ir_result.error.summary
    ir = ir_result.ok

    v2_body = v2_serialize(ir, deps)
    assert not hasattr(v2_body, "summary"), getattr(v2_body, "summary", "")

    chat_body = chat_serialize(ir, deps, "anthropic")
    assert isinstance(chat_body, dict)
    model_response = ModelResponse(**{"id": ir.id, "model": ir.model, **chat_body})
    v1_body = dict(
        LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
            response=model_response
        )
    )

    assert json.dumps(v2_body, sort_keys=True, default=str) == json.dumps(
        v1_body, sort_keys=True, default=str
    ), name
