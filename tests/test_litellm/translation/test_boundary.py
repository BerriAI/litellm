"""The typed boundary: freeze/thaw round-trips and failures-as-values."""

import json

from expression.collections import Block, Map

from litellm.translation import translate_chat_request
from litellm.translation.boundary import freeze, thaw

MODEL = "claude-3-5-sonnet-20241022"


def test_freeze_thaw_round_trips_nested_json() -> None:
    original = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "tags": [1, 2, {"k": "v"}]},
        "required": ["city"],
        "flag": True,
        "nothing": None,
    }
    frozen = freeze(original)
    assert isinstance(frozen, Map)
    assert isinstance(frozen["properties"], Map)
    assert isinstance(frozen["required"], Block)
    assert json.dumps(thaw(frozen), sort_keys=True) == json.dumps(
        original, sort_keys=True
    )


def test_missing_model_and_messages_lists_every_failure() -> None:
    result = translate_chat_request({}, "anthropic")
    assert result.is_error()
    summary = result.error.summary
    assert "model" in summary
    assert "messages" in summary


def test_invalid_tool_call_arguments_is_an_error_not_an_exception() -> None:
    request = {
        "model": MODEL,
        "max_tokens": 10,
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{not json"},
                    }
                ],
            }
        ],
    }
    result = translate_chat_request(request, "anthropic")
    assert result.is_error()
    assert "JSON" in result.error.summary


def test_unsupported_provider_is_an_error_value() -> None:
    result = translate_chat_request(
        {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        "bedrock_converse",
    )
    assert result.is_error()
    assert "bedrock_converse" in result.error.summary
