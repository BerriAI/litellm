import json
from typing import Final

import pytest
from pydantic import JsonValue

from litellm.litellm_core_utils.tool_fields import DropPolicy
from litellm.responses.litellm_completion_transformation.count_request import ChatCountRequest, chat_count_request

_MARK: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
_MARKED_RESPONSES_REQUEST: Final[dict[str, JsonValue]] = {
    "model": "claude-opus-5-5",
    "instructions": "Be terse.",
    "input": [
        {"type": "message", "role": "user", "content": "Weather in SF?", "cache_control": _MARK},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "And in LA?", "cache_control": _MARK}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
            "cache_control": _MARK,
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny", "cache_control": _MARK},
    ],
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "cache_control": _MARK,
        }
    ],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", (False, True))
async def test_claude_count_request_leaves_out_every_cache_control_mark(drop_params: bool) -> None:
    request: Final = await chat_count_request(
        model="claude-opus-5-5",
        custom_llm_provider="anthropic",
        api_base=None,
        responses_request=_MARKED_RESPONSES_REQUEST,
        policy=DropPolicy(drop_params=drop_params, additional_drop_params=frozenset()),
    )

    assert isinstance(request, ChatCountRequest)
    counted: Final = json.dumps({"messages": request.messages, "tools": request.tools}, default=str)
    assert "cache_control" not in counted
    for kept in ("Weather in SF?", "And in LA?", "call_1", "sunny", "get_weather"):
        assert kept in counted
