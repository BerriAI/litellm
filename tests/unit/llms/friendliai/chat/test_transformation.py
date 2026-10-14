from typing import Final

from litellm.llms.friendliai.chat.transformation import FriendliaiChatConfig
from litellm.types.llms.openai import AllMessageValues

_TOOL_CALLS: Final = [
    {"id": "toolu_01", "type": "function", "function": {"name": "Bash", "arguments": '{"command": "echo hi"}'}}
]


def _replayed_thinking_tool_turn() -> list[AllMessageValues]:
    return [
        {"role": "user", "content": "run echo hi"},
        {
            "role": "assistant",
            "thinking_blocks": [{"type": "thinking", "thinking": "run the Bash tool", "signature": ""}],
            "tool_calls": _TOOL_CALLS,
            "reasoning_content": "run the Bash tool",
            "provider_specific_fields": {},
        },
        {"role": "tool", "tool_call_id": "toolu_01", "content": "hi"},
    ]


async def test_transform_request_drops_internal_fields_friendliai_rejects():
    messages: Final = _replayed_thinking_tool_turn()
    config: Final = FriendliaiChatConfig()

    sync_request: Final = config.transform_request(
        model="zai-org/GLM-5.3", messages=messages, optional_params={}, litellm_params={}, headers={}
    )
    async_request: Final = await config.async_transform_request(
        model="zai-org/GLM-5.3", messages=messages, optional_params={}, litellm_params={}, headers={}
    )

    expected_assistant: Final = {
        "role": "assistant",
        "tool_calls": _TOOL_CALLS,
        "reasoning_content": "run the Bash tool",
    }
    assert sync_request["messages"] == async_request["messages"] == [messages[0], expected_assistant, messages[2]]
    assert messages == _replayed_thinking_tool_turn()
