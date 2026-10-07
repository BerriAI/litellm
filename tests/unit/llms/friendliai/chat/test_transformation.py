from typing import Final

from litellm.llms.friendliai.chat.transformation import FriendliaiChatConfig
from litellm.types.llms.openai import AllMessageValues


def _replayed_thinking_turn() -> list[AllMessageValues]:
    return [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "thinking_blocks": [{"type": "thinking", "thinking": "greet briefly", "signature": ""}],
            "reasoning_content": "greet briefly",
            "provider_specific_fields": {},
        },
        {"role": "user", "content": "again"},
    ]


async def test_transform_request_drops_internal_fields_friendliai_rejects():
    messages: Final = _replayed_thinking_turn()
    config: Final = FriendliaiChatConfig()

    sync_request: Final = config.transform_request(
        model="zai-org/GLM-5.3", messages=messages, optional_params={}, litellm_params={}, headers={}
    )
    async_request: Final = await config.async_transform_request(
        model="zai-org/GLM-5.3", messages=messages, optional_params={}, litellm_params={}, headers={}
    )

    expected_assistant: Final = {"role": "assistant", "content": "hello", "reasoning_content": "greet briefly"}
    assert sync_request["messages"] == async_request["messages"] == [messages[0], expected_assistant, messages[2]]
    assert messages == _replayed_thinking_turn()
