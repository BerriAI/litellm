from typing import List, cast

from litellm.llms.groq.chat.transformation import GroqChatConfig
from litellm.types.llms.openai import AllMessageValues


def test_transform_messages_strips_assistant_provider_specific_fields() -> None:
    """Groq payloads should not include LiteLLM internal assistant metadata."""
    config = GroqChatConfig()
    messages = cast(
        List[AllMessageValues],
        [
            {
                "role": "assistant",
                "content": "Tool metadata",
                "provider_specific_fields": {
                    "mcp_list_tools": [{"name": "weather"}],
                    "mcp_tool_calls": [{"id": "call_123"}],
                },
            }
        ],
    )

    transformed = cast(
        List[AllMessageValues],
        config._transform_messages(messages=messages, model="qwen/qwen3-32b"),
    )

    assert transformed[0]["role"] == "assistant"
    assert transformed[0].get("content") == "Tool metadata"
    assert "provider_specific_fields" not in transformed[0]
