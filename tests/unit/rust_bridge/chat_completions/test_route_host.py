from types import MappingProxyType
from typing import Final

from litellm.rust_bridge.chat_completions.route_host import arguments, response
from litellm.rust_bridge.chat_completions.entrypoints import LiteLLMChatCompletionsRequest
from litellm.types.utils import ModelResponse


def test_response_builds_the_public_model_response() -> None:
    built: Final = response(
        MappingProxyType(
            {
                "id": "chatcmpl-native",
                "object": "chat.completion",
                "created": 1,
                "model": "claude-sonnet-4-5",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "native"},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        )
    )

    assert isinstance(built, ModelResponse)
    assert built.id == "chatcmpl-native"
    assert built.choices[0].message.content == "native"
    assert built.usage is not None
    assert built.usage.total_tokens == 5


def test_arguments_are_the_public_kwargs_view() -> None:
    kwargs: Final = MappingProxyType({"metadata": {"user_id": "u"}})
    request: Final = LiteLLMChatCompletionsRequest(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=None,
        api_key=None,
        api_base=None,
        custom_llm_provider="anthropic",
        extra_headers=None,
        kwargs=kwargs,
    )

    assert arguments(request) is kwargs
