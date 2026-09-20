from types import MappingProxyType
from typing import Final

from litellm.rust_bridge.messages.route_host import arguments, response
from litellm.rust_bridge.messages.entrypoints import LiteLLMMessagesRequest


def test_response_is_a_detached_public_messages_dict() -> None:
    native: Final = MappingProxyType(
        {
            "id": "msg_native",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "native"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    )

    built: Final = response(native)

    assert built == dict(native)
    assert isinstance(built, dict)
    built["_hidden_params"] = {"annotated": True}
    assert "_hidden_params" not in native


def test_arguments_are_the_public_kwargs_view() -> None:
    kwargs: Final = MappingProxyType({"litellm_metadata": {"user_id": "u"}})
    request: Final = LiteLLMMessagesRequest(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16,
        stream=None,
        api_key=None,
        api_base=None,
        custom_llm_provider="anthropic",
        kwargs=kwargs,
    )

    assert arguments(request) is kwargs
