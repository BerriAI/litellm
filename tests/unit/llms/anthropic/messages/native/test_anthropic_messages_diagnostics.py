from typing import Final

import pytest

from litellm.llms.anthropic.messages.native.utils import AnthropicMessagesRequestUtils

REQUEST: Final[dict[str, object]] = {
    "max_tokens": 64,
    "diagnostics": {"previous_message_id": "msg_abc"},
    "messages": [{"role": "user", "content": "Hello"}],
}


@pytest.mark.parametrize("custom_llm_provider", ["anthropic", None])
def test_messages_forwards_diagnostics_on_the_anthropic_route(custom_llm_provider: str | None) -> None:
    optional_params = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params=dict(REQUEST), model="claude-sonnet-4-6", drop_params=False, custom_llm_provider=custom_llm_provider
    )
    assert dict(optional_params).get("diagnostics") == {"previous_message_id": "msg_abc"}


@pytest.mark.parametrize("custom_llm_provider", ["vertex_ai", "bedrock", "bedrock_mantle", "azure_ai"])
def test_messages_does_not_forward_diagnostics_off_the_anthropic_route(custom_llm_provider: str) -> None:
    optional_params = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params=dict(REQUEST), model="claude-sonnet-4-6", drop_params=False, custom_llm_provider=custom_llm_provider
    )
    assert "diagnostics" not in optional_params
    assert dict(optional_params).get("max_tokens") == 64
