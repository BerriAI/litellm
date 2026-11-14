from typing import Final

import pytest

from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig
from litellm.llms.anthropic.messages.native.utils import (
    AnthropicMessagesRequestUtils,
    anthropic_messages_optional_param_keys,
)

REQUEST: Final[dict[str, object]] = {
    "max_tokens": 64,
    "diagnostics": {"previous_message_id": "msg_abc"},
    "messages": [{"role": "user", "content": "Hello"}],
}


def test_anthropic_messages_optional_param_keys_contains_diagnostics() -> None:
    keys = anthropic_messages_optional_param_keys()
    assert "diagnostics" in keys


def test_get_supported_anthropic_messages_params_contains_diagnostics() -> None:
    config = AnthropicMessagesConfig()
    supported = config.get_supported_anthropic_messages_params("claude-sonnet-4-6")
    assert "diagnostics" in supported


@pytest.mark.parametrize("custom_llm_provider", ["anthropic", "azure_ai", None])
def test_messages_forwards_diagnostics_on_anthropic_api_routes(custom_llm_provider: str | None) -> None:
    optional_params = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params=dict(REQUEST), model="claude-sonnet-4-6", drop_params=False, custom_llm_provider=custom_llm_provider
    )
    assert dict(optional_params).get("diagnostics") == {"previous_message_id": "msg_abc"}


@pytest.mark.parametrize("custom_llm_provider", ["vertex_ai", "bedrock"])
def test_messages_drops_diagnostics_where_the_provider_rejects_it(custom_llm_provider: str) -> None:
    optional_params = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params=dict(REQUEST), model="claude-sonnet-4-6", drop_params=False, custom_llm_provider=custom_llm_provider
    )
    assert "diagnostics" not in optional_params
    assert dict(optional_params).get("max_tokens") == 64
