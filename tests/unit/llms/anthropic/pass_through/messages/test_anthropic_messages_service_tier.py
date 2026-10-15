from typing import Final

import pytest

from litellm.llms.anthropic.pass_through.messages.utils import AnthropicMessagesRequestUtils

REQUEST: Final[dict[str, object]] = {
    "max_tokens": 64,
    "service_tier": "standard_only",
    "messages": [{"role": "user", "content": "Hello"}],
}


@pytest.mark.parametrize("custom_llm_provider", ["anthropic", "azure_ai", "bedrock_mantle", None])
def test_messages_forwards_service_tier_on_anthropic_api_routes(custom_llm_provider: str | None) -> None:
    optional_params = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params=dict(REQUEST), model="claude-sonnet-4-6", drop_params=False, custom_llm_provider=custom_llm_provider
    )
    assert dict(optional_params).get("service_tier") == "standard_only"


@pytest.mark.parametrize("custom_llm_provider", ["vertex_ai", "bedrock"])
def test_messages_drops_service_tier_where_the_provider_rejects_it(custom_llm_provider: str) -> None:
    optional_params = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params=dict(REQUEST), model="claude-sonnet-4-6", drop_params=False, custom_llm_provider=custom_llm_provider
    )
    assert "service_tier" not in optional_params
    assert dict(optional_params).get("max_tokens") == 64
