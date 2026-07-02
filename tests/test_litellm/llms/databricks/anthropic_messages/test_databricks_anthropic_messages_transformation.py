import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.databricks.anthropic_messages.transformation import (
    DatabricksAnthropicMessagesConfig,
)
from litellm.utils import ProviderConfigManager

HOST = "https://my-ws.cloud.databricks.com"


@pytest.fixture(autouse=True)
def _clear_config_cache():
    ProviderConfigManager._get_provider_anthropic_messages_config_cached.cache_clear()
    yield
    ProviderConfigManager._get_provider_anthropic_messages_config_cached.cache_clear()


class TestProviderRegistration:
    def test_databricks_claude_uses_native_anthropic_config(self):
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="databricks/databricks-claude-3-7-sonnet",
            provider=litellm.LlmProviders.DATABRICKS,
        )
        assert isinstance(config, DatabricksAnthropicMessagesConfig)
        assert config.custom_llm_provider == "databricks"

    def test_databricks_non_claude_returns_none(self):
        # Non-claude databricks models have no native Anthropic surface -> None,
        # so the handler falls back to the chat/completions emulation path.
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="databricks/databricks-llama-4-maverick",
            provider=litellm.LlmProviders.DATABRICKS,
        )
        assert config is None

    def test_anthropic_provider_unaffected(self):
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="claude-3-7-sonnet",
            provider=litellm.LlmProviders.ANTHROPIC,
        )
        assert isinstance(config, AnthropicMessagesConfig)
        assert not isinstance(config, DatabricksAnthropicMessagesConfig)


class TestGetCompleteUrl:
    def _url(self, api_base):
        return DatabricksAnthropicMessagesConfig().get_complete_url(
            api_base=api_base,
            api_key="dapi-test",
            model="databricks/databricks-claude-3-7-sonnet",
            optional_params={},
            litellm_params={},
            stream=False,
        )

    def test_bare_host_builds_gateway_anthropic_path(self):
        assert self._url(HOST) == f"{HOST}/ai-gateway/anthropic/v1/messages"

    def test_serving_endpoints_base_rewritten_to_gateway(self):
        # Native Anthropic Messages exists only on the gateway, so the host is
        # recovered and the gateway anthropic path is always used.
        assert (
            self._url(f"{HOST}/serving-endpoints")
            == f"{HOST}/ai-gateway/anthropic/v1/messages"
        )

    def test_gateway_base_builds_path(self):
        assert self._url(f"{HOST}/ai-gateway") == (
            f"{HOST}/ai-gateway/anthropic/v1/messages"
        )

    def test_full_messages_url_is_idempotent(self):
        full = f"{HOST}/ai-gateway/anthropic/v1/messages"
        assert self._url(full) == full


class TestValidateEnvironment:
    def test_pat_sets_bearer_auth_and_anthropic_version(self):
        config = DatabricksAnthropicMessagesConfig()
        headers, api_base = config.validate_anthropic_messages_environment(
            headers={},
            model="databricks/databricks-claude-3-7-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="dapi-secret-token",
            api_base=HOST,
        )
        assert headers["Authorization"] == "Bearer dapi-secret-token"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["content-type"] == "application/json"
        assert "User-Agent" in headers
        assert api_base == HOST
