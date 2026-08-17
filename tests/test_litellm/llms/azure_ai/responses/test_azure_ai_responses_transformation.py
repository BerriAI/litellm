import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.azure_ai.common_utils import (
    is_agents_v2_model,
    parse_agent_reference,
)
from litellm.llms.azure_ai.responses.transformation import AzureAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

PROJECT_API_BASE = "https://example.services.ai.azure.com/api/projects/my-project"


class TestAgentsV2Helpers:
    def test_is_agents_v2_model_true_for_name_version(self):
        assert is_agents_v2_model("azure_ai/agents/my-agent:1") is True
        assert is_agents_v2_model("agents/my-agent:2") is True

    def test_is_agents_v2_model_false_for_v1_assistant_ids(self):
        assert is_agents_v2_model("azure_ai/agents/asst_123") is False
        assert is_agents_v2_model("agents/asst_abc") is False

    def test_is_agents_v2_model_false_for_non_agents(self):
        assert is_agents_v2_model("azure_ai/gpt-4o") is False

    def test_parse_agent_reference(self):
        assert parse_agent_reference("azure_ai/agents/my-agent:1") == ("my-agent", "1")
        assert parse_agent_reference("agents/other-agent:42") == ("other-agent", "42")


class TestAzureAIResponsesAPIConfig:
    def test_custom_llm_provider(self):
        config = AzureAIResponsesAPIConfig()
        assert config.custom_llm_provider == LlmProviders.AZURE_AI

    def test_get_complete_url(self):
        config = AzureAIResponsesAPIConfig()
        url = config.get_complete_url(
            api_base=PROJECT_API_BASE,
            litellm_params={"api_version": "2025-05-01"},
        )
        assert (
            url
            == "https://example.services.ai.azure.com/api/projects/my-project/openai/responses?api-version=2025-05-01"
        )

    def test_get_complete_url_default_api_version(self):
        config = AzureAIResponsesAPIConfig()
        url = config.get_complete_url(api_base=PROJECT_API_BASE, litellm_params={})
        assert url.endswith("/openai/responses?api-version=2025-05-01")

    def test_validate_environment_bearer_token(self):
        config = AzureAIResponsesAPIConfig()
        headers = config.validate_environment(
            headers={},
            model="azure_ai/agents/my-agent:1",
            litellm_params=GenericLiteLLMParams(api_key="test-azure-ad-token"),
        )
        assert headers["Authorization"] == "Bearer test-azure-ad-token"
        assert headers["Content-Type"] == "application/json"

    def test_transform_request_injects_agent_reference(self):
        config = AzureAIResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="azure_ai/agents/my-agent:1",
            input=[{"role": "user", "content": "Hello"}],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request["agent_reference"] == {
            "name": "my-agent",
            "version": "1",
            "type": "agent_reference",
        }

    def test_transform_request_strips_model_for_agent_reference(self):
        config = AzureAIResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="azure_ai/agents/my-agent:1",
            input="Hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert "model" not in request

    def test_transform_request_passthrough_for_non_v2_model(self):
        config = AzureAIResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="azure_ai/gpt-4o",
            input="Hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request["model"] == "azure_ai/gpt-4o"
        assert "agent_reference" not in request


class TestProviderConfigManagerAzureAIResponses:
    def test_agents_v2_model_returns_responses_config(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE_AI,
            model="azure_ai/agents/my-agent:1",
        )
        assert config is not None
        assert isinstance(config, AzureAIResponsesAPIConfig)

    def test_v1_agents_model_returns_none(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE_AI,
            model="azure_ai/agents/asst_123",
        )
        assert config is None

    def test_default_azure_ai_model_returns_none(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE_AI,
            model="azure_ai/gpt-4o",
        )
        assert config is None


class TestV1AgentsCompletionRoutingUnchanged:
    def test_v1_agents_still_detected_as_agents_route(self):
        from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo

        assert (
            AzureFoundryModelInfo.get_azure_ai_route("azure_ai/agents/asst_123") == "agents"
        )

    def test_v2_agents_not_routed_to_v1_completion(self):
        from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

        assert AzureAIAgentsConfig.is_azure_ai_agents_route("azure_ai/agents/my-agent:1") is True
        assert is_agents_v2_model("azure_ai/agents/my-agent:1") is True
