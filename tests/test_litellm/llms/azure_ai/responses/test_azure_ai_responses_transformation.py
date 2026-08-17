import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

import litellm
from litellm.llms.azure_ai.common_utils import (
    is_agents_v2_model,
    parse_agent_reference,
)
from litellm.llms.azure_ai.responses.transformation import AzureAIResponsesAPIConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

PROJECT_API_BASE = "https://example.services.ai.azure.com/api/projects/my-project"

AGENT_RESPONSE = {
    "id": "resp_agent123",
    "object": "response",
    "created_at": 1234567890,
    "status": "completed",
    "model": "my-agent",
    "output": [
        {
            "type": "message",
            "id": "msg_agent123",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi there", "annotations": []}],
        }
    ],
}


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

    def test_get_complete_url_falls_back_to_env_api_base(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_BASE", PROJECT_API_BASE)
        config = AzureAIResponsesAPIConfig()
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url.startswith(PROJECT_API_BASE)

    def test_get_complete_url_requires_api_base(self, monkeypatch):
        monkeypatch.delenv("AZURE_AI_API_BASE", raising=False)
        monkeypatch.setattr("litellm.api_base", None)
        config = AzureAIResponsesAPIConfig()
        with pytest.raises(ValueError, match="api_base is required"):
            config.get_complete_url(api_base=None, litellm_params={})

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


class TestAgentReferenceCannotBeOverridden:
    def test_extra_body_cannot_replace_model_derived_agent(self):
        config = AzureAIResponsesAPIConfig()
        data = config.transform_responses_api_request(
            model="azure_ai/agents/my-agent:1",
            input="Hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        merged = config.merge_extra_body(
            data=data,
            extra_body={
                "agent_reference": {
                    "name": "someone-elses-agent",
                    "version": "9",
                    "type": "agent_reference",
                }
            },
        )

        assert merged["agent_reference"] == {
            "name": "my-agent",
            "version": "1",
            "type": "agent_reference",
        }

    def test_extra_body_still_merges_other_keys(self):
        config = AzureAIResponsesAPIConfig()
        data = config.transform_responses_api_request(
            model="azure_ai/agents/my-agent:1",
            input="Hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        merged = config.merge_extra_body(data=data, extra_body={"custom_flag": True})

        assert merged["custom_flag"] is True
        assert merged["agent_reference"]["name"] == "my-agent"

    def test_extra_body_wins_when_no_agent_reference(self):
        config = AzureAIResponsesAPIConfig()
        merged = config.merge_extra_body(
            data={"model": "azure_ai/gpt-4o"},
            extra_body={"model": "azure_ai/gpt-4o-mini"},
        )
        assert merged["model"] == "azure_ai/gpt-4o-mini"


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

    def test_missing_model_returns_none(self):
        from litellm.llms.azure_ai.responses.transformation import (
            get_azure_ai_responses_api_config,
        )

        assert get_azure_ai_responses_api_config(None) is None

    def test_default_azure_ai_model_returns_none(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE_AI,
            model="azure_ai/gpt-4o",
        )
        assert config is None


class TestOutboundRequest:
    def _call_responses(self, **overrides):
        requests: list[httpx.Request] = []

        def capture(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=AGENT_RESPONSE)

        client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(capture)))
        litellm.responses(
            model="azure_ai/agents/my-agent:1",
            input="Hello",
            api_base=PROJECT_API_BASE,
            api_key="test-azure-ad-token",
            client=client,
            **overrides,
        )
        return requests[0]

    def test_request_targets_project_responses_route_with_agent_reference(self):
        request = self._call_responses()
        body = json.loads(request.content)

        assert str(request.url) == (
            "https://example.services.ai.azure.com/api/projects/my-project"
            "/openai/responses?api-version=2025-05-01"
        )
        assert request.headers["authorization"] == "Bearer test-azure-ad-token"
        assert body["agent_reference"] == {
            "name": "my-agent",
            "version": "1",
            "type": "agent_reference",
        }
        assert "model" not in body

    def test_hostile_extra_body_cannot_switch_agent(self):
        request = self._call_responses(
            extra_body={
                "agent_reference": {
                    "name": "someone-elses-agent",
                    "version": "9",
                    "type": "agent_reference",
                }
            }
        )
        body = json.loads(request.content)

        assert body["agent_reference"]["name"] == "my-agent"
        assert body["agent_reference"]["version"] == "1"


class TestV1AgentsCompletionRoutingUnchanged:
    def test_v1_agents_still_detected_as_agents_route(self):
        from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo

        assert (
            AzureFoundryModelInfo.get_azure_ai_route("azure_ai/agents/asst_123") == "agents"
        )

    def test_v2_agents_completion_raises_pointing_at_responses_api(self):
        from litellm.llms.azure_ai.agents.transformation import (
            AzureAIAgentsConfig,
            AzureAIAgentsError,
        )
        from litellm.utils import ModelResponse

        with pytest.raises(AzureAIAgentsError, match="Responses API"):
            AzureAIAgentsConfig.completion(
                model="azure_ai/agents/my-agent:1",
                messages=[{"role": "user", "content": "Hello"}],
                api_base=PROJECT_API_BASE,
                api_key="test-azure-ad-token",
                model_response=ModelResponse(),
                logging_obj=None,
                optional_params={},
                litellm_params={},
                timeout=60.0,
                acompletion=False,
            )
