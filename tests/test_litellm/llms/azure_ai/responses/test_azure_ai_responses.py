"""Unit tests for the Azure AI (Azure AI Foundry) Responses API config."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.azure_ai.responses.transformation import (
    AzureAIResponsesAPIConfig,
    _is_azure_foundry_host,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_registration():
    """azure_ai resolves to AzureAIResponsesAPIConfig for the responses API."""
    cfg = ProviderConfigManager.get_provider_responses_api_config(provider=LlmProviders.AZURE_AI, model="gpt-5.6")
    assert isinstance(cfg, AzureAIResponsesAPIConfig)
    assert cfg.custom_llm_provider == LlmProviders.AZURE_AI


@pytest.mark.parametrize(
    "api_base,expected",
    [
        # Commercial
        (
            "https://res.services.ai.azure.com/api/projects/proj",
            "https://res.services.ai.azure.com/api/projects/proj/openai/v1/responses",
        ),
        (
            "https://res.services.ai.azure.com",
            "https://res.services.ai.azure.com/models/responses",
        ),
        (
            "https://res.openai.azure.com",
            "https://res.openai.azure.com/models/responses",
        ),
        # Azure Government (.azure.us)
        (
            "https://res.services.ai.azure.us/api/projects/proj",
            "https://res.services.ai.azure.us/api/projects/proj/openai/v1/responses",
        ),
        (
            "https://res.services.ai.azure.us",
            "https://res.services.ai.azure.us/models/responses",
        ),
        (
            "https://res.openai.azure.us",
            "https://res.openai.azure.us/models/responses",
        ),
        # Generic OpenAI-compatible base
        (
            "https://gateway.example.com/v1",
            "https://gateway.example.com/v1/responses",
        ),
    ],
)
def test_get_complete_url_paths(api_base, expected):
    cfg = AzureAIResponsesAPIConfig()
    url = cfg.get_complete_url(api_base=api_base, litellm_params={})
    assert url == expected


def test_get_complete_url_project_check_precedes_host_check():
    """A project URL on an Azure Foundry host must use /openai/v1/responses,
    not /models/responses (the /projects/ branch must win)."""
    cfg = AzureAIResponsesAPIConfig()
    url = cfg.get_complete_url(
        api_base="https://res.services.ai.azure.us/api/projects/proj",
        litellm_params={},
    )
    assert url.endswith("/api/projects/proj/openai/v1/responses")


def test_get_complete_url_propagates_api_version():
    cfg = AzureAIResponsesAPIConfig()
    url = cfg.get_complete_url(
        api_base="https://res.services.ai.azure.us",
        litellm_params={"api_version": "preview"},
    )
    assert "api-version=preview" in url


def test_get_complete_url_requires_api_base(monkeypatch):
    monkeypatch.delenv("AZURE_AI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None, raising=False)
    cfg = AzureAIResponsesAPIConfig()
    with pytest.raises(ValueError, match="api_base is required"):
        cfg.get_complete_url(api_base=None, litellm_params={})


@pytest.mark.parametrize(
    "api_base,expect_api_key_header",
    [
        ("https://res.services.ai.azure.com", True),
        ("https://res.openai.azure.com", True),
        ("https://res.cognitiveservices.azure.com", True),
        ("https://res.services.ai.azure.us", True),
        ("https://res.openai.azure.us", True),
        ("https://res.cognitiveservices.azure.us", True),
        ("https://gateway.example.com/v1", False),
    ],
)
def test_validate_environment_auth_header_selection(api_base, expect_api_key_header):
    cfg = AzureAIResponsesAPIConfig()
    headers = cfg.validate_environment(
        headers={},
        model="gpt-5.6",
        litellm_params=GenericLiteLLMParams(api_key="secret", api_base=api_base),
    )
    if expect_api_key_header:
        assert headers.get("api-key") == "secret"
        assert "Authorization" not in headers
    else:
        assert headers.get("Authorization") == "Bearer secret"
        assert "api-key" not in headers
    assert headers["Content-Type"] == "application/json"


@pytest.mark.parametrize(
    "api_base,expected",
    [
        ("https://res.services.ai.azure.com", True),
        ("https://res.openai.azure.us", True),
        ("https://res.cognitiveservices.azure.us", True),
        ("https://gateway.example.com/v1", False),
        (None, False),
        ("", False),
    ],
)
def test_is_azure_foundry_host(api_base, expected):
    assert _is_azure_foundry_host(api_base) is expected


def test_projects_check_ignores_query_string():
    """A ``/projects/`` substring in the query string must not be treated as a
    project endpoint (path-only match)."""
    cfg = AzureAIResponsesAPIConfig()
    url = cfg.get_complete_url(
        api_base="https://res.services.ai.azure.com?foo=/projects/x",
        litellm_params={},
    )
    # Host-based route, not the project route.
    assert "/models/responses" in url
    assert "/openai/v1/responses" not in url


def test_transform_request_flattens_function_tools():
    """Inherited from AzureOpenAIResponsesAPIConfig: Azure's Responses API needs
    function tool params at the top level, not nested under 'function'. This is
    essential for the reasoning + tools use case."""
    cfg = AzureAIResponsesAPIConfig()
    params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
    }
    out = cfg.transform_responses_api_request(
        model="gpt-5.6",
        input="weather in Paris?",
        response_api_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    tool = out["tools"][0]
    assert tool["name"] == "get_weather"  # flattened to top level
    assert "function" not in tool
