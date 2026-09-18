"""
Tests for hosted_vllm responses API support.

Regression test for: https://github.com/BerriAI/litellm/issues
Bug: client.responses.create() raised TypeError: 'NoneType' object is not a mapping
when extra_body=None was passed through the responses→completion pipeline for
hosted_vllm (and any OpenAI-compatible provider using add_provider_specific_params_to_optional_params).
"""

from unittest.mock import MagicMock

import pytest

from litellm.llms.hosted_vllm.responses.transformation import (
    HostedVLLMResponsesAPIConfig,
)
from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.main import _resolve_responses_api_provider_config
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_hosted_vllm_responses_create_with_explicit_none_extra_body():
    """
    Directly verify the fix in add_provider_specific_params_to_optional_params:
    extra_body=None must not crash when building optional_params.
    """
    from litellm.utils import get_optional_params

    # This should not raise TypeError: 'NoneType' object is not a mapping
    optional_params = get_optional_params(
        model="Qwen/Qwen3-8B",
        custom_llm_provider="hosted_vllm",
        extra_body=None,
    )

    # extra_body=None should be normalized to an empty dict (or absent)
    assert (
        optional_params.get("extra_body") is not None
        or "extra_body" not in optional_params
    )


def test_hosted_vllm_provider_defaults_to_chat_completions_bridge():
    """Hosted vLLM backends vary in Responses API fidelity, so default to the bridge.

    A deployment may explicitly opt into native /v1/responses via model_info.
    """
    config = ProviderConfigManager.get_provider_responses_api_config(
        model="hosted_vllm/Qwen/Qwen3-8B",
        provider=LlmProviders.HOSTED_VLLM,
    )

    assert config is None


def test_hosted_vllm_provider_supports_deployment_opt_in():
    config = _resolve_responses_api_provider_config(
        "Qwen/Qwen3-8B",
        "hosted_vllm",
        {"supported_endpoints": ["/v1/responses"]},
    )

    assert isinstance(config, OpenAILikeResponsesConfig)


@pytest.mark.parametrize("tool_type", ["tool_search", "local_shell"])
def test_hosted_vllm_bridge_drops_unsupported_server_tools(tool_type):
    """Codex sends these server-side tools to every Responses backend.

    Hosted vLLM's chat completions schema only accepts function tools, so
    passing them through turns a valid agent turn into a 400 request.
    """
    tools, web_search_options = (
        LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
                {"type": tool_type},
            ]
        )
    )

    assert [tool["function"]["name"] for tool in tools] == ["get_weather"]
    assert web_search_options is None


def test_hosted_vllm_responses_api_url():
    """Test get_complete_url() constructs the correct URL."""
    config = HostedVLLMResponsesAPIConfig()

    # api_base without /v1
    url = config.get_complete_url(
        api_base="http://localhost:8000",
        litellm_params={},
    )
    assert url == "http://localhost:8000/v1/responses"

    # api_base with /v1
    url_with_v1 = config.get_complete_url(
        api_base="http://localhost:8000/v1",
        litellm_params={},
    )
    assert url_with_v1 == "http://localhost:8000/v1/responses"

    # api_base with trailing slash
    url_with_slash = config.get_complete_url(
        api_base="http://localhost:8000/v1/",
        litellm_params={},
    )
    assert url_with_slash == "http://localhost:8000/v1/responses"


def test_hosted_vllm_responses_api_url_requires_api_base():
    """Test get_complete_url() raises ValueError when api_base is not set."""
    config = HostedVLLMResponsesAPIConfig()

    with pytest.raises(ValueError, match="api_base not set"):
        config.get_complete_url(
            api_base=None,
            litellm_params={},
        )


def test_hosted_vllm_validate_environment_default_api_key():
    """Test validate_environment() defaults to 'fake-api-key' when no key is provided."""
    config = HostedVLLMResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="Qwen/Qwen3-8B",
        litellm_params=GenericLiteLLMParams(),
    )

    assert headers.get("Authorization") == "Bearer fake-api-key"


def test_hosted_vllm_validate_environment_custom_api_key():
    """Test validate_environment() uses the provided api_key."""
    config = HostedVLLMResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="Qwen/Qwen3-8B",
        litellm_params=GenericLiteLLMParams(api_key="my-custom-key"),
    )

    assert headers.get("Authorization") == "Bearer my-custom-key"
