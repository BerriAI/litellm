import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.azure_ai.azure_model_router.transformation import (
    AzureModelRouterConfig,
)
from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig


@pytest.mark.asyncio
async def test_get_openai_compatible_provider_info():
    """
    Test that Azure AI requests are formatted correctly with the proper endpoint and parameters
    for both synchronous and asynchronous calls
    """
    config = AzureAIStudioConfig()

    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model="azure_ai/gpt-4o-mini",
        api_base="https://my-base",
        api_key="my-key",
        custom_llm_provider="azure_ai",
    )

    assert custom_llm_provider == "azure"


def test_azure_ai_validate_environment():
    config = AzureAIStudioConfig()
    headers = config.validate_environment(
        headers={},
        model="azure_ai/gpt-4o-mini",
        messages=[],
        optional_params={},
        litellm_params={},
    )
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_validate_environment_with_api_key():
    """
    Test that when api_key is provided, it is set in the api-key header
    for Azure Foundry endpoints (.services.ai.azure.com).
    """
    config = AzureAIStudioConfig()
    headers = config.validate_environment(
        headers={},
        model="Kimi-K2.5",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-api-key",
        api_base="https://my-endpoint.services.ai.azure.com",
    )
    assert headers["api-key"] == "test-api-key"
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_validate_environment_with_azure_ad_token():
    """
    Test that when no api_key is provided but Azure AD credentials are available,
    the Authorization header is set with a Bearer token.

    Regression test for https://github.com/BerriAI/litellm/issues/20759
    """
    import litellm

    config = AzureAIStudioConfig()
    with patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token",
        return_value="fake-azure-ad-token",
    ), patch(
        "litellm.llms.azure.common_utils.get_secret_str",
        return_value=None,
    ), patch.object(litellm, "api_key", None), patch.object(
        litellm, "azure_key", None
    ):
        headers = config.validate_environment(
            headers={},
            model="Kimi-K2.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="https://my-endpoint.services.ai.azure.com",
        )
    assert headers.get("Authorization") == "Bearer fake-azure-ad-token"
    assert "api-key" not in headers
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_grok_stop_parameter_handling():
    """
    Test that Grok models properly handle stop parameter filtering in Azure AI Studio.
    """
    config = AzureAIStudioConfig()
    
    # Test Grok model detection
    assert config._supports_stop_reason("grok-4-fast") == False
    assert config._supports_stop_reason("grok-4") == False
    assert config._supports_stop_reason("grok-3-mini") == False
    assert config._supports_stop_reason("grok-code-fast") == False
    assert config._supports_stop_reason("gpt-4") == True
    
    # Test supported parameters for Grok models
    grok_params = config.get_supported_openai_params("grok-4-fast")
    assert "stop" not in grok_params, "Grok models should not support stop parameter"
    
    # Test supported parameters for non-Grok models
    gpt_params = config.get_supported_openai_params("gpt-4")
    assert "stop" in gpt_params, "GPT models should support stop parameter"


def test_azure_model_router_response_shows_actual_model():
    """
    Test that Azure Model Router returns the actual model used in the response,
    not the router model.
    
    According to the documentation, when using Azure Model Router, the response
    should show the actual model that handled the request (e.g., gpt-5-nano-2025-08-07)
    rather than the router model (e.g., model-router).
    
    Regression test for: Azure Model Router should show actual model in response
    """
    from httpx import Response

    from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse
    
    config = AzureModelRouterConfig()
    
    # Mock raw response from Azure that includes the actual model used
    raw_response_json = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-5-nano-2025-08-07",  # Actual model used by the router
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    
    # Create mock Response object
    mock_response = MagicMock(spec=Response)
    mock_response.json.return_value = raw_response_json
    mock_response.text = json.dumps(raw_response_json)
    mock_response.headers = {}
    
    # Create ModelResponse object
    model_response = ModelResponse()
    
    # Create mock logging object with required methods
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.post_call = MagicMock()
    logging_obj.model_call_details = {}
    
    # Call transform_response with router model
    result = config.transform_response(
        model="model-router",  # This is the router model (without prefix)
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={"model": "azure_ai/model-router"},  # Original request model
        encoding=None,
        api_key="test-key",
        json_mode=False,
    )
    
    # Verify that the response contains the actual model used, not the router model
    assert result.model == "azure_ai/gpt-5-nano-2025-08-07", (
        f"Expected model to be 'azure_ai/gpt-5-nano-2025-08-07' (actual model used), "
        f"but got '{result.model}'"
    )


def test_azure_ai_map_openai_params_renames_max_completion_tokens():
    """
    Azure AI Foundry's Model Inference endpoint only accepts `max_tokens`, so
    `max_completion_tokens` must be rewritten before the request is sent.

    Regression test for https://github.com/BerriAI/litellm/issues/26322
    """
    config = AzureAIStudioConfig()
    optional_params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 1000},
        optional_params={},
        model="mistral-large-3",
        drop_params=False,
    )
    assert optional_params.get("max_tokens") == 1000
    assert "max_completion_tokens" not in optional_params


def test_azure_ai_map_openai_params_preserves_max_tokens():
    """
    Plain `max_tokens` must still pass through unchanged.
    """
    config = AzureAIStudioConfig()
    optional_params = config.map_openai_params(
        non_default_params={"max_tokens": 500},
        optional_params={},
        model="mistral-large-3",
        drop_params=False,
    )
    assert optional_params.get("max_tokens") == 500
    assert "max_completion_tokens" not in optional_params


def test_azure_ai_map_openai_params_max_completion_tokens_takes_priority():
    """
    When both are provided, `max_completion_tokens` should win — matching
    `MistralConfig.map_openai_params` precedence.
    """
    config = AzureAIStudioConfig()
    optional_params = config.map_openai_params(
        non_default_params={"max_tokens": 500, "max_completion_tokens": 1000},
        optional_params={},
        model="mistral-large-3",
        drop_params=False,
    )
    assert optional_params.get("max_tokens") == 1000
    assert "max_completion_tokens" not in optional_params


def test_azure_ai_map_openai_params_does_not_mutate_caller_dict():
    """
    The incoming non_default_params dict belongs to the caller — the rename must
    not mutate it.
    """
    config = AzureAIStudioConfig()
    non_default_params = {"max_completion_tokens": 1000}
    config.map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="mistral-large-3",
        drop_params=False,
    )
    assert non_default_params == {"max_completion_tokens": 1000}


@pytest.mark.parametrize(
    "model,non_default_params",
    [
        # o-series models: parent dispatches to openaiOSeriesConfig which
        # actively maps max_tokens → max_completion_tokens; our post-super
        # normalization must still land on max_tokens for the Foundry endpoint.
        ("o3-mini", {"max_completion_tokens": 1000}),
        ("o3-mini", {"max_tokens": 1000}),
        # gpt-5 models: parent dispatches to openAIGPT5Config with the same
        # max_tokens → max_completion_tokens rename.
        ("gpt-5-mini", {"max_completion_tokens": 1000}),
        ("gpt-5-mini", {"max_tokens": 1000}),
    ],
)
def test_azure_ai_map_openai_params_handles_openai_family_models(
    model, non_default_params
):
    """
    Azure AI Foundry can host OpenAI-family model names (o-series, gpt-5) whose
    parent configs flip max_tokens ↔ max_completion_tokens internally. The
    override must normalize to max_tokens regardless of which direction the
    parent dispatch took.
    """
    config = AzureAIStudioConfig()
    optional_params = config.map_openai_params(
        non_default_params=dict(non_default_params),
        optional_params={},
        model=model,
        drop_params=False,
    )
    assert optional_params.get("max_tokens") == 1000
    assert "max_completion_tokens" not in optional_params


def test_azure_model_router_inherits_max_completion_tokens_rewrite():
    """
    `AzureModelRouterConfig` subclasses `AzureAIStudioConfig` and shares the same
    Model Inference endpoint, so it must also rewrite `max_completion_tokens`.
    """
    config = AzureModelRouterConfig()
    optional_params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 1000},
        optional_params={},
        model="model-router",
        drop_params=False,
    )
    assert optional_params.get("max_tokens") == 1000
    assert "max_completion_tokens" not in optional_params
