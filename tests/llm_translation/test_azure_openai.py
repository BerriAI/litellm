import os


import pytest
from base_embedding_unit_tests import BaseLLMEmbeddingTest


from unittest.mock import patch
import litellm
from litellm import completion


class TestAzureEmbedding(BaseLLMEmbeddingTest):
    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "azure/text-embedding-ada-002",
            "api_key": os.getenv("AZURE_AI_API_KEY"),
            "api_base": os.getenv("AZURE_AI_API_BASE"),
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.AZURE


@pytest.mark.parametrize("max_retries", [0, 4])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("sync_mode", [True, False])
@patch("litellm.llms.azure.common_utils.select_azure_base_url_or_endpoint")
@pytest.mark.asyncio
async def test_azure_instruct(
    mock_select_azure_base_url_or_endpoint, max_retries, stream, sync_mode
):
    import litellm
    from litellm import completion, acompletion

    # Clear the LLM clients cache to ensure select_azure_base_url_or_endpoint is called
    litellm.in_memory_llm_clients_cache.flush_cache()

    args = {
        "model": "azure_text/instruct-model",
        "messages": [
            {"role": "user", "content": "What is the weather like in Boston?"}
        ],
        "max_tokens": 10,
        "max_retries": max_retries,
    }

    try:
        if sync_mode:
            completion(**args)
        else:
            await acompletion(**args)
    except Exception:
        pass

    mock_select_azure_base_url_or_endpoint.assert_called_once()
    assert (
        mock_select_azure_base_url_or_endpoint.call_args.kwargs["azure_client_params"][
            "max_retries"
        ]
        == max_retries
    )


@pytest.mark.parametrize("max_retries", [0, 4])
@pytest.mark.parametrize("sync_mode", [True, False])
@patch("litellm.llms.azure.common_utils.select_azure_base_url_or_endpoint")
@pytest.mark.asyncio
async def test_azure_embedding_max_retries_0(
    mock_select_azure_base_url_or_endpoint, max_retries, sync_mode
):
    import litellm
    from litellm import aembedding, embedding

    # Clear the LLM clients cache to ensure select_azure_base_url_or_endpoint is called
    litellm.in_memory_llm_clients_cache.flush_cache()

    args = {
        "model": "azure/text-embedding-ada-002",
        "input": "Hello world",
        "max_retries": max_retries,
    }

    try:
        if sync_mode:
            embedding(**args)
        else:
            await aembedding(**args)
    except Exception as e:
        print(e)

    mock_select_azure_base_url_or_endpoint.assert_called_once()
    print(
        "mock_select_azure_base_url_or_endpoint.call_args.kwargs",
        mock_select_azure_base_url_or_endpoint.call_args.kwargs,
    )
    assert (
        mock_select_azure_base_url_or_endpoint.call_args.kwargs["azure_client_params"][
            "max_retries"
        ]
        == max_retries
    )


def test_azure_safety_result():
    """Bubble up safety result from Azure OpenAI"""
    from litellm import completion

    litellm.turn_on_debug()

    response = completion(
        model="azure/gpt-4.1-mini",
        api_key=os.getenv("AZURE_AI_API_KEY"),
        api_base=os.getenv("AZURE_AI_API_BASE"),
        api_version="2024-12-01-preview",
        messages=[{"role": "user", "content": "Hello world"}],
    )
    print(f"response: {response}")
    assert response.choices[0].message.content is not None
    assert response.choices[0].provider_specific_fields is not None


def test_completion_azure_deployment_id():
    """
    Ensure deployment_id takes precedence over model.
    """
    litellm.set_verbose = True
    response = completion(
        deployment_id="gpt-4.1-mini",
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?",
            }
        ],
    )
    # Add any assertions here to check the response
    print(response)
