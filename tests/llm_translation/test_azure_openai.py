import sys
import os

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path

import pytest
from litellm.llms.azure.common_utils import process_azure_headers
from httpx import Headers
from base_embedding_unit_tests import BaseLLMEmbeddingTest


def test_process_azure_headers_empty():
    result = process_azure_headers({})
    assert result == {}, "Expected empty dictionary for no input"


def test_process_azure_headers_with_all_headers():
    input_headers = Headers(
        {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "90",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-remaining-tokens": "9000",
            "other-header": "value",
        }
    )

    expected_output = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "90",
        "x-ratelimit-limit-tokens": "10000",
        "x-ratelimit-remaining-tokens": "9000",
        "llm_provider-x-ratelimit-limit-requests": "100",
        "llm_provider-x-ratelimit-remaining-requests": "90",
        "llm_provider-x-ratelimit-limit-tokens": "10000",
        "llm_provider-x-ratelimit-remaining-tokens": "9000",
        "llm_provider-other-header": "value",
    }

    result = process_azure_headers(input_headers)
    assert result == expected_output, "Unexpected output for all Azure headers"


def test_process_azure_headers_with_partial_headers():
    input_headers = Headers(
        {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-tokens": "9000",
            "other-header": "value",
        }
    )

    expected_output = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-tokens": "9000",
        "llm_provider-x-ratelimit-limit-requests": "100",
        "llm_provider-x-ratelimit-remaining-tokens": "9000",
        "llm_provider-other-header": "value",
    }

    result = process_azure_headers(input_headers)
    assert result == expected_output, "Unexpected output for partial Azure headers"


def test_process_azure_headers_with_no_matching_headers():
    input_headers = Headers(
        {"unrelated-header-1": "value1", "unrelated-header-2": "value2"}
    )

    expected_output = {
        "llm_provider-unrelated-header-1": "value1",
        "llm_provider-unrelated-header-2": "value2",
    }

    result = process_azure_headers(input_headers)
    assert result == expected_output, "Unexpected output for non-matching headers"


def test_process_azure_headers_with_dict_input():
    input_headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "90",
        "other-header": "value",
    }

    expected_output = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "90",
        "llm_provider-x-ratelimit-limit-requests": "100",
        "llm_provider-x-ratelimit-remaining-requests": "90",
        "llm_provider-other-header": "value",
    }

    result = process_azure_headers(input_headers)
    assert result == expected_output, "Unexpected output for dict input"


from httpx import Client
from unittest.mock import MagicMock, patch
from openai import AzureOpenAI
import litellm
from litellm import completion
import os


@pytest.mark.parametrize(
    "input, call_type",
    [
        ({"messages": [{"role": "user", "content": "Hello world"}]}, "completion"),
        ({"input": "Hello world"}, "embedding"),
        ({"prompt": "Hello world"}, "image_generation"),
    ],
)
@pytest.mark.parametrize(
    "header_value",
    [
        "headers",
        "extra_headers",
    ],
)
def test_azure_extra_headers(input, call_type, header_value):
    from litellm import embedding, image_generation

    http_client = Client()

    messages = [{"role": "user", "content": "Hello world"}]
    with patch.object(http_client, "send", new=MagicMock()) as mock_client:
        litellm.client_session = http_client
        try:
            if call_type == "completion":
                func = completion
            elif call_type == "embedding":
                func = embedding
            elif call_type == "image_generation":
                func = image_generation

            data = {
                "model": "azure/chatgpt-v-2",
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com",
                "api_version": "2023-07-01-preview",
                "api_key": "my-azure-api-key",
                header_value: {
                    "Authorization": "my-bad-key",
                    "Ocp-Apim-Subscription-Key": "hello-world-testing",
                },
                **input,
            }
            response = func(**data)
            print(response)

        except Exception as e:
            print(e)

        mock_client.assert_called()

        print(f"mock_client.call_args: {mock_client.call_args}")
        request = mock_client.call_args[0][0]
        print(request.method)  # This will print 'POST'
        print(request.url)  # This will print the full URL
        print(request.headers)  # This will print the full URL
        auth_header = request.headers.get("Authorization")
        apim_key = request.headers.get("Ocp-Apim-Subscription-Key")
        print(auth_header)
        assert auth_header == "my-bad-key"
        assert apim_key == "hello-world-testing"


@pytest.mark.parametrize(
    "api_base, model, expected_endpoint",
    [
        (
            "https://my-endpoint-sweden-berri992.openai.azure.com",
            "dall-e-3-test",
            "https://my-endpoint-sweden-berri992.openai.azure.com/openai/deployments/dall-e-3-test/images/generations?api-version=2023-12-01-preview",
        ),
        (
            "https://my-endpoint-sweden-berri992.openai.azure.com/openai/deployments/my-custom-deployment",
            "dall-e-3",
            "https://my-endpoint-sweden-berri992.openai.azure.com/openai/deployments/my-custom-deployment/images/generations?api-version=2023-12-01-preview",
        ),
    ],
)
def test_process_azure_endpoint_url(api_base, model, expected_endpoint):
    from litellm.llms.azure.azure import AzureChatCompletion

    azure_chat_completion = AzureChatCompletion()
    input_args = {
        "azure_client_params": {
            "api_version": "2023-12-01-preview",
            "azure_endpoint": api_base,
            "azure_deployment": model,
            "max_retries": 2,
            "timeout": 600,
            "api_key": "f28ab7b695af4154bc53498e5bdccb07",
        },
        "model": model,
    }
    result = azure_chat_completion.create_azure_base_url(**input_args)
    assert result == expected_endpoint, "Unexpected endpoint"


class TestAzureEmbedding(BaseLLMEmbeddingTest):
    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "azure/azure-embedding-model",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv("AZURE_API_BASE"),
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.AZURE
