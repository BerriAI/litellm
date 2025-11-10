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
                "model": "azure/gpt-4.1-mini",
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
            "model": "azure/text-embedding-ada-002",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv("AZURE_API_BASE"),
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.AZURE


@patch("azure.identity.UsernamePasswordCredential")
@patch("azure.identity.get_bearer_token_provider")
def test_get_azure_ad_token_from_username_password(
    mock_get_bearer_token_provider, mock_credential
):
    from litellm.llms.azure.common_utils import (
        get_azure_ad_token_from_username_password,
    )

    # Test inputs
    client_id = "test-client-id"
    username = "test-username"
    password = "test-password"

    # Mock the token provider function
    mock_token_provider = lambda: "mock-token"
    mock_get_bearer_token_provider.return_value = mock_token_provider

    # Call the function
    result = get_azure_ad_token_from_username_password(
        client_id=client_id, azure_username=username, azure_password=password
    )

    # Verify UsernamePasswordCredential was called with correct arguments
    mock_credential.assert_called_once_with(
        client_id=client_id, username=username, password=password
    )

    # Verify get_bearer_token_provider was called
    mock_get_bearer_token_provider.assert_called_once_with(
        mock_credential.return_value, "https://cognitiveservices.azure.com/.default"
    )

    # Verify the result is the mock token provider
    assert result == mock_token_provider


def test_azure_openai_gpt_4o_naming(monkeypatch):
    from openai import AzureOpenAI
    from pydantic import BaseModel, Field

    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")

    client = AzureOpenAI(
        api_key="test-api-key",
        base_url="https://my-endpoint-sweden-berri992.openai.azure.com",
        api_version="2023-12-01-preview",
    )

    class ResponseFormat(BaseModel):

        number: str = Field(description="total number of days in a week")
        days: list[str] = Field(description="name of days in a week")

    with patch.object(client.chat.completions.with_raw_response, "create") as mock_post:
        try:
            completion(
                model="azure/gpt4o",
                messages=[{"role": "user", "content": "Hello world"}],
                response_format=ResponseFormat,
                client=client,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        print(mock_post.call_args.kwargs)

        assert "tool_calls" not in mock_post.call_args.kwargs


@pytest.mark.parametrize(
    "api_version",
    [
        "2024-10-21",
        # "2024-02-15-preview",
    ],
)
def test_azure_gpt_4o_with_tool_call_and_response_format(api_version):
    from litellm import completion
    from typing import Optional
    from pydantic import BaseModel
    import litellm

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key="fake-key",
        base_url="https://fake-azure.openai.azure.com",
        api_version=api_version,
    )

    class InvestigationOutput(BaseModel):
        alert_explanation: Optional[str] = None
        investigation: Optional[str] = None
        conclusions_and_possible_root_causes: Optional[str] = None
        next_steps: Optional[str] = None
        related_logs: Optional[str] = None
        app_or_infra: Optional[str] = None
        external_links: Optional[str] = None

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Returns the current date and time",
                "strict": True,
                "parameters": {
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone to get the current time for (e.g., 'UTC', 'America/New_York')",
                        }
                    },
                    "required": ["timezone"],
                    "type": "object",
                    "additionalProperties": False,
                },
            },
        }
    ]

    with patch.object(client.chat.completions.with_raw_response, "create") as mock_post:
        response = litellm.completion(
            model="azure/gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a tool-calling AI assist provided with common devops and IT tools that you can use to troubleshoot problems or answer questions.\nWhenever possible you MUST first use tools to investigate then answer the question.",
                },
                {
                    "role": "user",
                    "content": "What is the current date and time in NYC?",
                },
            ],
            drop_params=True,
            temperature=0.00000001,
            tools=tools,
            tool_choice="auto",
            response_format=InvestigationOutput,  # commenting this line will cause the output to be correct
            api_version=api_version,
            client=client,
        )

        mock_post.assert_called_once()

        if api_version == "2024-10-21":
            assert "response_format" in mock_post.call_args.kwargs
        else:
            assert "response_format" not in mock_post.call_args.kwargs


def test_map_openai_params():
    """
    Ensure response_format does not override tools
    """
    from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig

    azure_openai_config = AzureOpenAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Returns the current date and time",
                "strict": True,
                "parameters": {
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone to get the current time for (e.g., 'UTC', 'America/New_York')",
                        }
                    },
                    "required": ["timezone"],
                    "type": "object",
                    "additionalProperties": False,
                },
            },
        }
    ]
    received_args = {
        "non_default_params": {
            "temperature": 1e-08,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "properties": {
                            "alert_explanation": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "Alert Explanation",
                            },
                            "investigation": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "Investigation",
                            },
                            "conclusions_and_possible_root_causes": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "Conclusions And Possible Root Causes",
                            },
                            "next_steps": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "Next Steps",
                            },
                            "related_logs": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "Related Logs",
                            },
                            "app_or_infra": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "App Or Infra",
                            },
                            "external_links": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "External Links",
                            },
                        },
                        "title": "InvestigationOutput",
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "alert_explanation",
                            "investigation",
                            "conclusions_and_possible_root_causes",
                            "next_steps",
                            "related_logs",
                            "app_or_infra",
                            "external_links",
                        ],
                    },
                    "name": "InvestigationOutput",
                    "strict": True,
                },
            },
            "tools": tools,
            "tool_choice": "auto",
        },
        "optional_params": {},
        "model": "gpt-4o",
        "drop_params": True,
        "api_version": "2024-02-15-preview",
    }
    optional_params = azure_openai_config.map_openai_params(**received_args)
    assert "tools" in optional_params
    assert len(optional_params["tools"]) > 1


@pytest.mark.parametrize("max_retries", [0, 4])
@pytest.mark.parametrize("stream", [True, False])
@patch(
    "litellm.main.azure_chat_completions.make_sync_azure_openai_chat_completion_request"
)
def test_azure_max_retries_0(
    mock_make_sync_azure_openai_chat_completion_request, max_retries, stream
):
    from litellm import completion

    try:
        completion(
            model="azure/gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello world"}],
            max_retries=max_retries,
            stream=stream,
        )
    except Exception as e:
        print(e)

    mock_make_sync_azure_openai_chat_completion_request.assert_called_once()
    assert (
        mock_make_sync_azure_openai_chat_completion_request.call_args.kwargs[
            "azure_client"
        ].max_retries
        == max_retries
    )


@pytest.mark.parametrize("max_retries", [0, 4])
@pytest.mark.parametrize("stream", [True, False])
@patch("litellm.main.azure_chat_completions.make_azure_openai_chat_completion_request")
@pytest.mark.asyncio
async def test_async_azure_max_retries_0(
    make_azure_openai_chat_completion_request, max_retries, stream
):
    from litellm import acompletion

    try:
        await acompletion(
            model="azure/gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello world"}],
            max_retries=max_retries,
            stream=stream,
        )
    except Exception as e:
        print(e)

    make_azure_openai_chat_completion_request.assert_called_once()
    assert (
        make_azure_openai_chat_completion_request.call_args.kwargs[
            "azure_client"
        ].max_retries
        == max_retries
    )


@pytest.mark.parametrize("max_retries", [0, 4])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("sync_mode", [True, False])
@patch("litellm.llms.azure.common_utils.select_azure_base_url_or_endpoint")
@pytest.mark.asyncio
async def test_azure_instruct(
    mock_select_azure_base_url_or_endpoint, max_retries, stream, sync_mode
):
    from litellm import completion, acompletion

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
    from litellm import aembedding, embedding

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

    litellm._turn_on_debug()

    response = completion(
        model="azure/gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hello world"}],
    )
    print(f"response: {response}")
    assert response.choices[0].message.content is not None
    assert response.choices[0].provider_specific_fields is not None


def test_azure_openai_responses_bridge():
    from litellm import completion
    import litellm

    litellm._turn_on_debug()

    with patch.object(litellm, "responses") as mock_responses:
        try:
            response = completion(
                model="azure/responses/test-azure-computer-use-preview",
                messages=[{"role": "user", "content": "Hello world"}],
                api_base=os.getenv("AZURE_COMPUTER_USE_API_BASE"),
                api_version="2025-04-01-preview",
                api_key=os.getenv("AZURE_COMPUTER_USE_API_KEY"),
            )
        except Exception as e:
            print(e)

        mock_responses.assert_called_once()
        assert (
            mock_responses.call_args.kwargs["model"]
            == "test-azure-computer-use-preview"
        )
        assert mock_responses.call_args.kwargs["custom_llm_provider"] == "azure"


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
def test_azure_with_content_safety_error():
    """
    Verify user can access innererror from the Azure OpenAI exception
    """
    from litellm import completion
    from litellm.exceptions import ContentPolicyViolationError
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type
    from unittest.mock import MagicMock
    
    mock_exception = Exception("The response was filtered due to the prompt triggering Azure OpenAI's content management policy")
    mock_exception.body = {
        "innererror": {
            "code": "ResponsibleAIPolicyViolation",
            "content_filter_result": {
                "hate": {
                    "filtered": False,
                    "severity": "safe"
                },
                "jailbreak": {
                    "filtered": False,
                    "detected": False
                },
                "self_harm": {
                    "filtered": False,
                    "severity": "safe"
                },
                "sexual": {
                    "filtered": False,
                    "severity": "safe"
                },
                "violence": {
                    "filtered": True,
                    "severity": "high"
                }
            }
        }
    }
    
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_exception.response = mock_response
    
    with pytest.raises(ContentPolicyViolationError) as exc_info:
        exception_type(
            model="azure/gpt-4o-new-test",
            original_exception=mock_exception,
            custom_llm_provider="azure"
        )
    
    e = exc_info.value
    print("got exception=", e)
    assert e.provider_specific_fields is not None
    print("got provider_specific_fields=", e.provider_specific_fields)
    assert e.provider_specific_fields.get("innererror") is not None
    assert e.provider_specific_fields["innererror"]["code"] == "ResponsibleAIPolicyViolation"
    assert e.provider_specific_fields["innererror"]["content_filter_result"]["violence"]["filtered"] is True
    assert e.provider_specific_fields["innererror"]["content_filter_result"]["violence"]["severity"] == "high"
