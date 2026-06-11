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


from unittest.mock import MagicMock
import litellm
from litellm import completion
import os


@pytest.mark.parametrize(
    "api_base, model, expected_endpoint",
    [
        (
            "https://fake-azure-endpoint.invalid",
            "dall-e-3-test",
            "https://fake-azure-endpoint.invalid/openai/deployments/dall-e-3-test/images/generations?api-version=2023-12-01-preview",
        ),
        (
            "https://fake-azure-endpoint.invalid/openai/deployments/my-custom-deployment",
            "dall-e-3",
            "https://fake-azure-endpoint.invalid/openai/deployments/my-custom-deployment/images/generations?api-version=2023-12-01-preview",
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
            "api_key": "sk-test-mock-key-505",
        },
        "model": model,
    }
    result = azure_chat_completion.create_azure_base_url(**input_args)
    assert result == expected_endpoint, "Unexpected endpoint"


class TestAzureEmbedding(BaseLLMEmbeddingTest):
    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "azure/text-embedding-ada-002",
            "api_key": os.getenv("AZURE_AI_API_KEY"),
            "api_base": os.getenv("AZURE_AI_API_BASE"),
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.AZURE


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


def test_azure_safety_result():
    """Bubble up safety result from Azure OpenAI"""
    from litellm import completion

    litellm._turn_on_debug()

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


def test_azure_with_content_safety_error():
    """
    Verify user can access innererror from the Azure OpenAI exception
    """
    from litellm.exceptions import ContentPolicyViolationError
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type

    mock_exception = Exception(
        "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"
    )
    mock_exception.body = {
        "innererror": {
            "code": "ResponsibleAIPolicyViolation",
            "content_filter_result": {
                "hate": {"filtered": False, "severity": "safe"},
                "jailbreak": {"filtered": False, "detected": False},
                "self_harm": {"filtered": False, "severity": "safe"},
                "sexual": {"filtered": False, "severity": "safe"},
                "violence": {"filtered": True, "severity": "high"},
            },
        }
    }

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_exception.response = mock_response

    with pytest.raises(ContentPolicyViolationError) as exc_info:
        exception_type(
            model="azure/gpt-4o-new-test",
            original_exception=mock_exception,
            custom_llm_provider="azure",
        )

    e = exc_info.value
    print("got exception=", e)
    assert e.provider_specific_fields is not None
    print("got provider_specific_fields=", e.provider_specific_fields)
    assert e.provider_specific_fields.get("innererror") is not None
    assert (
        e.provider_specific_fields["innererror"]["code"]
        == "ResponsibleAIPolicyViolation"
    )
    assert (
        e.provider_specific_fields["innererror"]["content_filter_result"]["violence"][
            "filtered"
        ]
        is True
    )
    assert (
        e.provider_specific_fields["innererror"]["content_filter_result"]["violence"][
            "severity"
        ]
        == "high"
    )


def test_azure_openai_with_prompt_cache_key():
    """
    E2E test for Azure OpenAI with prompt cache key param on /chat/completions API.
    """
    litellm._turn_on_debug()
    response = litellm.completion(
        model="azure/gpt-4.1-mini",
        api_key=os.getenv("AZURE_AI_API_KEY"),
        api_base=os.getenv("AZURE_AI_API_BASE"),
        api_version="2024-12-01-preview",
        messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
        prompt_cache_key="test_streaming_azure_openai",
    )
