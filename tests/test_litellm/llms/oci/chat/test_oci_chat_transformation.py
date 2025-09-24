import datetime
import os
import sys
import httpx
import pytest
import json

import litellm

# Adds the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm import ModelResponse
from litellm.llms.oci.chat.transformation import OCIChatConfig, version

TEST_MODEL_NAME = "xai.grok-4"
TEST_MODEL = f"oci/{TEST_MODEL_NAME}"
TEST_MESSAGES = [{"role": "user", "content": "Hello, how are you?"}]
TEST_COMPARTMENT_ID = "ocid1.compartment.oc1..xxxxxx"
BASE_OCI_PARAMS = {
    "oci_region": "us-ashburn-1",
    "oci_user": "ocid1.user.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_fingerprint": "4f:29:77:cc:b1:3e:55:ab:61:2a:de:47:f1:38:4c:90",
    "oci_tenancy": "ocid1.tenancy.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_compartment_id": TEST_COMPARTMENT_ID,
}

TEST_OCI_PARAMS_KEY = {
    **BASE_OCI_PARAMS,
    "oci_key": "<private_key.pem as string>",
}

TEST_OCI_PARAMS_KEY_FILE = {
    **BASE_OCI_PARAMS,
    "oci_key_file": "<private_key.pem as a Path>",
}

@pytest.fixture(params=[TEST_OCI_PARAMS_KEY, TEST_OCI_PARAMS_KEY_FILE])
def supplied_params(request):
    """Fixture for passing in optional_parameters"""
    return request.param


class TestOCIChatConfig:
    def test_validate_environment_with_oci_region(self, supplied_params):
        config = OCIChatConfig()
        headers = {}

        result = config.validate_environment(
            headers=headers,
            model=TEST_MODEL,
            messages=TEST_MESSAGES,  # type: ignore
            optional_params=supplied_params,
            litellm_params={},
        )

        assert result["content-type"] == "application/json"
        assert result["user-agent"] == f"litellm/{version}"

    def test_missing_oci_auth_parameters(self, supplied_params):
        params = supplied_params.copy()  # safely copy, no reassignment
        params.pop("oci_region")

        for key in list(params.keys()):
            modified_params = params.copy()
            del modified_params[key]

            with pytest.raises(Exception) as excinfo:
                config = OCIChatConfig()
                headers = {}

                config.validate_environment(
                    headers=headers,
                    model=TEST_MODEL,
                    messages=TEST_MESSAGES,  # type: ignore
                    optional_params=modified_params,
                    api_base="https://api.oci.example.com",
                    litellm_params={},
                )
            assert ("Missing required parameters:") in str(excinfo.value)

    def test_transform_request_simple(self):
        """
        Tests if a simple request is transformed correctly.
        """
        config = OCIChatConfig()
        optional_params = {"oci_compartment_id": TEST_COMPARTMENT_ID}
        transformed_request = config.transform_request(
            model=TEST_MODEL_NAME,
            messages=TEST_MESSAGES, # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        expected_output = {
            "compartmentId": TEST_COMPARTMENT_ID,
            "servingMode": {"servingType": "ON_DEMAND", "modelId": TEST_MODEL_NAME},
            "chatRequest": {
                "apiFormat": "GENERIC",
                "isStream": False,
                "messages": [
                    {
                        "role": "USER",
                        "content": [{"type": "TEXT", "text": "Hello, how are you?"}],
                    }
                ],
            },
        }
        assert transformed_request == expected_output

    def test_transform_request_with_tools(self):
        """
        Tests if a request with tools is transformed correctly.
        """
        config = OCIChatConfig()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "tools": tools,
        }
        transformed_request = config.transform_request(
            model=TEST_MODEL_NAME,
            messages=TEST_MESSAGES, # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert "tools" in transformed_request["chatRequest"]
        assert transformed_request["chatRequest"]["tools"][0]["name"] == "get_current_weather"
        assert transformed_request["chatRequest"]["tools"][0]["type"] == "FUNCTION"
        assert transformed_request["chatRequest"]["tools"][0]["description"] == "Get the current weather in a given location"
        assert transformed_request["chatRequest"]["tools"][0]["parameters"] is not None

    def test_transform_response_simple_text(self):
        """
        Tests if a simple text response is transformed correctly.
        """
        config = OCIChatConfig()
        created_time = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        mock_oci_response = {
            "modelId": TEST_MODEL_NAME,
            "modelVersion": "1.0",
            "chatResponse": {
                "apiFormat": "GENERIC",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "I am doing well, thank you!"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "timeCreated": created_time,
                "usage": {
                    "promptTokens": 10,
                    "completionTokens": 20,
                    "totalTokens": 30,
                    "completionTokensDetails": {
                        "acceptedPredictionTokens": 20,
                        "reasoningTokens": 20,
                    },
                    "promptTokensDetails": {
                        "cachedTokens": 10,
                    },
                },
            },
        }
        response = httpx.Response(
            status_code=200, json=mock_oci_response, headers={"Content-Type": "application/json"}
        )
        result = config.transform_response(
            model=TEST_MODEL_NAME,
            raw_response=response,
            model_response=ModelResponse(),
            logging_obj={},  # type: ignore
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        assert isinstance(result, ModelResponse)
        assert len(result.choices) == 1
        assert isinstance(result.choices[0], litellm.Choices)
        assert result.choices[0].message
        assert result.choices[0].message.content == "I am doing well, thank you!"
        assert result.choices[0].finish_reason == "stop"
        assert result.model == TEST_MODEL_NAME
        assert hasattr(result, "usage")
        assert isinstance(result.usage, litellm.Usage) # type: ignore
        assert result.usage.prompt_tokens == 10 # type: ignore
        assert result.usage.completion_tokens == 20 # type: ignore
        assert result.usage.total_tokens == 30 # type: ignore
        # These are not handled in the transformer, TBH no idea why they are here
        # but, for now, they seem to be always None
        assert result.usage.completion_tokens_details is None
        assert result.usage.prompt_tokens_details is None

    def test_transform_response_with_tool_calls(self):
        """
        Tests if a response with tool calls is transformed correctly.
        """
        config = OCIChatConfig()
        created_time = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        mock_oci_response = {
            "modelId": TEST_MODEL_NAME,
            "modelVersion": "1.0",
            "chatResponse": {
                "apiFormat": "GENERIC",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "ASSISTANT",
                            "content": None,
                            "toolCalls": [
                                {
                                    "id": "call_abc123",
                                    "type": "FUNCTION",
                                    "name": "get_weather",
                                    "arguments": '{"location": "Vila Velha, BR"}',
                                }
                            ],
                        },
                        "finishReason": "stop",
                    }
                ],
                "timeCreated": created_time,
                "usage": {
                    "promptTokens": 10,
                    "completionTokens": 20,
                    "totalTokens": 30,
                    "completionTokensDetails": {
                        "acceptedPredictionTokens": 20,
                        "reasoningTokens": 20,
                    },
                    "promptTokensDetails": {
                        "cachedTokens": 10,
                    },
                },
            },
        }
        response = httpx.Response(status_code=200, json=mock_oci_response)
        model_response = ModelResponse(
            choices=[litellm.Choices(index=0, message=litellm.Message())]
        )

        result = config.transform_response(
            model=TEST_MODEL_NAME,
            raw_response=response,
            model_response=model_response,
            logging_obj={},  # type: ignore
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        # General assertions
        assert isinstance(result, ModelResponse)
        assert len(result.choices) == 1

        choice = result.choices[0]
        assert isinstance(choice, litellm.Choices)
        assert choice.finish_reason == "stop"

        # Message and tool_calls assertions
        message = choice.message
        assert isinstance(message, litellm.Message)
        assert hasattr(message, "tool_calls")
        assert isinstance(message.tool_calls, list)
        assert len(message.tool_calls) == 1

        # Specific tool_call assertions
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, litellm.utils.ChatCompletionMessageToolCall)
        assert tool_call.id == "call_abc123"
        assert tool_call.type == "function"
        assert tool_call.function["name"] == "get_weather"
        assert tool_call.function["arguments"] == '{"location": "Vila Velha, BR"}'

        # Usage assertions
        assert hasattr(result, "usage")
        usage = result.usage # type: ignore
        assert isinstance(usage, litellm.Usage) # type: ignore
        assert usage.prompt_tokens == 10 # type: ignore
        assert usage.completion_tokens == 20 # type: ignore
        assert usage.total_tokens == 30 # type: ignore
