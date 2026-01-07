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
from litellm.llms.oci.chat.transformation import OCIChatConfig, OCIRequestWrapper, version

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

    def test_transform_request_dedicated_mode_with_endpoint_id(self):
        """
        Tests if a request with DEDICATED serving mode and explicit oci_endpoint_id is transformed correctly.
        """
        config = OCIChatConfig()
        test_endpoint_id = "ocid1.generativeaiendpoint.oc1.us-chicago-1.xxxxxx"
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "oci_serving_mode": "DEDICATED",
            "oci_endpoint_id": test_endpoint_id,
        }
        transformed_request = config.transform_request(
            model=TEST_MODEL_NAME,
            messages=TEST_MESSAGES, # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        expected_serving_mode = {
            "servingType": "DEDICATED",
            "endpointId": test_endpoint_id,
        }
        assert transformed_request["servingMode"] == expected_serving_mode
        assert transformed_request["compartmentId"] == TEST_COMPARTMENT_ID

    def test_transform_request_dedicated_mode_without_endpoint_id(self):
        """
        Tests if a request with DEDICATED serving mode falls back to model name when oci_endpoint_id is not provided.
        """
        config = OCIChatConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "oci_serving_mode": "DEDICATED",
        }
        transformed_request = config.transform_request(
            model=TEST_MODEL_NAME,
            messages=TEST_MESSAGES, # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Should use model name as endpoint ID when oci_endpoint_id is not provided
        expected_serving_mode = {
            "servingType": "DEDICATED",
            "endpointId": TEST_MODEL_NAME,
        }
        assert transformed_request["servingMode"] == expected_serving_mode
        assert transformed_request["compartmentId"] == TEST_COMPARTMENT_ID

    def test_transform_request_on_demand_mode(self):
        """
        Tests if a request with ON_DEMAND serving mode uses modelId correctly.
        """
        config = OCIChatConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "oci_serving_mode": "ON_DEMAND",
        }
        transformed_request = config.transform_request(
            model=TEST_MODEL_NAME,
            messages=TEST_MESSAGES, # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        expected_serving_mode = {
            "servingType": "ON_DEMAND",
            "modelId": TEST_MODEL_NAME,
        }
        assert transformed_request["servingMode"] == expected_serving_mode
        assert transformed_request["compartmentId"] == TEST_COMPARTMENT_ID

    def test_transform_request_invalid_serving_mode(self):
        """
        Tests if an invalid serving mode raises an exception.
        """
        config = OCIChatConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "oci_serving_mode": "INVALID_MODE",
        }

        with pytest.raises(Exception) as excinfo:
            config.transform_request(
                model=TEST_MODEL_NAME,
                messages=TEST_MESSAGES, # type: ignore
                optional_params=optional_params,
                litellm_params={},
                headers={},
            )

        assert "must be either 'ON_DEMAND' or 'DEDICATED'" in str(excinfo.value)

    def test_transform_request_dedicated_cohere_with_endpoint_id(self):
        """
        Tests that Cohere vendor detection works correctly with DEDICATED mode and oci_endpoint_id.
        This is critical because the model parameter determines the API format even when using oci_endpoint_id.
        """
        config = OCIChatConfig()
        cohere_model = "cohere.command-latest"
        test_endpoint_id = "ocid1.generativeaiendpoint.oc1.us-chicago-1.xxxxxx"
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "oci_serving_mode": "DEDICATED",
            "oci_endpoint_id": test_endpoint_id,
        }

        messages = [
            {"role": "user", "content": "What is quantum computing?"},
        ]

        transformed_request = config.transform_request(
            model=cohere_model,
            messages=messages, # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Verify DEDICATED mode with correct endpoint ID
        assert transformed_request["servingMode"]["servingType"] == "DEDICATED"
        assert transformed_request["servingMode"]["endpointId"] == test_endpoint_id

        # Verify Cohere API format is used (not GENERIC)
        assert transformed_request["chatRequest"]["apiFormat"] == "COHERE"

        # Verify Cohere-specific request structure
        assert "message" in transformed_request["chatRequest"]  # Cohere uses "message"
        assert "chatHistory" in transformed_request["chatRequest"]  # Cohere uses "chatHistory"
        assert "messages" not in transformed_request["chatRequest"]  # Generic uses "messages"

        # Verify the message content
        assert transformed_request["chatRequest"]["message"] == "What is quantum computing?"

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


class TestOCISignerSupport:
    """Tests for OCI SDK Signer integration."""

    def test_validate_environment_with_oci_signer(self):
        """Test validation when using oci_signer instead of manual credentials."""
        config = OCIChatConfig()
        headers = {}

        # Mock signer object
        class MockSigner:
            def do_request_sign(self, request, enforce_content_headers=True):
                request.headers["authorization"] = "Signature version=\"1\""

        optional_params = {
            "oci_signer": MockSigner(),
            "oci_region": "us-ashburn-1"
        }

        result = config.validate_environment(
            headers=headers,
            model=TEST_MODEL,
            messages=TEST_MESSAGES,  # type: ignore
            optional_params=optional_params,
            litellm_params={},
        )

        assert result["content-type"] == "application/json"
        assert result["user-agent"] == f"litellm/{version}"

    def test_validate_environment_with_oci_signer_no_compartment_id_in_validate(self):
        """Test that oci_compartment_id is not required in validate_environment when using signer."""
        config = OCIChatConfig()
        headers = {}

        class MockSigner:
            def do_request_sign(self, request, enforce_content_headers=True):
                request.headers["authorization"] = "Signature version=\"1\""

        optional_params = {
            "oci_signer": MockSigner(),
            "oci_region": "us-phoenix-1"
        }

        # Should not raise an exception even without oci_compartment_id
        result = config.validate_environment(
            headers=headers,
            model=TEST_MODEL,
            messages=TEST_MESSAGES,  # type: ignore
            optional_params=optional_params,
            litellm_params={},
        )

        assert result["content-type"] == "application/json"

    def test_sign_request_with_oci_signer(self):
        """Test request signing with oci_signer."""
        config = OCIChatConfig()

        class MockSigner:
            def do_request_sign(self, request, enforce_content_headers=True):
                request.headers["authorization"] = "Signature version=\"1\""
                request.headers["date"] = "Mon, 01 Jan 2024 00:00:00 GMT"

        optional_params = {
            "oci_signer": MockSigner(),
            "method": "POST"
        }

        headers, body = config.sign_request(
            headers={},
            optional_params=optional_params,
            request_data={"test": "data"},
            api_base="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat"
        )

        assert "authorization" in headers
        assert "Signature" in headers["authorization"]
        assert body is not None  # oci_signer path returns body
        assert json.loads(body.decode("utf-8")) == {"test": "data"}

    def test_sign_request_with_oci_signer_updates_headers(self):
        """Test that signer properly updates request headers."""
        config = OCIChatConfig()

        class MockSigner:
            def do_request_sign(self, request, enforce_content_headers=True):
                # Verify the request has the expected attributes
                assert hasattr(request, "method")
                assert hasattr(request, "url")
                assert hasattr(request, "headers")
                assert hasattr(request, "body")
                assert hasattr(request, "path_url")

                # Add signature headers
                request.headers["authorization"] = "Signature keyId=\"test\""
                request.headers["date"] = "Mon, 01 Jan 2024 00:00:00 GMT"
                request.headers["x-content-sha256"] = "test-hash"

        optional_params = {
            "oci_signer": MockSigner(),
        }

        headers, body = config.sign_request(
            headers={"custom-header": "custom-value"},
            optional_params=optional_params,
            request_data={"message": "Hello"},
            api_base="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat"
        )

        # Check that all signer-added headers are present
        assert headers["authorization"] == "Signature keyId=\"test\""
        assert headers["date"] == "Mon, 01 Jan 2024 00:00:00 GMT"
        assert headers["x-content-sha256"] == "test-hash"
        # Original headers should be preserved
        assert headers["custom-header"] == "custom-value"

    def test_sign_request_with_failing_oci_signer(self):
        """Test error handling when oci_signer fails."""
        config = OCIChatConfig()

        class FailingSigner:
            def do_request_sign(self, request, enforce_content_headers=True):
                raise RuntimeError("Signing failed due to invalid credentials")

        optional_params = {
            "oci_signer": FailingSigner(),
        }

        from litellm.llms.oci.common_utils import OCIError

        with pytest.raises(OCIError) as excinfo:
            config.sign_request(
                headers={},
                optional_params=optional_params,
                request_data={"test": "data"},
                api_base="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat"
            )

        assert "Failed to sign request with provided oci_signer" in str(excinfo.value)
        assert excinfo.value.status_code == 500

    def test_sign_request_with_invalid_http_method(self):
        """Test that invalid HTTP methods are rejected."""
        config = OCIChatConfig()

        class MockSigner:
            def do_request_sign(self, request, enforce_content_headers=True):
                pass

        optional_params = {
            "oci_signer": MockSigner(),
            "method": "INVALID"
        }

        with pytest.raises(ValueError) as excinfo:
            config.sign_request(
                headers={},
                optional_params=optional_params,
                request_data={"test": "data"},
                api_base="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat"
            )

        assert "Unsupported HTTP method: INVALID" in str(excinfo.value)

    def test_oci_request_wrapper_path_url(self):
        """Test OCIRequestWrapper path_url property."""
        wrapper = OCIRequestWrapper(
            method="POST",
            url="https://example.com/api/v1/chat?param1=value1&param2=value2",
            headers={},
            body=b"test"
        )

        assert wrapper.path_url == "/api/v1/chat?param1=value1&param2=value2"

    def test_oci_request_wrapper_path_url_no_query(self):
        """Test OCIRequestWrapper path_url property without query string."""
        wrapper = OCIRequestWrapper(
            method="POST",
            url="https://example.com/api/v1/chat",
            headers={},
            body=b"test"
        )

        assert wrapper.path_url == "/api/v1/chat"
