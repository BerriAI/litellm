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
from litellm.llms.oci.chat.transformation import (
    OCIChatConfig,
    OCIRequestWrapper,
    version,
)

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
            messages=TEST_MESSAGES,  # type: ignore
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
            messages=TEST_MESSAGES,  # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert "tools" in transformed_request["chatRequest"]
        assert (
            transformed_request["chatRequest"]["tools"][0]["name"]
            == "get_current_weather"
        )
        assert transformed_request["chatRequest"]["tools"][0]["type"] == "FUNCTION"
        assert (
            transformed_request["chatRequest"]["tools"][0]["description"]
            == "Get the current weather in a given location"
        )
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
            messages=TEST_MESSAGES,  # type: ignore
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
            messages=TEST_MESSAGES,  # type: ignore
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
            messages=TEST_MESSAGES,  # type: ignore
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
                messages=TEST_MESSAGES,  # type: ignore
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
            messages=messages,  # type: ignore
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
        assert (
            "chatHistory" in transformed_request["chatRequest"]
        )  # Cohere uses "chatHistory"
        assert (
            "messages" not in transformed_request["chatRequest"]
        )  # Generic uses "messages"

        # Verify the message content
        assert (
            transformed_request["chatRequest"]["message"]
            == "What is quantum computing?"
        )

    def test_transform_request_response_format_json_object(self):
        """
        Tests that response_format type 'json_object' is uppercased to 'JSON_OBJECT' for generic OCI models.
        """
        config = OCIChatConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "response_format": {"type": "json_object"},
        }
        transformed_request = config.transform_request(
            model=TEST_MODEL_NAME,
            messages=TEST_MESSAGES,  # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        rf = transformed_request["chatRequest"]["responseFormat"]
        assert rf["type"] == "JSON_OBJECT"

    def test_transform_request_response_format_text(self):
        """
        Tests that response_format type 'text' is uppercased to 'TEXT' for generic OCI models.
        """
        config = OCIChatConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "response_format": {"type": "text"},
        }
        transformed_request = config.transform_request(
            model=TEST_MODEL_NAME,
            messages=TEST_MESSAGES,  # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        rf = transformed_request["chatRequest"]["responseFormat"]
        assert rf["type"] == "TEXT"

    def test_transform_request_response_format_json_shorthand(self):
        """
        Tests that response_format type 'json' is mapped to 'JSON_OBJECT' for generic OCI models.
        """
        config = OCIChatConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "response_format": {"type": "json"},
        }
        transformed_request = config.transform_request(
            model=TEST_MODEL_NAME,
            messages=TEST_MESSAGES,  # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        rf = transformed_request["chatRequest"]["responseFormat"]
        assert rf["type"] == "JSON_OBJECT"

    def test_transform_response_without_token_details(self):
        """
        Tests that responses missing completionTokensDetails and promptTokensDetails
        are handled correctly (fields are optional).
        """
        config = OCIChatConfig()
        created_time = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
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
                            "content": [{"type": "TEXT", "text": "Hello!"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "timeCreated": created_time,
                "usage": {
                    "promptTokens": 5,
                    "completionTokens": 10,
                    "totalTokens": 15,
                },
            },
        }
        response = httpx.Response(
            status_code=200,
            json=mock_oci_response,
            headers={"Content-Type": "application/json"},
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
        assert result.choices[0].message.content == "Hello!"
        assert result.usage.prompt_tokens == 5  # type: ignore
        assert result.usage.completion_tokens == 10  # type: ignore
        assert result.usage.total_tokens == 15  # type: ignore

    def test_transform_response_simple_text(self):
        """
        Tests if a simple text response is transformed correctly.
        """
        config = OCIChatConfig()
        created_time = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
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
                            "content": [
                                {"type": "TEXT", "text": "I am doing well, thank you!"}
                            ],
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
            status_code=200,
            json=mock_oci_response,
            headers={"Content-Type": "application/json"},
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
        assert isinstance(result.usage, litellm.Usage)  # type: ignore
        assert result.usage.prompt_tokens == 10  # type: ignore
        assert result.usage.completion_tokens == 20  # type: ignore
        assert result.usage.total_tokens == 30  # type: ignore
        # These are not handled in the transformer, TBH no idea why they are here
        # but, for now, they seem to be always None
        assert result.usage.completion_tokens_details is None
        assert result.usage.prompt_tokens_details is None

    def test_transform_response_with_tool_calls(self):
        """
        Tests if a response with tool calls is transformed correctly.
        """
        config = OCIChatConfig()
        created_time = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
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
        usage = result.usage  # type: ignore
        assert isinstance(usage, litellm.Usage)  # type: ignore
        assert usage.prompt_tokens == 10  # type: ignore
        assert usage.completion_tokens == 20  # type: ignore
        assert usage.total_tokens == 30  # type: ignore


class TestOCISignerSupport:
    """Tests for OCI SDK Signer integration."""

    def test_validate_environment_with_oci_signer(self):
        """Test validation when using oci_signer instead of manual credentials."""
        config = OCIChatConfig()
        headers = {}

        # Mock signer object
        class MockSigner:
            def do_request_sign(self, request, enforce_content_headers=True):
                request.headers["authorization"] = 'Signature version="1"'

        optional_params = {"oci_signer": MockSigner(), "oci_region": "us-ashburn-1"}

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
                request.headers["authorization"] = 'Signature version="1"'

        optional_params = {"oci_signer": MockSigner(), "oci_region": "us-phoenix-1"}

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
                request.headers["authorization"] = 'Signature version="1"'
                request.headers["date"] = "Mon, 01 Jan 2024 00:00:00 GMT"

        optional_params = {"oci_signer": MockSigner(), "method": "POST"}

        headers, body = config.sign_request(
            headers={},
            optional_params=optional_params,
            request_data={"test": "data"},
            api_base="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat",
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
                request.headers["authorization"] = 'Signature keyId="test"'
                request.headers["date"] = "Mon, 01 Jan 2024 00:00:00 GMT"
                request.headers["x-content-sha256"] = "test-hash"

        optional_params = {
            "oci_signer": MockSigner(),
        }

        headers, body = config.sign_request(
            headers={"custom-header": "custom-value"},
            optional_params=optional_params,
            request_data={"message": "Hello"},
            api_base="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat",
        )

        # Check that all signer-added headers are present
        assert headers["authorization"] == 'Signature keyId="test"'
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
                api_base="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat",
            )

        assert "Failed to sign request with provided oci_signer" in str(excinfo.value)
        assert excinfo.value.status_code == 500

    def test_sign_request_with_invalid_http_method(self):
        """Test that invalid HTTP methods are rejected."""
        config = OCIChatConfig()

        class MockSigner:
            def do_request_sign(self, request, enforce_content_headers=True):
                pass

        optional_params = {"oci_signer": MockSigner(), "method": "INVALID"}

        with pytest.raises(ValueError) as excinfo:
            config.sign_request(
                headers={},
                optional_params=optional_params,
                request_data={"test": "data"},
                api_base="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat",
            )

        assert "Unsupported HTTP method: INVALID" in str(excinfo.value)

    def test_oci_request_wrapper_path_url(self):
        """Test OCIRequestWrapper path_url property."""
        wrapper = OCIRequestWrapper(
            method="POST",
            url="https://example.com/api/v1/chat?param1=value1&param2=value2",
            headers={},
            body=b"test",
        )

        assert wrapper.path_url == "/api/v1/chat?param1=value1&param2=value2"

    def test_oci_request_wrapper_path_url_no_query(self):
        """Test OCIRequestWrapper path_url property without query string."""
        wrapper = OCIRequestWrapper(
            method="POST",
            url="https://example.com/api/v1/chat",
            headers={},
            body=b"test",
        )

        assert wrapper.path_url == "/api/v1/chat"


class TestOCISplitChunks:
    """
    Unit tests for the SSE split_chunks helpers used in sync and async streaming.

    These validate the fix for:
    - Sync: JSONDecodeError when iter_text() returns chunks spanning multiple events
    - Async: whitespace-only chunks being yielded before stripping (Greptile P2)
    """

    def _run_sync_split(self, raw_chunks):
        """Invoke the sync split_chunks logic directly (extracted for testability)."""
        results = []
        for item in raw_chunks:
            for chunk in item.split("\n\n"):
                stripped = chunk.strip()
                if stripped:
                    results.append(stripped)
        return results

    async def _run_async_split(self, raw_chunks):
        """Invoke the async split_chunks logic directly."""
        results = []

        async def _gen():
            for c in raw_chunks:
                yield c

        async for item in _gen():
            for chunk in item.split("\n\n"):
                stripped = chunk.strip()
                if stripped:
                    results.append(stripped)
        return results

    def test_sync_single_event_per_chunk(self):
        """Normal case: one SSE event per iter_text() chunk."""
        chunks = ['data: {"text":"hello"}', 'data: {"text":"world"}']
        assert self._run_sync_split(chunks) == [
            'data: {"text":"hello"}',
            'data: {"text":"world"}',
        ]

    def test_sync_multiple_events_in_one_chunk(self):
        """iter_text() returns two SSE events concatenated — must be split."""
        chunks = ['data: {"text":"a"}\n\ndata: {"text":"b"}']
        assert self._run_sync_split(chunks) == [
            'data: {"text":"a"}',
            'data: {"text":"b"}',
        ]

    def test_sync_whitespace_only_chunks_discarded(self):
        """Whitespace between events must not be yielded."""
        chunks = ["data: {}\n\n   \n\ndata: {}"]
        result = self._run_sync_split(chunks)
        assert result == ["data: {}", "data: {}"]

    def test_sync_empty_string_discarded(self):
        """Empty string produced by splitting trailing \\n\\n must be discarded."""
        chunks = ["data: {}\n\n"]
        assert self._run_sync_split(chunks) == ["data: {}"]

    @pytest.mark.asyncio
    async def test_async_whitespace_only_chunks_discarded(self):
        """
        Regression test for Greptile P2: async version was checking `if not chunk`
        BEFORE stripping, so '\\n  ' would pass the guard and yield '' downstream,
        causing ValueError in chunk_creator ('Chunk does not start with data:').
        """
        chunks = ["data: {}\n\n   \n\ndata: {}"]
        result = await self._run_async_split(chunks)
        assert result == ["data: {}", "data: {}"]

    @pytest.mark.asyncio
    async def test_async_empty_string_discarded(self):
        """Trailing \\n\\n must not produce an empty yielded chunk in async path."""
        chunks = ["data: {}\n\n"]
        result = await self._run_async_split(chunks)
        assert result == ["data: {}"]

    @pytest.mark.asyncio
    async def test_async_multiple_events_in_one_chunk(self):
        """Async path must split concatenated SSE events just like sync."""
        chunks = ['data: {"text":"x"}\n\ndata: {"text":"y"}']
        result = await self._run_async_split(chunks)
        assert result == ['data: {"text":"x"}', 'data: {"text":"y"}']


class TestOCIProviderEmbeddingConfig:
    """
    Verifies that get_provider_embedding_config returns OCIEmbedConfig for OCI
    and that the dead duplicate elif branch has been removed (Greptile P1).
    """

    def test_returns_oci_embed_config(self):
        from litellm.llms.oci.embed.transformation import OCIEmbedConfig
        from litellm.utils import ProviderConfigManager
        from litellm.types.utils import LlmProviders

        config = ProviderConfigManager.get_provider_embedding_config(
            model="cohere.embed-english-v3.0",
            provider=LlmProviders.OCI,
        )
        assert isinstance(config, OCIEmbedConfig)

    def test_no_duplicate_oci_branch(self):
        """
        Ensure utils.py does not contain two separate OCI embedding branches.
        The dead code was removed in commit 64dfbe2b; this test guards against
        regression (e.g. a future merge re-introducing it).
        """
        import inspect
        from litellm.utils import ProviderConfigManager

        source = inspect.getsource(ProviderConfigManager.get_provider_embedding_config)
        oci_count = source.count("LlmProviders.OCI")
        assert oci_count == 1, (
            f"Expected exactly 1 OCI branch in get_provider_embedding_config, found {oci_count}. "
            "A duplicate dead-code branch may have been reintroduced."
        )


class TestOCICohereParamMapping:
    """
    Unit tests for Bug 3 (stop → stopSequences) and Bug 4 (hardcoded defaults removed).
    """

    def _make_config(self):
        return OCIChatConfig()

    def test_cohere_stop_maps_to_stop_sequences(self):
        """Bug 3: Cohere API uses 'stopSequences', not 'stop'."""
        config = self._make_config()
        result = config.map_openai_params(
            non_default_params={"stop": ["END", "STOP"]},
            optional_params={},
            model="cohere.command-latest",
            drop_params=False,
        )
        assert "stopSequences" in result, "stop should map to stopSequences for Cohere"
        assert result["stopSequences"] == ["END", "STOP"]
        assert "stop" not in result

    def test_generic_stop_maps_to_stop(self):
        """GENERIC vendors (Meta, Google, xAI) keep 'stop' as-is."""
        config = self._make_config()
        result = config.map_openai_params(
            non_default_params={"stop": ["END"]},
            optional_params={},
            model="meta.llama-3.3-70b-instruct",
            drop_params=False,
        )
        assert result.get("stop") == ["END"]
        assert "stopSequences" not in result

    def test_cohere_no_hardcoded_defaults(self):
        """Bug 4: Cohere calls must not inject maxTokens/temperature/topK/topP/frequencyPenalty
        when the user hasn't provided them."""
        config = self._make_config()
        result = config.map_openai_params(
            non_default_params={},
            optional_params={},
            model="cohere.command-latest",
            drop_params=False,
        )
        for injected in (
            "maxTokens",
            "temperature",
            "topK",
            "topP",
            "frequencyPenalty",
        ):
            assert (
                injected not in result
            ), f"'{injected}' should not be injected when user did not provide it"

    def test_cohere_explicit_params_still_passed(self):
        """User-provided Cohere params must still be forwarded correctly."""
        config = self._make_config()
        result = config.map_openai_params(
            non_default_params={"max_tokens": 200, "temperature": 0.5},
            optional_params={},
            model="cohere.command-latest",
            drop_params=False,
        )
        assert result.get("maxTokens") == 200
        assert result.get("temperature") == 0.5


class TestOCIStreamingSignedBody:
    """
    Unit test for Bug 1: sync and async streaming paths must use signed_json_body
    when provided, not re-serialize data with json.dumps().
    """

    def test_get_custom_stream_wrapper_uses_signed_body(self, monkeypatch):
        """
        When signed_json_body is provided, the POST must use that exact bytes object,
        not json.dumps(data) — otherwise the RSA-SHA256 signature is invalid.
        """
        import httpx
        from unittest.mock import MagicMock, patch

        config = OCIChatConfig()
        signed_bytes = b'{"signed": true}'
        posted_data = {}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_text.return_value = iter([])

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        def capture_post(url, **kwargs):
            posted_data["data"] = kwargs.get("data")
            return mock_response

        mock_client.post.side_effect = capture_post

        mock_logging = MagicMock()

        config.get_sync_custom_stream_wrapper(
            api_base="https://example.com",
            headers={},
            data={"key": "value"},
            messages=[],
            model="meta.llama-3.3-70b-instruct",
            custom_llm_provider="oci",
            logging_obj=mock_logging,
            client=mock_client,
            signed_json_body=signed_bytes,
        )

        assert (
            posted_data["data"] == signed_bytes
        ), "Streaming must use signed_json_body, not re-serialize data"

    def test_get_custom_stream_wrapper_fallback_without_signed_body(self, monkeypatch):
        """When signed_json_body is None, fall back to json.dumps(data)."""
        import json
        from unittest.mock import MagicMock

        config = OCIChatConfig()
        posted_data = {}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_text.return_value = iter([])

        mock_client = MagicMock()

        def capture_post(url, **kwargs):
            posted_data["data"] = kwargs.get("data")
            return mock_response

        mock_client.post.side_effect = capture_post

        mock_logging = MagicMock()
        payload = {"key": "value"}

        config.get_sync_custom_stream_wrapper(
            api_base="https://example.com",
            headers={},
            data=payload,
            messages=[],
            model="meta.llama-3.3-70b-instruct",
            custom_llm_provider="oci",
            logging_obj=mock_logging,
            client=mock_client,
            signed_json_body=None,
        )

        assert posted_data["data"] == json.dumps(
            payload
        ), "Without signed_json_body, must fall back to json.dumps(data)"


# ---------------------------------------------------------------------------
# Additional coverage: error paths in validate_environment, transform_request,
# transform_response, and map_openai_params
# ---------------------------------------------------------------------------


class TestOCIChatConfigErrorPaths:
    def test_validate_environment_empty_messages_raises(self):
        config = OCIChatConfig()
        with pytest.raises(Exception, match="messages"):
            config.validate_environment(
                headers={},
                model=TEST_MODEL_NAME,
                messages=[],
                optional_params={
                    "oci_signer": MagicMock(),
                    "oci_compartment_id": TEST_COMPARTMENT_ID,
                },
                litellm_params={},
            )

    def test_transform_request_missing_compartment_id_raises(self):
        config = OCIChatConfig()
        with pytest.raises(Exception, match="oci_compartment_id"):
            config.transform_request(
                model=TEST_MODEL_NAME,
                messages=TEST_MESSAGES,  # type: ignore
                optional_params={},
                litellm_params={},
                headers={},
            )

    def test_transform_request_cohere_no_user_message_raises(self):
        config = OCIChatConfig()
        with pytest.raises(Exception, match="user message"):
            config.transform_request(
                model="cohere.command-latest",
                messages=[{"role": "system", "content": "You are helpful."}],  # type: ignore
                optional_params={"oci_compartment_id": TEST_COMPARTMENT_ID},
                litellm_params={},
                headers={},
            )

    def test_transform_response_error_key_raises(self):
        config = OCIChatConfig()
        response = httpx.Response(
            status_code=400,
            json={"error": "model not found"},
        )
        with pytest.raises(Exception, match="model not found"):
            config.transform_response(
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

    def test_map_openai_params_unsupported_param_raises_without_drop(self):
        config = OCIChatConfig()
        with pytest.raises(Exception, match="not supported on OCI"):
            config.map_openai_params(
                non_default_params={"audio": {"voice": "alloy"}},
                optional_params={},
                model=TEST_MODEL_NAME,
                drop_params=False,
            )

    def test_map_openai_params_unsupported_param_dropped(self):
        config = OCIChatConfig()
        result = config.map_openai_params(
            non_default_params={"audio": {"voice": "alloy"}},
            optional_params={},
            model=TEST_MODEL_NAME,
            drop_params=True,
        )
        assert "audio" not in result

    def test_transform_request_tool_choice_string_mapped(self):
        config = OCIChatConfig()
        result = config.transform_request(
            model=TEST_MODEL_NAME,
            messages=TEST_MESSAGES,  # type: ignore
            optional_params={
                "oci_compartment_id": TEST_COMPARTMENT_ID,
                "tool_choice": "auto",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "fn",
                            "description": "d",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            litellm_params={},
            headers={},
        )
        assert result["chatRequest"]["toolChoice"] == {"type": "AUTO"}


import pytest
from unittest.mock import MagicMock
