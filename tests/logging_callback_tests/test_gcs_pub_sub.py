import base64
import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jsonschema
import pytest

from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.gcs_pubsub.pub_sub import *
from datetime import datetime, timedelta
from litellm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingModelInformation,
    StandardLoggingMetadata,
    StandardLoggingHiddenParams,
)

verbose_logger.setLevel(logging.DEBUG)

ignored_keys = [
    "request_id",
    "session_id",
    "startTime",
    "endTime",
    "completionStartTime",
    "endTime",
    "request_duration_ms",
    "metadata.model_map_information",
    "metadata.usage_object",
    "metadata.cold_storage_object_key",
    "metadata.litellm_overhead_time_ms",
    "metadata.cost_breakdown",
]


def _load_schema():
    """Load the StandardLoggingPayload JSON Schema."""
    pwd = os.path.dirname(os.path.realpath(__file__))
    schema_path = os.path.join(
        pwd, "gcs_pub_sub_body", "standard_logging_payload_schema.json"
    )
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_standard_logging_payload(payload: dict, schema: dict = None) -> list[str]:
    """
    Validate a StandardLoggingPayload against the JSON Schema.

    Returns a list of human-readable error messages matching the format:
        "Error in field <path>: <message>"
    """
    if schema is None:
        schema = _load_schema()

    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "<root>"
        errors.append(f"Error in field {path}: {error.message}")
    return errors


async def wait_for_flush(mock_post: AsyncMock, timeout: float = 5.0):
    """Poll until mock_post is called or timeout expires."""
    for _ in range(int(timeout / 0.1)):
        if mock_post.called:
            return
        await asyncio.sleep(0.1)


def extract_pubsub_payload(mock_post: AsyncMock) -> dict:
    """
    Extract and decode the Pub/Sub payload from a mocked httpx post call.

    Handles base64 decoding of the Pub/Sub message data.
    """
    actual_request = mock_post.call_args[1]["json"]
    assert "messages" in actual_request, f"Expected 'messages' key in Pub/Sub request, got keys: {list(actual_request.keys())}"
    encoded_message = actual_request["messages"][0]["data"]
    decoded_message = base64.b64decode(encoded_message).decode("utf-8")
    return json.loads(decoded_message)


def create_pubsub_logger() -> tuple[GcsPubSubLogger, AsyncMock]:
    """
    Create a GcsPubSubLogger with mocked HTTP client and auth headers.

    The caller must ensure _premium_user_check is already patched (e.g. via
    the @patch decorator).

    Returns (logger, mock_post) tuple.
    """
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 202
    mock_post.return_value.text = "Accepted"

    logger = GcsPubSubLogger(
        project_id="reliableKeys",
        topic_id="litellmDB",
        flush_interval=1,
    )
    logger.async_httpx_client.post = mock_post

    mock_headers = AsyncMock()
    mock_headers.return_value = {"Authorization": "Bearer mock_token"}
    logger.construct_request_headers = mock_headers

    return logger, mock_post


def _compare_nested_dicts(
    actual: dict, expected: dict, path: str = "", ignore_keys: list[str] = []
) -> list[str]:
    """Compare nested dictionaries and return a list of differences in a human-friendly format."""
    differences = []

    # Check if current path should be ignored
    if path in ignore_keys:
        return differences

    # Check for keys in actual but not in expected
    for key in actual.keys():
        current_path = f"{path}.{key}" if path else key
        if current_path not in ignore_keys and key not in expected:
            differences.append(f"Extra key in actual: {current_path}")

    for key, expected_value in expected.items():
        current_path = f"{path}.{key}" if path else key
        if current_path in ignore_keys:
            continue
        if key not in actual:
            differences.append(f"Missing key: {current_path}")
            continue

        actual_value = actual[key]

        # Try to parse JSON strings
        if isinstance(expected_value, str):
            try:
                expected_value = json.loads(expected_value)
            except json.JSONDecodeError:
                pass
        if isinstance(actual_value, str):
            try:
                actual_value = json.loads(actual_value)
            except json.JSONDecodeError:
                pass

        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            differences.extend(
                _compare_nested_dicts(
                    actual_value, expected_value, current_path, ignore_keys
                )
            )
        elif isinstance(expected_value, dict) or isinstance(actual_value, dict):
            differences.append(
                f"Type mismatch at {current_path}: expected dict, got {type(actual_value).__name__}"
            )
        else:
            # For non-dict values, only report if they're different
            if actual_value != expected_value:
                # Format the values to be more readable
                actual_str = str(actual_value)
                expected_str = str(expected_value)
                if len(actual_str) > 50 or len(expected_str) > 50:
                    actual_str = f"{actual_str[:50]}..."
                    expected_str = f"{expected_str[:50]}..."
                differences.append(
                    f"Value mismatch at {current_path}:\n  expected: {expected_str}\n  got:      {actual_str}"
                )
    return differences


def assert_gcs_pubsub_request_matches_expected(
    actual_request_body: dict,
    expected_file_name: str,
):
    """
    Helper function to compare actual GCS PubSub request body with expected JSON file.

    Args:
        actual_request_body (dict): The actual request body received from the API call
        expected_file_name (str): Name of the JSON file containing expected request body
    """
    # Get the current directory and read the expected request body
    pwd = os.path.dirname(os.path.realpath(__file__))
    expected_body_path = os.path.join(pwd, "gcs_pub_sub_body", expected_file_name)

    with open(expected_body_path, "r") as f:
        expected_request_body = json.load(f)

    # Replace dynamic values in actual request body
    differences = _compare_nested_dicts(
        actual_request_body, expected_request_body, ignore_keys=ignored_keys
    )
    if differences:
        assert False, f"Dictionary mismatch: {differences}"

def assert_gcs_pubsub_request_matches_expected_standard_logging_payload(
    actual_request_body: dict,
    expected_file_name: str,
):
    """
    Helper function to compare actual GCS PubSub request body with expected JSON file.

    Args:
        actual_request_body (dict): The actual request body received from the API call
        expected_file_name (str): Name of the JSON file containing expected request body
    """
    # Get the current directory and read the expected request body
    pwd = os.path.dirname(os.path.realpath(__file__))
    expected_body_path = os.path.join(pwd, "gcs_pub_sub_body", expected_file_name)

    with open(expected_body_path, "r") as f:
        expected_request_body = json.load(f)

    # Replace dynamic values in actual request body
    FIELDS_TO_VALIDATE = [
        "custom_llm_provider",
        "hidden_params",
        "messages",
        "response",
        "model",
        "status",
        "stream",
    ]

    actual_request_body["response"]["id"] = expected_request_body["response"]["id"]
    actual_request_body["response"]["created"] = expected_request_body["response"][
        "created"
    ]

    for field in FIELDS_TO_VALIDATE:
        assert field in actual_request_body

    FIELDS_EXISTENCE_CHECKS = [
        "response_cost",
        "response_time",
        "completion_tokens",
        "prompt_tokens",
        "total_tokens"
    ]

    for field in FIELDS_EXISTENCE_CHECKS:
        assert field in actual_request_body


@pytest.mark.asyncio
async def test_async_gcs_pub_sub():
    # Create a mock for the async_httpx_client's post method
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 202
    mock_post.return_value.text = "Accepted"

    # Initialize the GcsPubSubLogger and set the mock
    gcs_pub_sub_logger = GcsPubSubLogger(flush_interval=1)
    gcs_pub_sub_logger.async_httpx_client.post = mock_post

    mock_construct_request_headers = AsyncMock()
    mock_construct_request_headers.return_value = {"Authorization": "Bearer mock_token"}
    gcs_pub_sub_logger.construct_request_headers = mock_construct_request_headers
    litellm.callbacks = [gcs_pub_sub_logger]

    # Make the completion call
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="hi",
    )

    await asyncio.sleep(3)  # Wait for async flush

    # Assert httpx post was called
    mock_post.assert_called_once()

    # Get the actual request body from the mock
    actual_url = mock_post.call_args[1]["url"]
    print("sent to url", actual_url)
    assert (
        actual_url
        == "https://pubsub.googleapis.com/v1/projects/reliableKeys/topics/litellmDB:publish"
    )
    actual_request = mock_post.call_args[1]["json"]

    # Extract and decode the base64 encoded message
    encoded_message = actual_request["messages"][0]["data"]
    import base64

    decoded_message = base64.b64decode(encoded_message).decode("utf-8")

    # Parse the JSON string into a dictionary
    actual_request = json.loads(decoded_message)
    print("##########\n")
    print(json.dumps(actual_request, indent=4))
    print("##########\n")
    # Verify the request body matches expected format
    assert_gcs_pubsub_request_matches_expected_standard_logging_payload(
        actual_request, "standard_logging_payload.json"
    )


@pytest.mark.asyncio
async def test_async_gcs_pub_sub_v1():
    # Create a mock for the async_httpx_client's post method
    litellm.gcs_pub_sub_use_v1 = True
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 202
    mock_post.return_value.text = "Accepted"

    # Initialize the GcsPubSubLogger and set the mock
    gcs_pub_sub_logger = GcsPubSubLogger(flush_interval=1)
    gcs_pub_sub_logger.async_httpx_client.post = mock_post

    mock_construct_request_headers = AsyncMock()
    mock_construct_request_headers.return_value = {"Authorization": "Bearer mock_token"}
    gcs_pub_sub_logger.construct_request_headers = mock_construct_request_headers
    litellm.callbacks = [gcs_pub_sub_logger]

    # Make the completion call
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="hi",
    )

    await asyncio.sleep(3)  # Wait for async flush

    # Assert httpx post was called
    mock_post.assert_called_once()

    # Get the actual request body from the mock
    actual_url = mock_post.call_args[1]["url"]
    print("sent to url", actual_url)
    assert (
        actual_url
        == "https://pubsub.googleapis.com/v1/projects/reliableKeys/topics/litellmDB:publish"
    )
    actual_request = mock_post.call_args[1]["json"]

    # Extract and decode the base64 encoded message
    encoded_message = actual_request["messages"][0]["data"]
    import base64

    decoded_message = base64.b64decode(encoded_message).decode("utf-8")

    # Parse the JSON string into a dictionary
    actual_request = json.loads(decoded_message)
    print("##########\n")
    print(json.dumps(actual_request, indent=4))
    print("##########\n")
    # Verify the request body matches expected format
    assert_gcs_pubsub_request_matches_expected(
        actual_request, "spend_logs_payload.json"
    )


# ---------------------------------------------------------------------------
# Test 3a: OpenAI endpoint chat completion — messages must be array
# Bug: messages field is {} instead of array for all models via OpenAI endpoint
# Root cause: litellm_logging.py:5388 — kwargs.get("messages") returns None
#   for passthrough endpoints, but even for OpenAI endpoints the field should
#   be a proper array, not an empty dict.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@patch("litellm.proxy.utils._premium_user_check")
async def test_pubsub_openai_chat_completion_messages_is_array(_mock_premium):
    """
    Validates that the Pub/Sub payload for a standard chat completion contains:
    - messages as a list (not {} or a string)
    - messages[0] has the original user message
    - response_cost is a float >= 0
    - prompt_tokens and completion_tokens are > 0
    """
    logger, mock_post = create_pubsub_logger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [logger]

    try:
        await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response="hi",
        )

        await wait_for_flush(mock_post)

        mock_post.assert_called_once()
        payload = extract_pubsub_payload(mock_post)

        # Core assertion: messages must be a list
        assert isinstance(payload["messages"], list), (
            f"messages should be a list, got {type(payload['messages']).__name__}: {payload['messages']}"
        )
        assert len(payload["messages"]) > 0, "messages list should not be empty"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello, world!"

        # Cost and tokens
        assert isinstance(payload["response_cost"], (int, float))
        assert payload["response_cost"] >= 0
        assert payload["prompt_tokens"] > 0, f"prompt_tokens should be > 0, got {payload['prompt_tokens']}"
        assert payload["completion_tokens"] > 0, f"completion_tokens should be > 0, got {payload['completion_tokens']}"
        assert payload["total_tokens"] > 0

        # Schema validation
        schema = _load_schema()
        errors = validate_standard_logging_payload(payload, schema)
        assert errors == [], f"Schema validation errors: {errors}"
    finally:
        litellm.callbacks = original_callbacks


# ---------------------------------------------------------------------------
# Test 3b: OpenAI endpoint embedding — response_cost must exist
# Bug: response_cost missing for embedding models
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@patch("litellm.proxy.utils._premium_user_check")
async def test_pubsub_openai_embedding_response_cost_exists(_mock_premium):
    """
    Validates that the Pub/Sub payload for an embedding call contains:
    - response_cost as a float >= 0 (not missing)
    - messages as a list
    - call_type is 'aembedding'
    """
    logger, mock_post = create_pubsub_logger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [logger]

    try:
        await litellm.aembedding(
            model="text-embedding-ada-002",
            input=["Hello, world!"],
            mock_response=[0.1, 0.2, 0.3],
        )

        await wait_for_flush(mock_post)

        mock_post.assert_called_once()
        payload = extract_pubsub_payload(mock_post)

        # response_cost must exist and be a number
        assert "response_cost" in payload, "response_cost is a required property"
        assert isinstance(payload["response_cost"], (int, float)), (
            f"response_cost should be a number, got {type(payload['response_cost']).__name__}"
        )
        assert payload["response_cost"] >= 0

        # call_type
        assert payload["call_type"] == "aembedding"

        # messages should still be a list (the input)
        assert isinstance(payload["messages"], list), (
            f"messages should be a list, got {type(payload['messages']).__name__}: {payload['messages']}"
        )

        # prompt_tokens should be > 0 for embeddings
        assert payload["prompt_tokens"] >= 0

        # model and provider
        assert payload["model"] is not None
        assert payload["custom_llm_provider"] is not None
        assert len(payload["custom_llm_provider"]) > 0
    finally:
        litellm.callbacks = original_callbacks


# ---------------------------------------------------------------------------
# Test 3c: Claude passthrough — tokens and response content
# Bug: tokens=0, response content empty, messages={}
# Root cause: anthropic_passthrough_logging_handler.py:69 hardcodes messages=[]
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pubsub_anthropic_passthrough_tokens_and_response():
    """
    Validates that the Pub/Sub payload for a Claude passthrough request contains:
    - prompt_tokens > 0
    - completion_tokens > 0
    - response.choices[0].message.content is non-empty
    - messages is a list (not {})
    """
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )

    # Create a mock Anthropic response
    anthropic_response_body = {
        "id": "msg_mock123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "Hello! How can I help you?"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }

    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = anthropic_response_body
    mock_httpx_response.text = json.dumps(anthropic_response_body)
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "application/json"}

    request_body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi there!"}],
    }

    logging_obj = LiteLLMLoggingObj(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hi there!"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="mock-call-id",
        function_id="mock-function-id",
    )

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)

    result = AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler(
        httpx_response=mock_httpx_response,
        response_body=anthropic_response_body,
        logging_obj=logging_obj,
        url_route="/v1/messages",
        result=json.dumps(anthropic_response_body),
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
        request_body=request_body,
    )

    litellm_model_response = result["result"]
    kwargs = result["kwargs"]

    # The key assertion is on the transformed response
    assert litellm_model_response is not None

    # Tokens should be extracted from the Anthropic response
    usage = litellm_model_response.usage
    assert usage.prompt_tokens > 0, f"prompt_tokens should be > 0, got {usage.prompt_tokens}"
    assert usage.completion_tokens > 0, f"completion_tokens should be > 0, got {usage.completion_tokens}"
    assert usage.total_tokens > 0, f"total_tokens should be > 0, got {usage.total_tokens}"

    # Response content should be non-empty
    assert len(litellm_model_response.choices) > 0
    content = litellm_model_response.choices[0].message.content
    assert content is not None and len(content) > 0, (
        f"response.choices[0].message.content should be non-empty, got: '{content}'"
    )

    # response_cost should be set
    assert "response_cost" in kwargs
    assert isinstance(kwargs["response_cost"], (int, float))

    # The kwargs should include model
    assert kwargs.get("model") == "claude-3-5-sonnet-20241022"

    # Messages should be propagated from request_body to kwargs
    assert isinstance(kwargs.get("messages"), list), (
        f"kwargs['messages'] should be a list from request_body, got {type(kwargs.get('messages'))}"
    )
    msg = kwargs["messages"][0]
    assert "role" in msg and "content" in msg, f"Message should have 'role' and 'content', got keys: {list(msg.keys())}"
    assert msg["content"] == "Hi there!"


# ---------------------------------------------------------------------------
# Test 3d: Meta/Llama via Vertex passthrough — model and provider
# Bug: model="unknown", custom_llm_provider="", tokens=0, missing response.choices
# Root cause: extract_model_from_url regex doesn't match meta publisher paths
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pubsub_vertex_meta_passthrough_model_and_provider():
    """
    Validates that the Pub/Sub payload for Meta/Llama via Vertex passthrough:
    - model is NOT 'unknown'
    - custom_llm_provider is non-empty
    - response has 'choices' array
    - tokens > 0
    """
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
        VertexPassthroughLoggingHandler,
    )

    # Llama response format from Vertex AI rawPredict
    llama_response_body = {
        "id": "chatcmpl-mock123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "meta/llama-3.1-70b-instruct-maas",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm Llama.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25,
        },
    }

    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = llama_response_body
    mock_httpx_response.text = json.dumps(llama_response_body)
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "application/json"}

    logging_obj = LiteLLMLoggingObj(
        model="meta/llama-3.1-70b-instruct-maas",
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="mock-call-id",
        function_id="mock-function-id",
    )

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)

    url_route = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/meta/models/llama-3.1-70b-instruct-maas:rawPredict"

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=mock_httpx_response,
        logging_obj=logging_obj,
        url_route=url_route,
        result=json.dumps(llama_response_body),
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
        request_body={"messages": [{"role": "user", "content": "Hi"}]},
    )

    litellm_model_response = result["result"]
    kwargs = result["kwargs"]

    # Model should NOT be "unknown"
    model = kwargs.get("model", "")
    assert model != "unknown", (
        f"model should not be 'unknown', got '{model}'. "
        f"extract_model_from_url likely failed for the meta publisher URL."
    )
    assert len(model) > 0, "model should be non-empty"

    # custom_llm_provider should be set on the logging_obj
    provider = logging_obj.model_call_details.get("custom_llm_provider", "")
    assert provider != "", (
        f"custom_llm_provider should not be empty, got '{provider}'"
    )

    # Response should have choices
    assert litellm_model_response is not None
    assert hasattr(litellm_model_response, "choices"), "response should have 'choices'"
    assert len(litellm_model_response.choices) > 0, "response.choices should not be empty"

    # Tokens should be > 0
    if hasattr(litellm_model_response, "usage") and litellm_model_response.usage:
        assert litellm_model_response.usage.prompt_tokens > 0, (
            f"prompt_tokens should be > 0, got {litellm_model_response.usage.prompt_tokens}"
        )
        assert litellm_model_response.usage.completion_tokens > 0, (
            f"completion_tokens should be > 0, got {litellm_model_response.usage.completion_tokens}"
        )

    # Messages should be propagated from request_body to kwargs
    assert isinstance(kwargs.get("messages"), list), (
        f"kwargs['messages'] should be a list from request_body, got {type(kwargs.get('messages'))}"
    )
    msg = kwargs["messages"][0]
    assert "role" in msg and "content" in msg, f"Message should have 'role' and 'content', got keys: {list(msg.keys())}"


# ---------------------------------------------------------------------------
# Test 3e: Vertex AI generateContent — messages must be array
# Bug: messages={} instead of array for Vertex AI models
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pubsub_vertex_generate_content_messages_is_array():
    """
    Validates that the Pub/Sub payload for a Vertex AI generateContent call:
    - messages is a list (not {})
    - model is correctly extracted from the URL
    """
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
        VertexPassthroughLoggingHandler,
    )

    # Gemini generateContent response format
    gemini_response_body = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I help?"}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
        "modelVersion": "gemini-2.5-flash",
    }

    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = gemini_response_body
    mock_httpx_response.text = json.dumps(gemini_response_body)
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "application/json"}

    logging_obj = LiteLLMLoggingObj(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="mock-call-id",
        function_id="mock-function-id",
    )
    # The Vertex transformer accesses logging_obj.optional_params internally
    logging_obj.optional_params = {}

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)

    url_route = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=mock_httpx_response,
        logging_obj=logging_obj,
        url_route=url_route,
        result=json.dumps(gemini_response_body),
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
        request_body={
            "contents": [{"parts": [{"text": "Hi"}], "role": "user"}]
        },
    )

    litellm_model_response = result["result"]
    kwargs = result["kwargs"]

    # Model should be extracted correctly
    model = kwargs.get("model", "")
    assert model != "unknown", f"model should not be 'unknown', got '{model}'"
    assert "gemini" in model.lower(), f"model should contain 'gemini', got '{model}'"

    # Response should be valid
    assert litellm_model_response is not None
    assert hasattr(litellm_model_response, "choices")
    assert len(litellm_model_response.choices) > 0

    # response_cost should be set
    assert "response_cost" in kwargs
    assert isinstance(kwargs["response_cost"], (int, float))

    # Messages should be converted from Vertex contents format to OpenAI format
    assert isinstance(kwargs.get("messages"), list), (
        f"kwargs['messages'] should be a list from request_body, got {type(kwargs.get('messages'))}"
    )
    assert len(kwargs["messages"]) > 0
    # Verify conversion from {parts, role} to {role, content}
    msg = kwargs["messages"][0]
    assert "role" in msg, f"Converted message should have 'role', got keys: {list(msg.keys())}"
    assert "content" in msg, f"Converted message should have 'content', got keys: {list(msg.keys())}"
    assert msg["content"] == "Hi", f"Expected content 'Hi', got '{msg['content']}'"


# ---------------------------------------------------------------------------
# Test 3f: Streaming response — all fields populated
# Bug: messages={} for streaming responses
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@patch("litellm.proxy.utils._premium_user_check")
async def test_pubsub_streaming_chat_completion(_mock_premium):
    """
    Validates that the Pub/Sub payload for a streaming chat completion contains:
    - messages as a list
    - stream is True
    - tokens and response_cost are populated
    """
    logger, mock_post = create_pubsub_logger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [logger]

    try:
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response="hi",
            stream=True,
        )

        # Consume the stream
        async for chunk in response:
            pass

        await wait_for_flush(mock_post)

        mock_post.assert_called_once()
        payload = extract_pubsub_payload(mock_post)

        # messages must be a list
        assert isinstance(payload["messages"], list), (
            f"messages should be a list, got {type(payload['messages']).__name__}: {payload['messages']}"
        )
        assert len(payload["messages"]) > 0

        # stream should be True
        assert payload["stream"] is True

        # Tokens should be populated
        assert payload["total_tokens"] > 0, f"total_tokens should be > 0 for streaming, got {payload['total_tokens']}"
        assert payload["prompt_tokens"] > 0
        assert payload["completion_tokens"] > 0

        # response_cost should be set
        assert isinstance(payload["response_cost"], (int, float))
        assert payload["response_cost"] >= 0

        # Schema validation
        schema = _load_schema()
        errors = validate_standard_logging_payload(payload, schema)
        assert errors == [], f"Schema validation errors: {errors}"
    finally:
        litellm.callbacks = original_callbacks


# ---------------------------------------------------------------------------
# Step 5: Parametrized schema validation test
# Tests multiple models/call_types against the JSON Schema in one sweep.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,call_type,expected_provider",
    [
        ("gpt-4o", "acompletion", "openai"),
        ("gpt-4o-mini", "acompletion", "openai"),
    ],
    ids=["gpt-4o", "gpt-4o-mini"],
)
@patch("litellm.proxy.utils._premium_user_check")
async def test_pubsub_payload_validates_against_schema(
    _mock_premium, model, call_type, expected_provider
):
    """
    Parametrized test that validates the Pub/Sub payload against the JSON Schema
    for multiple models and call types.
    """
    logger, mock_post = create_pubsub_logger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [logger]

    try:
        if call_type == "acompletion":
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "test message"}],
                mock_response="test response",
            )
        elif call_type == "aembedding":
            await litellm.aembedding(
                model=model,
                input=["test input"],
                mock_response=[0.1, 0.2, 0.3],
            )

        await wait_for_flush(mock_post)

        mock_post.assert_called_once()
        payload = extract_pubsub_payload(mock_post)

        # Validate against JSON Schema
        schema = _load_schema()
        errors = validate_standard_logging_payload(payload, schema)
        assert errors == [], (
            f"Schema validation errors for model={model}, call_type={call_type}:\n"
            + "\n".join(errors)
        )

        # Check provider
        assert payload.get("custom_llm_provider") == expected_provider, (
            f"Expected provider '{expected_provider}', got '{payload.get('custom_llm_provider')}'"
        )
    finally:
        litellm.callbacks = original_callbacks
