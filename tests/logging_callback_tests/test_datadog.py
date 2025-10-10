import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.datadog.datadog import *
from datetime import datetime, timedelta
from litellm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingModelInformation,
    StandardLoggingMetadata,
    StandardLoggingHiddenParams,
    LiteLLMCommonStrings,
)
from litellm.types.integrations.datadog import DatadogInitParams

verbose_logger.setLevel(logging.DEBUG)


def create_standard_logging_payload() -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        call_type="completion",
        response_cost=0.1,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=30,
        prompt_tokens=20,
        completion_tokens=10,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        model_map_information=StandardLoggingModelInformation(
            model_map_key="gpt-3.5-turbo", model_map_value=None
        ),
        model="gpt-3.5-turbo",
        model_id="model-123",
        model_group="openai-gpt",
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_org_id=None,
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_team_alias="test_team_alias",
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
        ),
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        model_parameters={"stream": True},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.1",
            additional_headers=None,
        ),
    )


@pytest.mark.asyncio
async def test_create_datadog_logging_payload():
    """Test creating a DataDog logging payload from a standard logging object"""
    dd_logger = DataDogLogger()
    standard_payload = create_standard_logging_payload()

    # Create mock kwargs with the standard logging object
    kwargs = {"standard_logging_object": standard_payload}

    # Test payload creation
    dd_payload = dd_logger.create_datadog_logging_payload(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    # Verify payload structure
    assert dd_payload["ddsource"] == os.getenv("DD_SOURCE", "litellm")
    assert dd_payload["service"] == "litellm-server"
    assert dd_payload["status"] == DataDogStatus.INFO

    # verify the message field == standard_payload
    dict_payload = json.loads(dd_payload["message"])
    assert dict_payload == standard_payload


@pytest.mark.asyncio
async def test_datadog_failure_logging():
    """Test logging a failure event to DataDog"""
    dd_logger = DataDogLogger()
    standard_payload = create_standard_logging_payload()
    standard_payload["status"] = "failure"  # Set status to failure
    standard_payload["error_str"] = "Test error"

    kwargs = {"standard_logging_object": standard_payload}

    dd_payload = dd_logger.create_datadog_logging_payload(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert (
        dd_payload["status"] == DataDogStatus.ERROR
    )  # Verify failure maps to warning status

    # verify the message field == standard_payload
    dict_payload = json.loads(dd_payload["message"])
    assert dict_payload == standard_payload

    # verify error_str is in the message field
    assert "error_str" in dict_payload
    assert dict_payload["error_str"] == "Test error"


@pytest.mark.asyncio
async def test_datadog_logging_http_request():
    """
    - Test that the HTTP request is made to Datadog
    - sent to the /api/v2/logs endpoint
    - the payload is batched
    - each element in the payload is a DatadogPayload
    - each element in a DatadogPayload.message contains all the valid fields
    """
    try:
        from litellm.integrations.datadog.datadog import DataDogLogger

        os.environ["DD_SITE"] = "https://fake.datadoghq.com"
        os.environ["DD_API_KEY"] = "anything"
        dd_logger = DataDogLogger()

        litellm.callbacks = [dd_logger]

        litellm.set_verbose = True

        # Create a mock for the async_client's post method
        mock_post = AsyncMock()
        mock_post.return_value.status_code = 202
        mock_post.return_value.text = "Accepted"
        dd_logger.async_client.post = mock_post

        # Make the completion call
        for _ in range(5):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "what llm are u"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="Accepted",
            )
            print(response)

        # Wait for 5 seconds
        await asyncio.sleep(6)

        # Assert that the mock was called
        assert mock_post.called, "HTTP request was not made"

        # Get the arguments of the last call
        args, kwargs = mock_post.call_args

        print("CAll args and kwargs", args, kwargs)

        # Print the request body

        # You can add more specific assertions here if needed
        # For example, checking if the URL is correct
        assert kwargs["url"].endswith("/api/v2/logs"), "Incorrect DataDog endpoint"

        body = kwargs["data"]

        # use gzip to unzip the body
        with gzip.open(io.BytesIO(body), "rb") as f:
            body = f.read().decode("utf-8")
        print(body)

        # body is string parse it to dict
        body = json.loads(body)
        print(body)

        assert len(body) == 5  # 5 logs should be sent to DataDog

        # Assert that the first element in body has the expected fields and shape
        assert isinstance(body[0], dict), "First element in body should be a dictionary"

        # Get the expected fields and their types from DatadogPayload
        expected_fields = DatadogPayload.__annotations__
        # Assert that all elements in body have the fields of DatadogPayload with correct types
        for log in body:
            assert isinstance(log, dict), "Each log should be a dictionary"
            for field, expected_type in expected_fields.items():
                assert field in log, f"Field '{field}' is missing from the log"
                assert isinstance(
                    log[field], expected_type
                ), f"Field '{field}' has incorrect type. Expected {expected_type}, got {type(log[field])}"

        # Additional assertion to ensure no extra fields are present
        for log in body:
            assert set(log.keys()) == set(
                expected_fields.keys()
            ), f"Log contains unexpected fields: {set(log.keys()) - set(expected_fields.keys())}"

        # Parse the 'message' field as JSON and check its structure
        message = json.loads(body[0]["message"])
        print("logged message", json.dumps(message, indent=4))

        expected_message_fields = StandardLoggingPayload.__annotations__.keys()

        for field in expected_message_fields:
            assert field in message, f"Field '{field}' is missing from the message"

        # Check specific fields
        assert message["call_type"] == "acompletion"
        assert message["model"] == "gpt-3.5-turbo"
        assert isinstance(message["model_parameters"], dict)
        assert "temperature" in message["model_parameters"]
        assert "max_tokens" in message["model_parameters"]
        assert isinstance(message["response"], dict)
        assert isinstance(message["metadata"], dict)

    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")


@pytest.mark.asyncio
async def test_datadog_log_redis_failures():
    """
    Test that poorly configured Redis is logged as Warning on DataDog
    """
    try:
        from litellm.caching.caching import Cache
        from litellm.integrations.datadog.datadog import DataDogLogger

        litellm.cache = Cache(
            type="redis", host="badhost", port="6379", password="badpassword"
        )

        os.environ["DD_SITE"] = "https://fake.datadoghq.com"
        os.environ["DD_API_KEY"] = "anything"
        dd_logger = DataDogLogger()

        litellm.callbacks = [dd_logger]
        litellm.service_callback = ["datadog"]

        litellm.set_verbose = True

        # Create a mock for the async_client's post method
        mock_post = AsyncMock()
        mock_post.return_value.status_code = 202
        mock_post.return_value.text = "Accepted"
        dd_logger.async_client.post = mock_post

        # Make the completion call
        for _ in range(3):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "what llm are u"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="Accepted",
            )
            print(response)

        # Wait for 5 seconds
        await asyncio.sleep(6)

        # Assert that the mock was called
        assert mock_post.called, "HTTP request was not made"

        # Get the arguments of the last call
        args, kwargs = mock_post.call_args
        print("CAll args and kwargs", args, kwargs)

        # For example, checking if the URL is correct
        assert kwargs["url"].endswith("/api/v2/logs"), "Incorrect DataDog endpoint"

        body = kwargs["data"]

        # use gzip to unzip the body
        with gzip.open(io.BytesIO(body), "rb") as f:
            body = f.read().decode("utf-8")
        print(body)

        # body is string parse it to dict
        body = json.loads(body)
        print(body)

        failure_events = [log for log in body if log["status"] == "warning"]
        assert len(failure_events) > 0, "No failure events logged"

        print("ALL FAILURE/WARN EVENTS", failure_events)

        for event in failure_events:
            message = json.loads(event["message"])
            assert (
                event["status"] == "warning"
            ), f"Event status is not 'warning': {event['status']}"
            assert (
                message["service"] == "redis"
            ), f"Service is not 'redis': {message['service']}"
            assert "error" in message, "No 'error' field in the message"
            assert message["error"], "Error field is empty"
    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")


@pytest.mark.asyncio
@pytest.mark.skip(reason="local-only test, to test if everything works fine.")
async def test_datadog_logging():
    try:
        litellm.success_callback = ["datadog"]
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)

        await asyncio.sleep(5)
    except Exception as e:
        print(e)


@pytest.mark.asyncio
async def test_datadog_payload_environment_variables():
    """Test that DataDog payload correctly includes environment variables in the payload structure"""
    try:
        # Set test environment variables
        test_env = {
            "DD_ENV": "test-env",
            "DD_SERVICE": "test-service",
            "DD_VERSION": "1.0.0",
            "DD_SOURCE": "test-source",
            "DD_API_KEY": "fake-key",
            "DD_SITE": "datadoghq.com",
        }

        with patch.dict(os.environ, test_env):
            dd_logger = DataDogLogger()
            standard_payload = create_standard_logging_payload()

            # Create the payload
            dd_payload = dd_logger.create_datadog_logging_payload(
                kwargs={"standard_logging_object": standard_payload},
                response_obj=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

            print("dd payload=", json.dumps(dd_payload, indent=2))

            # Verify payload structure and environment variables
            assert (
                dd_payload["ddsource"] == "test-source"
            ), "Incorrect source in payload"
            assert (
                dd_payload["service"] == "test-service"
            ), "Incorrect service in payload"

            assert (
                "env:test-env,service:test-service,version:1.0.0,HOSTNAME:"
                in dd_payload["ddtags"]
            ), "Incorrect tags in payload"

    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")


@pytest.mark.asyncio
async def test_datadog_payload_content_truncation():
    """
    Test that DataDog payload correctly truncates long content

    DataDog has a limit of 1MB for the logged payload size.
    """
    dd_logger = DataDogLogger()

    # Create a standard payload with very long content
    standard_payload = create_standard_logging_payload()
    long_content = "x" * 80_000  # Create string longer than MAX_STR_LENGTH (10_000)

    # Modify payload with long content
    standard_payload["error_str"] = long_content
    standard_payload["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": long_content,
                        "detail": "low",
                    },
                }
            ],
        }
    ]
    standard_payload["response"] = {"choices": [{"message": {"content": long_content}}]}

    # Create the payload
    dd_payload = dd_logger.create_datadog_logging_payload(
        kwargs={"standard_logging_object": standard_payload},
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    print("dd_payload", json.dumps(dd_payload, indent=2))

    # Parse the message back to dict to verify truncation
    message_dict = json.loads(dd_payload["message"])

    # Verify truncation of fields
    assert len(message_dict["error_str"]) < 10_100, "error_str not truncated correctly"
    assert (
        len(str(message_dict["messages"])) < 10_100
    ), "messages not truncated correctly"
    assert (
        len(str(message_dict["response"])) < 10_100
    ), "response not truncated correctly"


def test_datadog_static_methods():
    """Test the static helper methods in DataDogLogger class"""

    # Test with default environment variables
    assert DataDogLogger._get_datadog_source() == "litellm"
    assert DataDogLogger._get_datadog_service() == "litellm-server"
    assert DataDogLogger._get_datadog_hostname() is not None
    assert DataDogLogger._get_datadog_env() == "unknown"
    assert DataDogLogger._get_datadog_pod_name() == "unknown"

    # Test tags format with default values
    assert (
        "env:unknown,service:litellm,version:unknown,HOSTNAME:"
        in DataDogLogger._get_datadog_tags()
    )

    # Test with custom environment variables
    test_env = {
        "DD_SOURCE": "custom-source",
        "DD_SERVICE": "custom-service",
        "HOSTNAME": "test-host",
        "DD_ENV": "production",
        "DD_VERSION": "1.0.0",
        "POD_NAME": "pod-123",
    }

    with patch.dict(os.environ, test_env):
        assert DataDogLogger._get_datadog_source() == "custom-source"
        print(
            "DataDogLogger._get_datadog_source()", DataDogLogger._get_datadog_source()
        )
        assert DataDogLogger._get_datadog_service() == "custom-service"
        print(
            "DataDogLogger._get_datadog_service()", DataDogLogger._get_datadog_service()
        )
        assert DataDogLogger._get_datadog_hostname() == "test-host"
        print(
            "DataDogLogger._get_datadog_hostname()",
            DataDogLogger._get_datadog_hostname(),
        )
        assert DataDogLogger._get_datadog_env() == "production"
        print("DataDogLogger._get_datadog_env()", DataDogLogger._get_datadog_env())
        assert DataDogLogger._get_datadog_pod_name() == "pod-123"
        print(
            "DataDogLogger._get_datadog_pod_name()",
            DataDogLogger._get_datadog_pod_name(),
        )

        # Test tags format with custom values
        expected_custom_tags = "env:production,service:custom-service,version:1.0.0,HOSTNAME:test-host,POD_NAME:pod-123"
        print("DataDogLogger._get_datadog_tags()", DataDogLogger._get_datadog_tags())
        assert DataDogLogger._get_datadog_tags() == expected_custom_tags


@pytest.mark.asyncio
async def test_datadog_non_serializable_messages():
    """Test logging events with non-JSON-serializable messages"""
    dd_logger = DataDogLogger()

    # Create payload with non-serializable content
    standard_payload = create_standard_logging_payload()
    non_serializable_obj = datetime.now()  # datetime objects aren't JSON serializable
    standard_payload["messages"] = [{"role": "user", "content": non_serializable_obj}]
    standard_payload["response"] = {
        "choices": [{"message": {"content": non_serializable_obj}}]
    }

    kwargs = {"standard_logging_object": standard_payload}

    # Test payload creation
    dd_payload = dd_logger.create_datadog_logging_payload(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    # Verify payload can be serialized
    assert dd_payload["status"] == DataDogStatus.INFO

    # Verify the message can be parsed back to dict
    dict_payload = json.loads(dd_payload["message"])

    # Check that the non-serializable objects were converted to strings
    assert isinstance(dict_payload["messages"][0]["content"], str)
    assert isinstance(dict_payload["response"]["choices"][0]["message"]["content"], str)


def test_get_datadog_tags():
    """Test the _get_datadog_tags static method with various inputs"""
    # Test with no standard_logging_object and default env vars
    base_tags = DataDogLogger._get_datadog_tags()
    assert "env:" in base_tags
    assert "service:" in base_tags
    assert "version:" in base_tags
    assert "POD_NAME:" in base_tags
    assert "HOSTNAME:" in base_tags

    # Test with custom env vars
    test_env = {
        "DD_ENV": "production",
        "DD_SERVICE": "custom-service",
        "DD_VERSION": "1.0.0",
        "HOSTNAME": "test-host",
        "POD_NAME": "pod-123",
    }
    with patch.dict(os.environ, test_env):
        custom_tags = DataDogLogger._get_datadog_tags()
        assert "env:production" in custom_tags
        assert "service:custom-service" in custom_tags
        assert "version:1.0.0" in custom_tags
        assert "HOSTNAME:test-host" in custom_tags
        assert "POD_NAME:pod-123" in custom_tags

    # Test with standard_logging_object containing request_tags
    standard_logging_obj = create_standard_logging_payload()
    standard_logging_obj["request_tags"] = ["tag1", "tag2"]

    tags_with_request = DataDogLogger._get_datadog_tags(standard_logging_obj)
    assert "request_tag:tag1" in tags_with_request
    assert "request_tag:tag2" in tags_with_request

    # Test with empty request_tags
    standard_logging_obj["request_tags"] = []
    tags_empty_request = DataDogLogger._get_datadog_tags(standard_logging_obj)
    assert "request_tag:" not in tags_empty_request

    # Test with None request_tags
    standard_logging_obj["request_tags"] = None
    tags_none_request = DataDogLogger._get_datadog_tags(standard_logging_obj)
    assert "request_tag:" not in tags_none_request


@pytest.mark.asyncio
async def test_datadog_message_redaction():
    """
    Test that DataDog logger correctly initializes with turn_off_message_logging=True 
    from litellm.datadog_params
    """
    try:
        # Test using litellm.datadog_params pattern
        litellm.datadog_params = DatadogInitParams(turn_off_message_logging=True)
        
        os.environ["DD_SITE"] = "https://fake.datadoghq.com"
        os.environ["DD_API_KEY"] = "anything"
        
        # Mock the periodic flush to avoid async issues
        with patch("asyncio.create_task"):
            dd_logger = DataDogLogger()

        # Verify that turn_off_message_logging was set correctly from litellm.datadog_params
        assert hasattr(dd_logger, 'turn_off_message_logging'), "DataDogLogger should have turn_off_message_logging attribute"
        assert dd_logger.turn_off_message_logging is True, f"Expected turn_off_message_logging=True, got {dd_logger.turn_off_message_logging}"
        
        # Test the redaction method inherited from CustomLogger
        model_call_details = {
            "standard_logging_object": {
                "messages": [{"role": "user", "content": "This is sensitive information that should be redacted"}],
                "response": {"choices": [{"message": {"content": "This is a sensitive response that should be redacted"}}]}
            }
        }
        
        # Apply redaction using the inherited method
        redacted_details = dd_logger.redact_standard_logging_payload_from_model_call_details(model_call_details)
        redacted_str = LiteLLMCommonStrings.redacted_by_litellm.value
        
        # Verify that messages are redacted
        redacted_standard_obj = redacted_details["standard_logging_object"]
        assert redacted_standard_obj["messages"][0]["content"] == redacted_str, f"Messages not redacted. Got: {redacted_standard_obj['messages'][0]['content']}"
        
        # Verify that response is redacted
        assert redacted_standard_obj["response"]["choices"][0]["message"]["content"] == redacted_str, f"Response not redacted. Got: {redacted_standard_obj['response']['choices'][0]['message']['content']}"

        print("✅ DataDog message redaction test passed")

    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")
    finally:
        # Clean up
        litellm.datadog_params = None
        litellm.callbacks = []