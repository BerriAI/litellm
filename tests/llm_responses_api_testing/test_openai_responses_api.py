import os
import sys
import pytest
import asyncio
from typing import Optional, cast
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponseTextConfig,
    ResponseAPIUsage,
    IncompleteDetails,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from base_responses_api import BaseResponsesAPITest, validate_responses_api_response

class TestOpenAIResponsesAPITest(BaseResponsesAPITest):
    def get_base_completion_call_args(self):
        return {
            "model": "openai/gpt-4o",
        }


class TestCustomLogger(CustomLogger):
    def __init__(
        self,
    ):
        self.standard_logging_object: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("in async_log_success_event")
        print("kwargs=", json.dumps(kwargs, indent=4, default=str))
        self.standard_logging_object = kwargs["standard_logging_object"]
        pass


def validate_standard_logging_payload(
    slp: StandardLoggingPayload, response: ResponsesAPIResponse, request_model: str
):
    """
    Validate that a StandardLoggingPayload object matches the expected response

    Args:
        slp (StandardLoggingPayload): The standard logging payload object to validate
        response (dict): The litellm response to compare against
        request_model (str): The model name that was requested
    """
    # Validate payload exists
    assert slp is not None, "Standard logging payload should not be None"

    # Validate token counts
    print("response=", json.dumps(response, indent=4, default=str))
    assert (
        slp["prompt_tokens"] == response["usage"]["input_tokens"]
    ), "Prompt tokens mismatch"
    assert (
        slp["completion_tokens"] == response["usage"]["output_tokens"]
    ), "Completion tokens mismatch"
    assert (
        slp["total_tokens"]
        == response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
    ), "Total tokens mismatch"

    # Validate spend and response metadata
    assert slp["response_cost"] > 0, "Response cost should be greater than 0"
    assert slp["id"] == response["id"], "Response ID mismatch"
    assert slp["model"] == request_model, "Model name mismatch"

    # Validate messages
    assert slp["messages"] == [{"content": "hi", "role": "user"}], "Messages mismatch"

    # Validate complete response structure
    validate_responses_match(slp["response"], response)


@pytest.mark.asyncio
async def test_basic_openai_responses_api_streaming_with_logging():
    litellm._turn_on_debug()
    litellm.set_verbose = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    request_model = "gpt-4o"
    response = await litellm.aresponses(
        model=request_model,
        input="hi",
        stream=True,
    )
    final_response: Optional[ResponseCompletedEvent] = None
    async for event in response:
        if event.type == "response.completed":
            final_response = event
        print("litellm response=", json.dumps(event, indent=4, default=str))

    print("sleeping for 2 seconds...")
    await asyncio.sleep(2)
    print(
        "standard logging payload=",
        json.dumps(test_custom_logger.standard_logging_object, indent=4, default=str),
    )

    assert final_response is not None
    assert test_custom_logger.standard_logging_object is not None

    validate_standard_logging_payload(
        slp=test_custom_logger.standard_logging_object,
        response=final_response.response,
        request_model=request_model,
    )


def validate_responses_match(slp_response, litellm_response):
    """Validate that the standard logging payload OpenAI response matches the litellm response"""
    # Validate core fields
    assert slp_response["id"] == litellm_response["id"], "ID mismatch"
    assert slp_response["model"] == litellm_response["model"], "Model mismatch"
    assert (
        slp_response["created_at"] == litellm_response["created_at"]
    ), "Created at mismatch"

    # Validate usage
    assert (
        slp_response["usage"]["input_tokens"]
        == litellm_response["usage"]["input_tokens"]
    ), "Input tokens mismatch"
    assert (
        slp_response["usage"]["output_tokens"]
        == litellm_response["usage"]["output_tokens"]
    ), "Output tokens mismatch"
    assert (
        slp_response["usage"]["total_tokens"]
        == litellm_response["usage"]["total_tokens"]
    ), "Total tokens mismatch"

    # Validate output/messages
    assert len(slp_response["output"]) == len(
        litellm_response["output"]
    ), "Output length mismatch"
    for slp_msg, litellm_msg in zip(slp_response["output"], litellm_response["output"]):
        assert slp_msg["role"] == litellm_msg.role, "Message role mismatch"
        # Access the content's text field for the litellm response
        litellm_content = litellm_msg.content[0].text if litellm_msg.content else ""
        assert (
            slp_msg["content"][0]["text"] == litellm_content
        ), f"Message content mismatch. Expected {litellm_content}, Got {slp_msg['content']}"
        assert slp_msg["status"] == litellm_msg.status, "Message status mismatch"


@pytest.mark.asyncio
async def test_basic_openai_responses_api_non_streaming_with_logging():
    litellm._turn_on_debug()
    litellm.set_verbose = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    request_model = "gpt-4o"
    response = await litellm.aresponses(
        model=request_model,
        input="hi",
    )

    print("litellm response=", json.dumps(response, indent=4, default=str))
    print("response hidden params=", response._hidden_params)

    print("sleeping for 2 seconds...")
    await asyncio.sleep(2)
    print(
        "standard logging payload=",
        json.dumps(test_custom_logger.standard_logging_object, indent=4, default=str),
    )

    assert response is not None
    assert test_custom_logger.standard_logging_object is not None

    validate_standard_logging_payload(
        test_custom_logger.standard_logging_object, response, request_model
    )


def validate_stream_event(event):
    """
    Validate that a streaming event from litellm.responses() or litellm.aresponses()
    with stream=True conforms to the expected structure based on its event type.

    Args:
        event: The streaming event object to validate

    Raises:
        AssertionError: If the event doesn't match the expected structure for its type
    """
    # Common validation for all event types
    assert hasattr(event, "type"), "Event should have a 'type' attribute"

    # Type-specific validation
    if event.type == "response.created" or event.type == "response.in_progress":
        assert hasattr(
            event, "response"
        ), f"{event.type} event should have a 'response' attribute"
        validate_responses_api_response(event.response, final_chunk=False)

    elif event.type == "response.completed":
        assert hasattr(
            event, "response"
        ), "response.completed event should have a 'response' attribute"
        validate_responses_api_response(event.response, final_chunk=True)
        # Usage is guaranteed only on the completed event
        assert (
            "usage" in event.response
        ), "response.completed event should have usage information"
        print("Usage in event.response=", event.response["usage"])
        assert isinstance(event.response["usage"], ResponseAPIUsage)
    elif event.type == "response.failed" or event.type == "response.incomplete":
        assert hasattr(
            event, "response"
        ), f"{event.type} event should have a 'response' attribute"

    elif (
        event.type == "response.output_item.added"
        or event.type == "response.output_item.done"
    ):
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "item"
        ), f"{event.type} event should have an 'item' attribute"

    elif (
        event.type == "response.content_part.added"
        or event.type == "response.content_part.done"
    ):
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "part"
        ), f"{event.type} event should have a 'part' attribute"

    elif event.type == "response.output_text.delta":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "delta"
        ), f"{event.type} event should have a 'delta' attribute"

    elif event.type == "response.output_text.annotation.added":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "annotation_index"
        ), f"{event.type} event should have an 'annotation_index' attribute"
        assert hasattr(
            event, "annotation"
        ), f"{event.type} event should have an 'annotation' attribute"

    elif event.type == "response.output_text.done":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "text"
        ), f"{event.type} event should have a 'text' attribute"

    elif event.type == "response.refusal.delta":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "delta"
        ), f"{event.type} event should have a 'delta' attribute"

    elif event.type == "response.refusal.done":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "refusal"
        ), f"{event.type} event should have a 'refusal' attribute"

    elif event.type == "response.function_call_arguments.delta":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "delta"
        ), f"{event.type} event should have a 'delta' attribute"

    elif event.type == "response.function_call_arguments.done":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "arguments"
        ), f"{event.type} event should have an 'arguments' attribute"

    elif event.type in [
        "response.file_search_call.in_progress",
        "response.file_search_call.searching",
        "response.file_search_call.completed",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
    ]:
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"

    elif event.type == "error":
        assert hasattr(
            event, "message"
        ), "Error event should have a 'message' attribute"
    return True  # Return True if validation passes


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_openai_responses_api_streaming_validation(sync_mode):
    """Test that validates each streaming event from the responses API"""
    litellm._turn_on_debug()

    event_types_seen = set()

    if sync_mode:
        response = litellm.responses(
            model="gpt-4o",
            input="Tell me about artificial intelligence in 3 sentences.",
            stream=True,
        )
        for event in response:
            print(f"Validating event type: {event.type}")
            validate_stream_event(event)
            event_types_seen.add(event.type)
    else:
        response = await litellm.aresponses(
            model="gpt-4o",
            input="Tell me about artificial intelligence in 3 sentences.",
            stream=True,
        )
        async for event in response:
            print(f"Validating event type: {event.type}")
            validate_stream_event(event)
            event_types_seen.add(event.type)

    # At minimum, we should see these core event types
    required_events = {"response.created", "response.completed"}

    missing_events = required_events - event_types_seen
    assert not missing_events, f"Missing required event types: {missing_events}"

    print(f"Successfully validated all event types: {event_types_seen}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_openai_responses_litellm_router(sync_mode):
    """
    Test the OpenAI responses API with LiteLLM Router in both sync and async modes
    """
    litellm._turn_on_debug()
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt4o-special-alias",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            }
        ]
    )

    # Call the handler
    if sync_mode:
        response = router.responses(
            model="gpt4o-special-alias",
            input="Hello, can you tell me a short joke?",
            max_output_tokens=100,
        )
        print("SYNC MODE RESPONSE=", response)
    else:
        response = await router.aresponses(
            model="gpt4o-special-alias",
            input="Hello, can you tell me a short joke?",
            max_output_tokens=100,
        )

    print(
        f"Router {'sync' if sync_mode else 'async'} response=",
        json.dumps(response, indent=4, default=str),
    )

    # Use the helper function to validate the response
    validate_responses_api_response(response, final_chunk=True)

    return response


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_openai_responses_litellm_router_streaming(sync_mode):
    """
    Test the OpenAI responses API with streaming through LiteLLM Router
    """
    litellm._turn_on_debug()
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt4o-special-alias",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            }
        ]
    )

    event_types_seen = set()

    if sync_mode:
        response = router.responses(
            model="gpt4o-special-alias",
            input="Tell me about artificial intelligence in 2 sentences.",
            stream=True,
        )
        for event in response:
            print(f"Validating event type: {event.type}")
            validate_stream_event(event)
            event_types_seen.add(event.type)
    else:
        response = await router.aresponses(
            model="gpt4o-special-alias",
            input="Tell me about artificial intelligence in 2 sentences.",
            stream=True,
        )
        async for event in response:
            print(f"Validating event type: {event.type}")
            validate_stream_event(event)
            event_types_seen.add(event.type)

    # At minimum, we should see these core event types
    required_events = {"response.created", "response.completed"}

    missing_events = required_events - event_types_seen
    assert not missing_events, f"Missing required event types: {missing_events}"

    print(f"Successfully validated all event types: {event_types_seen}")


@pytest.mark.asyncio
async def test_openai_responses_litellm_router_no_metadata():
    """
    Test that metadata is not passed through when using the Router for responses API
    """
    mock_response = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello world!", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        # Adding all required fields
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = str(json_data)

        def json(self):  # Changed from async to sync
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm._turn_on_debug()
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt4o-special-alias",
                    "litellm_params": {
                        "model": "gpt-4o",
                        "api_key": "fake-key",
                    },
                }
            ]
        )

        # Call the handler with metadata
        await router.aresponses(
            model="gpt4o-special-alias",
            input="Hello, can you tell me a short joke?",
        )

        # Check the request body
        request_body = mock_post.call_args.kwargs["json"]
        print("Request body:", json.dumps(request_body, indent=4))



        # Assert metadata is not in the request
        assert (
            "metadata" not in request_body
        ), "metadata should not be in the request body"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_openai_responses_litellm_router_with_metadata():
    """
    Test that metadata is correctly passed through when explicitly provided to the Router for responses API
    """
    test_metadata = {
        "user_id": "123",
        "conversation_id": "abc",
        "custom_field": "test_value",
    }

    mock_response = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello world!", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": test_metadata,  # Include the test metadata in response
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = str(json_data)

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm._turn_on_debug()
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt4o-special-alias",
                    "litellm_params": {
                        "model": "gpt-4o",
                        "api_key": "fake-key",
                    },
                }
            ]
        )

        # Call the handler with metadata
        await router.aresponses(
            model="gpt4o-special-alias",
            input="Hello, can you tell me a short joke?",
            metadata=test_metadata,
        )

        # Check the request body
        request_body = mock_post.call_args.kwargs["json"]
        print("Request body:", json.dumps(request_body, indent=4))

        # Assert metadata matches exactly what was passed
        assert (
            request_body["metadata"] == test_metadata
        ), "metadata in request body should match what was passed"
        mock_post.assert_called_once()


def test_bad_request_bad_param_error():
    """Raise a BadRequestError when an invalid parameter value is provided"""
    try:
        litellm.responses(model="gpt-4o", input="This should fail", temperature=2000)
        pytest.fail("Expected BadRequestError but no exception was raised")
    except litellm.BadRequestError as e:
        print(f"Exception raised: {e}")
        print(f"Exception type: {type(e)}")
        print(f"Exception args: {e.args}")
        print(f"Exception details: {e.__dict__}")
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.asyncio()
async def test_async_bad_request_bad_param_error():
    """Raise a BadRequestError when an invalid parameter value is provided"""
    try:
        await litellm.aresponses(
            model="gpt-4o", input="This should fail", temperature=2000
        )
        pytest.fail("Expected BadRequestError but no exception was raised")
    except litellm.BadRequestError as e:
        print(f"Exception raised: {e}")
        print(f"Exception type: {type(e)}")
        print(f"Exception args: {e.args}")
        print(f"Exception details: {e.__dict__}")
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_openai_o1_pro_response_api(sync_mode):
    """
    Test that LiteLLM correctly handles an incomplete response from OpenAI's o1-pro model
    due to reaching max_output_tokens limit.
    """
    # Mock response from o1-pro
    mock_response = {
        "id": "resp_67dc3dd77b388190822443a85252da5a0e13d8bdc0e28d88",
        "object": "response",
        "created_at": 1742486999,
        "status": "incomplete",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"},
        "instructions": None,
        "max_output_tokens": 20,
        "model": "o1-pro-2025-03-19",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_67dc3de50f64819097450ed50a33d5f90e13d8bdc0e28d88",
                "summary": [],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "generate_summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 73,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 93,
        },
        "user": None,
        "metadata": {},
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):  # Changed from async to sync
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm._turn_on_debug()
        litellm.set_verbose = True

        # Call o1-pro with max_output_tokens=20
        response = await litellm.aresponses(
            model="openai/o1-pro",
            input="Write a detailed essay about artificial intelligence and its impact on society",
            max_output_tokens=20,
        )

        # Verify the request was made correctly
        mock_post.assert_called_once()
        request_body = mock_post.call_args.kwargs["json"]
        assert request_body["model"] == "o1-pro"
        assert request_body["max_output_tokens"] == 20

        # Validate the response
        print("Response:", json.dumps(response, indent=4, default=str))

        # Check that the response has the expected structure
        assert response["id"] is not None
        assert response["status"] == "incomplete"
        assert response["incomplete_details"].reason == "max_output_tokens"
        assert response["max_output_tokens"] == 20

        # Validate usage information
        assert response["usage"]["input_tokens"] == 73
        assert response["usage"]["output_tokens"] == 20
        assert response["usage"]["total_tokens"] == 93

        # Validate that the response is properly identified as incomplete
        validate_responses_api_response(response, final_chunk=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_openai_o1_pro_response_api_streaming(sync_mode):
    """
    Test that LiteLLM correctly handles an incomplete response from OpenAI's o1-pro model
    due to reaching max_output_tokens limit in both sync and async streaming modes.
    """
    # Mock response from o1-pro
    mock_response = {
        "id": "resp_67dc3dd77b388190822443a85252da5a0e13d8bdc0e28d88",
        "object": "response",
        "created_at": 1742486999,
        "status": "incomplete",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"},
        "instructions": None,
        "max_output_tokens": 20,
        "model": "o1-pro-2025-03-19",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_67dc3de50f64819097450ed50a33d5f90e13d8bdc0e28d88",
                "summary": [],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "generate_summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 73,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 93,
        },
        "user": None,
        "metadata": {},
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm._turn_on_debug()
        litellm.set_verbose = True

        # Verify the request was made correctly
        if sync_mode:
            # For sync mode, we need to patch the sync HTTP handler
            with patch(
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
                return_value=MockResponse(mock_response, 200),
            ) as mock_sync_post:
                response = litellm.responses(
                    model="openai/o1-pro",
                    input="Write a detailed essay about artificial intelligence and its impact on society",
                    max_output_tokens=20,
                    stream=True,
                )

                # Process the sync stream
                event_count = 0
                for event in response:
                    print(
                        f"Sync litellm response #{event_count}:",
                        json.dumps(event, indent=4, default=str),
                    )
                    event_count += 1

                # Verify the sync request was made correctly
                mock_sync_post.assert_called_once()
                request_body = mock_sync_post.call_args.kwargs["json"]
                assert request_body["model"] == "o1-pro"
                assert request_body["max_output_tokens"] == 20
                assert "stream" not in request_body
        else:
            # For async mode
            response = await litellm.aresponses(
                model="openai/o1-pro",
                input="Write a detailed essay about artificial intelligence and its impact on society",
                max_output_tokens=20,
                stream=True,
            )

            # Process the async stream
            event_count = 0
            async for event in response:
                print(
                    f"Async litellm response #{event_count}:",
                    json.dumps(event, indent=4, default=str),
                )
                event_count += 1

            # Verify the async request was made correctly
            mock_post.assert_called_once()
            request_body = mock_post.call_args.kwargs["json"]
            assert request_body["model"] == "o1-pro"
            assert request_body["max_output_tokens"] == 20
            assert "stream" not in request_body


def test_basic_computer_use_preview_tool_call():
    """
    Test that LiteLLM correctly handles a computer_use_preview tool call where the environment is set to "linux"

    linux is an unsupported environment for the computer_use_preview tool, but litellm users should still be able to pass it to openai
    """
    # Mock response from OpenAI

    mock_response = {
        "id": "resp_67dc3dd77b388190822443a85252da5a0e13d8bdc0e28d88",
        "object": "response",
        "created_at": 1742486999,
        "status": "incomplete",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"},
        "instructions": None,
        "max_output_tokens": 20,
        "model": "o1-pro-2025-03-19",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_67dc3de50f64819097450ed50a33d5f90e13d8bdc0e28d88",
                "summary": [],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "generate_summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 73,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 93,
        },
        "user": None,
        "metadata": {},
    }
    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=MockResponse(mock_response, 200),
    ) as mock_post:
        litellm._turn_on_debug()
        litellm.set_verbose = True

        # Call the responses API with computer_use_preview tool
        response = litellm.responses(
            model="openai/computer-use-preview",
            tools=[{
                "type": "computer_use_preview",
                "display_width": 1024,
                "display_height": 768,
                "environment": "linux"  # other possible values: "mac", "windows", "ubuntu"
            }],
            input="Check the latest OpenAI news on bing.com.",
            reasoning={"summary": "concise"},
            truncation="auto"
        )

        # Verify the request was made correctly
        mock_post.assert_called_once()
        request_body = mock_post.call_args.kwargs["json"]
        
        # Validate the request structure
        assert request_body["model"] == "computer-use-preview"
        assert len(request_body["tools"]) == 1
        assert request_body["tools"][0]["type"] == "computer_use_preview"
        assert request_body["tools"][0]["display_width"] == 1024
        assert request_body["tools"][0]["display_height"] == 768
        assert request_body["tools"][0]["environment"] == "linux"
        
        # Check that reasoning was passed correctly
        assert request_body["reasoning"]["summary"] == "concise"
        assert request_body["truncation"] == "auto"
        
        # Validate the input format
        assert isinstance(request_body["input"], str)
        assert request_body["input"] == "Check the latest OpenAI news on bing.com."
        


def test_mcp_tools_with_responses_api():
    litellm._turn_on_debug()
    MCP_TOOLS = [
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "allowed_tools": ["ask_question"]
        }
    ]
    MODEL = "openai/gpt-4.1"
    USER_QUERY = "What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?"
    #########################################################
    # Step 1: OpenAI will use MCP LIST, and return a list of MCP calls for our approval 
    response = litellm.responses(
        model=MODEL,
        tools=MCP_TOOLS,
        input=USER_QUERY
    )
    print(response)

    response = cast(ResponsesAPIResponse, response)

    mcp_approval_id: Optional[str]
    for output in response.output:
        if output.type == "mcp_approval_request":
            mcp_approval_id = output.id
            break

    # Step 2: Send followup with approval for the MCP call
    response_with_mcp_call = litellm.responses(
        model=MODEL,
        tools=MCP_TOOLS,
        input=[
            {
                "type": "mcp_approval_response",
                "approve": True,
                "approval_request_id": mcp_approval_id
            }
        ],
        previous_response_id=response.id,
    )
    print(response_with_mcp_call)


