import os
import sys
import pytest
import asyncio
from typing import Optional
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


def validate_responses_api_response(response, final_chunk: bool = False):
    """
    Validate that a response from litellm.responses() or litellm.aresponses()
    conforms to the expected ResponsesAPIResponse structure.

    Args:
        response: The response object to validate

    Raises:
        AssertionError: If the response doesn't match the expected structure
    """
    # Validate response structure
    print("response=", json.dumps(response, indent=4, default=str))
    assert isinstance(
        response, ResponsesAPIResponse
    ), "Response should be an instance of ResponsesAPIResponse"

    # Required fields
    assert "id" in response and isinstance(
        response["id"], str
    ), "Response should have a string 'id' field"
    assert "created_at" in response and isinstance(
        response["created_at"], (int, float)
    ), "Response should have a numeric 'created_at' field"
    assert "output" in response and isinstance(
        response["output"], list
    ), "Response should have a list 'output' field"
    assert "parallel_tool_calls" in response and isinstance(
        response["parallel_tool_calls"], bool
    ), "Response should have a boolean 'parallel_tool_calls' field"

    # Optional fields with their expected types
    optional_fields = {
        "error": (dict, type(None)),  # error can be dict or None
        "incomplete_details": (IncompleteDetails, type(None)),
        "instructions": (str, type(None)),
        "metadata": dict,
        "model": str,
        "object": str,
        "temperature": (int, float),
        "tool_choice": (dict, str),
        "tools": list,
        "top_p": (int, float),
        "max_output_tokens": (int, type(None)),
        "previous_response_id": (str, type(None)),
        "reasoning": dict,
        "status": str,
        "text": ResponseTextConfig,
        "truncation": str,
        "usage": ResponseAPIUsage,
        "user": (str, type(None)),
    }
    if final_chunk is False:
        optional_fields["usage"] = type(None)

    for field, expected_type in optional_fields.items():
        if field in response:
            assert isinstance(
                response[field], expected_type
            ), f"Field '{field}' should be of type {expected_type}, but got {type(response[field])}"

    # Check if output has at least one item
    if final_chunk is True:
        assert (
            len(response["output"]) > 0
        ), "Response 'output' field should have at least one item"

    return True  # Return True if validation passes


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_basic_openai_responses_api(sync_mode):
    litellm._turn_on_debug()

    if sync_mode:
        response = litellm.responses(
            model="gpt-4o", input="Basic ping", max_output_tokens=20
        )
    else:
        response = await litellm.aresponses(
            model="gpt-4o", input="Basic ping", max_output_tokens=20
        )

    print("litellm response=", json.dumps(response, indent=4, default=str))

    # Use the helper function to validate the response
    validate_responses_api_response(response, final_chunk=True)


@pytest.mark.parametrize("sync_mode", [True])
@pytest.mark.asyncio
async def test_basic_openai_responses_api_streaming(sync_mode):
    litellm._turn_on_debug()

    if sync_mode:
        response = litellm.responses(
            model="gpt-4o",
            input="Basic ping",
            stream=True,
        )
        for event in response:
            print("litellm response=", json.dumps(event, indent=4, default=str))
    else:
        response = await litellm.aresponses(
            model="gpt-4o",
            input="Basic ping",
            stream=True,
        )
        async for event in response:
            print("litellm response=", json.dumps(event, indent=4, default=str))


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


@pytest.mark.parametrize("model", ["gpt-4o"])
def test_sync_retrieve_delete(model):
    '''
    Test creation, then getting, then deleting.
    '''
    response = litellm.responses(
            model=model,
            input="What color is the sky?",
        )
    assert response is not None

    id = response['id']

    # Test retrieve functionality
    retrieved_response = litellm.responses_retrieve(id, custom_llm_provider="openai")
    assert retrieved_response["text"] == response["text"]

    # Test delete functionality
    deleted_response = litellm.responses_delete(id, custom_llm_provider="openai")
    assert deleted_response is not None


@pytest.mark.parametrize("model", ["gpt-4o"])
@pytest.mark.asyncio  
async def test_async_retrieve_delete(model):
    '''
    Test async creation, then getting, then deleting.
    '''
    response = await litellm.aresponses(
            model=model,
            input="What color is the sky?",
        )
    assert response is not None

    id = response['id']

    # Test async retrieve functionality
    retrieved_response = await litellm.aresponses_retrieve(id, custom_llm_provider="openai")
    assert retrieved_response["text"] == response["text"]

    # Test async delete functionality
    deleted_response = await litellm.aresponses_delete(id, custom_llm_provider="openai")
    assert deleted_response is not None

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
        request_body = mock_post.call_args.kwargs["data"]
        print("Request body:", json.dumps(request_body, indent=4))

        loaded_request_body = json.loads(request_body)
        print("Loaded request body:", json.dumps(loaded_request_body, indent=4))

        # Assert metadata is not in the request
        assert (
            loaded_request_body["metadata"] == None
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
        request_body = mock_post.call_args.kwargs["data"]
        loaded_request_body = json.loads(request_body)
        print("Request body:", json.dumps(loaded_request_body, indent=4))

        # Assert metadata matches exactly what was passed
        assert (
            loaded_request_body["metadata"] == test_metadata
        ), "metadata in request body should match what was passed"
        mock_post.assert_called_once()

