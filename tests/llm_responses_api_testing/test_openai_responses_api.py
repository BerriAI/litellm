import os
import sys
import pytest
from unittest.mock import patch, AsyncMock
import httpx
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
import time
import json

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from base_responses_api import validate_responses_api_response


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
        "model": "gpt-5.5",
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
            self.headers = httpx.Headers({})

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
                        "model": "gpt-5.5",
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
            self.headers = httpx.Headers({})

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
            self.headers = httpx.Headers({})

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


@pytest.mark.asyncio
async def test_store_field_transformation():
    """Test store field transformation with mocked API responses"""
    config = OpenAIResponsesAPIConfig()

    # Initialize logging object with required parameters
    logging_obj = LiteLLMLoggingObj(
        model="gpt-5.5",
        messages=[],
        stream=False,
        call_type="aresponses",
        start_time=time.time(),
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )

    # Base response data with all required fields
    base_response = {
        "id": "test_id",
        "created_at": 1751443898,
        "model": "gpt-5.5",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "error": None,
        "incomplete_details": None,
        "instructions": "test instructions",
        "metadata": {},
        "temperature": 0.7,
        "top_p": 1.0,
        "max_output_tokens": 100,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": None,
        "truncation": "auto",
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "user": "test_user",
    }

    # Test case 1: API returns store=True
    mock_response_store_true = httpx.Response(
        status_code=200, content=json.dumps({**base_response, "store": True}).encode()
    )

    # Test case 2: API returns store=False
    mock_response_store_false = httpx.Response(
        status_code=200, content=json.dumps({**base_response, "store": False}).encode()
    )

    # Test case 3: API returns store=null
    mock_response_store_null = httpx.Response(
        status_code=200, content=json.dumps({**base_response, "store": None}).encode()
    )

    # Test case 4: API omits store field
    mock_response_no_store = httpx.Response(
        status_code=200, content=json.dumps(base_response).encode()
    )

    # Test when store=True in request
    logging_obj.optional_params = {"store": True}
    response = config.transform_response_api_response(
        model="gpt-5.5", raw_response=mock_response_store_true, logging_obj=logging_obj
    )
    assert (
        response.store is True
    ), "store should be True when specified in request and API returns True"

    # Test when store=False in request
    logging_obj.optional_params = {"store": False}
    response = config.transform_response_api_response(
        model="gpt-5.5", raw_response=mock_response_store_false, logging_obj=logging_obj
    )
    assert (
        response.store is False
    ), "store should be False when specified in request and API returns False"

    # Test when store not in request but API returns null
    response = config.transform_response_api_response(
        model="gpt-5.5", raw_response=mock_response_store_null, logging_obj=logging_obj
    )
    assert (
        response.store is None
    ), "store should be None when not specified in request and API returns null"

    # Test when store not in request and API omits store field
    response = config.transform_response_api_response(
        model="gpt-5.5", raw_response=mock_response_no_store, logging_obj=logging_obj
    )
    assert (
        response.store is None
    ), "store should be None when not specified in request and API omits store"

    # Verify created_at is always converted to integer
    assert isinstance(
        response.created_at, int
    ), "created_at should always be converted to integer"
    assert (
        response.created_at == 1751443898
    ), "created_at should maintain the same value after conversion"


