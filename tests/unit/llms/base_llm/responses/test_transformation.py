"""The shared Responses API config contract."""

import asyncio, importlib, json

import httpx
import pytest

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm import Router
from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import(
    OutputTextDeltaEvent,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch


@pytest.mark.asyncio
async def test_default_async_transform_delegates_to_the_sync_transform():
    """A config that overrides only the sync transform gets the same request from the async hook,
    so the async handler can always await the hook."""
    cfg = OpenAIResponsesAPIConfig()
    input_with_cache_marker = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    sync_body = cfg.transform_responses_api_request(
        model="gpt-5",
        input=input_with_cache_marker,
        response_api_optional_request_params={"max_output_tokens": 64},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    async_body = await cfg.async_transform_responses_api_request(
        model="gpt-5",
        input=input_with_cache_marker,
        response_api_optional_request_params={"max_output_tokens": 64},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert async_body == sync_body
    assert "cache_control" not in async_body["input"][0]["content"][0]


def test_responses_sends_a_caller_extra_body_over_the_request_unchanged(respx_mock, monkeypatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "m",
                "output": [],
            },
        )
    )

    litellm.responses(
        model="openai/gpt-5",
        input="hi",
        api_key="sk-test",
        metadata={"a": "1"},
        extra_body={"foo": 1, "metadata": {"b": "2"}},
    )

    body = json.loads(route.calls.last.request.content)
    assert body["foo"] == 1
    assert body["metadata"] == {"b": "2"}


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="function")
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception:
        pass
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestBaseResponsesAPIStreamingIterator:
    """Test cases for BaseResponsesAPIStreamingIterator"""

    @pytest.mark.asyncio
    async def test_responses_streaming_iterator_parses_u2028_in_sse_json(self):
        """
        U+2028 inside JSON must not split the SSE event. httpx aiter_lines uses
        str.splitlines() and drops response.completed; OpenAI SSEDecoder does not.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        u2028 = "\u2028"
        payload = json.dumps(
            {
                "type": "response.completed",
                "response": {"instructions": f"eligible{u2028}promo"},
            }
        )
        sse_bytes = f"data: {payload}\n\n".encode("utf-8")

        async def mock_aiter_bytes():
            yield sse_bytes

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = mock_aiter_bytes

        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "resp_u2028"
        mock_responses_api_response.usage = ResponseAPIUsage(input_tokens=3, output_tokens=2, total_tokens=5)
        mock_completed_event = Mock(spec=ResponseCompletedEvent)
        mock_completed_event.type = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        mock_completed_event.response = mock_responses_api_response
        mock_config.transform_streaming_response.return_value = mock_completed_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        chunks = []
        with (
            patch("asyncio.create_task"),
            patch("litellm.responses.streaming_iterator.executor"),
        ):
            async for chunk in iterator:
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        assert iterator.completed_response is not None

    def test_process_chunk_with_response_completed_event(self):
        """
        Test that _process_chunk correctly processes a ResponseCompletedEvent
        and calls _update_responses_api_response_id_with_model_id for the final chunk.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create a mock ResponsesAPIResponse for the completed event
        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "original_response_id"

        # Create a mock ResponseCompletedEvent
        mock_completed_event = Mock(spec=ResponseCompletedEvent)
        mock_completed_event.type = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        mock_completed_event.response = mock_responses_api_response

        # Set up the mock transform method to return our completed event
        mock_config.transform_streaming_response.return_value = mock_completed_event

        # Mock the update_responses_api_response_id_with_model_id method
        updated_response = Mock(spec=ResponsesAPIResponse)
        updated_response.id = "updated_response_id"
        updated_response.usage = ResponseAPIUsage(input_tokens=3, output_tokens=2, total_tokens=5)

        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        # Prepare test chunk data
        test_chunk_data = {
            "type": "response.completed",
            "response": {
                "id": "original_response_id",
                "output": [{"type": "message", "content": [{"text": "Hello World"}]}],
            },
        }

        with patch.object(
            ResponsesAPIRequestUtils,
            "update_responses_api_response_id_with_model_id",
            return_value=updated_response,
        ) as mock_update_id:
            # Process the chunk
            result = iterator._process_chunk(json.dumps(test_chunk_data))

            # Assertions
            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED

            # Verify that update_responses_api_response_id_with_model_id was called
            mock_update_id.assert_called_once_with(
                responses_api_response=mock_responses_api_response,
                litellm_metadata={"model_info": {"id": "model_123"}},
                custom_llm_provider="openai",
            )

            # Verify the completed response was stored
            assert iterator.completed_response == result

            # Verify the response was updated on the event
            assert result.response == updated_response

    def test_process_chunk_with_delta_event_no_id_update(self):
        """
        Test that _process_chunk correctly processes a delta event
        and does NOT call _update_responses_api_response_id_with_model_id.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create a mock OutputTextDeltaEvent (not a completed event)
        mock_delta_event = Mock(spec=OutputTextDeltaEvent)
        mock_delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        mock_delta_event.delta = "Hello"
        # Delta events don't have a response attribute
        (delattr(mock_delta_event, "response") if hasattr(mock_delta_event, "response") else None)

        # Set up the mock transform method to return our delta event
        mock_config.transform_streaming_response.return_value = mock_delta_event

        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        # Prepare test chunk data for a delta event
        test_chunk_data = {
            "type": "response.output_text.delta",
            "delta": "Hello",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
        }

        with patch.object(
            ResponsesAPIRequestUtils, "update_responses_api_response_id_with_model_id"
        ) as mock_update_id:
            # Process the chunk
            result = iterator._process_chunk(json.dumps(test_chunk_data))

            # Assertions
            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA

            # Verify that update_responses_api_response_id_with_model_id was NOT called
            mock_update_id.assert_not_called()

            # Verify no completed response was stored (since this is not a completed event)
            assert iterator.completed_response is None

    def test_process_chunk_handles_invalid_json(self):
        """
        Test that _process_chunk gracefully handles invalid JSON.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
        )

        # Test with invalid JSON
        result = iterator._process_chunk("invalid json {")

        # Should return None for invalid JSON
        assert result is None
        assert iterator.completed_response is None

    def test_process_chunk_handles_done_marker(self):
        """
        Test that _process_chunk correctly handles the [DONE] marker.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
        )

        # Test with [DONE] marker
        result = iterator._process_chunk(STREAM_SSE_DONE_STRING)

        # Should return None and set finished flag
        assert result is None
        assert iterator.finished is True

    def test_process_chunk_handles_empty_chunk(self):
        """
        Test that _process_chunk correctly handles empty or None chunks.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
        )

        # Test with empty chunk
        result = iterator._process_chunk("")
        assert result is None

        # Test with None chunk
        result = iterator._process_chunk(None)
        assert result is None

    def test_handle_logging_completed_response_with_unpickleable_objects(self):
        """
        Test that _handle_logging_completed_response handles responses containing
        objects that cannot be pickled (like Pydantic ValidatorIterator).

        This test verifies the fix for issue #17192 where streaming with tool_choice
        containing allowed_tools would fail with:
        "cannot pickle 'pydantic_core._pydantic_core.ValidatorIterator' object"

        The fix uses model_dump + model_validate instead of copy.deepcopy.
        """
        import asyncio

        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_logging_obj.async_success_handler = Mock()
        mock_logging_obj.success_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create the iterator instance
        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        # Create a ResponseCompletedEvent with tool_choice that has model_dump
        mock_completed_response = Mock()
        mock_completed_response.model_dump.return_value = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [{"type": "function_call", "name": "search_web"}],
                "tool_choice": {"type": "function", "name": "search_web"},
            },
        }
        # model_validate should return a new mock (the copy)
        type(mock_completed_response).model_validate = Mock(return_value=Mock())

        iterator.completed_response = mock_completed_response

        # This should NOT raise an exception
        # Previously it would fail with: TypeError: cannot pickle 'ValidatorIterator'
        # Mock asyncio.create_task and executor.submit since we're not in async context
        with (
            patch("asyncio.create_task") as mock_create_task,
            patch("litellm.responses.streaming_iterator.executor") as mock_executor,
        ):
            try:
                iterator._handle_logging_completed_response()
            except TypeError as e:
                if "pickle" in str(e):
                    pytest.fail(f"_handle_logging_completed_response failed with pickle error: {e}")
                raise

    @staticmethod
    def _config_completing_after_one_delta() -> Mock:
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        completed_response = ResponsesAPIResponse(
            id="resp_123",
            created_at=0,
            status="completed",
            model="gpt-5.5",
            object="response",
            output=[],
            usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

        def _transform(model, parsed_chunk, logging_obj):
            if parsed_chunk.get("type") == "response.completed":
                return ResponseCompletedEvent(
                    type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                    response=completed_response,
                )
            return OutputTextDeltaEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                item_id="msg_123",
                output_index=0,
                content_index=0,
                delta=parsed_chunk["delta"],
            )

        mock_config.transform_streaming_response.side_effect = _transform
        return mock_config

    @pytest.mark.asyncio
    async def test_stop_async_iteration_not_logged_as_failure(self):
        """
        Test that StopAsyncIteration is NOT logged as a failure.

        This test verifies that when streaming completes normally with StopAsyncIteration,
        the _handle_failure method is NOT called, preventing false error logs in Langfuse
        and other logging integrations.

        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}

        async def mock_aiter_bytes():
            yield b'data: {"type": "response.output_text.delta", "delta": "test"}\n\n'
            yield b'data: {"type": "response.completed", "response": {"id": "resp_123"}}\n\n'

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()

        mock_config = self._config_completing_after_one_delta()

        # Create the iterator instance
        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        # Consume the iterator until StopAsyncIteration
        chunks_received = []
        try:
            async for chunk in iterator:
                chunks_received.append(chunk)
        except StopAsyncIteration:
            pass  # This is expected

        # Verify we got the delta and the terminal event
        assert len(chunks_received) == 2
        assert iterator.completed_response is not None

        # CRITICAL: Verify that failure handlers were NOT called
        # StopAsyncIteration is a normal end of stream, not a failure
        mock_logging_obj.async_failure_handler.assert_not_called()
        mock_logging_obj.failure_handler.assert_not_called()

    def test_stop_iteration_not_logged_as_failure(self):
        """
        Test that StopIteration is NOT logged as a failure in sync iterator.

        This test verifies that when streaming completes normally with StopIteration,
        the _handle_failure method is NOT called, preventing false error logs in Langfuse
        and other logging integrations.

        Regression test for: https://github.com/BerriAI/litellm/issues/XXXXX
        """
        from litellm.responses.streaming_iterator import (
            SyncResponsesAPIStreamingIterator,
        )

        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}

        def mock_iter_bytes():
            yield b'data: {"type": "response.output_text.delta", "delta": "test"}\n\n'
            yield b'data: {"type": "response.completed", "response": {"id": "resp_123"}}\n\n'

        mock_response.iter_bytes = mock_iter_bytes

        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()

        mock_config = self._config_completing_after_one_delta()

        # Create the iterator instance
        iterator = SyncResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        # Consume the iterator until StopIteration
        chunks_received = []
        try:
            for chunk in iterator:
                chunks_received.append(chunk)
        except StopIteration:
            pass  # This is expected

        # Verify we got the delta and the terminal event
        assert len(chunks_received) == 2
        assert iterator.completed_response is not None

        # CRITICAL: Verify that failure handlers were NOT called
        # StopIteration is a normal end of stream, not a failure
        mock_logging_obj.async_failure_handler.assert_not_called()
        mock_logging_obj.failure_handler.assert_not_called()

    def test_process_chunk_response_failed_calls_failure_handler(self):
        """
        Test that a RESPONSE_FAILED event routes to failure handlers,
        not success handlers. Failed responses represent genuine LLM-level
        errors and should be logged as failures.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        mock_logging_obj.async_success_handler = Mock()
        mock_logging_obj.success_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "resp_failed_123"
        mock_responses_api_response.error = {
            "type": "server_error",
            "message": "The model encountered an error",
        }
        mock_responses_api_response.usage = ResponseAPIUsage(input_tokens=3, output_tokens=2, total_tokens=5)

        mock_failed_event = Mock(spec=ResponseFailedEvent)
        mock_failed_event.type = ResponsesAPIStreamEvents.RESPONSE_FAILED
        mock_failed_event.response = mock_responses_api_response

        mock_config.transform_streaming_response.return_value = mock_failed_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        test_chunk_data = {
            "type": "response.failed",
            "response": {
                "id": "resp_failed_123",
                "error": {
                    "type": "server_error",
                    "message": "The model encountered an error",
                },
            },
        }

        with (
            patch.object(
                ResponsesAPIRequestUtils,
                "update_responses_api_response_id_with_model_id",
                return_value=mock_responses_api_response,
            ),
            patch("litellm.responses.streaming_iterator.run_async_function") as mock_run_async,
            patch("litellm.responses.streaming_iterator.executor") as mock_executor,
        ):
            result = iterator._process_chunk(json.dumps(test_chunk_data))

            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.RESPONSE_FAILED
            assert iterator.completed_response == result

            # Failure handler should have been called via _handle_failure
            mock_run_async.assert_called_once()
            call_kwargs = mock_run_async.call_args
            assert call_kwargs[1]["async_function"] == mock_logging_obj.async_failure_handler

            mock_executor.submit.assert_called_once()
            submit_args = mock_executor.submit.call_args
            assert submit_args[0][0] == mock_logging_obj.failure_handler

    def test_process_chunk_response_incomplete_calls_success_handler(self):
        """
        Test that a RESPONSE_INCOMPLETE event routes to success handlers.
        Incomplete responses (e.g. max_output_tokens reached) are still valid
        responses with usage data — analogous to finish_reason='length' in chat.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        mock_logging_obj.async_success_handler = Mock()
        mock_logging_obj.success_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "resp_incomplete_123"
        mock_responses_api_response.incomplete_details = {"reason": "max_output_tokens"}
        mock_responses_api_response.usage = ResponseAPIUsage(input_tokens=3, output_tokens=2, total_tokens=5)

        mock_incomplete_event = Mock(spec=ResponseIncompleteEvent)
        mock_incomplete_event.type = ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE
        mock_incomplete_event.response = mock_responses_api_response

        mock_config.transform_streaming_response.return_value = mock_incomplete_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-5.5",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        test_chunk_data = {
            "type": "response.incomplete",
            "response": {
                "id": "resp_incomplete_123",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        }

        with (
            patch.object(
                ResponsesAPIRequestUtils,
                "update_responses_api_response_id_with_model_id",
                return_value=mock_responses_api_response,
            ),
            patch("asyncio.create_task") as mock_create_task,
            patch("litellm.responses.streaming_iterator.executor") as mock_executor,
        ):
            result = iterator._process_chunk(json.dumps(test_chunk_data))

            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE
            assert iterator.completed_response == result

            # Success handlers are dispatched as one async task (via _handle_logging_completed_response);
            # the sync handler must never be submitted to the executor concurrently (LIT-4210)
            mock_create_task.assert_called_once()
            mock_executor.submit.assert_not_called()

            # Failure handlers should NOT have been called
            mock_logging_obj.async_failure_handler.assert_not_called()
            mock_logging_obj.failure_handler.assert_not_called()

@pytest.fixture()
def _vcr_outcome_gate_router_unit(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="function")
def setup_and_teardown_router_unit():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception:
        pass
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)

def _make_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                },
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-test",
                },
            },
        ]
    )

def _make_completed_event(input_tokens: int, output_tokens: int, total_tokens: int) -> ResponseCompletedEvent:
    response = ResponsesAPIResponse.model_construct(
        usage=ResponseAPIUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    )
    return ResponseCompletedEvent.model_construct(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=response,
    )

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
def test_extract_partial_responses_usage_native_completed():
    """Native path: completed_response carries usage → returned as-is."""
    completed = _make_completed_event(11, 7, 18)
    source = MagicMock()
    source.completed_response = completed

    usage = Router._extract_partial_responses_usage(source)
    assert usage is not None
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert usage.total_tokens == 18

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
def test_extract_partial_responses_usage_no_completed_response():
    """Native path: no completed_response → returns None."""
    source = MagicMock()
    source.completed_response = None

    usage = Router._extract_partial_responses_usage(source)
    assert usage is None

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
def test_extract_partial_responses_usage_bridge_iterator_no_completed_response():
    """
    Regression for #35411: the bridge iterator
    (LiteLLMCompletionStreamingIterator) overrides __init__ without calling
    super().__init__(), so completed_response was never set until the stream
    reached RESPONSE_COMPLETED. On a mid-stream provider error (before
    completion) the fallback recovery path read source_iterator.completed_response
    and raised AttributeError, masking the real provider error and bypassing
    fallbacks. The attribute must always exist and default to None.
    """
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )

    wrapper = MagicMock()
    wrapper.logging_obj = MagicMock()
    iterator = LiteLLMCompletionStreamingIterator(
        model="anthropic/claude-sonnet-4-5",
        litellm_custom_stream_wrapper=wrapper,
        request_input="hi",
        responses_api_request={},
    )

    assert iterator.completed_response is None
    # No chat chunks collected yet and no completed_response → must return
    # None instead of raising AttributeError.
    assert Router._extract_partial_responses_usage(iterator) is None

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
def test_combine_responses_fallback_usage_sums_completed_event():
    """Partial-stream usage is summed into the fallback event's usage."""
    fallback_event = _make_completed_event(5, 3, 8)
    partial = ResponseAPIUsage(input_tokens=11, output_tokens=7, total_tokens=18)

    Router._combine_responses_fallback_usage(fallback_event, partial)

    combined = fallback_event.response.usage
    assert combined is not None
    assert combined.input_tokens == 16
    assert combined.output_tokens == 10
    assert combined.total_tokens == 26

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
def test_combine_responses_fallback_usage_passthrough_for_unknown_event():
    """Events that are not completed/failed/incomplete are not mutated."""
    other = MagicMock()  # not a ResponseCompletedEvent etc. → isinstance false
    partial = ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    Router._combine_responses_fallback_usage(other, partial)

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
def test_build_responses_continuation_input_from_string():
    out = Router._build_responses_continuation_input("Hello world", "partial assistant text")
    assert len(out) == 3
    assert out[0]["role"] == "user"
    assert out[0]["content"][0]["text"] == "Hello world"
    assert out[1]["role"] == "developer"
    assert out[2]["role"] == "assistant"
    assert out[2]["content"][0]["text"] == "partial assistant text"

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
def test_build_responses_continuation_input_from_list_preserves_items():
    existing: List[Any] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "msg1"}],
        }
    ]
    out = Router._build_responses_continuation_input(existing, "partial")
    assert len(out) == 3
    assert out[0]["content"][0]["text"] == "msg1"
    assert out[1]["role"] == "developer"
    assert out[2]["role"] == "assistant"

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
def test_build_responses_continuation_input_from_none():
    out = Router._build_responses_continuation_input(None, "partial")
    assert len(out) == 2
    assert out[0]["role"] == "developer"
    assert out[1]["role"] == "assistant"

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_passthrough():
    """
    Without MidStreamFallbackError, the wrapper yields source events
    unchanged and returns a BaseResponsesAPIStreamingIterator subclass.
    """
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    events = [_make_completed_event(1, 1, 2)]

    class _FakeSource:
        """Minimal source iterator. Provides every attribute the wrapper
        constructor reads from source_iterator."""

        def __init__(self) -> None:
            self._i = 0
            self.completed_response = None
            self.response = MagicMock()
            self.model = "openai/gpt-4o-mini"
            self.logging_obj = MagicMock()
            self.responses_api_provider_config = MagicMock()
            self.start_time = 0.0
            self.litellm_metadata = {}
            self.custom_llm_provider = "openai"
            self.request_data = {}
            self.call_type = "aresponses"
            self._hidden_params: dict = {}

        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self):
            if self._i >= len(events):
                raise StopAsyncIteration
            ev = events[self._i]
            self._i += 1
            return ev

        async def aclose(self):
            return None

    router = _make_router()
    source = _FakeSource()

    wrapper = await router._aresponses_streaming_iterator(source, initial_kwargs={"model": "primary"})
    assert isinstance(wrapper, BaseResponsesAPIStreamingIterator)

    collected = [ev async for ev in wrapper]
    assert len(collected) == 1
    assert collected[0].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_with_streaming_fallbacks_non_streaming_passthrough():
    """Non-streaming response is returned unchanged, no wrap."""
    router = _make_router()
    plain_response = MagicMock()

    async def fake_original(**_kwargs):
        return plain_response

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks_helper",
        new=AsyncMock(return_value=plain_response),
    ):
        out = await router._aresponses_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=False,
        )
    assert out is plain_response

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_with_streaming_fallbacks_wraps_streaming_iterator():
    """Streaming response is wrapped via _aresponses_streaming_iterator."""
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    router = _make_router()
    streaming_iter = MagicMock(spec=BaseResponsesAPIStreamingIterator)
    wrapped = MagicMock(spec=BaseResponsesAPIStreamingIterator)

    async def fake_original(**_kwargs):
        return streaming_iter

    with (
        patch.object(
            router,
            "_ageneric_api_call_with_fallbacks_helper",
            new=AsyncMock(return_value=streaming_iter),
        ),
        patch.object(
            router,
            "_aresponses_streaming_iterator",
            new=AsyncMock(return_value=wrapped),
        ) as mock_wrap,
    ):
        out = await router._aresponses_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=True,
        )
    assert out is wrapped
    mock_wrap.assert_awaited_once()

def _make_three_tier_router(**router_kwargs) -> Router:
    return Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "openai/primary-model", "api_key": "sk-test"}},
            {"model_name": "fb1", "litellm_params": {"model": "openai/fb1-model", "api_key": "sk-test"}},
            {"model_name": "fb2", "litellm_params": {"model": "openai/fb2-model", "api_key": "sk-test"}},
        ],
        num_retries=0,
        **router_kwargs,
    )

def _mid_stream_failure(model: str):
    import litellm
    from litellm.exceptions import MidStreamFallbackError

    return MidStreamFallbackError(
        message="stream dropped",
        model=model,
        llm_provider="openai",
        original_exception=litellm.InternalServerError(message="stream dropped", llm_provider="openai", model=model),
        is_pre_first_chunk=True,
    )

def _scripted_responses_stream(events: list, error: Exception | None = None):
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

    class _ScriptedStream(BaseResponsesAPIStreamingIterator):
        def __init__(self) -> None:
            self._events = list(events)
            self._hidden_params: dict = {}
            self.completed_response = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._events:
                return self._events.pop(0)
            if error is not None:
                raise error
            raise StopAsyncIteration

        async def aclose(self) -> None:
            return None

    return _ScriptedStream()

def _three_tier_original(calls: list, primary_fails_pre_stream: bool):
    import litellm

    completed_event = _make_completed_event(1, 1, 2)

    async def fake_original(**kwargs):
        model = kwargs["model"]
        calls.append(model)
        if model == "openai/primary-model":
            if primary_fails_pre_stream:
                raise litellm.InternalServerError(message="primary down", llm_provider="openai", model=model)
            return _scripted_responses_stream([], _mid_stream_failure(model))
        if model == "openai/fb1-model":
            return _scripted_responses_stream([], _mid_stream_failure(model))
        return _scripted_responses_stream([completed_event])

    return fake_original, completed_event

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_pre_stream_primary_failure_then_hop_stream_failure_reaches_second_entry():
    """Regression: fallbacks=[{"primary": ["fb1", "fb2"]}]. The primary fails before streaming,
    fb1 is reached through the regular fallback chain and then fails mid-stream. Only the
    primary's stream used to be wrapped, so fb1's mid-stream failure either re-raised or
    re-tried fb1 itself; fb2 was unreachable."""
    router = _make_three_tier_router(fallbacks=[{"primary": ["fb1", "fb2"]}])
    calls: list = []
    fake_original, completed_event = _three_tier_original(calls, primary_fails_pre_stream=True)

    stream = await router._aresponses_with_streaming_fallbacks(
        original_function=fake_original, model="primary", stream=True, input="hi"
    )
    collected = [event async for event in stream]

    assert calls == ["openai/primary-model", "openai/fb1-model", "openai/fb2-model"]
    assert collected == [completed_event]

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_two_consecutive_mid_stream_failures_reach_second_entry():
    """Regression: the primary and fb1 both fail mid-stream; fb2 must still be tried."""
    router = _make_three_tier_router(fallbacks=[{"primary": ["fb1", "fb2"]}])
    calls: list = []
    fake_original, completed_event = _three_tier_original(calls, primary_fails_pre_stream=False)

    stream = await router._aresponses_with_streaming_fallbacks(
        original_function=fake_original, model="primary", stream=True, input="hi"
    )
    collected = [event async for event in stream]

    assert calls == ["openai/primary-model", "openai/fb1-model", "openai/fb2-model"]
    assert collected == [completed_event]

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_per_request_fallbacks_survive_into_hop_streams():
    """Regression: a request-level fallbacks list (key or team router_settings) is popped
    before each attempt runs, so a hop's mid-stream re-entry used to see only the router's
    own (empty) list and gave up after fb1."""
    router = _make_three_tier_router()
    calls: list = []
    fake_original, completed_event = _three_tier_original(calls, primary_fails_pre_stream=False)

    stream = await router._aresponses_with_streaming_fallbacks(
        original_function=fake_original,
        model="primary",
        stream=True,
        input="hi",
        fallbacks=[{"primary": ["fb1", "fb2"]}],
    )
    collected = [event async for event in stream]

    assert calls == ["openai/primary-model", "openai/fb1-model", "openai/fb2-model"]
    assert collected == [completed_event]

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_attempt_strips_the_controls_carrier_and_wraps_every_hop_stream():
    """Each attempt of the chain, not only the primary's, comes back wrapped for mid-stream
    failover, and the per-request controls carrier rides into the wrapper's re-entry kwargs
    without ever reaching the provider call."""
    from types import MappingProxyType

    from litellm.router_utils.fallback_event_handlers import (
        MID_STREAM_FALLBACK_CONTROLS_KEY,
        MidStreamFallbackControls,
    )

    router = _make_three_tier_router()
    completed_event = _make_completed_event(1, 1, 2)
    hop_stream = _scripted_responses_stream([completed_event])
    seen: dict = {}

    async def fake_original(**kwargs):
        seen.update(kwargs)
        return hop_stream

    controls = MidStreamFallbackControls(MappingProxyType({"fallbacks": [{"primary": ["fb1", "fb2"]}]}))
    stream = await router._ageneric_api_call_with_fallbacks_responses_attempt(
        model="fb1",
        original_generic_function=fake_original,
        stream=True,
        input="hi",
        **{MID_STREAM_FALLBACK_CONTROLS_KEY: controls},
    )
    collected = [event async for event in stream]

    assert seen["model"] == "openai/fb1-model"
    assert MID_STREAM_FALLBACK_CONTROLS_KEY not in seen
    assert "fallbacks" not in seen
    assert stream is not hop_stream
    assert collected == [completed_event]

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_fallback_on_in_stream_error_event():
    """A retriable in-stream error event (429) must trigger the router's mid-stream
    fallback path: the wrapper catches MidStreamFallbackError raised by the source
    iterator and yields the fallback stream instead of surfacing the error."""
    import json
    from unittest.mock import Mock

    import litellm
    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
    from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ErrorEvent, ErrorEventError

    router = _make_router()

    error_payload = {
        "type": "error",
        "error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "rate limited"},
    }
    sse_bytes = f"data: {json.dumps(error_payload)}\n\n".encode()

    async def mock_aiter_bytes():
        yield sse_bytes

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=0,
        error=ErrorEventError(type="tokens", code="rate_limit_exceeded", message="rate limited"),
    )

    source = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    fallback_event = _make_completed_event(1, 1, 2)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_FallbackStream()),
    ) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary", "input": "original question"},
        )
        collected = [ev async for ev in wrapped]

    assert collected == [fallback_event]
    mock_fallback.assert_awaited_once()
    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.status_code == 429
    assert isinstance(raised.original_exception, litellm.RateLimitError)
    assert raised.original_exception.status_code == 429
    assert mock_fallback.await_args.kwargs["kwargs"]["input"] == "original question"

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_fallback_uses_continuation_input_after_partial_content():
    """When output text was already streamed before the error, the fallback re-entry
    must carry a continuation input with the partial assistant text instead of
    retrying the original input from scratch (which would duplicate streamed content)."""
    import json
    from unittest.mock import Mock

    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
    from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ErrorEvent, ErrorEventError

    router = _make_router()

    events = [
        {"type": "response.output_text.delta", "delta": "partial answer"},
        {"type": "error", "error": {"type": "server_error", "code": "internal_error", "message": "boom"}},
    ]
    sse_payload = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    async def mock_aiter_bytes():
        yield sse_payload

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def transform(model, parsed_chunk, logging_obj):
        if parsed_chunk.get("type") == "error":
            return ErrorEvent(
                type=ResponsesAPIStreamEvents.ERROR,
                sequence_number=0,
                error=ErrorEventError(**parsed_chunk["error"]),
            )
        delta_event = Mock()
        delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        delta_event.delta = parsed_chunk["delta"]
        return delta_event

    mock_config.transform_streaming_response.side_effect = transform

    source = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    fallback_event = _make_completed_event(1, 1, 2)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_FallbackStream()),
    ) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary", "input": "original question"},
        )
        collected = [ev async for ev in wrapped]

    assert collected[-1] == fallback_event
    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.is_pre_first_chunk is False
    assert raised.generated_content == "partial answer"
    continuation = mock_fallback.await_args.kwargs["kwargs"]["input"]
    assert isinstance(continuation, list)
    assert continuation[0]["content"][0]["text"] == "original question"
    assert continuation[-2]["role"] == "developer"
    assert continuation[-1]["role"] == "assistant"
    assert continuation[-1]["content"][0]["text"] == "partial answer"

@pytest.mark.usefixtures("_vcr_outcome_gate_router_unit", "setup_and_teardown_router_unit")
@pytest.mark.asyncio
async def test_aresponses_client_error_event_skips_fallback():
    """A 400-mapped in-stream error (raised as APIError, not MidStreamFallbackError)
    must surface to the caller without invoking the router's fallback path."""
    import litellm

    router = _make_router()

    class _ClientErrorSource:
        completed_response = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise litellm.APIError(
                status_code=400,
                message="bad request",
                llm_provider="openai",
                model="gpt-5",
            )

    wrapped = await router._aresponses_streaming_iterator(
        response=_ClientErrorSource(),
        initial_kwargs={"model": "primary"},
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        with pytest.raises(litellm.APIError) as exc_info:
            async for _ in wrapped:
                pass

    assert exc_info.value.status_code == 400
    mock_fallback.assert_not_awaited()
