"""The shared Responses API config contract."""

import asyncio, importlib, json
from collections.abc import AsyncIterator, Callable, Mapping

import httpx
import pytest
import respx

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm import Router
from pydantic import BaseModel, ConfigDict
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
from typing import Any, Final, List
from unittest.mock import Mock, patch


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
        with (
            patch("asyncio.get_running_loop"),
            patch("litellm.responses.streaming_iterator.spawn_logging_task"),
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
            patch("asyncio.get_running_loop"),
            patch("litellm.responses.streaming_iterator.spawn_logging_task") as mock_spawn_logging_task,
            patch("litellm.responses.streaming_iterator.executor") as mock_executor,
        ):
            result = iterator._process_chunk(json.dumps(test_chunk_data))

            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE
            assert iterator.completed_response == result

            # Success handlers are dispatched as one async task (via _handle_logging_completed_response);
            # the sync handler must never be submitted to the executor concurrently (LIT-4210)
            spawned_task_names: Final = [call.kwargs["task_name"] for call in mock_spawn_logging_task.call_args_list]
            assert spawned_task_names.count("Responses stream success logging") == 1
            mock_executor.submit.assert_not_called()

            # Failure handlers should NOT have been called
            mock_logging_obj.async_failure_handler.assert_not_called()
            mock_logging_obj.failure_handler.assert_not_called()

def _sse(*events: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events),
    )


def _openai_stream_cut_after(partial: str) -> httpx.Response:
    return _sse(
        *(
            ({
                "type": "response.output_text.delta", "sequence_number": 0,
                "item_id": "msg_primary", "output_index": 0, "content_index": 0, "delta": partial,
            },)
            if partial else ()
        ),
        {
            "type": "error", "sequence_number": 1,
            "error": {"type": "server_error", "code": "internal_error", "message": "interrupted"},
        },
    )


def _anthropic_stream(text: str) -> httpx.Response:
    return _sse(
        {"type": "message_start", "message": {
            "id": "msg_fallback", "type": "message", "role": "assistant",
            "model": "claude-fallback", "content": [], "usage": {"input_tokens": 10, "output_tokens": 0},
        }},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )


def _primary_with_fallback(fallback_model: str) -> Router:
    return Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "openai/gpt-6-astra", "api_key": "synthetic"}},
            {"model_name": "fallback", "litellm_params": {"model": fallback_model, "api_key": "synthetic"}},
        ],
        fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
    )


def _streamed_text(events: tuple[Any, ...]) -> str:
    return "".join(event.delta for event in events if event.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA)


@pytest.fixture
def claude_with_prefill() -> str:
    model: Final = "anthropic/claude-prefill-capable-test-model"
    litellm.register_model(
        {model: {"litellm_provider": "anthropic", "mode": "chat", "supports_assistant_prefill": True}},
        persist_across_reloads=False,
    )
    litellm.get_model_info.cache_clear()
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", ["", "1, 2, "])
async def test_responses_fallback_can_continue_on_a_provider_without_prefill(
    partial: str, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    prompt: Final = "Print the integers 1 through 5, separated by commas."
    primary: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=_openai_stream_cut_after(partial)
    )

    def no_prefill_provider(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        if body["messages"][-1]["role"] != "user":
            return httpx.Response(
                400,
                json={"type": "error", "error": {
                    "type": "invalid_request_error", "message": "The conversation must end with a user message",
                }},
            )
        return _anthropic_stream("3, 4, 5" if partial else "1, 2, 3, 4, 5")

    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(side_effect=no_prefill_provider)
    router: Final = _primary_with_fallback("anthropic/claude-opus-5-5")
    try:
        stream: Final = await router.aresponses(model="primary", input=prompt, stream=True, max_output_tokens=100)
        events: Final = tuple([event async for event in stream])
        assert _streamed_text(events) == "1, 2, 3, 4, 5"
        assert events[-1].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        assert primary.call_count == fallback.call_count == 1
        messages: Final = json.loads(fallback.calls.last.request.content)["messages"]
        assert messages[0] == {"role": "user", "content": [{"type": "text", "text": prompt}]}
        if partial:
            assert [message["role"] for message in messages] == ["user", "assistant", "user"]
            assert messages[-2] == {"role": "assistant", "content": [{"type": "text", "text": partial}]}
            assert json.dumps(partial) in messages[-1]["content"][0]["text"]
        else:
            assert len(messages) == 1
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_responses_fallback_continues_a_claude_that_takes_prefill_from_its_own_partial(
    claude_with_prefill: str, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(return_value=_openai_stream_cut_after("1, 2, "))
    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_anthropic_stream(" 3, 4, 5")
    )
    router: Final = _primary_with_fallback(claude_with_prefill)
    try:
        stream: Final = await router.aresponses(
            model="primary", input="Print the integers 1 through 5.", stream=True, max_output_tokens=100
        )
        events: Final = tuple([event async for event in stream])
        assert _streamed_text(events) == "1, 2, 3, 4, 5"
        assert events[-1].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        messages: Final = json.loads(fallback.calls.last.request.content)["messages"]
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert messages[-1]["content"][0]["text"].startswith("1, 2,")
    finally:
        router.discard()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_takes_prefill", [True, False], ids=["prefill", "request_to_continue"])
async def test_responses_fallback_continuation_drops_reasoning_the_fallback_cannot_decrypt(
    fallback_takes_prefill: bool,
    claude_with_prefill: str,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(return_value=_openai_stream_cut_after("1, 2, "))
    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_anthropic_stream(" 3, 4, 5")
    )
    router: Final = _primary_with_fallback(claude_with_prefill if fallback_takes_prefill else "anthropic/claude-opus-5-5")
    try:
        stream: Final = await router.aresponses(
            model="primary",
            input=[
                {"role": "user", "content": "Print the integers 1 through 5."},
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "primary-only-ciphertext"},
            ],
            stream=True,
            max_output_tokens=100,
        )
        _ = [event async for event in stream]
        assert fallback.call_count == 1
        assert "primary-only-ciphertext" not in fallback.calls.last.request.content.decode()
    finally:
        router.discard()


class _Numbers(BaseModel):
    model_config = ConfigDict(frozen=True)

    numbers: tuple[int, ...]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "constraint",
    [
        {"text": {"format": {"type": "json_schema", "name": "numbers", "strict": True, "schema": {"type": "object"}}}},
        {"text": {"format": {"type": "json_object"}}},
        {"tool_choice": "required", "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]},
        {"text_format": _Numbers},
    ],
    ids=["json_schema", "json_object", "required_tool", "sdk_text_format"],
)
async def test_responses_fallback_surfaces_the_primary_error_once_constrained_output_was_streamed(
    constraint: dict[str, object],
    claude_with_prefill: str,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=_openai_stream_cut_after('{"numbers": [1, 2, ')
    )
    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_anthropic_stream("3]}")
    )
    router: Final = _primary_with_fallback(claude_with_prefill)
    try:
        stream: Final = await router.aresponses(
            model="primary", input="Numbers 1 to 3 as JSON.", stream=True, max_output_tokens=100, **constraint
        )
        with pytest.raises(litellm.InternalServerError, match="interrupted"):
            [event async for event in stream]
        assert fallback.call_count == 0
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_responses_fallback_continues_within_the_callers_output_ceiling(
    claude_with_prefill: str, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(return_value=_openai_stream_cut_after("1, 2, "))
    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_anthropic_stream("3, 4, 5")
    )
    router: Final = _primary_with_fallback(claude_with_prefill)
    try:
        stream: Final = await router.aresponses(model="primary", input="Count to 5.", stream=True, max_output_tokens=100)
        events: Final = tuple([event async for event in stream])
        assert _streamed_text(events) == "1, 2, 3, 4, 5"
        assert json.loads(fallback.calls.last.request.content)["max_tokens"] == 100 - litellm.token_counter(
            text="1, 2, "
        )
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_responses_fallback_surfaces_the_primary_error_once_the_cut_stream_used_the_output_ceiling(
    claude_with_prefill: str, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(return_value=_openai_stream_cut_after("1, 2, "))
    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_anthropic_stream("3, 4, 5")
    )
    router: Final = _primary_with_fallback(claude_with_prefill)
    try:
        stream: Final = await router.aresponses(
            model="primary", input="Count to 5.", stream=True, max_output_tokens=litellm.token_counter(text="1, 2, ")
        )
        with pytest.raises(litellm.InternalServerError, match="interrupted"):
            [event async for event in stream]
        assert fallback.call_count == 0
    finally:
        router.discard()


def _openai_stream_cut_mid_tool_call(preamble: str, *, announce_item: bool = True) -> httpx.Response:
    text: Final = {
        "type": "response.output_text.delta", "sequence_number": 0,
        "item_id": "msg_primary", "output_index": 0, "content_index": 0, "delta": preamble,
    }
    item_added: Final = {
        "type": "response.output_item.added", "sequence_number": 1, "output_index": 1,
        "item": {
            "type": "function_call", "id": "fc_primary", "call_id": "call_primary",
            "name": "lookup", "arguments": "", "status": "in_progress",
        },
    }
    arguments: Final = {
        "type": "response.function_call_arguments.delta", "sequence_number": 2,
        "item_id": "fc_primary", "output_index": 1, "delta": '{"city": "Par',
    }
    error: Final = {
        "type": "error", "sequence_number": 3,
        "error": {"type": "server_error", "code": "internal_error", "message": "interrupted"},
    }
    return _sse(*((text,) if preamble else ()), *((item_added,) if announce_item else ()), arguments, error)


def _anthropic_stream_cut_mid_tool_call(preamble: str, tool_block: Mapping[str, object] | None = None) -> httpx.Response:
    return _sse(
        {"type": "message_start", "message": {
            "id": "msg_primary", "type": "message", "role": "assistant",
            "model": "claude-primary", "content": [], "usage": {"input_tokens": 10, "output_tokens": 0},
        }},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": preamble}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": tool_block or {
            "type": "tool_use", "id": "toolu_primary", "name": "lookup", "input": {},
        }},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"city": "Par'}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "interrupted"}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("announce_item", [True, False], ids=["announced_item", "arguments_only"])
@pytest.mark.parametrize("preamble", ["Let me look that up. ", ""], ids=["after_text", "without_text"])
async def test_responses_fallback_surfaces_the_primary_error_once_the_cut_stream_began_a_tool_call(
    announce_item: bool, preamble: str, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=_openai_stream_cut_mid_tool_call(preamble, announce_item=announce_item)
    )
    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_anthropic_stream("It is sunny in Paris.")
    )
    router: Final = _primary_with_fallback("anthropic/claude-opus-5-5")
    try:
        stream: Final = await router.aresponses(
            model="primary",
            input="What is the weather in Paris?",
            stream=True,
            tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        )
        with pytest.raises(litellm.InternalServerError, match="interrupted"):
            [event async for event in stream]
        assert fallback.call_count == 0
    finally:
        router.discard()


def _openai_stream_cut_after_event(preamble: str, event: Mapping[str, object]) -> httpx.Response:
    return _sse(
        {
            "type": "response.output_text.delta", "sequence_number": 0,
            "item_id": "msg_primary", "output_index": 0, "content_index": 0, "delta": preamble,
        },
        {**event, "sequence_number": 1},
        {
            "type": "error", "sequence_number": 2,
            "error": {"type": "server_error", "code": "internal_error", "message": "interrupted"},
        },
    )


def _openai_stream_cut_after_output_item(preamble: str, item: Mapping[str, object]) -> httpx.Response:
    return _openai_stream_cut_after_event(
        preamble, {"type": "response.output_item.added", "output_index": 1, "item": item}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item",
    [
        {"type": "shell_call", "id": "sh_1", "call_id": "call_1", "action": {"commands": ["ls"]}, "status": "in_progress"},
        {
            "type": "apply_patch_call", "id": "apc_1", "call_id": "call_1", "status": "in_progress",
            "operation": {"type": "update_file", "path": "a.py", "diff": "@@"},
        },
        {
            "type": "local_shell_call", "id": "lsh_1", "call_id": "call_1", "status": "in_progress",
            "action": {"type": "exec", "command": ["ls"], "env": {}},
        },
        {
            "type": "computer_call", "id": "cu_1", "call_id": "call_1", "status": "in_progress",
            "action": {"type": "screenshot"}, "pending_safety_checks": [],
        },
        {"type": "mcp_approval_request", "id": "mcpr_1", "name": "lookup", "arguments": "{}", "server_label": "docs"},
    ],
    ids=lambda item: item["type"],
)
async def test_responses_fallback_surfaces_the_primary_error_once_the_cut_stream_began_a_client_run_tool_call(
    item: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=_openai_stream_cut_after_output_item("Let me check the repo. ", item)
    )
    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_anthropic_stream("The repo has two files.")
    )
    router: Final = _primary_with_fallback("anthropic/claude-opus-5-5")
    try:
        stream: Final = await router.aresponses(model="primary", input="What is in the repo?", stream=True)
        with pytest.raises(litellm.InternalServerError, match="interrupted"):
            [event async for event in stream]
        assert fallback.call_count == 0
    finally:
        router.discard()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_run_event",
    [
        {
            "type": "response.output_item.added", "output_index": 1,
            "item": {
                "type": "mcp_call", "id": "mcp_1", "name": "ask_question", "arguments": "{}",
                "server_label": "docs", "status": "in_progress",
            },
        },
        {"type": "response.mcp_call_arguments.delta", "output_index": 1, "item_id": "mcp_1", "delta": '{"question": "set'},
    ],
    ids=["mcp_call_item", "mcp_call_arguments"],
)
async def test_responses_fallback_continues_after_a_server_run_mcp_call(
    server_run_event: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=_openai_stream_cut_after_event("Let me check the docs. ", server_run_event)
    )
    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_anthropic_stream("The docs cover setup.")
    )
    router: Final = _primary_with_fallback("anthropic/claude-opus-5-5")
    try:
        stream: Final = await router.aresponses(model="primary", input="What do the docs cover?", stream=True)
        events: Final = tuple([event async for event in stream])
        assert _streamed_text(events).endswith("The docs cover setup.")
        assert fallback.call_count == 1
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_responses_fallback_continues_when_allowed_tools_in_auto_mode_leaves_the_reply_open(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    openai: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        side_effect=[
            _openai_stream_cut_after("1, 2, "),
            _sse(
                {
                    "type": "response.output_text.delta", "sequence_number": 0,
                    "item_id": "msg_fallback", "output_index": 0, "content_index": 0, "delta": "3, 4, 5",
                },
                {
                    "type": "response.completed", "sequence_number": 1,
                    "response": {
                        "id": "resp_fallback", "object": "response", "created_at": 0, "status": "completed",
                        "model": "gpt-6.1-terra", "output": [],
                        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
                    },
                },
            ),
        ]
    )
    router: Final = _primary_with_fallback("openai/gpt-6.1-terra")
    try:
        stream: Final = await router.aresponses(
            model="primary",
            input="Print the integers 1 through 5.",
            stream=True,
            tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
            tool_choice={"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "lookup"}]},
        )
        events: Final = tuple([event async for event in stream])
        assert _streamed_text(events) == "1, 2, 3, 4, 5"
        assert openai.call_count == 2
        continued_input: Final = json.loads(openai.calls.last.request.content)["input"]
        assert [item["role"] for item in continued_input[-2:]] == ["assistant", "user"]
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_responses_fallback_surfaces_a_content_filter_that_cut_the_stream_mid_answer(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=_sse(
            {
                "type": "response.output_text.delta", "sequence_number": 0,
                "item_id": "msg_primary", "output_index": 0, "content_index": 0, "delta": "Step one is to obtain the",
            },
            {
                "type": "error", "sequence_number": 1,
                "error": {
                    "type": "invalid_request_error", "code": "content_policy_violation",
                    "message": "The response was halted mid-stream.", "param": None,
                },
            },
        )
    )
    fallback: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_anthropic_stream(" reagents.")
    )
    router: Final = Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "openai/gpt-6-astra", "api_key": "synthetic"}},
            {"model_name": "fallback", "litellm_params": {"model": "anthropic/claude-opus-5-5", "api_key": "synthetic"}},
        ],
        fallbacks=[{"primary": ["fallback"]}],
        content_policy_fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
    )
    try:
        stream: Final = await router.aresponses(model="primary", input="How do I make it?", stream=True)
        with pytest.raises(litellm.ContentPolicyViolationError, match="halted mid-stream"):
            [event async for event in stream]
        assert fallback.call_count == 0
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_bridged_responses_fallback_surfaces_the_primary_error_once_the_cut_stream_began_a_tool_call(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    provider: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=[
            _anthropic_stream_cut_mid_tool_call("Let me look that up. "),
            _anthropic_stream("It is sunny in Paris."),
        ]
    )
    router: Final = Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "anthropic/claude-primary", "api_key": "synthetic"}},
            {"model_name": "fallback", "litellm_params": {"model": "anthropic/claude-opus-5-5", "api_key": "synthetic"}},
        ],
        fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
    )
    try:
        stream: Final = await router.aresponses(
            model="primary",
            input="What is the weather in Paris?",
            stream=True,
            tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        )
        with pytest.raises(litellm.InternalServerError, match="interrupted"):
            [event async for event in stream]
        assert provider.call_count == 1
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_bridged_responses_fallback_continues_after_a_server_run_tool_search(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    provider: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=[
            _anthropic_stream_cut_mid_tool_call(
                "Let me find the right tool. ",
                {"type": "server_tool_use", "id": "srvtoolu_primary", "name": "tool_search_tool_regex", "input": {}},
            ),
            _anthropic_stream("It is sunny in Paris."),
        ]
    )
    router: Final = Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "anthropic/claude-primary", "api_key": "synthetic"}},
            {"model_name": "fallback", "litellm_params": {"model": "anthropic/claude-opus-5-5", "api_key": "synthetic"}},
        ],
        fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
    )
    try:
        stream: Final = await router.aresponses(
            model="primary",
            input="What is the weather in Paris?",
            stream=True,
            tools=[
                {"type": "tool_search", "execution": "server"},
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}, "defer_loading": True},
            ],
        )
        events: Final = tuple([event async for event in stream])
        assert _streamed_text(events).endswith("It is sunny in Paris.")
        assert provider.call_count == 2
    finally:
        router.discard()


def _openai_completed(model: str) -> Mapping[str, object]:
    return {
        "type": "response.completed", "sequence_number": 1,
        "response": {
            "id": f"resp_{model}", "object": "response", "created_at": 0, "status": "completed", "model": model,
            "output": [], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary_fails_pre_stream", "request_level_fallbacks"),
    [(True, False), (False, False), (False, True)],
    ids=["pre_stream_then_hop_stream_failure", "two_mid_stream_failures", "request_level_fallbacks"],
)
async def test_responses_fallback_chain_reaches_the_second_entry_after_a_hop_stream_fails(
    primary_fails_pre_stream: bool,
    request_level_fallbacks: bool,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: fallbacks=[{"primary": ["fb1", "fb2"]}]. fb1 is reached (after the primary failed before or
    during streaming) and then fails mid-stream; fb2 must still be tried, including when the fallbacks list
    came with the request rather than the router."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    calls: Final[list[str]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        model: Final = json.loads(request.content)["model"]
        calls.append(model)
        if model == "primary-model" and primary_fails_pre_stream:
            return httpx.Response(500, json={"error": {"message": "primary down", "type": "server_error"}})
        if model in ("primary-model", "fb1-model"):
            return _openai_stream_cut_after("")
        return _sse(_openai_completed(model))

    respx_mock.post("https://api.openai.com/v1/responses").mock(side_effect=upstream)
    fallbacks: Final = [{"primary": ["fb1", "fb2"]}]
    router: Final = Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": f"openai/{name}-model", "api_key": "sk-test"}}
            for name in ("primary", "fb1", "fb2")
        ],
        num_retries=0,
        **({} if request_level_fallbacks else {"fallbacks": fallbacks}),
    )
    try:
        stream: Final = await router.aresponses(
            model="primary", input="hi", stream=True, **({"fallbacks": fallbacks} if request_level_fallbacks else {})
        )
        events: Final = [event async for event in stream]
    finally:
        router.discard()

    assert calls == ["primary-model", "fb1-model", "fb2-model"]
    assert [event.type for event in events] == [ResponsesAPIStreamEvents.RESPONSE_COMPLETED]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "falls_back"),
    [
        ({"type": "tokens", "code": "rate_limit_exceeded", "message": "rate limited"}, True),
        ({"type": "invalid_request_error", "code": "invalid_request_error", "message": "bad request"}, False),
    ],
    ids=["retriable_429_falls_back", "client_400_surfaces"],
)
async def test_responses_in_stream_error_event_falls_back_only_when_retriable(
    error: Mapping[str, object], falls_back: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retriable in-stream error event (429) takes the router's mid-stream fallback path; a 400-mapped one
    surfaces to the caller without invoking a fallback."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    calls: Final[list[str]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        model: Final = json.loads(request.content)["model"]
        calls.append(model)
        if model == "primary-model":
            return _sse({"type": "error", "sequence_number": 0, "error": error})
        return _sse(_openai_completed(model))

    respx_mock.post("https://api.openai.com/v1/responses").mock(side_effect=upstream)
    router: Final = Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": f"openai/{name}-model", "api_key": "sk-test"}}
            for name in ("primary", "fallback")
        ],
        fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
    )
    try:
        stream: Final = await router.aresponses(model="primary", input="hi", stream=True)
        if falls_back:
            events: Final = [event async for event in stream]
            assert [event.type for event in events] == [ResponsesAPIStreamEvents.RESPONSE_COMPLETED]
            assert calls == ["primary-model", "fallback-model"]
        else:
            with pytest.raises(litellm.BadRequestError) as exc_info:
                async for _ in stream:
                    pass
            assert exc_info.value.status_code == 400
            assert calls == ["primary-model"]
    finally:
        router.discard()


def _openai_response(model: str, status: str) -> Mapping[str, object]:
    return {
        "id": f"resp_{model}",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": model,
        "output": [],
    }


def _openai_created(model: str) -> Mapping[str, object]:
    return {"type": "response.created", "sequence_number": 0, "response": _openai_response(model, "in_progress")}


_INTERRUPTED: Final = {"code": "server_error", "message": "interrupted"}
_IN_STREAM_FAILURES: Final = (
    {"type": "error", "error": {"type": "server_error", **_INTERRUPTED}},
    {"type": "response.failed", "response": {**_openai_response("primary-model", "failed"), "error": _INTERRUPTED}},
)
_COMPLETED_TOOL_CALL: Final = {
    "type": "response.output_item.done",
    "output_index": 0,
    "item": {
        "type": "function_call",
        "id": "fc_primary",
        "call_id": "call_primary",
        "name": "shell",
        "arguments": '{"command": "echo run"}',
        "status": "completed",
    },
}
_SHELL_TOOL: Final = {"type": "function", "name": "shell", "parameters": {"type": "object"}}


def _upstream_streaming(*primary_events: Mapping[str, object]) -> Callable[[httpx.Request], httpx.Response]:
    def upstream(request: httpx.Request) -> httpx.Response:
        model: Final = json.loads(request.content)["model"]
        if model == "primary-model":
            return _sse(*({**event, "sequence_number": index} for index, event in enumerate(primary_events)))
        return _sse(_openai_created(model), _openai_completed(model))

    return upstream


def _requested_models(route: respx.Route) -> list[str]:
    return [json.loads(call.request.content)["model"] for call in route.calls]


def _openai_primary_with_fallback() -> Router:
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": f"openai/{name}-model", "api_key": "sk-test"}}
            for name in ("primary", "fallback")
        ],
        fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", _IN_STREAM_FAILURES, ids=["error_event", "response_failed"])
async def test_responses_fallback_surfaces_the_primary_error_once_a_completed_tool_call_was_delivered(
    failure: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        side_effect=_upstream_streaming(_openai_created("primary-model"), _COMPLETED_TOOL_CALL, failure)
    )
    router: Final = _openai_primary_with_fallback()
    try:
        stream: Final = await router.aresponses(
            model="primary", input="Run echo run.", stream=True, tools=[_SHELL_TOOL], tool_choice="required"
        )
        assert isinstance(stream, AsyncIterator)
        events: Final = aiter(stream)
        created: Final = await anext(events)
        done: Final = await anext(events)
        with pytest.raises(litellm.InternalServerError, match="interrupted"):
            await anext(events)
    finally:
        router.discard()
    assert created.type == ResponsesAPIStreamEvents.RESPONSE_CREATED
    assert done.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
    assert getattr(getattr(done, "item"), "call_id") == "call_primary"
    assert _requested_models(route) == ["primary-model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", _IN_STREAM_FAILURES, ids=["error_event", "response_failed"])
async def test_responses_fallback_takes_over_a_stream_that_failed_before_any_output_item(
    failure: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        side_effect=_upstream_streaming(_openai_created("primary-model"), failure)
    )
    router: Final = _openai_primary_with_fallback()
    try:
        stream: Final = await router.aresponses(
            model="primary", input="Run echo run.", stream=True, tools=[_SHELL_TOOL], tool_choice="required"
        )
        events: Final = [event async for event in stream]
    finally:
        router.discard()
    assert [event.type for event in events] == [
        ResponsesAPIStreamEvents.RESPONSE_CREATED,
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    ]
    assert getattr(getattr(events[0], "response"), "model") == "fallback-model"
    assert _requested_models(route) == ["primary-model", "fallback-model"]
