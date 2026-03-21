"""
Unit tests for BaseResponsesAPIStreamingIterator

Tests core functionality including:
1. Processing chunks and handling ResponseCompletedEvent 
2. Ensuring _update_responses_api_response_id_with_model_id is called for final chunk
3. Verifying ID update is NOT called for non-final chunks (delta events)
4. Edge case handling for invalid JSON, empty chunks, and [DONE] markers

These tests ensure the streaming iterator correctly processes response chunks 
and applies model ID updates only to completed responses, as required for proper
response tracking and logging.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    OutputTextDeltaEvent
)


class TestBaseResponsesAPIStreamingIterator:
    """Test cases for BaseResponsesAPIStreamingIterator"""

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
        
        # Mock the _update_responses_api_response_id_with_model_id method
        updated_response = Mock(spec=ResponsesAPIResponse)
        updated_response.id = "updated_response_id"
        
        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )
        
        # Prepare test chunk data
        test_chunk_data = {
            "type": "response.completed",
            "response": {
                "id": "original_response_id",
                "output": [{"type": "message", "content": [{"text": "Hello World"}]}]
            }
        }
        
        with patch.object(
            ResponsesAPIRequestUtils, 
            '_update_responses_api_response_id_with_model_id',
            return_value=updated_response
        ) as mock_update_id:
            # Process the chunk
            result = iterator._process_chunk(json.dumps(test_chunk_data))
            
            # Assertions
            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
            
            # Verify that _update_responses_api_response_id_with_model_id was called
            mock_update_id.assert_called_once_with(
                responses_api_response=mock_responses_api_response,
                litellm_metadata={"model_info": {"id": "model_123"}},
                custom_llm_provider="openai"
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
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        
        # Create a mock OutputTextDeltaEvent (not a completed event)
        mock_delta_event = Mock(spec=OutputTextDeltaEvent)
        mock_delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        mock_delta_event.delta = "Hello"
        # Delta events don't have a response attribute
        delattr(mock_delta_event, 'response') if hasattr(mock_delta_event, 'response') else None
        
        # Set up the mock transform method to return our delta event
        mock_config.transform_streaming_response.return_value = mock_delta_event
        
        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )
        
        # Prepare test chunk data for a delta event
        test_chunk_data = {
            "type": "response.output_text.delta",
            "delta": "Hello",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0
        }
        
        with patch.object(
            ResponsesAPIRequestUtils, 
            '_update_responses_api_response_id_with_model_id'
        ) as mock_update_id:
            # Process the chunk
            result = iterator._process_chunk(json.dumps(test_chunk_data))
            
            # Assertions
            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
            
            # Verify that _update_responses_api_response_id_with_model_id was NOT called
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
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        
        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj
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
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        
        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj
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
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj
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
        mock_response.aiter_lines = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_success_handler = Mock()
        mock_logging_obj.success_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create the iterator instance
        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )

        # Create a ResponseCompletedEvent with tool_choice that has model_dump
        mock_completed_response = Mock()
        mock_completed_response.model_dump.return_value = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [{"type": "function_call", "name": "search_web"}],
                "tool_choice": {"type": "function", "name": "search_web"}
            }
        }
        # model_validate should return a new mock (the copy)
        type(mock_completed_response).model_validate = Mock(return_value=Mock())

        iterator.completed_response = mock_completed_response

        # This should NOT raise an exception
        # Previously it would fail with: TypeError: cannot pickle 'ValidatorIterator'
        # Mock asyncio.create_task and executor.submit since we're not in async context
        with patch('asyncio.create_task') as mock_create_task, \
             patch('litellm.responses.streaming_iterator.executor') as mock_executor:
            try:
                iterator._handle_logging_completed_response()
            except TypeError as e:
                if "pickle" in str(e):
                    pytest.fail(f"_handle_logging_completed_response failed with pickle error: {e}")
                raise

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
        
        # Create an async iterator that raises StopAsyncIteration after yielding one chunk
        async def mock_aiter_lines():
            yield 'data: {"type": "response.output_text.delta", "delta": "test"}'
            # Normal end of stream - raise StopAsyncIteration
        
        mock_response.aiter_lines = mock_aiter_lines
        
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        mock_delta_event = Mock()
        mock_delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        mock_delta_event.delta = "test"
        mock_config.transform_streaming_response.return_value = mock_delta_event
        
        # Create the iterator instance
        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )
        
        # Consume the iterator until StopAsyncIteration
        chunks_received = []
        try:
            async for chunk in iterator:
                chunks_received.append(chunk)
        except StopAsyncIteration:
            pass  # This is expected
        
        # Verify we got the chunk
        assert len(chunks_received) == 1
        
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
        from litellm.responses.streaming_iterator import SyncResponsesAPIStreamingIterator
        
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        
        # Create a sync iterator that raises StopIteration after yielding one chunk
        def mock_iter_lines():
            yield 'data: {"type": "response.output_text.delta", "delta": "test"}'
            # Normal end of stream - raise StopIteration
        
        mock_response.iter_lines = mock_iter_lines
        
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        mock_delta_event = Mock()
        mock_delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        mock_delta_event.delta = "test"
        mock_config.transform_streaming_response.return_value = mock_delta_event
        
        # Create the iterator instance
        iterator = SyncResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )
        
        # Consume the iterator until StopIteration
        chunks_received = []
        try:
            for chunk in iterator:
                chunks_received.append(chunk)
        except StopIteration:
            pass  # This is expected
        
        # Verify we got the chunk
        assert len(chunks_received) == 1
        
        # CRITICAL: Verify that failure handlers were NOT called
        # StopIteration is a normal end of stream, not a failure
        mock_logging_obj.async_failure_handler.assert_not_called()
        mock_logging_obj.failure_handler.assert_not_called()

