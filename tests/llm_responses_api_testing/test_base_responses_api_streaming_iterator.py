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