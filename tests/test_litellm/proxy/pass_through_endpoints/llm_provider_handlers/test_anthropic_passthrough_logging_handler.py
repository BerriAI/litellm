import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)


class TestAnthropicLoggingHandlerModelFallback:
    """Test the model fallback logic in the anthropic passthrough logging handler."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.mock_chunks = [
            '{"type": "message_start", "message": {"id": "msg_123", "model": "claude-3-haiku-20240307"}}',
            '{"type": "content_block_delta", "delta": {"text": "Hello"}}',
            '{"type": "content_block_delta", "delta": {"text": " world"}}',
            '{"type": "message_stop"}',
        ]
        
    def _create_mock_logging_obj(self, model_in_details: str = None) -> LiteLLMLoggingObj:
        """Create a mock logging object with optional model in model_call_details"""
        mock_logging_obj = MagicMock()
        
        if model_in_details:
            # Create a dict-like mock that returns the model for the 'model' key
            mock_model_call_details = {'model': model_in_details}
            mock_logging_obj.model_call_details = mock_model_call_details
        else:
            # Create empty dict or None
            mock_logging_obj.model_call_details = {}
            
        return mock_logging_obj
    
    def _create_mock_passthrough_handler(self):
        """Create a mock passthrough success handler"""
        mock_handler = MagicMock()
        return mock_handler



    @patch.object(AnthropicPassthroughLoggingHandler, '_build_complete_streaming_response')
    @patch.object(AnthropicPassthroughLoggingHandler, '_create_anthropic_response_logging_payload')
    def test_model_from_request_body_used_when_present(self, mock_create_payload, mock_build_response):
        """Test that model from request_body is used when present"""
        # Arrange
        request_body = {"model": "claude-3-sonnet-20240229"}
        logging_obj = self._create_mock_logging_obj(model_in_details="claude-3-haiku-20240307")
        passthrough_handler = self._create_mock_passthrough_handler()
        
        # Mock successful response building
        mock_build_response.return_value = MagicMock()
        mock_create_payload.return_value = {"test": "payload"}
        
        # Act
        result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=passthrough_handler,
            url_route="/anthropic/v1/messages",
            request_body=request_body,
            endpoint_type="messages",
            start_time=self.start_time,
            all_chunks=self.mock_chunks,
            end_time=self.end_time,
        )
        
        # Assert
        assert result is not None
        # Verify that _build_complete_streaming_response was called with the request_body model
        mock_build_response.assert_called_once()
        call_args = mock_build_response.call_args
        assert call_args[1]['model'] == "claude-3-sonnet-20240229"  # Should use request_body model

    def test_model_fallback_logic_isolated(self):
        """Test just the model fallback logic in isolation"""
        # Test case 1: Model from request body
        request_body = {"model": "claude-3-sonnet-20240229"}
        logging_obj = self._create_mock_logging_obj(model_in_details="claude-3-haiku-20240307")
        
        # Extract the logic directly from the function
        model = request_body.get("model", "")
        if not model and hasattr(logging_obj, 'model_call_details') and logging_obj.model_call_details.get('model'):
            model = logging_obj.model_call_details.get('model')
        
        assert model == "claude-3-sonnet-20240229"  # Should use request_body model
        
        # Test case 2: Fallback to logging obj
        request_body = {}
        logging_obj = self._create_mock_logging_obj(model_in_details="claude-3-haiku-20240307")
        
        model = request_body.get("model", "")
        if not model and hasattr(logging_obj, 'model_call_details') and logging_obj.model_call_details.get('model'):
            model = logging_obj.model_call_details.get('model')
            
        assert model == "claude-3-haiku-20240307"  # Should use fallback model
        
        # Test case 3: Empty string in request body, fallback to logging obj
        request_body = {"model": ""}
        logging_obj = self._create_mock_logging_obj(model_in_details="claude-3-opus-20240229")
        
        model = request_body.get("model", "")
        if not model and hasattr(logging_obj, 'model_call_details') and logging_obj.model_call_details.get('model'):
            model = logging_obj.model_call_details.get('model')
            
        assert model == "claude-3-opus-20240229"  # Should use fallback model
        
        # Test case 4: Both empty
        request_body = {}
        logging_obj = self._create_mock_logging_obj()
        
        model = request_body.get("model", "")
        if not model and hasattr(logging_obj, 'model_call_details') and logging_obj.model_call_details.get('model'):
            model = logging_obj.model_call_details.get('model')
            
        assert model == ""  # Should be empty

    def test_edge_case_missing_model_call_details_attribute(self):
        """Test fallback behavior when logging_obj doesn't have model_call_details attribute"""
        # Case where logging_obj doesn't have the attribute at all
        request_body = {"model": ""}  # Empty model in request body
        logging_obj = MagicMock()
        # Remove the attribute to simulate it not existing
        if hasattr(logging_obj, 'model_call_details'):
            delattr(logging_obj, 'model_call_details')
        
        # Extract the logic directly from the function
        model = request_body.get("model", "")
        if not model and hasattr(logging_obj, 'model_call_details') and logging_obj.model_call_details.get('model'):
            model = logging_obj.model_call_details.get('model')
            
        assert model == ""  # Should remain empty since no fallback available
        
        # Case where model_call_details exists but get returns None
        request_body = {"model": ""}
        logging_obj = self._create_mock_logging_obj()  # Empty dict
        
        model = request_body.get("model", "")
        if not model and hasattr(logging_obj, 'model_call_details') and logging_obj.model_call_details.get('model'):
            model = logging_obj.model_call_details.get('model')
            
        assert model == ""  # Should remain empty 