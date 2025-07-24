import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.integrations.openmeter import OpenMeterLogger


class TestOpenMeterIntegration:
    """Test suite for OpenMeter integration"""

    def setup_method(self):
        """Set up test environment"""
        # Set required environment variables
        os.environ["OPENMETER_API_KEY"] = "test-api-key"
        os.environ["OPENMETER_API_ENDPOINT"] = "https://test.openmeter.com"
        
    def teardown_method(self):
        """Clean up test environment"""
        # Clean up environment variables
        os.environ.pop("OPENMETER_API_KEY", None)
        os.environ.pop("OPENMETER_API_ENDPOINT", None)
        os.environ.pop("OPENMETER_EVENT_TYPE", None)

    def test_openmeter_logger_initialization(self):
        """Test that OpenMeterLogger initializes correctly with required env vars"""
        logger = OpenMeterLogger()
        assert logger is not None

    def test_openmeter_logger_missing_api_key(self):
        """Test that OpenMeterLogger raises exception when API key is missing"""
        os.environ.pop("OPENMETER_API_KEY", None)
        with pytest.raises(Exception, match="Missing keys.*OPENMETER_API_KEY"):
            OpenMeterLogger()

    def test_common_logic_with_string_user(self):
        """Test that _common_logic correctly handles string user parameter"""
        logger = OpenMeterLogger()
        
        kwargs = {
            "user": "test-user-123",
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_call_id": "test-call-id"
        }
        
        # Mock response object
        response_obj = {
            "id": "test-response-id",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        result = logger._common_logic(kwargs, response_obj)
        
        # Verify subject is a string, not a tuple
        assert isinstance(result["subject"], str)
        assert result["subject"] == "test-user-123"
        assert result["data"]["model"] == "gpt-3.5-turbo"
        assert result["data"]["cost"] == 0.001

    def test_common_logic_with_integer_user(self):
        """Test that _common_logic correctly converts integer user to string"""
        logger = OpenMeterLogger()
        
        kwargs = {
            "user": 12345,  # Integer user ID
            "model": "gpt-4",
            "response_cost": 0.002,
            "litellm_call_id": "test-call-id-2"
        }
        
        response_obj = {
            "id": "test-response-id-2",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }
        
        result = logger._common_logic(kwargs, response_obj)
        
        # Verify subject is converted to string
        assert isinstance(result["subject"], str)
        assert result["subject"] == "12345"

    def test_common_logic_missing_user(self):
        """Test that _common_logic raises exception when user is missing"""
        logger = OpenMeterLogger()
        
        kwargs = {
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_call_id": "test-call-id"
        }
        
        response_obj = {"id": "test-response-id"}
        
        with pytest.raises(Exception, match="OpenMeter: user is required"):
            logger._common_logic(kwargs, response_obj)

    def test_common_logic_none_user(self):
        """Test that _common_logic raises exception when user is None"""
        logger = OpenMeterLogger()
        
        kwargs = {
            "user": None,
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_call_id": "test-call-id"
        }
        
        response_obj = {"id": "test-response-id"}
        
        with pytest.raises(Exception, match="OpenMeter: user is required"):
            logger._common_logic(kwargs, response_obj)

    def test_common_logic_empty_string_user(self):
        """Test that _common_logic correctly handles an empty string user"""
        logger = OpenMeterLogger()

        kwargs = {
            "user": "",
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_call_id": "test-call-id",
        }

        response_obj = {"id": "test-response-id"}

        result = logger._common_logic(kwargs, response_obj)
        assert isinstance(result["subject"], str)
        assert result["subject"] == ""

    @patch('litellm.integrations.openmeter.HTTPHandler')
    def test_log_success_event(self, mock_http_handler):
        """Test synchronous log_success_event method"""
        mock_post = MagicMock()
        mock_http_handler.return_value.post = mock_post
        
        logger = OpenMeterLogger()
        
        kwargs = {
            "user": "test-user",
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_call_id": "test-call-id"
        }
        
        response_obj = {
            "id": "test-response-id",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        logger.log_success_event(kwargs, response_obj, None, None)
        
        # Verify HTTP call was made
        mock_post.assert_called_once()
        
        # Verify the data structure
        call_args = mock_post.call_args
        data = json.loads(call_args[1]['data'])
        
        assert data["subject"] == "test-user"
        assert isinstance(data["subject"], str)
        assert data["data"]["model"] == "gpt-3.5-turbo"

    @patch('litellm.integrations.openmeter.get_async_httpx_client')
    @pytest.mark.asyncio
    async def test_async_log_success_event(self, mock_get_client):
        """Test asynchronous log_success_event method"""
        mock_post = AsyncMock()
        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_get_client.return_value = mock_client
        
        logger = OpenMeterLogger()
        
        kwargs = {
            "user": "async-test-user",
            "model": "gpt-4",
            "response_cost": 0.002,
            "litellm_call_id": "async-test-call-id"
        }
        
        response_obj = {
            "id": "async-test-response-id",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }
        
        await logger.async_log_success_event(kwargs, response_obj, None, None)
        
        # Verify async HTTP call was made
        mock_post.assert_called_once()
        
        # Verify the data structure  
        call_args = mock_post.call_args
        data = json.loads(call_args[1]['data'])
        
        assert data["subject"] == "async-test-user"
        assert isinstance(data["subject"], str)
        assert data["data"]["model"] == "gpt-4"

    def test_cloudevents_structure(self):
        """Test that the CloudEvents structure is correct"""
        logger = OpenMeterLogger()
        
        kwargs = {
            "user": "cloudevents-test-user",
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_call_id": "cloudevents-test-call-id"
        }
        
        response_data = {
            "id": "cloudevents-test-response-id",
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8,
                "total_tokens": 23
            }
        }
        response_obj = litellm.ModelResponse(**response_data)
        
        result = logger._common_logic(kwargs, response_obj)
        
        # Verify CloudEvents required fields
        assert result["specversion"] == "1.0"
        assert result["type"] == "litellm_tokens"  # default value
        assert result["id"] == "cloudevents-test-response-id"
        assert result["source"] == "litellm-proxy"
        assert "time" in result
        assert isinstance(result["subject"], str)
        assert result["subject"] == "cloudevents-test-user"
        
        # Verify data structure
        assert "data" in result
        assert result["data"]["model"] == "gpt-3.5-turbo"
        assert result["data"]["cost"] == 0.001
        assert result["data"]["prompt_tokens"] == 15
        assert result["data"]["completion_tokens"] == 8
        assert result["data"]["total_tokens"] == 23

    def test_custom_event_type(self):
        """Test that custom event type is used when set"""
        os.environ["OPENMETER_EVENT_TYPE"] = "custom_event_type"
        
        logger = OpenMeterLogger()
        
        kwargs = {
            "user": "custom-event-user",
            "model": "gpt-4",
            "response_cost": 0.003,
            "litellm_call_id": "custom-event-call-id"
        }
        
        response_obj = {
            "id": "custom-event-response-id",
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 12,
                "total_tokens": 37
            }
        }
        
        result = logger._common_logic(kwargs, response_obj)
        
        assert result["type"] == "custom_event_type"
