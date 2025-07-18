import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.integrations.cloudzero import CloudZeroLogger


class TestCloudZeroIntegration:
    """Test suite for CloudZero integration"""

    def setup_method(self):
        """Set up test environment"""
        # Set required environment variables
        os.environ["CLOUDZERO_API_KEY"] = "test-api-key"
        os.environ["CLOUDZERO_METRIC_NAME"] = "llm-cost"
        
    def teardown_method(self):
        """Clean up test environment"""
        # Clean up environment variables
        os.environ.pop("CLOUDZERO_API_KEY", None)
        os.environ.pop("CLOUDZERO_METRIC_NAME", None)
        os.environ.pop("CLOUDZERO_API_BASE", None)

    def test_cloudzero_logger_initialization(self):
        """Test that CloudZeroLogger initializes correctly with required env vars"""
        logger = CloudZeroLogger()
        assert logger is not None

    def test_cloudzero_logger_missing_api_key(self):
        """Test that CloudZeroLogger raises exception when API key is missing"""
        os.environ.pop("CLOUDZERO_API_KEY", None)
        with pytest.raises(Exception, match="Missing keys.*CLOUDZERO_API_KEY"):
            CloudZeroLogger()

    def test_cloudzero_logger_missing_metric_name(self):
        """Test that CloudZeroLogger raises exception when metric name is missing"""
        os.environ.pop("CLOUDZERO_METRIC_NAME", None)
        with pytest.raises(Exception, match="Missing keys.*CLOUDZERO_METRIC_NAME"):
            CloudZeroLogger()

    def test_get_api_url_default(self):
        """Test that _get_api_url returns correct URL with default base"""
        logger = CloudZeroLogger()
        url = logger._get_api_url()
        assert url == "https://api.cloudzero.com/unit-cost/v1/telemetry/metric/llm-cost"

    def test_get_api_url_custom_base(self):
        """Test that _get_api_url returns correct URL with custom base"""
        os.environ["CLOUDZERO_API_BASE"] = "https://custom.cloudzero.com/"
        logger = CloudZeroLogger()
        url = logger._get_api_url()
        assert url == "https://custom.cloudzero.com/unit-cost/v1/telemetry/metric/llm-cost"

    def test_common_logic_basic(self):
        """Test that _common_logic correctly formats basic telemetry data"""
        logger = CloudZeroLogger()
        
        kwargs = {
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
        
        # Verify structure
        assert "records" in result
        assert len(result["records"]) == 1
        
        record = result["records"][0]
        assert record["value"] == 0.001
        assert "timestamp" in record
        assert "filters" in record
        assert record["filters"]["model"] == "gpt-3.5-turbo"

    def test_common_logic_with_user_metadata(self):
        """Test that _common_logic includes user metadata in filters"""
        logger = CloudZeroLogger()
        
        kwargs = {
            "model": "gpt-4",
            "response_cost": 0.002,
            "litellm_call_id": "test-call-id",
            "litellm_params": {
                "metadata": {
                    "user_api_key_user_id": "user-123",
                    "user_api_key_team_id": "team-456",
                    "user_api_key_org_id": "org-789",
                    "user_api_key_end_user_id": "end-user-999"
                }
            }
        }
        
        response_obj = {
            "id": "test-response-id",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }
        
        result = logger._common_logic(kwargs, response_obj)
        
        record = result["records"][0]
        assert record["filters"]["user_id"] == "user-123"
        assert record["filters"]["team_id"] == "team-456"
        assert record["filters"]["org_id"] == "org-789"
        assert record["filters"]["end_user_id"] == "end-user-999"

    def test_common_logic_none_cost(self):
        """Test that _common_logic handles None cost gracefully"""
        logger = CloudZeroLogger()
        
        kwargs = {
            "model": "gpt-3.5-turbo",
            "response_cost": None,
            "litellm_call_id": "test-call-id"
        }
        
        response_obj = {"id": "test-response-id"}
        
        result = logger._common_logic(kwargs, response_obj)
        
        record = result["records"][0]
        assert record["value"] == 0.0

    @patch('litellm.integrations.cloudzero.HTTPHandler')
    def test_log_success_event(self, mock_http_handler):
        """Test synchronous log_success_event method"""
        mock_post = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Success"
        mock_post.return_value = mock_response
        mock_http_handler.return_value.post = mock_post
        
        logger = CloudZeroLogger()
        
        kwargs = {
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
        
        # Should not raise an exception
        logger.log_success_event(kwargs, response_obj, None, None)
        
        # Verify HTTP call was made
        mock_post.assert_called_once()
        
        # Verify the call details
        call_args = mock_post.call_args
        assert call_args[1]['url'] == "https://api.cloudzero.com/unit-cost/v1/telemetry/metric/llm-cost"
        assert call_args[1]['headers']['Authorization'] == "test-api-key"
        assert call_args[1]['headers']['Content-Type'] == "application/json"
        
        # Verify the data structure
        data = json.loads(call_args[1]['data'])
        assert "records" in data
        assert len(data["records"]) == 1
        assert data["records"][0]["value"] == 0.001

    @patch('litellm.integrations.cloudzero.HTTPHandler')
    def test_log_success_event_error_handling(self, mock_http_handler):
        """Test that log_success_event handles errors gracefully"""
        mock_post = MagicMock()
        mock_post.side_effect = Exception("Network error")
        mock_http_handler.return_value.post = mock_post
        
        logger = CloudZeroLogger()
        
        kwargs = {
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_call_id": "test-call-id"
        }
        
        response_obj = {"id": "test-response-id"}
        
        # Should not raise an exception (errors are logged but not raised)
        logger.log_success_event(kwargs, response_obj, None, None)

    @patch('litellm.integrations.cloudzero.get_async_httpx_client')
    @pytest.mark.asyncio
    async def test_async_log_success_event(self, mock_get_client):
        """Test asynchronous log_success_event method"""
        mock_post = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "Success"
        mock_post.return_value = mock_response
        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_get_client.return_value = mock_client
        
        logger = CloudZeroLogger()
        
        kwargs = {
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
        
        # Verify the call details
        call_args = mock_post.call_args
        assert call_args[1]['url'] == "https://api.cloudzero.com/unit-cost/v1/telemetry/metric/llm-cost"
        assert call_args[1]['headers']['Authorization'] == "test-api-key"
        assert call_args[1]['headers']['Content-Type'] == "application/json"
        
        # Verify the data structure  
        data = json.loads(call_args[1]['data'])
        assert "records" in data
        assert len(data["records"]) == 1
        assert data["records"][0]["value"] == 0.002

    @patch('litellm.integrations.cloudzero.get_async_httpx_client')
    @pytest.mark.asyncio
    async def test_async_log_success_event_error_handling(self, mock_get_client):
        """Test that async_log_success_event handles errors gracefully"""
        mock_post = AsyncMock()
        mock_post.side_effect = Exception("Network error")
        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_get_client.return_value = mock_client
        
        logger = CloudZeroLogger()
        
        kwargs = {
            "model": "gpt-4",
            "response_cost": 0.002,
            "litellm_call_id": "async-test-call-id"
        }
        
        response_obj = {"id": "async-test-response-id"}
        
        # Should not raise an exception (errors are logged but not raised)
        await logger.async_log_success_event(kwargs, response_obj, None, None)

    def test_telemetry_record_structure(self):
        """Test that the telemetry record structure is correct for CloudZero API"""
        logger = CloudZeroLogger()
        
        kwargs = {
            "model": "gpt-3.5-turbo",
            "response_cost": 0.0015,
            "litellm_call_id": "struct-test-call-id"
        }
        
        response_data = {
            "id": "struct-test-response-id",
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8,
                "total_tokens": 23
            }
        }
        response_obj = litellm.ModelResponse(**response_data)
        
        result = logger._common_logic(kwargs, response_obj)
        
        # Verify CloudZero telemetry structure
        assert "records" in result
        assert isinstance(result["records"], list)
        assert len(result["records"]) == 1
        
        record = result["records"][0]
        assert "value" in record
        assert "timestamp" in record
        assert record["value"] == 0.0015
        
        # Verify ISO format timestamp
        import datetime
        timestamp = record["timestamp"]
        # Should not raise an exception
        datetime.datetime.fromisoformat(timestamp.replace('+00:00', ''))