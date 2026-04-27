"""
Unit tests for VolcEngine Responses API cost calculation.
Tests coverage for cost calculation in transform_response_api_response and streaming.
"""
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.volcengine.responses.transformation import (
    VolcEngineResponsesAPIConfig,
)
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


class TestVolcEngineResponsesAPICostCalculation:
    """Test suite for VolcEngine cost calculation functionality"""

    def setup_method(self):
        self.config = VolcEngineResponsesAPIConfig()
        self.model = "doubao-pro"
        self.logging_obj = MagicMock()
        self.logging_obj.model = self.model

    @patch("litellm.llms.openai.responses.transformation.generic_cost_per_token")
    def test_transform_response_api_response_calculates_cost(
        self, mock_generic_cost
    ):
        """Test that transform_response_api_response calculates cost"""
        # Mock the cost calculation to return predictable values
        mock_generic_cost.return_value = (0.002, 0.001)  # prompt_cost, completion_cost

        raw_response_json = {
            "id": "resp_volcengine_123",
            "created_at": 1234567890,
            "model": self.model,
            "object": "response",
            "output": [],
            "status": "completed",
            "usage": {
                "input_tokens": 200,
                "output_tokens": 100,
                "total_tokens": 300,
            },
        }

        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = raw_response_json
        raw_response.text = json.dumps(raw_response_json)
        raw_response.headers = {}

        result = self.config.transform_response_api_response(
            model=self.model,
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        # Verify cost was calculated
        assert isinstance(result, ResponsesAPIResponse)
        assert "response_cost" in result._hidden_params
        assert result._hidden_params["response_cost"] == 0.003  # 0.002 + 0.001

    @patch("litellm.llms.openai.responses.transformation.generic_cost_per_token")
    def test_transform_response_api_response_with_cached_tokens(
        self, mock_generic_cost
    ):
        """Test cost calculation with cached tokens"""
        # Mock cost calculation including cached token discount
        mock_generic_cost.return_value = (0.005, 0.002)

        raw_response_json = {
            "id": "resp_volcengine_cached",
            "created_at": 1234567890,
            "model": self.model,
            "object": "response",
            "output": [],
            "status": "completed",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
                "input_tokens_details": {
                    "cached_tokens": 800,
                },
            },
        }

        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = raw_response_json
        raw_response.text = json.dumps(raw_response_json)
        raw_response.headers = {}

        result = self.config.transform_response_api_response(
            model=self.model,
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        # Verify cost was calculated with caching
        assert isinstance(result, ResponsesAPIResponse)
        assert "response_cost" in result._hidden_params
        assert result._hidden_params["response_cost"] == 0.007

    @patch("litellm.llms.openai.responses.transformation.generic_cost_per_token")
    def test_transform_streaming_response_completed_calculates_cost(
        self, mock_generic_cost
    ):
        """Test that streaming response.completed event calculates cost"""
        mock_generic_cost.return_value = (0.0015, 0.00075)

        completed_chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_stream_volcengine",
                "created_at": 1234567890,
                "model": self.model,
                "object": "response",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 75,
                    "total_tokens": 225,
                },
            },
        }

        result = self.config.transform_streaming_response(
            model=self.model,
            parsed_chunk=completed_chunk,
            logging_obj=self.logging_obj,
        )

        # Verify it's a completed event
        assert isinstance(result, ResponseCompletedEvent)
        assert result.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED

        # Verify cost was calculated and stored in the response object
        assert "response_cost" in result.response._hidden_params
        assert result.response._hidden_params["response_cost"] == 0.00225

    def test_transform_streaming_response_non_completed_no_cost(self):
        """Test that non-completed streaming events don't calculate cost"""
        delta_chunk = {
            "type": "response.output_text.delta",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello from VolcEngine",
        }

        # Should not crash, should not try to calculate cost
        result = self.config.transform_streaming_response(
            model=self.model,
            parsed_chunk=delta_chunk,
            logging_obj=self.logging_obj,
        )

        # Just verify it doesn't crash
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA

    @patch("litellm.llms.openai.responses.transformation.generic_cost_per_token")
    def test_transform_streaming_response_missing_output_field(
        self, mock_generic_cost
    ):
        """Test streaming response handles missing output field in response"""
        mock_generic_cost.return_value = (0.001, 0.0005)

        chunk_missing_output = {
            "type": "response.completed",
            "response": {
                "id": "resp_no_output",
                "created_at": 1234567890,
                "model": self.model,
                "object": "response",
                "status": "completed",
                # Missing 'output' field - VolcEngine patches this
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
            },
        }

        result = self.config.transform_streaming_response(
            model=self.model,
            parsed_chunk=chunk_missing_output,
            logging_obj=self.logging_obj,
        )

        # Should patch missing output and still calculate cost
        assert isinstance(result, ResponseCompletedEvent)
        assert "response_cost" in result.response._hidden_params
        assert result.response._hidden_params["response_cost"] == 0.0015

    @patch("litellm.llms.openai.responses.transformation.generic_cost_per_token")
    def test_cost_calculation_with_zero_tokens(self, mock_generic_cost):
        """Test cost calculation with zero tokens"""
        mock_generic_cost.return_value = (0.0, 0.0)

        raw_response_json = {
            "id": "resp_zero_tokens",
            "created_at": 1234567890,
            "model": self.model,
            "object": "response",
            "output": [],
            "status": "completed",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }

        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = raw_response_json
        raw_response.text = json.dumps(raw_response_json)
        raw_response.headers = {}

        result = self.config.transform_response_api_response(
            model=self.model,
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        # Should calculate cost (will be 0)
        assert isinstance(result, ResponsesAPIResponse)
        assert "response_cost" in result._hidden_params
        assert result._hidden_params["response_cost"] == 0.0

    def test_cost_calculation_error_handling(self):
        """Test that cost calculation errors are handled gracefully"""
        raw_response_json = {
            "id": "resp_invalid_model",
            "created_at": 1234567890,
            "model": "invalid-volcengine-model-xyz",
            "object": "response",
            "output": [],
            "status": "completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        }

        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = raw_response_json
        raw_response.text = json.dumps(raw_response_json)
        raw_response.headers = {}

        # Should not crash even with invalid model
        result = self.config.transform_response_api_response(
            model="invalid-volcengine-model-xyz",
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        assert isinstance(result, ResponsesAPIResponse)
        # Cost might not be set if model pricing not found, but shouldn't crash
