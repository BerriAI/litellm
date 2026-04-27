"""
Unit tests for Manus Responses API cost calculation.
Tests coverage for _get_usage_dict helper and cost calculation in transform methods.
"""
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.manus.responses.transformation import ManusResponsesAPIConfig
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse


class TestManusResponsesAPICostCalculation:
    """Test suite for Manus cost calculation functionality"""

    def setup_method(self):
        self.config = ManusResponsesAPIConfig()
        self.logging_obj = MagicMock()
        self.logging_obj.model = "manus/manus-1.6"

    def test_get_usage_dict_with_response_api_usage_object(self):
        """Test _get_usage_dict converts ResponseAPIUsage object to dict"""
        usage_obj = ResponseAPIUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        result = self.config._get_usage_dict(usage_obj)

        assert isinstance(result, dict)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_get_usage_dict_with_dict(self):
        """Test _get_usage_dict handles dict input"""
        usage_dict = {
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        }

        result = self.config._get_usage_dict(usage_dict)

        assert result == usage_dict

    def test_get_usage_dict_with_none(self):
        """Test _get_usage_dict handles None/empty input"""
        result = self.config._get_usage_dict(None)
        assert result == {}

        result = self.config._get_usage_dict({})
        assert result == {}

    @patch("litellm.llms.openai.responses.transformation.generic_cost_per_token")
    def test_transform_response_api_response_calculates_cost_with_usage_object(
        self, mock_generic_cost
    ):
        """Test cost calculation when Manus returns ResponseAPIUsage object"""
        mock_generic_cost.return_value = (0.0015, 0.00075)

        raw_response_json = {
            "id": "resp_manus_123",
            "created_at": 1234567890,
            "createdAt": 1234567890,
            "model": "manus/manus-1.6",
            "object": "response",
            "output": [],
            "status": "completed",
            "reasoning": {},
            "text": {},
            # Manus returns ResponseAPIUsage object when usage is missing
            "usage": ResponseAPIUsage(
                input_tokens=150,
                output_tokens=75,
                total_tokens=225,
            ),
        }

        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = raw_response_json
        raw_response.text = json.dumps(raw_response_json, default=str)
        raw_response.headers = {}

        result = self.config.transform_response_api_response(
            model="manus/manus-1.6",
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        # Verify cost was calculated
        assert isinstance(result, ResponsesAPIResponse)
        assert "response_cost" in result._hidden_params
        assert result._hidden_params["response_cost"] == pytest.approx(0.00225)

    @patch("litellm.llms.openai.responses.transformation.generic_cost_per_token")
    def test_transform_get_response_api_response_extracts_model_from_response(
        self, mock_generic_cost
    ):
        """Test that transform_get_response_api_response gets model from response JSON"""
        mock_generic_cost.return_value = (0.001, 0.0005)

        raw_response_json = {
            "id": "resp_get_123",
            "created_at": 1234567890,
            "createdAt": 1234567890,
            "model": "manus/manus-1.6",
            "object": "response",
            "output": [],
            "status": "completed",
            "reasoning": {},
            "text": {},
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

        result = self.config.transform_get_response_api_response(
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        # Verify cost was calculated using model from response
        assert isinstance(result, ResponsesAPIResponse)
        assert "response_cost" in result._hidden_params
        assert result._hidden_params["response_cost"] == pytest.approx(0.0015)

    @patch("litellm.llms.openai.responses.transformation.generic_cost_per_token")
    def test_transform_get_response_api_response_fallback_to_logging_obj(
        self, mock_generic_cost
    ):
        """Test that transform_get_response_api_response falls back to logging_obj.model"""
        mock_generic_cost.return_value = (0.001, 0.0005)

        raw_response_json = {
            "id": "resp_get_456",
            "created_at": 1234567890,
            "createdAt": 1234567890,
            # No model in response - should use logging_obj.model
            "object": "response",
            "output": [],
            "status": "completed",
            "reasoning": {},
            "text": {},
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

        self.logging_obj.model = "manus/manus-1.6"

        result = self.config.transform_get_response_api_response(
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        # Verify cost was calculated using model from logging_obj
        assert isinstance(result, ResponsesAPIResponse)
        assert "response_cost" in result._hidden_params
        assert result._hidden_params["response_cost"] == pytest.approx(0.0015)

    @patch("litellm.llms.openai.responses.transformation.generic_cost_per_token")
    def test_cost_calculation_with_missing_usage(self, mock_generic_cost):
        """Test that missing usage doesn't crash, just skips cost calculation"""
        mock_generic_cost.return_value = (0.0, 0.0)

        raw_response_json = {
            "id": "resp_no_usage",
            "created_at": 1234567890,
            "createdAt": 1234567890,
            "model": "manus/manus-1.6",
            "object": "response",
            "output": [],
            "status": "completed",
            "reasoning": {},
            "text": {},
            # No usage provided, will be auto-filled with ResponseAPIUsage(0,0,0)
        }

        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = raw_response_json
        raw_response.text = json.dumps(raw_response_json)
        raw_response.headers = {}

        # Should not crash
        result = self.config.transform_response_api_response(
            model="manus/manus-1.6",
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        assert isinstance(result, ResponsesAPIResponse)
        # Cost should be 0 for zero tokens
        assert "response_cost" in result._hidden_params
        assert result._hidden_params["response_cost"] == 0.0
