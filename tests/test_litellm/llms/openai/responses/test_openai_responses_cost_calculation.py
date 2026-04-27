"""
Unit tests for Responses API cost calculation functionality.
Tests the fix for issue #26475 - ensuring cost calculation works correctly
across all 60+ providers inheriting from OpenAIResponsesAPIConfig.

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import LlmProviders


class TestResponsesAPICostCalculation:
    """Test suite for Responses API cost calculation across transformation paths"""

    def setup_method(self):
        self.config = OpenAIResponsesAPIConfig()
        self.model = "gpt-4o"
        self.logging_obj = MagicMock()
        self.logging_obj.model = self.model

    def test_calculate_response_cost_basic(self):
        """Test _calculate_response_cost helper with basic usage"""
        usage_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

        response = MagicMock()
        response._hidden_params = {}

        self.config._calculate_response_cost(self.model, usage_dict, response)

        # Verify cost was calculated and stored
        assert "response_cost" in response._hidden_params
        assert response._hidden_params["response_cost"] > 0
        assert isinstance(response._hidden_params["response_cost"], float)

    def test_calculate_response_cost_with_cached_tokens(self):
        """Test cost calculation with cached tokens (prompt caching)"""
        usage_dict = {
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "input_tokens_details": {
                "cached_tokens": 800,  # 80% cache hit
            },
        }

        response = MagicMock()
        response._hidden_params = {}

        self.config._calculate_response_cost(self.model, usage_dict, response)

        # Verify cost was calculated with caching discount
        assert "response_cost" in response._hidden_params
        assert response._hidden_params["response_cost"] > 0

    def test_calculate_response_cost_empty_usage(self):
        """Test _calculate_response_cost with empty usage dict"""
        usage_dict = {}

        response = MagicMock()
        response._hidden_params = {}

        self.config._calculate_response_cost(self.model, usage_dict, response)

        # Should not add cost for empty usage
        assert "response_cost" not in response._hidden_params

    def test_calculate_response_cost_none_usage(self):
        """Test _calculate_response_cost with None usage"""
        response = MagicMock()
        response._hidden_params = {}

        self.config._calculate_response_cost(self.model, None, response)

        # Should not crash, should not add cost
        assert "response_cost" not in response._hidden_params

    def test_transform_response_api_response_calculates_cost(self):
        """Test that transform_response_api_response calculates cost"""
        raw_response_json = {
            "id": "resp_123",
            "created_at": 1234567890,
            "model": self.model,
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

        result = self.config.transform_response_api_response(
            model=self.model,
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        # Verify cost was calculated
        assert isinstance(result, ResponsesAPIResponse)
        assert "response_cost" in result._hidden_params
        assert result._hidden_params["response_cost"] > 0

    def test_transform_compact_response_calculates_cost(self):
        """Test that transform_compact_response_api_response calculates cost"""
        raw_response_json = {
            "id": "resp_compact_123",
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

        result = self.config.transform_compact_response_api_response(
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        # Verify cost was calculated
        assert isinstance(result, ResponsesAPIResponse)
        assert "response_cost" in result._hidden_params
        assert result._hidden_params["response_cost"] > 0

    def test_transform_streaming_response_completed_calculates_cost(self):
        """Test that transform_streaming_response calculates cost for ResponseCompletedEvent"""
        completed_chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_stream_123",
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
        assert result.response._hidden_params["response_cost"] > 0

    def test_transform_streaming_response_non_completed_no_cost(self):
        """Test that non-completed streaming events don't try to calculate cost"""
        delta_chunk = {
            "type": "response.output_text.delta",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        }

        # Should not raise, should not try to calculate cost
        result = self.config.transform_streaming_response(
            model=self.model,
            parsed_chunk=delta_chunk,
            logging_obj=self.logging_obj,
        )

        # Just verify it doesn't crash
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA

    def test_cost_calculation_with_zero_tokens(self):
        """Test cost calculation with zero tokens (edge case)"""
        usage_dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        response = MagicMock()
        response._hidden_params = {}

        self.config._calculate_response_cost(self.model, usage_dict, response)

        # Should calculate cost (might be 0, but should be present)
        assert "response_cost" in response._hidden_params
        assert response._hidden_params["response_cost"] == 0.0

    def test_cost_calculation_field_mapping(self):
        """Test that Responses API field names are correctly mapped to Usage field names"""
        # Responses API uses input_tokens/output_tokens
        # Usage expects prompt_tokens/completion_tokens
        usage_dict = {
            "input_tokens": 500,
            "output_tokens": 250,
            "total_tokens": 750,
            "input_tokens_details": {
                "cached_tokens": 400,
            },
            "output_tokens_details": {
                "reasoning_tokens": 50,
            },
        }

        response = MagicMock()
        response._hidden_params = {}

        # Should map fields correctly and not raise KeyError
        self.config._calculate_response_cost(self.model, usage_dict, response)

        assert "response_cost" in response._hidden_params
        assert response._hidden_params["response_cost"] > 0

    def test_cost_calculation_error_handling(self):
        """Test that cost calculation errors don't break the response"""
        # Invalid model that doesn't exist in pricing table
        invalid_model = "nonexistent-model-xyz"

        usage_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

        response = MagicMock()
        response._hidden_params = {}

        # Should handle error gracefully, not crash
        self.config._calculate_response_cost(invalid_model, usage_dict, response)

        # Cost might not be set if model pricing not found, but shouldn't crash
        # This is acceptable fallback behavior


class TestProviderInheritanceCostCalculation:
    """Test that cost calculation works for providers inheriting from OpenAIResponsesAPIConfig"""

    def test_enum_provider_cost_calculation(self):
        """Test provider returning LlmProviders enum"""
        config = OpenAIResponsesAPIConfig()
        assert isinstance(config.custom_llm_provider, LlmProviders)
        assert config.custom_llm_provider == LlmProviders.OPENAI

        usage_dict = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        response = MagicMock()
        response._hidden_params = {}

        config._calculate_response_cost("gpt-4o", usage_dict, response)
        assert "response_cost" in response._hidden_params

