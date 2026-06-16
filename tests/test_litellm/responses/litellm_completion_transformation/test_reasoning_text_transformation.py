"""
Test reasoning parameter transformation in transformation.py
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


class TestReasoningDictTransformation:
    """Test that reasoning dict parameters are correctly converted"""

    def test_reasoning_dict_with_effort_converted_to_string(self):
        """Test that reasoning dict with effort field is converted to string"""
        request = {
            "model": "test-model",
            "input": "Hello",
            "reasoning": {"effort": "medium", "summary": "detailed"},
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input=request["input"],
            responses_api_request=request,
        )

        # reasoning dict should be converted to reasoning_effort string
        reasoning_effort = result.get("reasoning_effort")
        assert reasoning_effort == "medium"
        # Original reasoning dict should NOT be in result
        assert "reasoning" not in result

    def test_reasoning_dict_only_effort(self):
        """Test reasoning dict with only effort field"""
        request = {
            "model": "test-model",
            "input": "Hello",
            "reasoning": {"effort": "high"},
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input=request["input"],
            responses_api_request=request,
        )

        assert result.get("reasoning_effort") == "high"

    def test_reasoning_string_preserved(self):
        """Test that reasoning as string is preserved"""
        request = {
            "model": "test-model",
            "input": "Hello",
            "reasoning": "medium",
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input=request["input"],
            responses_api_request=request,
        )

        assert result.get("reasoning_effort") == "medium"

    def test_reasoning_dict_only_summary_discarded(self):
        """Test that reasoning dict with only summary is discarded"""
        request = {
            "model": "test-model",
            "input": "Hello",
            "reasoning": {"summary": "detailed"},
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input=request["input"],
            responses_api_request=request,
        )

        # Should not have reasoning_effort (no effort field to extract)
        assert "reasoning_effort" not in result

    def test_no_reasoning_parameter(self):
        """Test request without reasoning parameter"""
        request = {
            "model": "test-model",
            "input": "Hello",
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input=request["input"],
            responses_api_request=request,
        )

        assert "reasoning_effort" not in result


class TestTextDictTransformation:
    """Test that text dict parameter is correctly transformed"""

    def test_text_json_object_format(self):
        """Test text dict with json_object format"""
        request = {
            "model": "test-model",
            "input": "Hello",
            "text": {"format": {"type": "json_object"}},
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input=request["input"],
            responses_api_request=request,
        )

        # text should be converted to response_format
        response_format = result.get("response_format")
        assert response_format == {"type": "json_object"}
        # Original text dict should NOT be in result
        assert "text" not in result

    def test_text_json_schema_format(self):
        """Test text dict with json_schema format"""
        request = {
            "model": "test-model",
            "input": "Hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "my_schema",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input=request["input"],
            responses_api_request=request,
        )

        response_format = result.get("response_format")
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "my_schema"

    def test_no_text_parameter(self):
        """Test request without text parameter"""
        request = {
            "model": "test-model",
            "input": "Hello",
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input=request["input"],
            responses_api_request=request,
        )

        assert "response_format" not in result
