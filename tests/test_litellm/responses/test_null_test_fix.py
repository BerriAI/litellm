"""
Test for fixing null text values in output_text content blocks.

This test verifies that LiteLLM properly handles streaming responses where
text content is None, preventing TypeErrors in downstream OpenAI-compatible SDKs.

Related issue: When using LiteLLM as an OpenAI-compatible proxy for self-hosted
models, streamed responses can contain output_text content blocks where text is null.
These responses are forwarded unchanged to downstream SDKs which expect text to always
be a string (or omitted), causing TypeErrors.
"""

import pytest

from litellm.types.llms.openai import ResponsesAPIResponse


class TestNullTextHandling:
    """Test suite for handling None/null text values in responses."""

    def test_output_text_with_none_text_dict_access(self):
        """
        Test that output_text property handles None text values correctly when using dict access.
        
        This simulates the scenario where a self-hosted model returns a response with
        text: null in the content block.
        """
        # Create a response with None text value (simulating gpt-oss-120b behavior)
        response_data = {
            "id": "resp_test123",
            "object": "response",
            "created_at": 1234567890,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": None,  # This is the problematic case
                            "annotations": []
                        }
                    ]
                }
            ]
        }
        
        response = ResponsesAPIResponse(**response_data)
        
        # Should not raise TypeError and should return empty string
        assert response.output_text == ""
        
    def test_output_text_with_none_text_object_access(self):
        """
        Test that output_text property handles None text values correctly.
        
        This test verifies the object access path (getattr) in the output_text property.
        """
        response_data = {
            "id": "resp_test456",
            "object": "response",
            "created_at": 1234567890,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test456",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": None,  # This is the problematic case
                            "annotations": []
                        }
                    ]
                }
            ]
        }
        
        response = ResponsesAPIResponse(**response_data)
        
        # Should not raise TypeError and should return empty string
        assert response.output_text == ""
    
    def test_output_text_with_mixed_none_and_valid_text(self):
        """
        Test that output_text properly concatenates when some text values are None.
        """
        response_data = {
            "id": "resp_test789",
            "object": "response",
            "created_at": 1234567890,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test789",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello ",
                            "annotations": []
                        },
                        {
                            "type": "output_text",
                            "text": None,  # Should be treated as empty string
                            "annotations": []
                        },
                        {
                            "type": "output_text",
                            "text": "world!",
                            "annotations": []
                        }
                    ]
                }
            ]
        }
        
        response = ResponsesAPIResponse(**response_data)
        
        # Should concatenate non-None values, treating None as empty string
        assert response.output_text == "Hello world!"
    
    def test_output_text_with_empty_string(self):
        """
        Test that empty strings are handled correctly (baseline test).
        """
        response_data = {
            "id": "resp_test_empty",
            "object": "response",
            "created_at": 1234567890,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test_empty",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "",
                            "annotations": []
                        }
                    ]
                }
            ]
        }
        
        response = ResponsesAPIResponse(**response_data)
        
        # Should return empty string
        assert response.output_text == ""
    
    def test_output_text_with_valid_text(self):
        """
        Test that valid text values work correctly (baseline test).
        """
        response_data = {
            "id": "resp_test_valid",
            "object": "response",
            "created_at": 1234567890,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test_valid",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "This is a valid response",
                            "annotations": []
                        }
                    ]
                }
            ]
        }
        
        response = ResponsesAPIResponse(**response_data)
        
        # Should return the text as-is
        assert response.output_text == "This is a valid response"
    
    def test_output_text_no_output_text_content(self):
        """
        Test that responses without output_text content return empty string.
        """
        response_data = {
            "id": "resp_test_no_content",
            "object": "response",
            "created_at": 1234567890,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test_no_content",
                    "status": "completed",
                    "role": "assistant",
                    "content": []
                }
            ]
        }
        
        response = ResponsesAPIResponse(**response_data)
        
        # Should return empty string when no output_text content exists
        assert response.output_text == ""


class TestStreamingIteratorTextHandling:
    """Test suite for streaming iterator text handling."""
    
    def test_content_part_added_event_has_empty_string_text(self):
        """
        Test that ContentPartAddedEvent is created with empty string, not None.
        """
        from unittest.mock import Mock

        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )
        from litellm.types.llms.openai import (
            ResponseInputParam,
            ResponsesAPIOptionalRequestParams,
        )

        # Create a mock stream wrapper
        mock_wrapper = Mock()
        mock_wrapper.logging_obj = Mock()
        
        iterator = LiteLLMCompletionStreamingIterator(
            model="gpt-oss-120b",
            litellm_custom_stream_wrapper=mock_wrapper,
            request_input="test input",
            responses_api_request={},
        )
        
        event = iterator.create_content_part_added_event()
        
        # Verify that the part has text field set to empty string, not None
        part_dict = event.part.model_dump() if hasattr(event.part, 'model_dump') else dict(event.part)
        assert "text" in part_dict
        assert part_dict["text"] == ""
        assert part_dict["text"] is not None
    
    def test_delta_string_from_none_content(self):
        """
        Test that _get_delta_string_from_streaming_choices returns empty string for None content.
        """
        from unittest.mock import Mock

        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )
        from litellm.types.utils import Delta, StreamingChoices

        # Create a mock stream wrapper
        mock_wrapper = Mock()
        mock_wrapper.logging_obj = Mock()
        
        iterator = LiteLLMCompletionStreamingIterator(
            model="gpt-oss-120b",
            litellm_custom_stream_wrapper=mock_wrapper,
            request_input="test input",
            responses_api_request={},
        )
        
        # Create a choice with None content
        choice = StreamingChoices(
            index=0,
            delta=Delta(content=None, role="assistant"),
            finish_reason=None
        )
        
        result = iterator._get_delta_string_from_streaming_choices([choice])
        
        # Should return empty string, not None
        assert result == ""
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
