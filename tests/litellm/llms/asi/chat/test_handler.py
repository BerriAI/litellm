"""
Test the ASI chat handler module
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

import litellm
from litellm.llms.asi.chat.handler import ASIChatCompletion
from litellm.utils import ModelResponse


class TestASIChatCompletion(unittest.TestCase):
    """Test the ASIChatCompletion class"""

    def setUp(self):
        """Set up the test"""
        self.handler = ASIChatCompletion()
        self.model = "asi1-mini"
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        self.mock_response = {
            "id": "chatcmpl-123456789",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "asi1-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm doing well, thank you for asking! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 15,
                "total_tokens": 40,
            },
        }

    @patch("litellm.llms.openai_like.chat.handler.OpenAILikeChatHandler.completion")
    def test_completion(self, mock_parent_completion):
        """Test the completion method"""
        # Create a mock response
        mock_response = ModelResponse(
            id="test-id",
            choices=[{"message": {"content": "This is a test response"}}],
            created=1234567890,
            model="asi-1",
            object="chat.completion"
        )
        # Set up the mock to return our predefined response
        mock_parent_completion.return_value = mock_response

        # Create mock objects for required parameters
        mock_print_verbose = MagicMock()
        mock_logging_obj = MagicMock()
        
        # Test the completion method with minimal parameters
        result = self.handler.completion(
            model=self.model,
            messages=self.messages,
            api_base="https://api.asi1.ai/v1",
            model_response=ModelResponse(id="", choices=[], created=0, model="", object=""),
            custom_llm_provider="asi",
            api_key="test-api-key",
            print_verbose=mock_print_verbose,
            logging_obj=mock_logging_obj,
            custom_prompt_dict={},
            encoding=None,
            optional_params={},
            litellm_params={}
        )

        # Check that the parent class's completion method was called
        mock_parent_completion.assert_called_once()
        
        # Verify the result matches our mock response
        self.assertEqual(result, mock_response)

    @patch("litellm.llms.openai_like.chat.handler.OpenAILikeChatHandler.completion")
    def test_completion_with_json_format(self, mock_parent_completion):
        """Test the completion method with JSON response format"""
        # Create a mock response with JSON content
        json_content = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
        mock_response = ModelResponse(
            id="test-id",
            choices=[{"message": {"content": json_content}}],
            created=1234567890,
            model="asi-1",
            object="chat.completion"
        )
        
        # Set up the mock to return our predefined response
        mock_parent_completion.return_value = mock_response

        # Create mock objects for required parameters
        mock_print_verbose = MagicMock()
        mock_logging_obj = MagicMock()
        
        # Test the completion method with JSON response format
        result = self.handler.completion(
            model=self.model,
            messages=self.messages,
            api_base="https://api.asi1.ai/v1",
            model_response=ModelResponse(id="", choices=[], created=0, model="", object=""),
            optional_params={"response_format": {"type": "json_object"}},
            custom_llm_provider="asi",
            api_key="test-api-key",
            print_verbose=mock_print_verbose,
            logging_obj=mock_logging_obj,
            custom_prompt_dict={},
            encoding=None,
            litellm_params={}
        )

        # Check that the parent class's completion method was called
        mock_parent_completion.assert_called_once()
        
        # Verify the result matches our mock response
        self.assertEqual(result, mock_response)
        
        # Check that the JSON format parameters were properly set
        call_args = mock_parent_completion.call_args[1]
        self.assertTrue(call_args["optional_params"].get("json_response_requested"))
        self.assertTrue(call_args["optional_params"].get("json_mode"))
        
        # Verify that the messages were properly modified to include JSON instructions
        messages = call_args["messages"]
        has_json_instruction = False
        for msg in messages:
            if msg.get("role") == "system" and "JSON" in msg.get("content", ""):
                has_json_instruction = True
                break
        self.assertTrue(has_json_instruction)

    @patch("litellm.llms.asi.chat.transformation.ASIChatConfig.transform_response")
    def test_transform_response(self, mock_transform):
        """Test the transform_response method"""
        # Create a mock response
        mock_raw_response = MagicMock()
        mock_raw_response.json.return_value = self.mock_response
        
        # Create a mock transformed response
        mock_transformed = ModelResponse(
            id="transformed-id",
            choices=[{"message": {"content": "Transformed content"}}],
            created=1234567890,
            model="asi-1",
            object="chat.completion"
        )
        mock_transform.return_value = mock_transformed
        
        # Test the transform_response method
        result = self.handler.transform_response(
            raw_response=mock_raw_response,
            model=self.model,
            optional_params={},
            logging_obj=MagicMock()
        )
        
        # Check that the transform method was called
        mock_transform.assert_called_once()
        
        # Check that the result matches our mock transformed response
        self.assertEqual(result, mock_transformed)
        
if __name__ == "__main__":
    unittest.main()
