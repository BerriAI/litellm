import unittest
import httpx
from unittest.mock import MagicMock

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import AnthropicMessagesConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

class TestAnthropicMessagesTransformation(unittest.TestCase):
    def setUp(self):
        self.config = AnthropicMessagesConfig()
        self.model = "claude-3-haiku-20240307"
        self.messages = [{"role": "user", "content": "Hello, world!"}]
        self.logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        
        # Create mock response
        self.mock_response = MagicMock(spec=httpx.Response)
        self.mock_response.json.return_value = {
            "id": "msg_123456",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I help you today?"
                }
            ],
            "model": self.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 15
            }
        }
    
    def test_map_openai_params(self):
        # Test max_tokens mapping
        params = {"max_completion_tokens": 100}
        mapped = self.config.map_openai_params(self.model, params)
        self.assertEqual(mapped["max_tokens"], 100)
        self.assertEqual(params, {})  # Ensure original param was popped
        
        # Test stop sequences mapping
        params = {"stop": ["END", "STOP"]}
        mapped = self.config.map_openai_params(self.model, params)
        self.assertEqual(mapped["stop_sequences"], ["END", "STOP"])
        self.assertEqual(params, {})  # Ensure original param was popped
        
        # Test response_format mapping
        params = {"response_format": {"type": "json"}}
        mapped = self.config.map_openai_params(self.model, params)
        self.assertTrue(mapped["json_mode"])
        self.assertEqual(params, {})  # Ensure original param was popped
    
    def test_get_complete_url(self):
        # Test with no API base
        url = self.config.get_complete_url(None, self.model)
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        
        # Test with API base without endpoint
        url = self.config.get_complete_url("https://custom-api.example.com", self.model)
        self.assertEqual(url, "https://custom-api.example.com/v1/messages")
        
        # Test with API base that already includes endpoint
        url = self.config.get_complete_url("https://custom-api.example.com/v1/messages", self.model)
        self.assertEqual(url, "https://custom-api.example.com/v1/messages")
    
    def test_validate_environment(self):
        # Test with empty headers
        headers = {}
        result = self.config.validate_environment(headers, self.model, "test-api-key")
        self.assertEqual(result["x-api-key"], "test-api-key")
        self.assertEqual(result["anthropic-version"], "2023-06-01")
        self.assertEqual(result["content-type"], "application/json")
        
        # Test with existing headers
        headers = {
            "x-api-key": "existing-key",
            "anthropic-version": "custom-version",
            "content-type": "custom-content-type"
        }
        result = self.config.validate_environment(headers, self.model, "test-api-key")
        self.assertEqual(result["x-api-key"], "existing-key")
        self.assertEqual(result["anthropic-version"], "custom-version")
        self.assertEqual(result["content-type"], "custom-content-type")
    
    def test_transform_request(self):
        optional_params = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.95,
            "stop": ["END"],
            "unsupported_param": "value"  # This should be excluded
        }
        litellm_params = {}
        headers = {}
        
        result = self.config.transform_request(
            model=self.model,
            messages=self.messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers
        )
        
        # Check that expected keys are present
        self.assertEqual(result["model"], self.model)
        self.assertEqual(result["messages"], self.messages)
        self.assertEqual(result["max_tokens"], 100)
        self.assertEqual(result["temperature"], 0.7)
        self.assertEqual(result["top_p"], 0.95)
        self.assertEqual(result["stop_sequences"], ["END"])
        
        # Check that unsupported params are excluded
        self.assertNotIn("unsupported_param", result)
    
    def test_transform_response(self):
        result = self.config.transform_response(
            model=self.model,
            raw_response=self.mock_response,
            model_response=None,
            logging_obj=self.logging_obj,
            api_key="test-api-key",
            request_data={},
            messages=self.messages,
            optional_params={},
            litellm_params={}
        )
        
        # Verify the result is the parsed JSON from the mock response
        self.assertEqual(result, self.mock_response.json.return_value)

if __name__ == "__main__":
    unittest.main() 