import unittest
import httpx
from unittest.mock import MagicMock

from litellm.llms.bedrock.anthropic_messages.invoke_transformation import BedrockInvokeAnthropicMessagesConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

class TestBedrockInvokeAnthropicMessagesTransformation(unittest.TestCase):
    def setUp(self):
        self.config = BedrockInvokeAnthropicMessagesConfig()
        self.model = "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0"
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
    
    def test_get_complete_url(self):
        # For Bedrock, the URL handling is deferred to the main handler
        url = self.config.get_complete_url(None, self.model)
        self.assertEqual(url, "")
        
        url = self.config.get_complete_url("https://custom-api.example.com", self.model)
        self.assertEqual(url, "https://custom-api.example.com")
    
    def test_validate_environment(self):
        # Test with empty headers - Bedrock doesn't use API keys
        headers = {}
        result = self.config.validate_environment(headers, self.model, None)
        self.assertEqual(result["content-type"], "application/json")
        self.assertNotIn("x-api-key", result)
        
        # Test with existing headers
        headers = {
            "content-type": "custom-content-type"
        }
        result = self.config.validate_environment(headers, self.model, None)
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
        self.assertEqual(result["messages"], self.messages)
        self.assertEqual(result["max_tokens"], 100)
        self.assertEqual(result["temperature"], 0.7)
        self.assertEqual(result["top_p"], 0.95)
        self.assertEqual(result["stop_sequences"], ["END"])
        self.assertEqual(result["anthropic_version"], "bedrock-2023-05-31")
        
        # Check that model is not included (handled by Bedrock differently)
        self.assertNotIn("model", result)
        
        # Check that unsupported params are excluded
        self.assertNotIn("unsupported_param", result)
    
    def test_transform_response(self):
        # This is just testing the placeholder implementation for now
        result = self.config.transform_response(
            model=self.model,
            raw_response=self.mock_response,
            model_response=None,
            logging_obj=self.logging_obj,
            api_key="",
            request_data={},
            messages=self.messages,
            optional_params={},
            litellm_params={}
        )
        
        # Verify the result is the parsed JSON from the mock response
        self.assertEqual(result, self.mock_response.json.return_value)

if __name__ == "__main__":
    unittest.main() 