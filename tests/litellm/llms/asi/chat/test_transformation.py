"""
Test the ASI chat transformation module
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

import litellm
from litellm.llms.asi.chat.transformation import ASIChatConfig
from litellm.llms.asi.chat.json_extraction import extract_json
from litellm.utils import ModelResponse


class TestASIChatConfig(unittest.TestCase):
    """Test the ASIChatConfig class"""

    def setUp(self):
        """Set up the test"""
        self.config = ASIChatConfig()
        self.model = "asi1-mini"
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

    def test_map_openai_params(self):
        """Test the map_openai_params method"""
        # Test with JSON response format
        non_default_params = {"response_format": {"type": "json_object"}}
        optional_params = {}
        
        result = self.config.map_openai_params(non_default_params, optional_params, self.model)
        
        # Check that json_response_requested and json_mode are set
        assert optional_params.get("json_response_requested") is True
        assert optional_params.get("json_mode") is True

    def test_get_api_key(self):
        """Test the get_api_key method"""
        # Test with provided API key
        api_key = "test-api-key"
        result = ASIChatConfig.get_api_key(api_key)
        assert result == api_key

    def test_json_extraction(self):
        """Test the JSON extraction functionality"""
        # Test with JSON in a code block
        json_content = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
        content_with_code_block = f"Here's the JSON data you requested:\n\n```json\n{json_content}\n```"
        
        extracted = extract_json(content_with_code_block)
        assert extracted is not None
        if extracted:
            assert json_content in extracted
        
        # Test with direct JSON
        direct_json = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
        extracted = extract_json(direct_json)
        assert extracted == direct_json
        
        # Test with plain text content
        plain_text = "This is just plain text without any JSON."
        extracted = extract_json(plain_text)
        assert extracted is not None
        # Our implementation wraps plain text in a JSON object with a "text" field
        import json
        parsed = json.loads(extracted)
        assert "text" in parsed
        assert parsed["text"] == plain_text
        

if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
