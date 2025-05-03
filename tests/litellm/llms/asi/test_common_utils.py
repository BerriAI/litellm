"""
Test the ASI common utilities module
"""

import unittest
from unittest.mock import MagicMock, patch

from litellm.llms.asi.common_utils import (
    is_asi_model,
    get_asi_model_name,
    validate_environment,
)


class TestASICommonUtils(unittest.TestCase):
    """Test the ASI common utilities"""

    def test_is_asi_model(self):
        """Test the is_asi_model function"""
        # Test with ASI model
        self.assertTrue(is_asi_model("asi1-mini"))
        self.assertTrue(is_asi_model("asi/asi1-mini"))
        
        # Test with non-ASI model
        self.assertFalse(is_asi_model("gpt-4"))
        self.assertFalse(is_asi_model("claude-3"))

    def test_get_asi_model_name(self):
        """Test the get_asi_model_name function"""
        # Test with provider prefix
        self.assertEqual(get_asi_model_name("asi/asi1-mini"), "asi1-mini")
        
        # Test without provider prefix
        self.assertEqual(get_asi_model_name("asi1-mini"), "asi1-mini")

    @patch("litellm.utils.get_secret")
    def test_validate_environment(self, mock_get_secret):
        """Test the validate_environment function"""
        # Test with provided API key
        api_key = "test-api-key"
        validate_environment(api_key)  # Should not raise an error
        
        # Test with environment variable
        mock_get_secret.return_value = "env-api-key"
        validate_environment()  # Should not raise an error
        mock_get_secret.assert_called_with("ASI_API_KEY")
        
        # Test with missing API key
        mock_get_secret.return_value = None
        with self.assertRaises(ValueError):
            validate_environment()


if __name__ == "__main__":
    unittest.main()
