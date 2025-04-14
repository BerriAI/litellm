import unittest
import litellm
from litellm import exceptions


class TestExceptionsMasking(unittest.TestCase):
    def test_api_key_masking_in_exceptions(self):
        """Test that API keys are properly masked in exception messages"""
        
        # Test with a message containing an API key
        api_key = "sk-12345678901234567890"
        message = f"Failed to authenticate with API key {api_key}"
        
        # Create an exception with this message
        exception = exceptions.AuthenticationError(
            message=message,
            llm_provider="test_provider",
            model="test_model"
        )
        
        # Check that the API key is not present in the exception message
        self.assertNotIn(api_key, exception.message)
        # Check that a masked version is present instead (should have the prefix and suffix)
        self.assertIn("sk-1", exception.message)
        self.assertIn("7890", exception.message)
        
    def test_multiple_sensitive_keys_masked(self):
        """Test that multiple sensitive keys in the same message are masked"""
        
        # Message with multiple sensitive information
        message = (
            "Error occurred. API key: sk-abc123def456, "
            "Secret key: secret_xyz987, "
            "Password: pass123word"
        )
        
        # Create an exception with this message
        exception = exceptions.BadRequestError(
            message=message,
            model="test_model",
            llm_provider="test_provider"
        )
        
        # Check that none of the sensitive data is present
        self.assertNotIn("sk-abc123def456", exception.message)
        self.assertNotIn("secret_xyz987", exception.message)
        self.assertNotIn("pass123word", exception.message)
        
        # Check that masked versions are present
        self.assertIn("sk-a", exception.message)
        self.assertIn("456", exception.message)
        self.assertIn("secr", exception.message)
        self.assertIn("987", exception.message)
        self.assertIn("pass", exception.message)
        self.assertIn("word", exception.message)


if __name__ == "__main__":
    unittest.main() 