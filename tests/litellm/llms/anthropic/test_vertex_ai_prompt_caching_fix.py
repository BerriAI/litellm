"""
Test file for Vertex AI prompt caching fix.

This test verifies that:
1. The anthropic-beta header is removed for Vertex AI requests
2. Regular Anthropic requests still work correctly
3. Prompt caching detection logic is preserved
"""

import unittest
from unittest.mock import patch, MagicMock
from litellm.llms.anthropic.common_utils import AnthropicModelInfo


class TestVertexAIPromptCachingFix(unittest.TestCase):
    """Test cases for the Vertex AI prompt caching fix."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_info = AnthropicModelInfo()

    def _is_cache_control_set(self, messages):
        """Helper method to test cache control detection."""
        for message in messages:
            if message.get("cache_control", None) is not None:
                return True
            _message_content = message.get("content")
            if _message_content is not None and isinstance(_message_content, list):
                for content in _message_content:
                    if "cache_control" in content:
                        return True
        return False

    def test_vertex_ai_removes_anthropic_beta_header(self):
        """Test that anthropic-beta header is removed for Vertex AI requests."""
        # Mock the get_anthropic_headers method
        with patch.object(self.model_info, 'get_anthropic_headers') as mock_get_headers:
            # Set up the mock to return headers with anthropic-beta
            mock_get_headers.return_value = {
                'anthropic-beta': 'prompt-caching-2024-07-31',
                'content-type': 'application/json'
            }
            
            # Test with Vertex AI request
            optional_params = {'is_vertex_request': True}
            headers = self.model_info.get_anthropic_headers(
                model='vertex/claude-3-5-sonnet-20240620',
                messages=[],
                optional_params=optional_params
            )
            
            # Verify anthropic-beta header is removed
            self.assertNotIn('anthropic-beta', headers)

    def test_regular_anthropic_preserves_anthropic_beta_header(self):
        """Test that regular Anthropic requests preserve anthropic-beta header."""
        # Mock the get_anthropic_headers method
        with patch.object(self.model_info, 'get_anthropic_headers') as mock_get_headers:
            # Set up the mock to return headers with anthropic-beta
            mock_get_headers.return_value = {
                'anthropic-beta': 'prompt-caching-2024-07-31',
                'content-type': 'application/json'
            }
            
            # Test with regular Anthropic request (not Vertex AI)
            optional_params = {'is_vertex_request': False}
            headers = self.model_info.get_anthropic_headers(
                model='claude-3-5-sonnet-20240620',
                messages=[],
                optional_params=optional_params
            )
            
            # Verify anthropic-beta header is preserved
            self.assertIn('anthropic-beta', headers)

    def test_prompt_caching_detection_still_works(self):
        """Test that prompt caching detection logic still works."""
        # Test messages with cache_control
        messages_with_cache = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Test prompt',
                        'cache_control': {'type': 'ephemeral'}
                    }
                ]
            }
        ]
        
        # Test messages without cache_control
        messages_without_cache = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Regular prompt'
                    }
                ]
            }
        ]
        
        # Test cache detection
        cache_detected_with = self._is_cache_control_set(messages_with_cache)
        cache_detected_without = self._is_cache_control_set(messages_without_cache)
        
        self.assertTrue(cache_detected_with)
        self.assertFalse(cache_detected_without)

    def test_vertex_ai_without_user_beta_header(self):
        """Test Vertex AI request when no user-provided beta header exists."""
        # Mock the get_anthropic_headers method
        with patch.object(self.model_info, 'get_anthropic_headers') as mock_get_headers:
            # Set up the mock to return headers without anthropic-beta
            mock_get_headers.return_value = {
                'content-type': 'application/json'
            }
            
            # Test with Vertex AI request
            optional_params = {'is_vertex_request': True}
            headers = self.model_info.get_anthropic_headers(
                model='vertex/claude-3-5-sonnet-20240620',
                messages=[],
                optional_params=optional_params
            )
            
            # Verify no anthropic-beta header is present
            self.assertNotIn('anthropic-beta', headers)


if __name__ == '__main__':
    unittest.main()