import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


class TestCustomApiBaseFix:
    """Test that custom api_base works correctly for both Gemini and Vertex AI models"""

    def test_gemini_custom_api_base_preserves_path(self):
        """Test that Gemini custom api_base preserves the full path structure"""
        vertex_base = VertexBase()
        
        # Test case from the issue: Cloudflare AI Gateway
        cloudflare_base = "https://gateway.ai.cloudflare.com/v1/my-id/my-gateway/google-ai-studio"
        original_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=test-key"
        
        token, final_url = vertex_base._check_custom_proxy(
            api_base=cloudflare_base,
            auth_header=None,
            custom_llm_provider="gemini",
            gemini_api_key="test-api-key",
            endpoint=":generateContent",
            stream=False,
            url=original_url,
        )
        
        # Should preserve the full path structure
        expected_url = "https://gateway.ai.cloudflare.com/v1/my-id/my-gateway/google-ai-studio/v1beta/models/gemini-2.5-flash:generateContent"
        assert final_url == expected_url
        assert token == "test-api-key"
    
    def test_gemini_custom_api_base_with_streaming(self):
        """Test that streaming adds the alt=sse parameter correctly"""
        vertex_base = VertexBase()
        
        cloudflare_base = "https://gateway.ai.cloudflare.com/v1/my-id/my-gateway/google-ai-studio"
        original_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent?key=test-key&alt=sse"
        
        token, final_url = vertex_base._check_custom_proxy(
            api_base=cloudflare_base,
            auth_header=None,
            custom_llm_provider="gemini",
            gemini_api_key="test-api-key",
            endpoint=":streamGenerateContent",
            stream=True,
            url=original_url,
        )
        
        # Should preserve path and add streaming parameter
        expected_url = "https://gateway.ai.cloudflare.com/v1/my-id/my-gateway/google-ai-studio/v1beta/models/gemini-pro:streamGenerateContent?alt=sse"
        assert final_url == expected_url
        assert token == "test-api-key"
    
    def test_vertex_ai_custom_api_base_unchanged(self):
        """Test that Vertex AI behavior remains unchanged"""
        vertex_base = VertexBase()
        
        # Vertex AI uses a different URL structure
        cloudflare_base = "https://gateway.ai.cloudflare.com/v1/my-id/my-gateway/google-vertex-ai/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash"
        original_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent"
        
        token, final_url = vertex_base._check_custom_proxy(
            api_base=cloudflare_base,
            auth_header="Bearer mock-token",
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=":generateContent",
            stream=False,
            url=original_url,
        )
        
        # Vertex AI should still use the old behavior (appending endpoint)
        expected_url = "https://gateway.ai.cloudflare.com/v1/my-id/my-gateway/google-vertex-ai/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent"
        assert final_url == expected_url
        assert token == "Bearer mock-token"  # Auth header unchanged for Vertex AI
    
    def test_full_flow_with_get_token_and_url(self):
        """Test the full flow through _get_token_and_url"""
        vertex_base = VertexBase()
        
        cloudflare_base = "https://gateway.ai.cloudflare.com/v1/my-id/my-gateway/google-ai-studio"
        
        token, url = vertex_base._get_token_and_url(
            model="gemini/gemini-2.5-flash",
            auth_header=None,
            gemini_api_key="test-api-key",
            vertex_project=None,
            vertex_location=None,
            vertex_credentials=None,
            stream=False,
            custom_llm_provider="gemini",
            api_base=cloudflare_base,
        )
        
        # Should produce the correct URL with custom base
        assert cloudflare_base in url
        assert "/v1beta/models/gemini/gemini-2.5-flash:generateContent" in url
        assert token == "test-api-key"
        
    def test_missing_gemini_api_key_raises_error(self):
        """Test that missing Gemini API key raises an error"""
        vertex_base = VertexBase()
        
        cloudflare_base = "https://gateway.ai.cloudflare.com/v1/my-id/my-gateway/google-ai-studio"
        original_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
        with pytest.raises(ValueError, match="Missing gemini_api_key"):
            vertex_base._check_custom_proxy(
                api_base=cloudflare_base,
                auth_header=None,
                custom_llm_provider="gemini",
                gemini_api_key=None,  # Missing API key
                endpoint=":generateContent",
                stream=False,
                url=original_url,
            )