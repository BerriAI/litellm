import asyncio
import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest import mock

import pytest

import litellm


def test_ollama_turbo_auth_header():
    """Test that ollama.com URLs get the correct auth header without Bearer prefix."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    from unittest.mock import patch

    with patch.object(client, "post") as mock_post:
        try:
            litellm.completion(
                model="ollama_chat/gpt-oss:120b",
                messages=[{"role": "user", "content": "test"}],
                api_base="https://ollama.com",
                api_key="test_key_123",
                client=client,
            )
        except Exception as e:
            print(e)
        
        mock_post.assert_called()
        
        # Check the headers just like test_ollama.py does
        print(mock_post.call_args.kwargs)
        headers = mock_post.call_args.kwargs.get("headers", {})
        
        # Should NOT have Bearer prefix for ollama.com
        assert headers.get("Authorization") == "test_key_123"
        assert "Bearer" not in headers.get("Authorization", "")


def test_ollama_localhost_auth_header():
    """Test that localhost URLs get the correct auth header with Bearer prefix."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    from unittest.mock import patch

    with patch.object(client, "post") as mock_post:
        try:
            litellm.completion(
                model="ollama_chat/llama2",
                messages=[{"role": "user", "content": "test"}],
                api_base="http://localhost:11434",
                api_key="test_key_456",
                client=client,
            )
        except Exception as e:
            print(e)
        
        mock_post.assert_called()
        
        # Check the headers just like test_ollama.py does
        print(mock_post.call_args.kwargs)
        headers = mock_post.call_args.kwargs.get("headers", {})
        
        # Should have Bearer prefix for localhost
        assert headers.get("Authorization") == "Bearer test_key_456"


@pytest.mark.skip(reason="Integration test - requires OLLAMA_TURBO_API_KEY")
def test_ollama_turbo_integration():
    """Integration test with real Ollama Turbo API."""
    api_key = os.environ.get("OLLAMA_TURBO_API_KEY")
    if not api_key:
        pytest.skip("OLLAMA_TURBO_API_KEY not set")
    
    try:
        response = litellm.completion(
            model="ollama_chat/gpt-oss:120b",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            api_base="https://ollama.com",
            api_key=api_key,
            max_tokens=10,
        )
        
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        
    except Exception as e:
        pytest.fail(f"Ollama Turbo API call failed: {str(e)}")


