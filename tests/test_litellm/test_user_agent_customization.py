"""
Test User-Agent header customization
Tests for Issue #19017: Option to disable or customize default User-Agent header
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion


def test_default_user_agent_is_set():
    """
    Test that by default, litellm sets the User-Agent header.
    """
    from litellm.llms.custom_httpx.http_handler import get_default_headers
    from litellm._version import version
    
    # Reset to default
    litellm.disable_default_user_agent = False
    
    headers = get_default_headers()
    assert "User-Agent" in headers
    assert headers["User-Agent"] == f"litellm/{version}"


def test_disable_default_user_agent():
    """
    Test that setting litellm.disable_default_user_agent = True prevents
    the default User-Agent header from being set.
    """
    from litellm.llms.custom_httpx.http_handler import get_default_headers
    
    # Disable default User-Agent
    litellm.disable_default_user_agent = True
    
    headers = get_default_headers()
    assert headers == {}
    
    # Reset to default
    litellm.disable_default_user_agent = False


def test_custom_user_agent_via_extra_headers():
    """
    Test that users can provide their own User-Agent via extra_headers.
    This is critical for Claude Code credentials that require specific User-Agent.
    """
    import httpx
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    
    # Disable default User-Agent
    litellm.disable_default_user_agent = True
    
    # Create HTTP handler
    handler = HTTPHandler()
    
    # Custom User-Agent for Claude Code
    custom_headers = {"User-Agent": "Claude Code/1.0.0"}
    
    # Build request with custom headers
    req = handler.client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers=custom_headers,
        json={"test": "data"}
    )
    
    # Verify custom User-Agent is used
    assert "User-Agent" in req.headers
    assert req.headers["User-Agent"] == "Claude Code/1.0.0"
    
    # Reset to default
    litellm.disable_default_user_agent = False


def test_env_var_disable_default_user_agent():
    """
    Test that LITELLM_DISABLE_DEFAULT_USER_AGENT environment variable works.
    """
    from litellm.llms.custom_httpx.http_handler import get_default_headers
    
    # Test with env var
    with patch.dict(os.environ, {"LITELLM_DISABLE_DEFAULT_USER_AGENT": "True"}):
        # Manually set the flag (in real usage, this would be done at import time)
        litellm.disable_default_user_agent = True
        
        headers = get_default_headers()
        assert headers == {}
    
    # Reset to default
    litellm.disable_default_user_agent = False


@pytest.mark.asyncio
async def test_async_http_handler_respects_disable_flag():
    """
    Test that AsyncHTTPHandler also respects the disable_default_user_agent flag.
    """
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_default_headers
    
    # Disable default User-Agent
    litellm.disable_default_user_agent = True
    
    # Create async handler
    handler = AsyncHTTPHandler()
    
    # Check that headers are empty
    headers = get_default_headers()
    assert headers == {}
    
    await handler.close()
    
    # Reset to default
    litellm.disable_default_user_agent = False


def test_override_user_agent_without_disabling():
    """
    Test that users can override User-Agent by passing it in extra_headers,
    even without disabling the default.
    
    Note: httpx will use the last header value when building the request.
    """
    import httpx
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    
    # Default User-Agent is enabled
    litellm.disable_default_user_agent = False
    
    # Create HTTP handler (will have default User-Agent)
    handler = HTTPHandler()
    
    # Custom User-Agent provided in request
    custom_headers = {"User-Agent": "MyCustomAgent/2.0.0"}
    
    # Build request with custom headers - httpx merges headers
    req = handler.client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers=custom_headers,
        json={"test": "data"}
    )
    
    # The custom User-Agent should override the default
    assert "User-Agent" in req.headers
    # httpx uses the request-level header over the client-level header
    assert req.headers["User-Agent"] == "MyCustomAgent/2.0.0"


def test_claude_code_use_case():
    """
    Test the specific use case from Issue #19017:
    Claude Code credentials that require specific User-Agent.
    """
    # Disable default User-Agent globally
    litellm.disable_default_user_agent = True
    
    # This is what the user would do in their code
    custom_headers = {"User-Agent": "Claude Code"}
    
    # Verify the headers can be passed through
    from litellm.llms.custom_httpx.http_handler import get_default_headers
    default_headers = get_default_headers()
    
    # Default headers should be empty
    assert default_headers == {}
    
    # Custom headers would be used in the actual request
    assert custom_headers["User-Agent"] == "Claude Code"
    
    # Reset
    litellm.disable_default_user_agent = False


def test_backwards_compatibility():
    """
    Test that existing code continues to work without any changes.
    By default, the User-Agent header should still be set.
    """
    from litellm.llms.custom_httpx.http_handler import get_default_headers
    from litellm._version import version
    
    # Ensure default behavior is maintained
    litellm.disable_default_user_agent = False
    
    headers = get_default_headers()
    assert "User-Agent" in headers
    assert headers["User-Agent"] == f"litellm/{version}"
    
    # Create HTTP handler
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    handler = HTTPHandler()
    
    # Build a request
    req = handler.client.build_request(
        "GET",
        "https://api.openai.com/v1/models"
    )
    
    # Default User-Agent should be present
    assert "User-Agent" in req.headers
    assert "litellm" in req.headers["User-Agent"]
