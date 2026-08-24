"""
Mock tests for LiteLLMA2ACardResolver.

Tests that the card resolver tries both old and new well-known paths.
"""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from litellm.a2a_protocol.card_resolver import (
    LiteLLMA2ACardResolver,
    fix_agent_card_url,
    is_localhost_or_internal_url,
    set_agent_card_url,
)


def test_a2a_protocol_imports_when_a2a_sdk_is_missing():
    """
    a2a-sdk is an optional dependency, so the proxy degrades by reading A2A_SDK_AVAILABLE and
    returning a JSON-RPC "'a2a' package not installed" error. Reading that flag imports this
    module, so importing it with the SDK absent must not raise.

    Runs in a subprocess because the check is about import time, and a2a-sdk is installed in CI.
    """
    script = textwrap.dedent(
        """
        import sys

        class _BlockA2A:
            def find_spec(self, name, path=None, target=None):
                if name == "a2a" or name.startswith("a2a."):
                    raise ModuleNotFoundError(f"No module named '{name}'")
                return None

        sys.meta_path.insert(0, _BlockA2A())

        from litellm.a2a_protocol import asend_message_streaming
        from litellm.a2a_protocol.main import A2A_SDK_AVAILABLE

        print(f"A2A_SDK_AVAILABLE={A2A_SDK_AVAILABLE}")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )

    assert result.returncode == 0, f"importing litellm.a2a_protocol without a2a-sdk failed:\n{result.stderr}"
    assert "A2A_SDK_AVAILABLE=False" in result.stdout


@pytest.mark.asyncio
async def test_card_resolver_fallback_from_new_to_old_path():
    """
    Test that the card resolver tries the new path (/.well-known/agent-card.json) first,
    and falls back to the old path (/.well-known/agent.json) if the new path fails.
    """
    # Mock the AgentCard
    mock_agent_card = MagicMock()
    mock_agent_card.name = "Test Agent"
    mock_agent_card.description = "A test agent"

    # Track which paths were called
    paths_called = []

    # Create a mock for the parent's get_agent_card method
    async def mock_parent_get_agent_card(
        self, relative_card_path=None, http_kwargs=None, signature_verifier=None
    ):
        paths_called.append(relative_card_path)
        if relative_card_path == "/.well-known/agent-card.json":
            # First call (new path) fails
            raise Exception("404 Not Found")
        else:
            # Second call (old path) succeeds
            return mock_agent_card

    # Create a mock httpx client
    mock_httpx_client = MagicMock()

    # Patch the parent class's get_agent_card method
    # We need to patch the actual parent class method that super() calls
    with patch.object(
        LiteLLMA2ACardResolver.__bases__[0],
        "get_agent_card",
        mock_parent_get_agent_card,
    ):
        resolver = LiteLLMA2ACardResolver(
            httpx_client=mock_httpx_client, base_url="http://test-agent:8000"
        )
        result = await resolver.get_agent_card()

        # Verify both paths were tried in correct order
        assert len(paths_called) == 2
        assert paths_called[0] == "/.well-known/agent-card.json"  # New path tried first
        assert paths_called[1] == "/.well-known/agent.json"  # Old path tried second

        # Verify the result
        assert result == mock_agent_card
        assert result.name == "Test Agent"


@pytest.mark.asyncio
async def test_get_agent_card_forwards_signature_verifier():
    """
    The SDK's get_agent_card accepts a signature_verifier. Dropping it from this override
    made the call raise TypeError for any caller passing it, so it must be forwarded.
    """
    received = {}

    async def mock_parent_get_agent_card(
        self, relative_card_path=None, http_kwargs=None, signature_verifier=None
    ):
        received["signature_verifier"] = signature_verifier
        return MagicMock()

    def verifier(card):
        return None

    with patch.object(
        LiteLLMA2ACardResolver.__bases__[0],
        "get_agent_card",
        mock_parent_get_agent_card,
    ):
        resolver = LiteLLMA2ACardResolver(
            httpx_client=MagicMock(), base_url="http://test-agent:8000"
        )
        await resolver.get_agent_card(
            relative_card_path="/.well-known/agent-card.json",
            signature_verifier=verifier,
        )

    assert received["signature_verifier"] is verifier


def test_is_localhost_or_internal_url():
    """Test that localhost/internal URLs are correctly detected."""
    # Should return True for localhost variants
    assert is_localhost_or_internal_url("http://localhost:8000/") is True
    assert is_localhost_or_internal_url("http://0.0.0.0:8001/") is True

    # Should return False for public URLs
    assert is_localhost_or_internal_url("https://my-agent.example.com/") is False
    assert is_localhost_or_internal_url(None) is False


def test_fix_agent_card_url_replaces_localhost():
    """Test that fix_agent_card_url replaces localhost URLs with base_url."""
    # Create mock agent card with localhost URL
    mock_card = MagicMock()
    mock_card.url = "http://0.0.0.0:8001/"

    # Fix the URL
    result = fix_agent_card_url(mock_card, "https://my-public-agent.example.com")

    # Verify localhost URL was replaced with base_url
    assert result.url == "https://my-public-agent.example.com/"


def test_set_agent_card_url_updates_top_level_and_supported_interface():
    card = SimpleNamespace(
        url="http://localhost:10001/",
        supported_interfaces=[SimpleNamespace(url="http://0.0.0.0:10001/")],
    )

    set_agent_card_url(card, "https://my-public-agent.example.com")

    assert card.url == "https://my-public-agent.example.com/"
    assert card.supported_interfaces[0].url == "https://my-public-agent.example.com/"


def test_fix_agent_card_url_updates_interface_when_top_level_is_localhost():
    card = SimpleNamespace(
        url="http://localhost:10001/",
        supported_interfaces=[SimpleNamespace(url="http://0.0.0.0:10001/")],
    )

    result = fix_agent_card_url(card, "https://my-public-agent.example.com")

    assert result.url == "https://my-public-agent.example.com/"
    assert result.supported_interfaces[0].url == "https://my-public-agent.example.com/"
