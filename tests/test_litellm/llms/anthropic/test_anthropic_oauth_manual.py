#!/usr/bin/env python3
"""
Manual integration tests for Anthropic OAuth token handling.

These tests require a real Anthropic OAuth token and make actual API calls.
They are skipped by default in CI - run manually with:

    ANTHROPIC_OAUTH_TOKEN=sk-ant-oat... poetry run pytest tests/test_litellm/llms/anthropic/test_anthropic_oauth_manual.py -v -s

Or run as a script:

    poetry run python tests/test_litellm/llms/anthropic/test_anthropic_oauth_manual.py sk-ant-oat...
"""

import json
import os
import sys
from typing import Any, Dict, Optional

import pytest

# Get OAuth token from env or skip
OAUTH_TOKEN: Optional[str] = os.getenv("ANTHROPIC_OAUTH_TOKEN")

# Skip all tests if no OAuth token is available
pytestmark = pytest.mark.skipif(
    OAUTH_TOKEN is None,
    reason="ANTHROPIC_OAUTH_TOKEN environment variable not set"
)


def _print_headers(headers: Dict[str, Any], title: str = "Headers") -> None:
    """Helper to print headers in a readable format."""
    print(f"\n{title}:")
    print(json.dumps(headers, indent=2))


def _print_response(response: Any, title: str = "Response") -> None:
    """Helper to print response in a readable format."""
    print(f"\n{title}:")
    if hasattr(response, "model_dump"):
        print(json.dumps(response.model_dump(), indent=2, default=str))
    elif hasattr(response, "json"):
        print(json.dumps(response.json(), indent=2, default=str))
    else:
        print(json.dumps(dict(response), indent=2, default=str))


class TestOAuthPassThroughHeaders:
    """Test OAuth header transformation for pass-through API."""

    def test_pass_through_headers_transformation(self) -> None:
        """Verify pass-through correctly transforms OAuth headers."""
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        assert OAUTH_TOKEN is not None
        config = AnthropicMessagesConfig()
        headers = {"authorization": f"Bearer {OAUTH_TOKEN}"}

        updated_headers, _ = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        _print_headers(updated_headers, "Pass-through Transformed Headers")

        # Verify OAuth headers
        assert "authorization" in updated_headers
        assert "x-api-key" not in updated_headers
        assert "oauth-2025-04-20" in updated_headers.get("anthropic-beta", "")
        assert updated_headers.get("anthropic-dangerous-direct-browser-access") == "true"


class TestOAuthDirectApiKey:
    """Test OAuth header transformation when token passed as api_key."""

    def test_direct_api_key_headers_transformation(self) -> None:
        """Verify direct api_key correctly transforms OAuth headers."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert OAUTH_TOKEN is not None
        config = AnthropicModelInfo()
        headers: Dict[str, str] = {}

        updated_headers = config.validate_environment(
            headers=headers,
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=OAUTH_TOKEN,
            api_base=None,
        )

        _print_headers(updated_headers, "Direct api_key Transformed Headers")

        # Verify OAuth headers
        assert "authorization" in updated_headers
        assert "x-api-key" not in updated_headers
        assert "oauth-2025-04-20" in updated_headers.get("anthropic-beta", "")
        assert updated_headers.get("anthropic-dangerous-direct-browser-access") == "true"


class TestOAuthRealAPICalls:
    """Test real API calls with OAuth token."""

    def test_litellm_completion(self) -> None:
        """Test litellm.completion() with OAuth token."""
        import litellm

        assert OAUTH_TOKEN is not None
        response = litellm.completion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Say 'OAuth test successful' in exactly 3 words"}],
            api_key=OAUTH_TOKEN,
            max_tokens=20,
        )

        _print_response(response, "litellm.completion Response")

        assert response.choices[0].message.content is not None

    def test_pass_through_api_call(self) -> None:
        """Test direct API call with pass-through transformed headers."""
        import httpx

        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        assert OAUTH_TOKEN is not None
        config = AnthropicMessagesConfig()
        headers = {"authorization": f"Bearer {OAUTH_TOKEN}"}

        updated_headers, _ = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=updated_headers,
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "Say 'pass-through works' in 3 words"}],
            },
            timeout=30.0,
        )

        print(f"\nStatus: {response.status_code}")
        _print_headers(dict(response.headers), "Response Headers")
        _print_response(response, "Response Body")

        assert response.status_code == 200
        assert "content" in response.json()


# Allow running as a script
if __name__ == "__main__":
    # Get token from CLI arg if provided
    if len(sys.argv) > 1:
        OAUTH_TOKEN = sys.argv[1]

    if not OAUTH_TOKEN:
        print("Usage: python test_anthropic_oauth_manual.py <oauth-token>")
        print("   or: ANTHROPIC_OAUTH_TOKEN=sk-ant-oat... python test_anthropic_oauth_manual.py")
        sys.exit(1)

    if not OAUTH_TOKEN.startswith("sk-ant-oat"):
        print(f"Warning: Token doesn't look like an OAuth token (expected sk-ant-oat prefix)")

    print("Anthropic OAuth Token Manual Test Suite")
    print(f"Token: {OAUTH_TOKEN[:15]}...{OAUTH_TOKEN[-5:]}")
    print("=" * 60)

    # Run tests manually
    test_passthrough = TestOAuthPassThroughHeaders()
    test_passthrough.test_pass_through_headers_transformation()

    test_direct = TestOAuthDirectApiKey()
    test_direct.test_direct_api_key_headers_transformation()

    test_api = TestOAuthRealAPICalls()
    test_api.test_litellm_completion()
    test_api.test_pass_through_api_call()

    print("\n" + "=" * 60)
    print("All tests passed!")
