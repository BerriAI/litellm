"""
Tests for Anthropic CountTokens API custom api_base handling.

Verifies that AnthropicTokenCounter correctly builds the count_tokens endpoint
from a custom api_base instead of hardcoding api.anthropic.com.
"""

import os
import sys
from typing import Optional
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter


class TestCountTokensApiBase:
    """Verify api_base is correctly extracted from litellm_params and used to build endpoint URL."""

    @pytest.mark.asyncio
    async def test_custom_api_base_builds_correct_count_tokens_url(self):
        """When api_base is provided in litellm_params, count_tokens endpoint should use it."""
        counter = AnthropicTokenCounter()
        captured_api_base: Optional[str] = None

        async def _fake_handle_count_tokens_request(
            model, messages, api_key, api_base=None, **kwargs
        ):
            nonlocal captured_api_base
            captured_api_base = api_base
            return {"input_tokens": 42}

        with patch(
            "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler.handle_count_tokens_request",
            side_effect=_fake_handle_count_tokens_request,
        ):
            result = await counter.count_tokens(
                model_to_use="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hello"}],
                contents=None,
                deployment={
                    "litellm_params": {
                        "api_key": "sk-test-key",
                        "api_base": "https://api.lkeap.cloud.tencent.com/plan/anthropic",
                    }
                },
                request_model="claude-3-5-sonnet",
            )

        assert (
            captured_api_base
            == "https://api.lkeap.cloud.tencent.com/plan/anthropic/v1/messages/count_tokens"
        )
        assert result is not None
        assert result.total_tokens == 42

    @pytest.mark.asyncio
    async def test_api_base_without_trailing_slash(self):
        """api_base without trailing slash should still produce correct URL."""
        counter = AnthropicTokenCounter()
        captured_api_base: Optional[str] = None

        async def _fake_handle_count_tokens_request(
            model, messages, api_key, api_base=None, **kwargs
        ):
            nonlocal captured_api_base
            captured_api_base = api_base
            return {"input_tokens": 10}

        with patch(
            "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler.handle_count_tokens_request",
            side_effect=_fake_handle_count_tokens_request,
        ):
            await counter.count_tokens(
                model_to_use="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                contents=None,
                deployment={
                    "litellm_params": {
                        "api_key": "sk-test-key",
                        "api_base": "https://custom.host/plan/anthropic",
                    }
                },
                request_model="claude-3-5-sonnet",
            )

        assert (
            captured_api_base
            == "https://custom.host/plan/anthropic/v1/messages/count_tokens"
        )

    @pytest.mark.asyncio
    async def test_no_api_base_uses_default_endpoint(self):
        """When api_base is not provided, api_base should be None (handler will use default)."""
        counter = AnthropicTokenCounter()
        captured_api_base: Optional[str] = None

        async def _fake_handle_count_tokens_request(
            model, messages, api_key, api_base=None, **kwargs
        ):
            nonlocal captured_api_base
            captured_api_base = api_base
            return {"input_tokens": 5}

        with patch(
            "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler.handle_count_tokens_request",
            side_effect=_fake_handle_count_tokens_request,
        ):
            await counter.count_tokens(
                model_to_use="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                contents=None,
                deployment={
                    "litellm_params": {
                        "api_key": "sk-test-key",
                    }
                },
                request_model="claude-3-5-sonnet",
            )

        assert captured_api_base is None

    @pytest.mark.asyncio
    async def test_api_base_with_trailing_slash_stripped(self):
        """api_base with trailing slash should have it stripped before appending path."""
        counter = AnthropicTokenCounter()
        captured_api_base: Optional[str] = None

        async def _fake_handle_count_tokens_request(
            model, messages, api_key, api_base=None, **kwargs
        ):
            nonlocal captured_api_base
            captured_api_base = api_base
            return {"input_tokens": 7}

        with patch(
            "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler.handle_count_tokens_request",
            side_effect=_fake_handle_count_tokens_request,
        ):
            await counter.count_tokens(
                model_to_use="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                contents=None,
                deployment={
                    "litellm_params": {
                        "api_key": "sk-test-key",
                        "api_base": "https://custom.host/plan/anthropic/",
                    }
                },
                request_model="claude-3-5-sonnet",
            )

        assert (
            captured_api_base
            == "https://custom.host/plan/anthropic/v1/messages/count_tokens"
        )
