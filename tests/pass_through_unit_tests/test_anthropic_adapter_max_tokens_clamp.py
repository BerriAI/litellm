"""
Tests for Anthropic adapter max_tokens clamping.

Ensures the Anthropic pass-through adapter clamps max_tokens to the backend
model's max_output_tokens when drop_params is enabled, preventing 400 errors
from providers with lower limits (e.g. DeepSeek 8192 vs Claude Opus 4 32000).

Related: https://github.com/BerriAI/litellm/issues/22249
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


class TestClampMaxTokens:
    """Tests for _clamp_max_tokens static method."""

    def test_clamp_when_exceeds_model_limit_and_drop_params_true(self):
        """max_tokens exceeding model limit should be clamped when drop_params=True."""
        result = LiteLLMMessagesToCompletionTransformationHandler._clamp_max_tokens(
            max_tokens=32000,
            model="deepseek/deepseek-chat",
            drop_params=True,
        )
        assert result == 8192

    def test_no_clamp_when_within_model_limit(self):
        """max_tokens within model limit should pass through unchanged."""
        result = LiteLLMMessagesToCompletionTransformationHandler._clamp_max_tokens(
            max_tokens=4096,
            model="deepseek/deepseek-chat",
            drop_params=True,
        )
        assert result == 4096

    def test_no_clamp_when_drop_params_false(self):
        """max_tokens should not be clamped when drop_params is False."""
        result = LiteLLMMessagesToCompletionTransformationHandler._clamp_max_tokens(
            max_tokens=32000,
            model="deepseek/deepseek-chat",
            drop_params=False,
        )
        assert result == 32000

    def test_no_clamp_when_drop_params_none(self):
        """max_tokens should not be clamped when drop_params is None (default)."""
        result = LiteLLMMessagesToCompletionTransformationHandler._clamp_max_tokens(
            max_tokens=32000,
            model="deepseek/deepseek-chat",
            drop_params=None,
        )
        assert result == 32000

    def test_clamp_respects_global_drop_params(self):
        """max_tokens should be clamped when litellm.drop_params is True."""
        original = litellm.drop_params
        try:
            litellm.drop_params = True
            result = LiteLLMMessagesToCompletionTransformationHandler._clamp_max_tokens(
                max_tokens=32000,
                model="deepseek/deepseek-chat",
                drop_params=None,
            )
            assert result == 8192
        finally:
            litellm.drop_params = original

    def test_no_clamp_for_unknown_model(self):
        """Unknown models should pass max_tokens through unchanged."""
        result = LiteLLMMessagesToCompletionTransformationHandler._clamp_max_tokens(
            max_tokens=32000,
            model="unknown-provider/unknown-model",
            drop_params=True,
        )
        assert result == 32000

    def test_exact_limit_not_clamped(self):
        """max_tokens exactly at model limit should not be clamped."""
        result = LiteLLMMessagesToCompletionTransformationHandler._clamp_max_tokens(
            max_tokens=8192,
            model="deepseek/deepseek-chat",
            drop_params=True,
        )
        assert result == 8192


class TestPrepareCompletionKwargsMaxTokens:
    """Tests that _prepare_completion_kwargs applies clamping correctly."""

    def test_prepare_kwargs_clamps_max_tokens_with_drop_params(self):
        """_prepare_completion_kwargs should clamp max_tokens when drop_params is in extra_kwargs."""
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=32000,
                messages=[{"role": "user", "content": "hello"}],
                model="deepseek/deepseek-chat",
                extra_kwargs={"drop_params": True},
            )
        )
        assert completion_kwargs["max_tokens"] == 8192

    def test_prepare_kwargs_preserves_max_tokens_without_drop_params(self):
        """_prepare_completion_kwargs should preserve max_tokens when drop_params is absent."""
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=32000,
                messages=[{"role": "user", "content": "hello"}],
                model="deepseek/deepseek-chat",
            )
        )
        assert completion_kwargs["max_tokens"] == 32000

    def test_prepare_kwargs_clamps_with_global_drop_params(self):
        """_prepare_completion_kwargs should respect litellm.drop_params global setting."""
        original = litellm.drop_params
        try:
            litellm.drop_params = True
            completion_kwargs, _ = (
                LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                    max_tokens=32000,
                    messages=[{"role": "user", "content": "hello"}],
                    model="deepseek/deepseek-chat",
                )
            )
            assert completion_kwargs["max_tokens"] == 8192
        finally:
            litellm.drop_params = original
