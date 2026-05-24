"""
Unit tests for the hosted_vllm Anthropic passthrough feature.

Verifies that:
- The payload stays in Anthropic format when the flag is enabled (no translation).
- The payload is translated to OpenAI format when the flag is disabled (default).
- Both the env-var and per-deployment litellm_param paths work correctly.
"""

import os
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.hosted_vllm.messages.transformation import (
    HostedVLLMAnthropicMessagesConfig,
    _should_skip_anthropic_translation,
)
from litellm.types.router import GenericLiteLLMParams


# ---------------------------------------------------------------------------
# _should_skip_anthropic_translation
# ---------------------------------------------------------------------------


def _make_params(**kwargs) -> GenericLiteLLMParams:
    return GenericLiteLLMParams(**kwargs)


class TestShouldSkipTranslation:
    def test_returns_false_by_default(self):
        params = _make_params()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION", None)
            assert _should_skip_anthropic_translation(params) is False

    def test_env_var_true_enables_skip(self):
        params = _make_params()
        with patch.dict(os.environ, {"DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION": "true"}):
            assert _should_skip_anthropic_translation(params) is True

    def test_env_var_1_enables_skip(self):
        params = _make_params()
        with patch.dict(os.environ, {"DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION": "1"}):
            assert _should_skip_anthropic_translation(params) is True

    def test_env_var_false_does_not_enable_skip(self):
        params = _make_params()
        with patch.dict(os.environ, {"DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION": "false"}):
            assert _should_skip_anthropic_translation(params) is False

    def test_litellm_param_true_overrides_env(self):
        # param=True wins even when env var is absent
        params = _make_params(disable_anthropic_translation=True)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION", None)
            assert _should_skip_anthropic_translation(params) is True

    def test_litellm_param_false_overrides_env(self):
        # param=False wins even when env var would enable skip
        params = _make_params(disable_anthropic_translation=False)
        with patch.dict(os.environ, {"DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION": "true"}):
            assert _should_skip_anthropic_translation(params) is False


# ---------------------------------------------------------------------------
# HostedVLLMAnthropicMessagesConfig — URL construction
# ---------------------------------------------------------------------------


class TestHostedVLLMConfig:
    def setup_method(self):
        self.config = HostedVLLMAnthropicMessagesConfig()

    def test_get_complete_url_strips_v1_then_appends_messages(self):
        # Standard api_base ends with /v1 — must not double up
        url = self.config.get_complete_url(
            api_base="http://vllm-host/v1",
            api_key="key",
            model="qwen",
            optional_params={},
            litellm_params={},
        )
        assert url == "http://vllm-host/v1/messages"

    def test_get_complete_url_bare_host_appends_v1_messages(self):
        url = self.config.get_complete_url(
            api_base="http://vllm-host",
            api_key="key",
            model="qwen",
            optional_params={},
            litellm_params={},
        )
        assert url == "http://vllm-host/v1/messages"

    def test_get_complete_url_no_double_suffix(self):
        url = self.config.get_complete_url(
            api_base="http://vllm-host/v1/messages",
            api_key="key",
            model="qwen",
            optional_params={},
            litellm_params={},
        )
        assert url == "http://vllm-host/v1/messages"

    def test_get_complete_url_raises_without_api_base(self):
        with pytest.raises(ValueError, match="api_base is required"):
            self.config.get_complete_url(
                api_base=None,
                api_key="key",
                model="qwen",
                optional_params={},
                litellm_params={},
            )

    # -----------------------------------------------------------------------
    # transform_anthropic_messages_request — payload stays in Anthropic format
    # -----------------------------------------------------------------------

    def test_request_payload_is_anthropic_format(self):
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "max_tokens": 100,
            "system": "You are helpful.",
            "temperature": 0.7,
        }
        payload = self.config.transform_anthropic_messages_request(
            model="qwen36-27b",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # Key Anthropic fields must be present
        assert payload["messages"] == messages
        assert payload["max_tokens"] == 100
        assert payload["model"] == "qwen36-27b"
        assert payload["system"] == "You are helpful."
        assert payload["temperature"] == 0.7

        # OpenAI-only fields must NOT appear
        assert "n" not in payload
        assert "frequency_penalty" not in payload
        assert "presence_penalty" not in payload

    def test_request_payload_missing_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens is required"):
            self.config.transform_anthropic_messages_request(
                model="qwen",
                messages=[],
                anthropic_messages_optional_request_params={},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    # -----------------------------------------------------------------------
    # validate_anthropic_messages_environment
    # -----------------------------------------------------------------------

    def test_environment_sets_auth_and_content_type(self):
        headers, returned_base = self.config.validate_anthropic_messages_environment(
            headers={},
            model="qwen",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="my-key",
            api_base="http://vllm-host/v1",
        )
        assert headers["authorization"] == "Bearer my-key"
        assert headers["content-type"] == "application/json"
        assert returned_base == "http://vllm-host/v1"

    def test_environment_does_not_overwrite_existing_auth(self):
        headers, _ = self.config.validate_anthropic_messages_environment(
            headers={"authorization": "Bearer already-set"},
            model="qwen",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="different-key",
            api_base=None,
        )
        assert headers["authorization"] == "Bearer already-set"


# ---------------------------------------------------------------------------
# Integration: anthropic_messages_handler routes to HostedVLLMAnthropicMessagesConfig
# ---------------------------------------------------------------------------


class TestHandlerRouting:
    """
    Verify that anthropic_messages_handler uses HostedVLLMAnthropicMessagesConfig
    when the flag is set, and falls back to the completions transformer otherwise.
    """

    @pytest.mark.asyncio
    async def test_flag_enabled_uses_passthrough_config(self):
        """When disable_anthropic_translation=True, ProviderConfigManager returns
        HostedVLLMAnthropicMessagesConfig and the native path is used instead of
        LiteLLMMessagesToCompletionTransformationHandler."""
        from litellm.llms.anthropic.experimental_pass_through.messages import handler as h
        from litellm.utils import ProviderConfigManager

        passthrough_config = HostedVLLMAnthropicMessagesConfig()
        mock_response = MagicMock()
        mock_response.id = "msg_test"

        with (
            patch.object(h.base_llm_http_handler, "anthropic_messages_handler", return_value=mock_response) as mock_native,
            patch.object(
                h.LiteLLMMessagesToCompletionTransformationHandler,
                "anthropic_messages_handler",
            ) as mock_translate,
            patch("litellm.get_llm_provider", return_value=("qwen36-27b-fp8", "hosted_vllm", "key", "http://vllm/v1")),
            patch.object(ProviderConfigManager, "get_provider_anthropic_messages_config", return_value=passthrough_config),
        ):
            h.anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
                model="hosted_vllm/qwen36-27b-fp8",
                api_key="key",
                api_base="http://vllm/v1",
                disable_anthropic_translation=True,
            )

        mock_native.assert_called_once()
        mock_translate.assert_not_called()

        call_kwargs = mock_native.call_args.kwargs
        assert isinstance(
            call_kwargs.get("anthropic_messages_provider_config"),
            HostedVLLMAnthropicMessagesConfig,
        )

    @pytest.mark.asyncio
    async def test_flag_disabled_uses_translation(self):
        """When disable_anthropic_translation is absent (default), the handler must
        route to LiteLLMMessagesToCompletionTransformationHandler."""
        from litellm.llms.anthropic.experimental_pass_through.messages import handler as h
        from litellm.utils import ProviderConfigManager

        mock_response = MagicMock()

        with (
            patch.object(
                h.LiteLLMMessagesToCompletionTransformationHandler,
                "anthropic_messages_handler",
                return_value=mock_response,
            ) as mock_translate,
            patch("litellm.get_llm_provider", return_value=("qwen36-27b-fp8", "hosted_vllm", "key", "http://vllm/v1")),
            patch.object(ProviderConfigManager, "get_provider_anthropic_messages_config", return_value=None),
        ):
            h.anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
                model="hosted_vllm/qwen36-27b-fp8",
                api_key="key",
                api_base="http://vllm/v1",
            )

        mock_translate.assert_called_once()
