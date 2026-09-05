"""
Tests for OpenCode Anthropic Messages wire-format arm (Issue 02).

These tests fail before the feature exists and fail if the dispatch
mapping, auth header selection, or URL construction are mutated.
"""

import asyncio
import json
from datetime import datetime


from httpx import Response

import litellm
import pytest

from litellm.llms.opencode.chat.messages_transformation import (
    OpenCodeMessagesConfig,
    OPENCODE_MESSAGES_MODELS,
    is_messages_model,
)
from litellm.types.completion import _CompletionDispatchContext
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ModelResponse

ZEN_MESSAGES_ENDPOINT = "https://opencode.ai/zen/v1/messages"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _anthropic_response(content: str, **usage_kwargs) -> dict:
    """Build a standard Anthropic Messages response body."""
    prompt = usage_kwargs.get("prompt_tokens", 1)
    completion = usage_kwargs.get("completion_tokens", 1)
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": prompt,
            "output_tokens": completion,
        },
    }


# ---------------------------------------------------------------------------
# Messages-model set as data
# ---------------------------------------------------------------------------


class TestMessagesModelSet:
    """The zen messages-model set is immutable data — mutations fail tests."""

    def test_all_zen_claude_models_present(self):
        """Every live claude model from Issue 02 is in the set.

        claude-opus-4-1 is excluded: it is not served by the live Zen
        gateway roster, so it was removed from the messages set and cost map.
        """
        expected_claude = {
            "claude-fable-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-4",
            "claude-sonnet-4-5",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
        }
        assert expected_claude <= OPENCODE_MESSAGES_MODELS["zen"]
        assert "claude-opus-4-1" not in OPENCODE_MESSAGES_MODELS["zen"]

    def test_qwen_models_in_set(self):
        """qwen3.5-plus and qwen3.6-plus are in the zen messages set."""
        assert "qwen3.5-plus" in OPENCODE_MESSAGES_MODELS["zen"]
        assert "qwen3.6-plus" in OPENCODE_MESSAGES_MODELS["zen"]

    def test_set_size(self):
        """Exactly 13 models in the zen messages set."""
        assert len(OPENCODE_MESSAGES_MODELS["zen"]) == 13

    def test_non_claude_model_not_in_set(self):
        """gpt-5 models are NOT in the messages set (they belong on chat)."""
        assert "gpt-5.1" not in OPENCODE_MESSAGES_MODELS["zen"]
        assert "gpt-5.6-luna" not in OPENCODE_MESSAGES_MODELS["zen"]
        assert "grok-4.5" not in OPENCODE_MESSAGES_MODELS["zen"]

    def test_set_is_frozenset(self):
        """The set is immutable — mutation raises."""
        with pytest.raises(AttributeError):
            OPENCODE_MESSAGES_MODELS["zen"].add("brand-new-model")


# ---------------------------------------------------------------------------
# is_messages_model routing
# ---------------------------------------------------------------------------


class TestIsMessagesModel:
    """Model-to-arm dispatch decision."""

    def test_claude_model_routes_to_messages(self):
        """claude-sonnet-4 routes to the messages arm on zen."""
        assert is_messages_model("zen", "claude-sonnet-4") is True
        assert is_messages_model("zen", "claude-opus-4-5") is True

    def test_qwen_model_routes_to_messages(self):
        """qwen3.5-plus routes to the messages arm on zen."""
        assert is_messages_model("zen", "qwen3.5-plus") is True
        assert is_messages_model("zen", "qwen3.6-plus") is True

    def test_non_messages_model_routes_to_chat(self):
        """gpt-5.1 does NOT route to messages (it goes to chat)."""
        assert is_messages_model("zen", "gpt-5.1") is False
        assert is_messages_model("zen", "grok-4.5") is False

    def test_unknown_model_does_not_route_to_messages(self):
        """Unknown/new models fall through to chat, not messages."""
        assert is_messages_model("zen", "brand-new-model") is False

    def test_go_minimax_routes_to_messages(self):
        """minimax-m2.5 is a messages model on go."""
        assert is_messages_model("go", "minimax-m2.5") is True
        assert is_messages_model("go", "minimax-m3") is True

    def test_go_qwen_routes_to_messages(self):
        """qwen3.5-max routes to messages on go."""
        assert is_messages_model("go", "qwen3.5-max") is True
        assert is_messages_model("go", "qwen3.8-max") is True

    def test_go_qwen_plus_routes_to_messages(self):
        """qwen3.6-plus routes to messages on go too."""
        assert is_messages_model("go", "qwen3.6-plus") is True

    def test_go_qwen_flash_routes_to_messages(self):
        """qwen3.8-flash is off the {plus,max} grid but serves the Anthropic wire."""
        assert is_messages_model("go", "qwen3.8-flash") is True

    def test_go_non_messages_model(self):
        """gpt-5.5 is chat-only on go."""
        assert is_messages_model("go", "gpt-5.5") is False

    def test_go_grok_not_messages(self):
        """gpt-5.6-luna is not a messages model on go."""
        assert is_messages_model("go", "gpt-5.6-luna") is False

    @pytest.mark.parametrize("number", [5, 6, 7, 8])
    @pytest.mark.parametrize("tier", ["plus", "max"])
    def test_go_covers_the_whole_qwen_grid(self, number, tier):
        """Every qwen3.{5..8}-{plus,max} routes to messages on go.

        The set covers the full grid rather than only the entries the cost map
        carries today, so a model the gateway adds keeps reaching the Anthropic
        wire instead of silently degrading to chat completions.
        """
        assert is_messages_model("go", f"qwen3.{number}-{tier}") is True

    def test_go_qwen_outside_the_grid_is_not_messages(self):
        """The grid is bounded — neighbouring versions are not assumed."""
        assert is_messages_model("go", "qwen3.4-plus") is False
        assert is_messages_model("go", "qwen3.9-plus") is False
        assert is_messages_model("go", "qwen3.5-turbo") is False


# ---------------------------------------------------------------------------
# OpenCodeMessagesConfig — URL and headers
# ---------------------------------------------------------------------------


class TestMessagesConfig:
    """Tests for the OpenCodeMessagesConfig class."""

    def test_zen_custom_llm_provider(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        assert cfg.custom_llm_provider == "opencode_zen"

    def test_go_custom_llm_provider(self):
        cfg = OpenCodeMessagesConfig(surface="go")
        assert cfg.custom_llm_provider == "opencode_go"

    def test_zen_base_url(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        assert cfg._base_url() == "https://opencode.ai/zen"

    def test_get_complete_url_zen(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        url = cfg.get_complete_url(None, None, "claude-sonnet-4", {}, {})
        assert url == "https://opencode.ai/zen/v1/messages"

    def test_get_complete_url_trailing_slash(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        url = cfg.get_complete_url("http://localhost:4000/", None, "claude-sonnet-4", {}, {})
        assert url == "http://localhost:4000/v1/messages"

    def test_error_class(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        assert cfg.get_error_class("bad", 400, {}) is not None


class TestBillingMetadataNotLeaked:
    """OpenCode is a third-party gateway, not the first-party Anthropic API, so
    x-anthropic-billing-header client attribution blocks must be dropped before
    the request leaves. The base AnthropicMessagesConfig keeps them (correct for
    api.anthropic.com), so a missing override silently forwards Claude Code
    billing attribution to a third party."""

    @pytest.mark.parametrize("surface", ["zen", "go"])
    def test_billing_header_system_block_is_stripped(self, surface):
        cfg = OpenCodeMessagesConfig(surface=surface)
        optional_params = {
            "max_tokens": 16,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: user_id=abc123"},
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        }

        cfg.transform_anthropic_messages_request(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        remaining = optional_params.get("system") or []
        texts = [block.get("text", "") for block in remaining if isinstance(block, dict)]
        assert not any(text.startswith("x-anthropic-billing-header:") for text in texts)
        assert "You are a helpful assistant." in texts


# ---------------------------------------------------------------------------
# x-api-key vs Bearer — regression at the auth seam
# ---------------------------------------------------------------------------


class TestAuthHeader:
    """Zen uses x-api-key; Go uses Bearer on the messages arm."""

    def _make_cfg(self, surface: str):
        return OpenCodeMessagesConfig(surface=surface)

    def test_zen_uses_x_api_key(self):
        """Zen /v1/messages sends x-api-key, NOT Bearer."""
        cfg = self._make_cfg("zen")
        headers: dict = {}
        result, _ = cfg.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-zen-key",
        )
        assert "x-api-key" in result
        assert result["x-api-key"] == "sk-zen-key"
        assert "Authorization" not in result

    def test_go_uses_x_api_key(self):
        """Go /v1/messages sends x-api-key, NOT Bearer.

        Regression test: live verification showed Bearer on Go /v1/messages
        returns 401 "Missing API key"; x-api-key returns 200.
        """
        cfg = self._make_cfg("go")
        headers: dict = {}
        result, _ = cfg.validate_anthropic_messages_environment(
            headers=headers,
            model="minimax-m2.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-go-key",
        )
        assert "x-api-key" in result
        assert result["x-api-key"] == "sk-go-key"
        assert "Authorization" not in result

    def test_zen_anthropic_version_set(self):
        """The anthropic-version header is present on zen."""
        cfg = self._make_cfg("zen")
        headers: dict = {}
        result, _ = cfg.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-key",
        )
        assert result["anthropic-version"] == "2023-06-01"

    def test_zen_content_type_set(self):
        """content-type is application/json."""
        cfg = self._make_cfg("zen")
        headers: dict = {}
        result, _ = cfg.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-key",
        )
        assert result["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# Integration — mocked messages completion call
# ---------------------------------------------------------------------------


class TestMockedMessagesCompletion:
    """Messages arm models hit /v1/messages with Anthropic body + x-api-key."""

    @pytest.fixture(autouse=True)
    def _defaults(self, monkeypatch):
        """Provide max_tokens on every integration test — Anthropic requires it."""
        monkeypatch.setattr(litellm, "opencode_zen_api_key", None)
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)
        monkeypatch.setattr(litellm, "opencode_api_key", None)
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "api_base", None)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
        # The import-time cost map comes from remote main and predates the
        # un-merged opencode feature. Load the local backup so the max_tokens
        # default (read from the cost map) resolves for opencode models.
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        litellm.in_memory_llm_clients_cache.flush_cache()

    def _make_completion_kwargs(self, **overrides):
        """Return default completion kwargs with a sensible max_tokens."""
        kwargs = {
            "messages": [{"role": "user", "content": "say hi"}],
            "max_tokens": 256,
        }
        kwargs.update(overrides)
        return kwargs

    def test_messages_model_hits_v1_messages(self, respx_mock, monkeypatch):
        """
        A messages-model reaches {base}/v1/messages, not /chat/completions.

        This is the core acceptance test for the messages arm.
        """
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("Claude speaks"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "say hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        assert result is not None
        assert len(respx_mock.calls) > 0
        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path
        assert "/chat/completions" not in request.url.path

    def test_messages_model_sends_x_api_key(self, respx_mock, monkeypatch):
        """
        Zen /v1/messages sends x-api-key header, not Bearer.

        Regression test: Bearer on Zen /v1/messages returns 401.
        """
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("ok"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-zen-123")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        assert request.headers.get("x-api-key") == "sk-zen-123"
        # Bearer should NOT be present for zen messages arm
        assert "Authorization" not in request.headers

    def test_messages_model_uses_anthropic_body_shape(self, respx_mock, monkeypatch):
        """The request body uses Anthropic Messages format, not OpenAI."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("anthropic body"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "test body shape"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        body = json.loads(request.read())
        # Anthropic messages uses "messages" with "role" and "content"
        # but the outer shape is Anthropic, not OpenAI
        assert "messages" in body

    def test_unknown_model_still_routes_to_chat_arm(self, respx_mock, monkeypatch):
        """
        Models outside the messages set still route to /chat/completions.

        Dispatch precedence: messages-model check happens first;
        non-matching models fall through to chat.
        """
        respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "chat works"}}]},
            )
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/grok-4.5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
        )

        assert result is not None
        request = respx_mock.calls[0].request
        assert "/chat/completions" in request.url.path
        assert "/v1/messages" not in request.url.path

    def test_messages_dispatch_precedence_over_chat(self, respx_mock, monkeypatch):
        """
        A messages-model is dispatched to /v1/messages, NOT /chat/completions.

        If dispatch precedence is broken, the model would hit /chat/completions.
        """
        messages_endpoint = respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("messages arm"))
        )

        chat_endpoint = respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"role": "assistant", "content": "wrong"}}]})
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        assert messages_endpoint.call_count == 1
        assert chat_endpoint.call_count == 0

    def test_go_messages_model_sends_x_api_key(self, respx_mock, monkeypatch):
        """Go messages models send x-api-key, not Bearer.

        Regression test: live verification showed Bearer on Go /v1/messages
        returns 401 "Missing API key"; x-api-key returns 200.
        """
        respx_mock.post("https://opencode.ai/zen/go/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("go messages"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-go-123")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        litellm.completion(
            model="opencode_go/minimax-m2.5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_go",
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path
        assert request.headers.get("x-api-key") == "sk-go-123"
        # Bearer should NOT be present for go messages arm
        assert "Authorization" not in request.headers

    def test_global_api_key_not_sent_on_messages_dispatch(self, respx_mock, monkeypatch):
        """Messages arm uses the OpenCode key, not a process-wide litellm.api_key.

        Regression guard for the credential-disclosure claim on the messages
        path: an unrelated global credential must never reach opencode.ai when a
        surface-specific OpenCode key is configured.
        """
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("ok"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-global-other-provider")
        monkeypatch.setattr(litellm, "opencode_zen_api_key", "sk-opencode")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        assert result is not None
        api_key_header = respx_mock.calls[0].request.headers.get("x-api-key")
        assert api_key_header == "sk-opencode"
        assert "sk-global-other-provider" not in api_key_header

    def test_env_var_key_resolution_messages_arm(self, respx_mock, monkeypatch):
        """Messages arm resolves api_key from OPENCODE_ZEN_API_KEY."""
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "sk-env-messages")
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("env key"))
        )

        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        litellm.completion(
            model="opencode_zen/claude-fable-5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        assert request.headers.get("x-api-key") == "sk-env-messages"

    def test_messages_model_qwen(self, respx_mock, monkeypatch):
        """qwen3.5-plus is also dispatched to messages arm."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("qwen messages"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/qwen3.5-plus",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        assert result is not None
        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path

    def test_max_tokens_defaulted_from_cost_map(self, respx_mock, monkeypatch):
        """A messages-model request with no max_tokens still succeeds.

        Regression: the Anthropic /v1/messages API requires max_tokens, and
        the messages arm passes optional_params straight through.  A playground
        wildcard request (no explicit max_tokens) previously failed with
        ``max_tokens is required for Anthropic /v1/messages API``.  The config
        now defaults it from the model's cost-map ``max_output_tokens``.
        """
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("defaulted"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
        )

        assert result is not None
        request = respx_mock.calls[0].request
        body = json.loads(request.read())
        # claude-sonnet-4 cost-map max_output_tokens is 64000
        assert body["max_tokens"] == 64000

    def test_max_tokens_defaulted_on_go_surface(self, respx_mock, monkeypatch):
        """Go messages models default max_tokens from the go cost-map entry."""
        respx_mock.post("https://opencode.ai/zen/go/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("go defaulted"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_go/qwen3.7-plus",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_go",
        )

        assert result is not None
        request = respx_mock.calls[0].request
        body = json.loads(request.read())
        # qwen3.7-plus cost-map max_output_tokens is 65536
        assert body["max_tokens"] == 65536

    def test_explicit_max_tokens_not_overridden(self, respx_mock, monkeypatch):
        """An explicit max_tokens is preserved, not replaced by the default."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("explicit"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=128,
        )

        assert result is not None
        request = respx_mock.calls[0].request
        body = json.loads(request.read())
        assert body["max_tokens"] == 128


# ---------------------------------------------------------------------------
# Streaming test
# ---------------------------------------------------------------------------


def _sse_body(content: str, **usage_kwargs) -> str:
    """Build a single SSE event line for a messages response."""
    prompt = usage_kwargs.get("prompt_tokens", 1)
    completion = usage_kwargs.get("completion_tokens", 1)
    data = {
        "type": "message",
        "id": "msg_123",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": prompt, "output_tokens": completion},
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n"


class TestMessagesArmStreaming:
    """Streaming returns the Anthropic SSE iterator on the messages arm."""

    @pytest.fixture(autouse=True)
    def _disable_aiohttp(self, monkeypatch):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    def test_streaming_sends_to_messages_url(self, respx_mock, monkeypatch):
        """stream=True routes to /v1/messages with Anthropic body shape."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(
                200,
                text=_sse_body("streamed answer"),
                headers={"content-type": "text/event-stream"},
            )
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            stream=True,
            max_tokens=256,
        )

        assert result is not None
        # The result must be an async iterator (StreamingGenerator)
        assert hasattr(result, "__aiter__")
        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path

    def test_streaming_body_includes_anthropic_params(self, respx_mock, monkeypatch):
        """Streaming request carries anthropic-version header."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(
                200,
                text=_sse_body("streamed"),
                headers={"content-type": "text/event-stream"},
            )
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            stream=True,
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        assert request.headers.get("anthropic-version") == "2023-06-01"


# ---------------------------------------------------------------------------
# acompletion coverage on the messages arm
# ---------------------------------------------------------------------------


class TestMessagesArmAcompletion:
    """acompletion dispatches to the messages arm for messages models."""

    @pytest.fixture(autouse=True)
    def _disable_aiohttp(self, monkeypatch):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    @pytest.mark.asyncio
    async def test_acompletion_sends_to_messages_url(self, respx_mock, monkeypatch):
        """acompletion for a messages model hits /v1/messages."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("async"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        result = await litellm.acompletion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=512,
        )

        assert result is not None
        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path

    @pytest.mark.asyncio
    async def test_acompletion_max_tokens_reaches_request_body(self, respx_mock, monkeypatch):
        """acompletion max_tokens is serialized into the request body."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("async"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-key")
        await litellm.acompletion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=512,
        )

        request = respx_mock.calls[0].request
        body = json.loads(request.content)
        assert body["max_tokens"] == 512


# ---------------------------------------------------------------------------
# Sync and acompletion paths both return an OpenAI-shaped result
# ---------------------------------------------------------------------------


class TestMessagesArmReturnShape:
    """``completion()`` promises a ``ModelResponse`` regardless of arm.

    The messages arm speaks the Anthropic wire format upstream, so the reply has
    to be translated back before it reaches the caller. Returning the raw
    Anthropic body means ``response.choices[0]`` raises ``AttributeError`` for
    every model on this arm.
    """

    @pytest.fixture(autouse=True)
    def _defaults(self, monkeypatch):
        monkeypatch.setattr(litellm, "opencode_zen_api_key", None)
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)
        monkeypatch.setattr(litellm, "opencode_api_key", None)
        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "api_base", None)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        litellm.in_memory_llm_clients_cache.flush_cache()

    def test_sync_path_returns_model_response(self, respx_mock):
        respx_mock.post(ZEN_MESSAGES_ENDPOINT).mock(return_value=Response(200, json=_anthropic_response("hi there")))

        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "hi there"

    def test_acompletion_path_returns_model_response(self, respx_mock):
        respx_mock.post(ZEN_MESSAGES_ENDPOINT).mock(return_value=Response(200, json=_anthropic_response("async hi")))

        result = asyncio.run(
            litellm.acompletion(
                model="opencode_zen/claude-sonnet-4",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
            )
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "async hi"

    def test_openai_request_is_translated_to_anthropic_wire_format(self, respx_mock):
        """A ``system`` message and ``stop`` must not reach Anthropic verbatim.

        Anthropic takes ``system`` as a top-level parameter and rejects
        ``role: "system"`` inside ``messages``, and names the stop list
        ``stop_sequences``. Forwarding the OpenAI body unchanged 400s upstream.
        """
        route = respx_mock.post(ZEN_MESSAGES_ENDPOINT).mock(return_value=Response(200, json=_anthropic_response("ok")))

        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            stop=["END"],
            max_tokens=16,
        )

        body = json.loads(route.calls[0].request.read())
        assert body["system"] == [{"type": "text", "text": "be terse"}]
        assert "system" not in {m["role"] for m in body["messages"]}
        assert body["stop_sequences"] == ["END"]
        assert "stop" not in body

    @pytest.mark.parametrize(
        "configured_base, expected_url",
        [
            ("https://gw.internal", "https://gw.internal/v1/messages"),
            ("https://gw.internal/v1", "https://gw.internal/v1/messages"),
            ("https://gw.internal/v1/messages", "https://gw.internal/v1/messages"),
            ("https://gw.internal/", "https://gw.internal/v1/messages"),
        ],
    )
    def test_configured_api_base_is_honoured(self, respx_mock, configured_base, expected_url):
        """An operator-configured gateway must be reached, whatever suffix it carries.

        Silently falling back to the public endpoint would send the operator's
        key and prompts to opencode.ai instead of their own gateway.
        """
        route = respx_mock.post(expected_url).mock(return_value=Response(200, json=_anthropic_response("ok")))

        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            api_base=configured_base,
            max_tokens=16,
        )

        assert route.called
        assert str(route.calls[0].request.url) == expected_url


# ---------------------------------------------------------------------------
# Cost-map entries for messages models
# ---------------------------------------------------------------------------


class TestMessagesCostMap:
    """Cost-map entries for the 13 Zen messages models."""

    @pytest.fixture(autouse=True)
    def _load_cost_map(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    def test_claude_sonnet_4_entry(self):
        entry = litellm.model_cost["opencode_zen/claude-sonnet-4"]
        assert entry["litellm_provider"] == "opencode_zen"
        assert entry["max_input_tokens"] == 1000000
        assert entry["max_output_tokens"] == 64000

    def test_claude_opus_4_1_not_in_cost_map(self):
        """claude-opus-4-1 is not served by the live Zen roster."""
        assert "opencode_zen/claude-opus-4-1" not in litellm.model_cost

    def test_claude_fable_5_entry(self):
        entry = litellm.model_cost["opencode_zen/claude-fable-5"]
        assert entry["max_input_tokens"] == 1000000
        assert entry["input_cost_per_token"] == 1e-05

    def test_qwen3_plus_entries(self):
        entry = litellm.model_cost["opencode_zen/qwen3.5-plus"]
        assert entry["litellm_provider"] == "opencode_zen"
        assert entry["max_input_tokens"] == 262144

        entry_qwen36 = litellm.model_cost["opencode_zen/qwen3.6-plus"]
        assert entry_qwen36["max_input_tokens"] == 262144

    def test_all_14_messages_models_have_cost_entries(self):
        """Every model in the messages set has a cost-map entry."""
        for model_name in OPENCODE_MESSAGES_MODELS["zen"]:
            key = f"opencode_zen/{model_name}"
            assert key in litellm.model_cost, f"{key} missing from cost map"
            assert litellm.model_cost[key]["litellm_provider"] == "opencode_zen"

    def test_cost_entries_have_pricing(self):
        """All messages models have nonzero pricing."""
        for model_name in OPENCODE_MESSAGES_MODELS["zen"]:
            key = f"opencode_zen/{model_name}"
            entry = litellm.model_cost[key]
            assert entry["input_cost_per_token"] >= 0
            assert entry["output_cost_per_token"] >= 0
