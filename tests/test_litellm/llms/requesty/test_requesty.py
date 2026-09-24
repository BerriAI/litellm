"""Tests for the Requesty provider (OpenAI-compatible gateway)."""

import pytest

import litellm
from litellm import get_llm_provider
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.requesty.chat.transformation import (
    RequestyChatCompletionStreamingHandler,
    RequestyConfig,
)
from litellm.llms.requesty.common_utils import RequestyException


def test_get_llm_provider_resolves_requesty():
    """requesty/<provider>/<model> routes to the requesty provider + base URL."""
    model, custom_llm_provider, _dynamic_api_key, api_base = get_llm_provider(model="requesty/openai/gpt-4o-mini")
    assert custom_llm_provider == "requesty"
    assert model == "openai/gpt-4o-mini"
    assert api_base == "https://router.requesty.ai/v1"


def test_requesty_in_provider_registries():
    """requesty is registered in the provider list and openai-compatible sets."""
    assert "requesty" in litellm.provider_list
    assert "requesty" in litellm.openai_compatible_providers
    assert "https://router.requesty.ai/v1" in litellm.openai_compatible_endpoints


def test_requesty_config_default_base_url():
    """RequestyConfig exposes the fixed Requesty router base URL."""
    api_base, _dynamic_api_key = RequestyConfig().get_openai_compatible_provider_info(api_base=None, api_key="test-key")
    assert api_base == "https://router.requesty.ai/v1"


def test_transform_request_extra_body_cannot_override_protected_fields():
    """Client-controlled extra_body must not clobber canonical model/messages.

    extra_body is caller-supplied and applied after model authorization/request
    inspection. Allowing it to overwrite `model` or `messages` would let a caller
    route to an unauthorized model, so those fields must be preserved.
    """
    messages = [{"role": "user", "content": "hello"}]
    optional_params = {
        "extra_body": {
            "model": "openai/unauthorized-model",
            "messages": [{"role": "user", "content": "evil"}],
            "custom_flag": True,
        }
    }

    result = RequestyConfig().transform_request(
        model="openai/gpt-4o-mini",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    # Canonical fields resolved by the transform are preserved.
    assert result["model"] == "openai/gpt-4o-mini"
    assert result["messages"] == messages
    # Non-protected extension params still pass through.
    assert result["custom_flag"] is True


def test_reasoning_model_registered_under_requesty_gets_reasoning_params(monkeypatch):
    """A model known only under litellm_provider="requesty" must still unlock reasoning params.

    The parent OpenRouter config checks supports_reasoning with provider "openrouter",
    which never matches models registered under "requesty".
    """
    monkeypatch.setitem(
        litellm.model_cost,
        "requesty/acme/thinker-1",
        {"litellm_provider": "requesty", "mode": "chat", "supports_reasoning": True},
    )

    supported_params = RequestyConfig().get_supported_openai_params(model="acme/thinker-1")

    assert "reasoning_effort" in supported_params
    assert "thinking" in supported_params
    assert supported_params.count("reasoning_effort") == 1
    assert "temperature" in supported_params


def test_non_reasoning_model_does_not_get_reasoning_params(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "requesty/acme/plain-1",
        {"litellm_provider": "requesty", "mode": "chat", "supports_reasoning": False},
    )

    supported_params = RequestyConfig().get_supported_openai_params(model="acme/plain-1")

    assert "reasoning_effort" not in supported_params
    assert "thinking" not in supported_params


def test_get_model_response_iterator_returns_requesty_handler():
    handler = RequestyConfig().get_model_response_iterator(streaming_response=iter(()), sync_stream=True)

    assert isinstance(handler, RequestyChatCompletionStreamingHandler)


class TestRequestyChatCompletionStreamingHandler:
    def test_chunk_parser_successful(self):
        handler = RequestyChatCompletionStreamingHandler(streaming_response=None, sync_stream=True)
        chunk = {
            "id": "chunk-1",
            "created": 1234567890,
            "model": "openai/gpt-4o-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [{"index": 0, "delta": {"content": "hello", "reasoning": "thinking"}}],
        }

        result = handler.chunk_parser(chunk)

        assert result.id == "chunk-1"
        assert result.object == "chat.completion.chunk"
        assert result.created == 1234567890
        assert result.model == "openai/gpt-4o-mini"
        assert result.usage.total_tokens == 30
        assert len(result.choices) == 1
        assert result.choices[0]["delta"]["content"] == "hello"
        assert result.choices[0]["delta"]["reasoning_content"] == "thinking"

    def test_chunk_parser_error_response_raises_requesty_exception(self):
        handler = RequestyChatCompletionStreamingHandler(streaming_response=None, sync_stream=True)
        error_chunk = {"error": {"message": "rate limited", "code": 429, "metadata": {"headers": {"Retry-After": "1"}}}}

        with pytest.raises(RequestyException) as exc_info:
            handler.chunk_parser(error_chunk)

        assert not isinstance(exc_info.value, OpenRouterException)
        assert "rate limited" in str(exc_info.value)
        assert exc_info.value.status_code == 429

    def test_chunk_parser_malformed_chunk_raises_requesty_exception(self):
        handler = RequestyChatCompletionStreamingHandler(streaming_response=None, sync_stream=True)

        with pytest.raises(RequestyException) as exc_info:
            handler.chunk_parser({"incomplete": "data"})

        assert not isinstance(exc_info.value, OpenRouterException)
        assert "KeyError" in str(exc_info.value)
        assert exc_info.value.status_code == 400


def test_requesty_config_reports_custom_llm_provider():
    assert RequestyConfig().custom_llm_provider == "requesty"


def test_requesty_config_reads_base_url_and_key_from_env(monkeypatch):
    """REQUESTY_API_BASE and REQUESTY_API_KEY are honoured when no explicit values are passed."""
    monkeypatch.setenv("REQUESTY_API_BASE", "https://router.eu.requesty.ai/v1")
    monkeypatch.setenv("REQUESTY_API_KEY", "env-test-key")

    _model, _custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(model="requesty/openai/gpt-4o-mini")

    assert api_base == "https://router.eu.requesty.ai/v1"
    assert dynamic_api_key == "env-test-key"


def test_supports_reasoning_swallows_lookup_errors(monkeypatch):
    """An unknown model must not raise; it simply gets no reasoning params."""

    def _boom(**_kwargs):
        raise ValueError("unknown model")

    monkeypatch.setattr(litellm, "supports_reasoning", _boom)

    supported_params = RequestyConfig().get_supported_openai_params("acme/unknown-1")

    assert "reasoning_effort" not in supported_params
    assert "thinking" not in supported_params


def test_map_openai_params_translates_max_reasoning_effort_to_xhigh():
    """Requesty accepts reasoning_effort=xhigh where OpenAI style clients send max."""
    optional_params = RequestyConfig().map_openai_params(
        non_default_params={"reasoning_effort": "max"},
        optional_params={},
        model="openai/gpt-5",
        drop_params=False,
    )

    assert optional_params["reasoning_effort"] == "xhigh"


def test_map_openai_params_keeps_other_reasoning_effort_values():
    optional_params = RequestyConfig().map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="openai/gpt-5",
        drop_params=False,
    )

    assert optional_params["reasoning_effort"] == "high"


def test_get_error_class_returns_requesty_exception():
    error = RequestyConfig().get_error_class(
        error_message="rate limited",
        status_code=429,
        headers={"x-request-id": "abc"},
    )

    assert isinstance(error, RequestyException)
    assert error.status_code == 429
    assert error.message == "rate limited"
