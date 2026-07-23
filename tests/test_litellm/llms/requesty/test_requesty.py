"""Tests for the Requesty provider (OpenAI-compatible gateway)."""

import litellm
from litellm import get_llm_provider
from litellm.llms.requesty.chat.transformation import RequestyConfig


def test_get_llm_provider_resolves_requesty():
    """requesty/<provider>/<model> routes to the requesty provider + base URL."""
    model, custom_llm_provider, _dynamic_api_key, api_base = get_llm_provider(
        model="requesty/openai/gpt-4o-mini"
    )
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
    api_base, _dynamic_api_key = RequestyConfig()._get_openai_compatible_provider_info(
        api_base=None, api_key="test-key"
    )
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
