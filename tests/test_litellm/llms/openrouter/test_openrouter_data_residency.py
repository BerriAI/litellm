"""Tests for OpenRouter in-region routing, data residency inference, and error handling."""

import httpx
import pytest

import litellm
from litellm.exceptions import NotFoundError, OpenRouterRegionNotSupportedError
from litellm.litellm_core_utils.exception_mapping_utils import _map_openrouter_exception
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.llms.openrouter.chat.transformation import OpenrouterConfig
from litellm.llms.openrouter.data_residency import (
    infer_openrouter_data_residency,
    resolve_openrouter_api_base,
)


@pytest.mark.parametrize(
    "api_base, region, expected",
    [
        (None, "eu", "https://eu.openrouter.ai/api/v1"),
        (None, "EU", "https://eu.openrouter.ai/api/v1"),
        (None, "europe", "https://eu.openrouter.ai/api/v1"),
        (None, "eu-west-1", "https://eu.openrouter.ai/api/v1"),
        (None, "eu-central-1", "https://eu.openrouter.ai/api/v1"),
        (None, "europe-west3", "https://eu.openrouter.ai/api/v1"),
        (None, "us", "https://us.openrouter.ai/api/v1"),
        (None, "US", "https://us.openrouter.ai/api/v1"),
        (None, "usa", "https://us.openrouter.ai/api/v1"),
        (None, "united states", "https://us.openrouter.ai/api/v1"),
        (None, "us-east-1", "https://us.openrouter.ai/api/v1"),
        (None, "us-west-2", "https://us.openrouter.ai/api/v1"),
        (None, "global", "https://openrouter.ai/api/v1"),
        (None, "default", "https://openrouter.ai/api/v1"),
        (None, None, "https://openrouter.ai/api/v1"),
        ("https://openrouter.ai/api/v1", "eu", "https://eu.openrouter.ai/api/v1"),
        ("https://openrouter.ai/api/v1", "us", "https://us.openrouter.ai/api/v1"),
        ("https://custom.openrouter.proxy/v1", "eu", "https://custom.openrouter.proxy/v1"),
        ("https://custom.openrouter.proxy/v1", "apac", "https://custom.openrouter.proxy/v1"),
        ("https://custom.openrouter.proxy/v1", None, "https://custom.openrouter.proxy/v1"),
    ],
)
def test_resolve_openrouter_api_base(api_base, region, expected):
    assert resolve_openrouter_api_base(api_base=api_base, region=region) == expected


def test_resolve_openrouter_api_base_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REGION", "eu")
    assert resolve_openrouter_api_base() == "https://eu.openrouter.ai/api/v1"


def test_resolve_openrouter_api_base_dynamic_region():
    assert resolve_openrouter_api_base(region="apac") == "https://apac.openrouter.ai/api/v1"


def test_resolve_openrouter_api_base_from_dict_params():
    assert resolve_openrouter_api_base(litellm_params={"region": "eu"}) == "https://eu.openrouter.ai/api/v1"
    assert resolve_openrouter_api_base(optional_params={"region": "us"}) == "https://us.openrouter.ai/api/v1"


@pytest.mark.parametrize(
    "api_base, expected",
    [
        ("https://eu.openrouter.ai/api/v1", "eu"),
        ("https://eu.openrouter.ai", "eu"),
        ("https://us.openrouter.ai/api/v1", "us"),
        ("https://us.openrouter.ai", "us"),
        ("https://EU.openrouter.ai/api/v1", "eu"),
        ("https://openrouter.ai/api/v1", None),
        ("https://openrouter.ai", None),
        ("https://example.com/v1", None),
        ("", None),
        (None, None),
        ("invalid-url", None),
    ],
)
def test_infer_openrouter_data_residency(api_base, expected):
    assert infer_openrouter_data_residency("openrouter", api_base) == expected


@pytest.mark.parametrize("custom_llm_provider", [None, "openai", "anthropic", "azure", "bedrock"])
def test_infer_openrouter_data_residency_non_openrouter(custom_llm_provider):
    assert (
        infer_openrouter_data_residency(custom_llm_provider, "https://eu.openrouter.ai/api/v1")
        is None
    )


def test_get_litellm_params_infers_openrouter_data_residency():
    params_eu = get_litellm_params(
        custom_llm_provider="openrouter",
        api_base="https://eu.openrouter.ai/api/v1",
    )
    assert params_eu["data_residency"] == "eu"

    params_us = get_litellm_params(
        custom_llm_provider="openrouter",
        api_base="https://us.openrouter.ai/api/v1",
    )
    assert params_us["data_residency"] == "us"

    params_global = get_litellm_params(
        custom_llm_provider="openrouter",
        api_base="https://openrouter.ai/api/v1",
    )
    assert params_global["data_residency"] is None


def test_openrouter_config_region():
    cfg = OpenrouterConfig(region="eu")
    assert cfg.region == "eu"


def test_openrouter_chat_config_get_complete_url():
    cfg = OpenrouterConfig(region="eu")
    url = cfg.get_complete_url(
        api_base=None,
        api_key="sk-test",
        model="anthropic/claude-3-opus",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://eu.openrouter.ai/api/v1/chat/completions"

    # Per-request parameter overrides config
    url_us = cfg.get_complete_url(
        api_base=None,
        api_key="sk-test",
        model="anthropic/claude-3-opus",
        optional_params={"region": "us"},
        litellm_params={},
    )
    assert url_us == "https://us.openrouter.ai/api/v1/chat/completions"


def test_openrouter_region_not_supported_error_mapping():
    """Verify that OpenRouter 404 with regional error raises OpenRouterRegionNotSupportedError."""
    fake_response = httpx.Response(
        status_code=404,
        request=httpx.Request("POST", "https://eu.openrouter.ai/api/v1/chat/completions"),
    )
    fake_exc = httpx.HTTPStatusError(
        message="404 Not Found",
        request=fake_response.request,
        response=fake_response,
    )
    fake_exc.status_code = 404

    # 1. Regional 404 message -> OpenRouterRegionNotSupportedError
    error_msg = '{"error": {"message": "No endpoints found supporting your data region", "code": 404}}'
    with pytest.raises(OpenRouterRegionNotSupportedError) as exc_info:
        _map_openrouter_exception(
            model="openrouter/anthropic/claude-sonnet-4",
            original_exception=fake_exc,
            custom_llm_provider="openrouter",
            error_str=error_msg,
            exception_type="NotFoundError",
            exception_provider="openrouter",
            extra_information="",
        )

    # Ensure it's also an instance of NotFoundError for backward compatibility
    assert isinstance(exc_info.value, NotFoundError)
    assert isinstance(exc_info.value, OpenRouterRegionNotSupportedError)
    assert exc_info.value.region == "eu"

    # 2. Standard 404 (model not found) -> standard NotFoundError
    standard_error_msg = '{"error": {"message": "Model not found", "code": 404}}'
    with pytest.raises(NotFoundError) as exc_info_standard:
        _map_openrouter_exception(
            model="openrouter/non-existent-model",
            original_exception=fake_exc,
            custom_llm_provider="openrouter",
            error_str=standard_error_msg,
            exception_type="NotFoundError",
            exception_provider="openrouter",
            extra_information="",
        )
    assert not isinstance(exc_info_standard.value, OpenRouterRegionNotSupportedError)


def test_openrouter_completion_with_region_parameter(monkeypatch):
    """Test that completion() resolves regional api_base and tracks data_residency."""
    captured_api_base = None
    captured_litellm_params = None

    def mock_completion(*args, **kwargs):
        nonlocal captured_api_base, captured_litellm_params
        captured_api_base = kwargs.get("api_base")
        captured_litellm_params = kwargs.get("litellm_params")
        return litellm.ModelResponse(
            choices=[
                litellm.Choices(
                    finish_reason="stop",
                    index=0,
                    message=litellm.Message(content="Hello from EU", role="assistant"),
                )
            ]
        )

    monkeypatch.setattr(
        "litellm.main.base_llm_http_handler.completion",
        mock_completion,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-mock-key")

    resp = litellm.completion(
        model="openrouter/anthropic/claude-sonnet-4",
        messages=[{"role": "user", "content": "Hello"}],
        region="eu",
    )
    assert captured_api_base == "https://eu.openrouter.ai/api/v1"
    assert captured_litellm_params is not None
    assert captured_litellm_params.get("api_base") == "https://eu.openrouter.ai/api/v1"
    assert captured_litellm_params.get("data_residency") == "eu"
    assert resp.choices[0].message.content == "Hello from EU"

