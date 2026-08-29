"""Tests for the AI Power Grid JSON provider integration."""

import os
from unittest import mock

import pytest

import litellm

AIPG_API_BASE = "https://api.aipowergrid.io/v1"


def test_aipg_json_registry():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = JSONProviderRegistry.get("aipg")
    assert config is not None
    assert config.base_url == AIPG_API_BASE
    assert config.api_key_env == "AIPG_API_KEY"
    assert config.api_base_env == "AIPG_API_BASE"
    assert config.require_explicit_key_for_custom_base is True
    assert config.supported_endpoints == [
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/images/generations",
    ]
    assert JSONProviderRegistry.supports_responses_api("aipg") is True


def test_aipg_get_openai_compatible_provider_info():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider = JSONProviderRegistry.get("aipg")
    assert provider is not None
    config = create_config_class(provider)()

    with mock.patch.dict(os.environ, {"AIPG_API_KEY": "grid-test"}, clear=True):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == AIPG_API_BASE
        assert api_key == "grid-test"

    with mock.patch.dict(
        os.environ,
        {
            "AIPG_API_KEY": "env-key",
            "AIPG_API_BASE": "https://operator.example/v1",
        },
        clear=True,
    ):
        api_base, api_key = config._get_openai_compatible_provider_info("https://explicit.example/v1", "explicit-key")
        assert api_base == "https://explicit.example/v1"
        assert api_key == "explicit-key"

    with (
        mock.patch.dict(os.environ, {"AIPG_API_KEY": "env-key"}, clear=True),
        pytest.raises(ValueError, match="api_key is required for custom api_base"),
    ):
        config._get_openai_compatible_provider_info("https://attacker.example/v1", None)

    with mock.patch.dict(
        os.environ,
        {
            "AIPG_API_KEY": "env-key",
            "AIPG_API_BASE": "https://operator.example/v1",
        },
        clear=True,
    ):
        api_base, api_key = config._get_openai_compatible_provider_info("https://operator.example/v1/", None)
        assert api_base == "https://operator.example/v1/"
        assert api_key == "env-key"

    mapped = config.map_openai_params(
        non_default_params={"max_completion_tokens": 12, "temperature": 0.2},
        optional_params={},
        model="gpt-oss-120b",
        drop_params=False,
    )
    assert mapped["max_tokens"] == 12
    assert mapped["temperature"] == 0.2


def test_get_llm_provider_aipg():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    with mock.patch.dict(os.environ, {"AIPG_API_KEY": "grid-test"}, clear=True):
        model, provider, api_key, api_base = get_llm_provider("aipg/gpt-oss-120b")

    assert model == "gpt-oss-120b"
    assert provider == "aipg"
    assert api_key == "grid-test"
    assert api_base == AIPG_API_BASE


def test_aipg_responses_rejects_env_key_with_custom_api_base():
    from litellm.llms.openai_like.dynamic_config import create_responses_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.types.router import GenericLiteLLMParams

    provider = JSONProviderRegistry.get("aipg")
    assert provider is not None
    config = create_responses_config_class(provider)()

    with (
        mock.patch.dict(os.environ, {"AIPG_API_KEY": "env-key"}, clear=True),
        pytest.raises(ValueError, match="api_key is required for custom api_base"),
    ):
        config.validate_environment(
            headers={},
            model="gpt-oss-120b",
            litellm_params=GenericLiteLLMParams(api_base="https://attacker.example/v1"),
        )

    with mock.patch.dict(os.environ, {"AIPG_API_KEY": "env-key"}, clear=True):
        headers = config.validate_environment(
            headers={},
            model="gpt-oss-120b",
            litellm_params=GenericLiteLLMParams(
                api_base="https://attacker.example/v1",
                api_key="explicit-key",
            ),
        )
        assert headers["Authorization"] == "Bearer explicit-key"


def test_aipg_image_generation_uses_native_endpoint():
    mock_response = mock.MagicMock()
    mock_response.model_dump.return_value = {
        "created": 1,
        "data": [{"url": "https://images.example/aipg.webp"}],
    }
    mock_client = mock.MagicMock()
    mock_client.images.generate.return_value = mock_response

    with (
        mock.patch.dict(os.environ, {"AIPG_API_KEY": "grid-test"}, clear=True),
        mock.patch("litellm.llms.openai.openai.OpenAI", return_value=mock_client) as constructor,
    ):
        response = litellm.image_generation(
            model="aipg/z-image-turbo",
            prompt="An amber square on black.",
            n=1,
            size="512x512",
        )

    constructor.assert_called_once()
    assert constructor.call_args.kwargs["api_key"] == "grid-test"
    assert str(constructor.call_args.kwargs["base_url"]).rstrip("/") == AIPG_API_BASE
    request = mock_client.images.generate.call_args.kwargs
    assert request["model"] == "z-image-turbo"
    assert request["prompt"] == "An amber square on black."
    assert request["n"] == 1
    assert request["size"] == "512x512"
    assert response.data[0]["url"] == "https://images.example/aipg.webp"


def test_aipg_image_generation_rejects_env_key_with_custom_api_base():
    with (
        mock.patch.dict(os.environ, {"AIPG_API_KEY": "env-key"}, clear=True),
        pytest.raises(litellm.BadRequestError, match="api_key is required for custom api_base"),
    ):
        litellm.image_generation(
            model="aipg/z-image-turbo",
            prompt="An amber square on black.",
            api_base="https://attacker.example/v1",
        )


def test_aipg_model_metadata():
    model_cost = litellm.get_model_cost_map(url="")
    expected = {
        "aipg/gpt-oss-120b": (60000, 32768, 7.5e-08, 3e-07),
        "aipg/deepseek-v4-flash-nvfp4": (262144, 32768, 7e-08, 1.4e-07),
        "aipg/Smollm-135m": (2048, 1024, 5e-09, 1e-08),
    }
    for model, (context, output_limit, input_cost, output_cost) in expected.items():
        info = model_cost[model]
        assert info["litellm_provider"] == "aipg"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == context
        assert info["max_output_tokens"] == output_limit
        assert info["max_tokens"] == output_limit
        assert info["input_cost_per_token"] == input_cost
        assert info["output_cost_per_token"] == output_cost

    image_prices = {
        "aipg/z-image-turbo": 0.003,
        "aipg/Krea 2 Turbo": 0.005,
        "aipg/FLUX.2 Klein 4B FP8": 0.01,
    }
    for model, input_cost_per_image in image_prices.items():
        info = model_cost[model]
        assert info["litellm_provider"] == "aipg"
        assert info["mode"] == "image_generation"
        assert info["input_cost_per_image"] == input_cost_per_image
        assert info["supported_endpoints"] == ["/v1/images/generations"]
