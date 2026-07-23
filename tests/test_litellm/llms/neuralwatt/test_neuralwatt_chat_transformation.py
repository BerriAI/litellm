"""
Tests for Neuralwatt provider integration
"""

import os
from unittest import mock

import httpx
import pytest

import litellm
from litellm.llms.neuralwatt.chat.transformation import NeuralwattChatConfig
from litellm.types.utils import ModelResponse


def _make_raw_response(body: dict) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=body,
        request=httpx.Request("POST", "https://api.neuralwatt.com/v1/chat/completions"),
    )


_COMPLETION_BODY = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1699000000,
    "model": "glm-5.2",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}


def test_neuralwatt_config_initialization():
    config = NeuralwattChatConfig()
    assert config.custom_llm_provider == "neuralwatt"


def test_neuralwatt_get_openai_compatible_provider_info():
    config = NeuralwattChatConfig()

    with mock.patch.dict(os.environ, {}, clear=True):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.neuralwatt.com/v1"
        assert api_key is None

    with mock.patch.dict(
        os.environ,
        {
            "NEURALWATT_API_KEY": "test-key",
            "NEURALWATT_API_BASE": "https://custom.neuralwatt.com/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://custom.neuralwatt.com/v1"
        assert api_key == "test-key"

    with mock.patch.dict(
        os.environ,
        {
            "NEURALWATT_API_KEY": "env-key",
            "NEURALWATT_API_BASE": "https://env.neuralwatt.com/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info("https://param.neuralwatt.com/v1", "param-key")
        assert api_base == "https://param.neuralwatt.com/v1"
        assert api_key == "param-key"


def test_neuralwatt_env_key_not_leaked_to_caller_supplied_base():
    config = NeuralwattChatConfig()

    with mock.patch.dict(os.environ, {"NEURALWATT_API_KEY": "env-secret"}):
        api_base, api_key = config._get_openai_compatible_provider_info("https://attacker.example/v1", None)

    assert api_base == "https://attacker.example/v1"
    assert api_key is None


def test_get_llm_provider_neuralwatt():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("neuralwatt/glm-5.2")
    assert model == "glm-5.2"
    assert provider == "neuralwatt"

    model, provider, api_key, api_base = get_llm_provider("glm-5.2", api_base="https://api.neuralwatt.com/v1")
    assert model == "glm-5.2"
    assert provider == "neuralwatt"
    assert api_base == "https://api.neuralwatt.com/v1"


def test_neuralwatt_in_provider_lists():
    assert "neuralwatt" in litellm.openai_compatible_providers
    assert "neuralwatt" in litellm.provider_list
    assert "https://api.neuralwatt.com/v1" in litellm.openai_compatible_endpoints


def test_neuralwatt_provider_config_resolution():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config("glm-5.2", LlmProviders.NEURALWATT)
    assert type(config).__name__ == "NeuralwattChatConfig"


def _reload_neuralwatt_models():
    with mock.patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
        litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.neuralwatt_models = set()
    litellm.add_known_models()


def test_neuralwatt_models_configuration():
    from litellm import get_model_info

    _reload_neuralwatt_models()

    for model in [
        "neuralwatt/glm-5.2",
        "neuralwatt/kimi-k2.6",
        "neuralwatt/qwen3.6-35b",
    ]:
        model_info = get_model_info(model)
        assert model_info is not None, f"Model info not found for {model}"
        assert model_info.get("litellm_provider") == "neuralwatt"
        assert model_info.get("mode") == "chat"
        assert model_info.get("supports_function_calling") is True
        assert model_info.get("supports_system_messages") is True

    assert get_model_info("neuralwatt/kimi-k2.6").get("supports_vision") is True
    assert get_model_info("neuralwatt/glm-5.2").get("supports_vision") is not True


def test_neuralwatt_cache_read_pricing_is_quarter_of_input():
    from litellm import get_model_info

    _reload_neuralwatt_models()

    assert len(litellm.neuralwatt_models) > 0

    for model in litellm.neuralwatt_models:
        info = get_model_info(model)
        input_cost = info["input_cost_per_token"]
        cache_read = info.get("cache_read_input_token_cost")
        assert cache_read is not None, f"{model} missing cache_read_input_token_cost"
        assert info.get("supports_prompt_caching") is True, f"{model} should advertise prompt caching"
        assert cache_read == pytest.approx(input_cost * 0.25, rel=1e-9), (
            f"{model} cache read {cache_read} must be 25% of input {input_cost}"
        )


def _transform(body: dict) -> ModelResponse:
    config = NeuralwattChatConfig()
    return config.transform_response(
        model="glm-5.2",
        raw_response=_make_raw_response(body),
        model_response=ModelResponse(),
        logging_obj=mock.MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_neuralwatt_energy_preserved_in_hidden_params():
    body = {
        **_COMPLETION_BODY,
        "energy": {
            "energy_joules": 5.23,
            "energy_kwh": 0.00000145,
            "duration_seconds": 0.0183,
            "measurement_available": True,
            "attribution_method": "time_weighted",
        },
    }

    response = _transform(body)

    assert response.choices[0].message.content == "hi"
    energy = response._hidden_params.get("energy")
    assert energy is not None
    assert energy["energy_joules"] == 5.23
    assert energy["energy_kwh"] == 0.00000145
    assert energy["duration_seconds"] == 0.0183
    assert energy["measurement_available"] is True
    assert energy["attribution_method"] == "time_weighted"


def test_neuralwatt_no_energy_block_is_absent():
    response = _transform(dict(_COMPLETION_BODY))
    assert "energy" not in response._hidden_params


def test_neuralwatt_malformed_energy_is_ignored():
    response = _transform({**_COMPLETION_BODY, "energy": "not-an-object"})
    assert response.choices[0].message.content == "hi"
    assert "energy" not in response._hidden_params


def test_neuralwatt_model_list_populated():
    _reload_neuralwatt_models()

    assert len(litellm.neuralwatt_models) > 0
    for model in litellm.neuralwatt_models:
        assert model.startswith("neuralwatt/")

    for model in ["neuralwatt/glm-5.2", "neuralwatt/kimi-k2.6"]:
        assert model in litellm.neuralwatt_models
