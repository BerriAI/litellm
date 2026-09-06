import pytest

import litellm
from litellm.llms.clf_ai_gateway.chat.transformation import ClfAiGatewayConfig
from litellm.router_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts
from litellm.types.utils import LlmProviders

MODELS = [
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5.2",
    "glm-4.7-flash",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "qwen3.8-27b",
]

GATEWAY_REASONING_EFFORTS = {
    "glm-5.3": ("none", "low", "medium", "high", "max"),
    "glm-5.3-flash": ("low", "medium", "high", "xhigh"),
    "glm-5.2": ("low", "medium", "high", "xhigh"),
    "glm-4.7-flash": ("low", "medium", "high"),
    "deepseek-v4-pro": ("low", "medium", "high", "xhigh"),
    "deepseek-v4-flash": ("low", "medium", "high", "xhigh"),
    "kimi-k2.6": ("low", "medium", "high"),
    "kimi-k2.7-code": ("low", "medium", "high"),
    "qwen3.8-27b": ("low", "medium", "xhigh"),
}


@pytest.fixture(autouse=True)
def _local_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


def test_provider_is_registered() -> None:
    assert LlmProviders.CLF_AI_GATEWAY.value == "clf_ai_gateway"
    assert "clf_ai_gateway" in litellm.openai_compatible_providers
    assert "api.clfaigateway.dev/v1" in litellm.openai_compatible_endpoints
    assert "clf_ai_gateway" in litellm.provider_list
    assert isinstance(
        litellm.ProviderConfigManager.get_provider_chat_config(model="glm-5.3", provider=LlmProviders.CLF_AI_GATEWAY),
        ClfAiGatewayConfig,
    )


def test_default_api_base_and_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLF_AI_GATEWAY_API_KEY", "sk-gw-test")
    monkeypatch.delenv("CLF_AI_GATEWAY_API_BASE", raising=False)
    api_base, api_key = ClfAiGatewayConfig()._get_openai_compatible_provider_info(api_base=None, api_key=None)
    assert api_base == "https://api.clfaigateway.dev/v1"
    assert api_key == "sk-gw-test"

    api_base, api_key = ClfAiGatewayConfig()._get_openai_compatible_provider_info(
        api_base="https://example.test/v1", api_key="explicit"
    )
    assert api_base == "https://example.test/v1"
    assert api_key == "explicit"


def test_get_llm_provider_resolves_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLF_AI_GATEWAY_API_KEY", "sk-gw-test")
    model, provider, key, api_base = litellm.get_llm_provider(model="clf_ai_gateway/glm-5.3")
    assert model == "glm-5.3"
    assert provider == "clf_ai_gateway"
    assert key == "sk-gw-test"
    assert api_base == "https://api.clfaigateway.dev/v1"


def test_get_llm_provider_infers_from_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLF_AI_GATEWAY_API_KEY", "sk-gw-test")
    _, provider, _, _ = litellm.get_llm_provider(model="glm-5.3", api_base="https://api.clfaigateway.dev/v1")
    assert provider == "clf_ai_gateway"

    _, provider, key, _ = litellm.get_llm_provider(
        model="glm-5.3", api_base="https://api.clfaigateway.dev/v1", api_key="explicit"
    )
    assert provider == "clf_ai_gateway"
    assert key == "explicit"


def test_supported_params_match_gateway_surface() -> None:
    params = ClfAiGatewayConfig().get_supported_openai_params(model="glm-5.3")
    assert "reasoning_effort" in params
    assert "tools" in params and "tool_choice" in params
    for rejected_by_gateway_with_400 in ("functions", "function_call", "logit_bias"):
        assert rejected_by_gateway_with_400 not in params


@pytest.mark.parametrize("model", MODELS)
def test_cost_map_entries(model: str) -> None:
    key = f"clf_ai_gateway/{model}"
    assert key in litellm.model_cost, f"missing cost map entry for {key}"
    entry = litellm.model_cost[key]
    assert entry["litellm_provider"] == "clf_ai_gateway"
    assert entry["mode"] == "chat"
    assert entry["max_output_tokens"] == 131072
    assert entry["input_cost_per_token"] > 0
    assert entry["output_cost_per_token"] > 0
    assert entry["supports_function_calling"] is True
    assert entry["supports_reasoning"] is True


def test_vision_flags_match_measured_surface() -> None:
    vision = {"glm-5.3-flash", "kimi-k2.6", "kimi-k2.7-code", "qwen3.8-27b"}
    for model in MODELS:
        entry = litellm.model_cost[f"clf_ai_gateway/{model}"]
        assert bool(entry.get("supports_vision")) is (model in vision), model


def test_validate_environment_reports_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLF_AI_GATEWAY_API_KEY", raising=False)
    missing = litellm.validate_environment(model="clf_ai_gateway/glm-5.3")
    assert missing["keys_in_environment"] is False
    assert "CLF_AI_GATEWAY_API_KEY" in missing["missing_keys"]

    monkeypatch.setenv("CLF_AI_GATEWAY_API_KEY", "sk-gw-test")
    present = litellm.validate_environment(model="clf_ai_gateway/glm-5.3")
    assert present["keys_in_environment"] is True
    assert present["missing_keys"] == []


def test_get_supported_openai_params_dispatches_to_provider_config() -> None:
    params = litellm.get_supported_openai_params(model="glm-5.3", custom_llm_provider="clf_ai_gateway")
    assert params is not None
    assert "reasoning_effort" in params
    assert "logit_bias" not in params


def test_completion_routes_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLF_AI_GATEWAY_API_KEY", "sk-gw-test")
    response = litellm.completion(
        model="clf_ai_gateway/glm-5.3",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello from mock",
    )
    assert response.choices[0].message.content == "hello from mock"


@pytest.mark.parametrize("model", MODELS)
def test_advertised_reasoning_efforts_match_the_gateway(model: str) -> None:
    key = f"clf_ai_gateway/{model}"
    entry = {**litellm.model_cost[key], "key": key}
    assert resolve_supported_reasoning_efforts(entry, deployment_is_mapped=True) == GATEWAY_REASONING_EFFORTS[model]
