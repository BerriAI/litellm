"""Tests for the JSON-configured StepFun provider."""

import litellm


def test_stepfun_in_provider_list():
    from litellm import LlmProviders

    assert LlmProviders.STEPFUN.value == "stepfun"
    assert "stepfun" in litellm.provider_list


def test_stepfun_json_config():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = JSONProviderRegistry.get("stepfun")

    assert config is not None
    assert config.base_url == "https://api.stepfun.ai/v1"
    assert config.api_key_env == "STEPFUN_API_KEY"
    assert config.api_base_env == "STEPFUN_API_BASE"
    assert config.param_mappings["max_completion_tokens"] == "max_tokens"
    assert config.supported_endpoints == [
        "/v1/chat/completions",
        "/v1/responses",
    ]


def test_stepfun_provider_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="stepfun/step-3.7-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "step-3.7-flash"
    assert provider == "stepfun"
    assert api_key is None
    assert api_base == "https://api.stepfun.ai/v1"


def test_stepfun_supports_responses_api():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert JSONProviderRegistry.supports_responses_api("stepfun") is True


def test_stepfun_router_config():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "step-3.7-flash",
                "litellm_params": {
                    "model": "stepfun/step-3.7-flash",
                    "api_key": "test-key",
                },
            }
        ]
    )

    assert len(router.model_list) == 1
    assert router.model_list[0]["model_name"] == "step-3.7-flash"
