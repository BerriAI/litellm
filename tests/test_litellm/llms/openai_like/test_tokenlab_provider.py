import pytest

import litellm


TOKENLAB_API_BASE = "https://api.tokenlab.sh/v1"
TOKENLAB_SAMPLE_MODELS = [
    "tokenlab/gpt-5.5",
    "tokenlab/claude-opus-4-8",
    "tokenlab/gemini-3.5-flash",
    "tokenlab/deepseek-v4-pro",
    "tokenlab/qwen3.7-max",
    "tokenlab/text-embedding-3-small",
]


def test_tokenlab_json_registry():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.TOKENLAB.value == "tokenlab"
    assert litellm.LlmProviders("tokenlab") == litellm.LlmProviders.TOKENLAB
    assert JSONProviderRegistry.exists("tokenlab")
    config = JSONProviderRegistry.get("tokenlab")
    assert config is not None
    assert config.base_url == TOKENLAB_API_BASE
    assert config.api_key_env == "TOKENLAB_API_KEY"
    assert config.api_base_env == "TOKENLAB_API_BASE"
    assert config.param_mappings["max_completion_tokens"] == "max_tokens"
    assert config.supported_endpoints == [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/embeddings",
    ]
    assert JSONProviderRegistry.supports_responses_api("tokenlab") is True
    assert JSONProviderRegistry.supports_anthropic_messages_api("tokenlab") is True


def test_tokenlab_provider_detection_by_prefix():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider("tokenlab/gpt-5.5")

    assert model == "gpt-5.5"
    assert provider == "tokenlab"
    assert api_base == TOKENLAB_API_BASE


def test_tokenlab_api_base_override():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    _, provider, api_key, api_base = get_llm_provider(
        model="tokenlab/gpt-5.5",
        api_base="https://custom.tokenlab.example/v1",
        api_key="sk-test",
    )

    assert provider == "tokenlab"
    assert api_base == "https://custom.tokenlab.example/v1"
    assert api_key == "sk-test"


def test_tokenlab_url_autodetection():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    _, provider, _, api_base = get_llm_provider(
        model="gpt-5.5",
        api_base=TOKENLAB_API_BASE,
    )

    assert provider == "tokenlab"
    assert api_base == TOKENLAB_API_BASE


def test_tokenlab_chat_complete_url():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("tokenlab"))()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gpt-5.5",
            optional_params={},
            litellm_params={},
        )
        == "https://api.tokenlab.sh/v1/chat/completions"
    )


def test_tokenlab_responses_api_config():
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="tokenlab",
        model="tokenlab/gpt-5.5",
    )

    assert isinstance(config, OpenAIResponsesAPIConfig)
    assert config.custom_llm_provider == "tokenlab"
    assert config.get_complete_url(api_base=None, litellm_params={}) == "https://api.tokenlab.sh/v1/responses"


def test_tokenlab_anthropic_messages_config():
    from litellm.llms.openai_like.messages.transformation import (
        OpenAILikeAnthropicMessagesConfig,
    )
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        provider=litellm.LlmProviders.TOKENLAB,
        model="claude-opus-4-8",
    )

    assert isinstance(config, OpenAILikeAnthropicMessagesConfig)
    assert (
        config.get_complete_url(
            api_base=TOKENLAB_API_BASE,
            api_key="sk-test",
            model="claude-opus-4-8",
            optional_params={},
            litellm_params={},
        )
        == "https://api.tokenlab.sh/v1/messages"
    )


class TestTokenLabCostMap:
    @pytest.fixture(autouse=True)
    def _use_local_model_cost_map(self, monkeypatch):
        original_model_cost = litellm.model_cost
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.get_model_info.cache_clear()
        try:
            yield
        finally:
            litellm.model_cost = original_model_cost
            litellm.get_model_info.cache_clear()

    def test_sample_models_registered(self):
        for model in TOKENLAB_SAMPLE_MODELS:
            info = litellm.get_model_info(model)
            assert info["litellm_provider"] == "tokenlab"
            assert info["mode"] in {"chat", "embedding"}

    def test_claude_models_advertise_messages_endpoint(self):
        endpoints = litellm.model_cost["tokenlab/claude-opus-4-8"]["supported_endpoints"]
        assert "/v1/messages" in endpoints

    def test_chat_cost_is_wired(self):
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="tokenlab/gpt-5.5",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert prompt_cost == pytest.approx(1.5)
        assert completion_cost == pytest.approx(9.0)

    def test_embedding_cost_is_wired(self):
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="tokenlab/text-embedding-3-small",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert prompt_cost == pytest.approx(0.02)
        assert completion_cost == 0
