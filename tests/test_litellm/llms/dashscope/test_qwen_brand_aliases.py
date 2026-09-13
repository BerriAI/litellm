import math

import pytest

import litellm
from litellm import completion, get_llm_provider
from litellm.llms.dashscope.chat.transformation import DashScopeChatConfig
from litellm.llms.dashscope.cost_calculator import (
    cost_per_token as dashscope_cost_per_token,
)
from litellm.llms.dashscope.embed.transformation import DashScopeEmbeddingConfig
from litellm.llms.dashscope.image_generation.transformation import (
    DashScopeImageGenerationConfig,
)
from litellm.llms.dashscope.qwen_ai_platform import (
    QWEN_AI_PLATFORM_API_BASE,
    QWEN_AI_PLATFORM_IMAGE_API_BASE,
    QWEN_AI_PLATFORM_RERANK_API_BASE,
    QwenAIPlatformChatConfig,
    QwenAIPlatformEmbeddingConfig,
    QwenAIPlatformImageGenerationConfig,
    QwenAIPlatformRerankConfig,
)
from litellm.llms.dashscope.qwencloud import (
    QWENCLOUD_API_BASE,
    QWENCLOUD_IMAGE_API_BASE,
    QWENCLOUD_RERANK_API_BASE,
    QwenCloudChatConfig,
    QwenCloudEmbeddingConfig,
    QwenCloudImageGenerationConfig,
    QwenCloudRerankConfig,
)
from litellm.llms.dashscope.rerank.transformation import DashScopeRerankConfig
from litellm.types.utils import LlmProviders, Usage
from litellm.utils import ProviderConfigManager

DASHSCOPE_FAMILY_ENV_VARS = [
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_API_BASE",
    "DASHSCOPE_API_BASE_RERANK",
    "DASHSCOPE_API_BASE_IMAGE",
    "QWENCLOUD_API_KEY",
    "QWENCLOUD_API_BASE",
    "QWENCLOUD_API_BASE_RERANK",
    "QWENCLOUD_API_BASE_IMAGE",
    "QWEN_AI_PLATFORM_API_KEY",
    "QWEN_AI_PLATFORM_API_BASE",
    "QWEN_AI_PLATFORM_API_BASE_RERANK",
    "QWEN_AI_PLATFORM_API_BASE_IMAGE",
]

BRAND_CASES = [
    pytest.param(
        {
            "provider": "qwencloud",
            "enum": LlmProviders.QWENCLOUD,
            "key_env": "QWENCLOUD_API_KEY",
            "base_env": "QWENCLOUD_API_BASE",
            "default_base": QWENCLOUD_API_BASE,
            "default_rerank_base": QWENCLOUD_RERANK_API_BASE,
            "default_image_base": QWENCLOUD_IMAGE_API_BASE,
            "chat_config": QwenCloudChatConfig,
            "embedding_config": QwenCloudEmbeddingConfig,
            "rerank_config": QwenCloudRerankConfig,
            "image_config": QwenCloudImageGenerationConfig,
        },
        id="qwencloud",
    ),
    pytest.param(
        {
            "provider": "qwen_ai_platform",
            "enum": LlmProviders.QWEN_AI_PLATFORM,
            "key_env": "QWEN_AI_PLATFORM_API_KEY",
            "base_env": "QWEN_AI_PLATFORM_API_BASE",
            "default_base": QWEN_AI_PLATFORM_API_BASE,
            "default_rerank_base": QWEN_AI_PLATFORM_RERANK_API_BASE,
            "default_image_base": QWEN_AI_PLATFORM_IMAGE_API_BASE,
            "chat_config": QwenAIPlatformChatConfig,
            "embedding_config": QwenAIPlatformEmbeddingConfig,
            "rerank_config": QwenAIPlatformRerankConfig,
            "image_config": QwenAIPlatformImageGenerationConfig,
        },
        id="qwen_ai_platform",
    ),
]


@pytest.fixture(autouse=True)
def clear_dashscope_family_env(monkeypatch):
    for env_var in DASHSCOPE_FAMILY_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


class TestQwenBrandProviderResolution:
    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_get_llm_provider_resolves_brand_default_base(self, brand):
        model, provider, api_key, api_base = get_llm_provider(f"{brand['provider']}/qwen-max", api_key="sk-explicit")
        assert model == "qwen-max"
        assert provider == brand["provider"]
        assert api_key == "sk-explicit"
        assert api_base == brand["default_base"]

    def test_dashscope_resolution_unchanged(self):
        model, provider, api_key, api_base = get_llm_provider("dashscope/qwen-max", api_key="sk-explicit")
        assert model == "qwen-max"
        assert provider == "dashscope"
        assert api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_brand_env_key_wins_over_dashscope_key(self, monkeypatch, brand):
        monkeypatch.setenv(brand["key_env"], "sk-brand")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope")
        _, _, api_key, _ = get_llm_provider(f"{brand['provider']}/qwen-max")
        assert api_key == "sk-brand"

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_dashscope_key_is_fallback(self, monkeypatch, brand):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope")
        _, _, api_key, _ = get_llm_provider(f"{brand['provider']}/qwen-max")
        assert api_key == "sk-dashscope"

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_dashscope_api_base_does_not_leak_into_brand(self, monkeypatch, brand):
        monkeypatch.setenv("DASHSCOPE_API_BASE", "https://legacy.example.com/v1")
        _, _, _, api_base = get_llm_provider(f"{brand['provider']}/qwen-max", api_key="sk-explicit")
        assert api_base == brand["default_base"]

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_brand_api_base_env_wins(self, monkeypatch, brand):
        monkeypatch.setenv(brand["base_env"], "https://brand.example.com/v1")
        _, _, _, api_base = get_llm_provider(f"{brand['provider']}/qwen-max", api_key="sk-explicit")
        assert api_base == "https://brand.example.com/v1"


class TestQwenBrandConfigDispatch:
    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_chat_config(self, brand):
        config = ProviderConfigManager.get_provider_chat_config("qwen-max", brand["enum"])
        assert isinstance(config, brand["chat_config"])
        assert isinstance(config, DashScopeChatConfig)

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_embedding_config(self, brand):
        config = ProviderConfigManager.get_provider_embedding_config(model="text-embedding-v3", provider=brand["enum"])
        assert isinstance(config, brand["embedding_config"])
        assert isinstance(config, DashScopeEmbeddingConfig)

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_rerank_config(self, brand):
        config = ProviderConfigManager.get_provider_rerank_config(
            model="gte-rerank-v2",
            provider=brand["enum"],
            api_base=None,
            present_version_params=[],
        )
        assert isinstance(config, brand["rerank_config"])
        assert isinstance(config, DashScopeRerankConfig)

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_image_generation_config(self, brand):
        config = ProviderConfigManager.get_provider_image_generation_config(model="qwen-image", provider=brand["enum"])
        assert isinstance(config, brand["image_config"])
        assert isinstance(config, DashScopeImageGenerationConfig)


class TestQwenBrandDefaultUrls:
    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_chat_complete_url(self, brand):
        url = brand["chat_config"]().get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="qwen-max",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{brand['default_base']}/chat/completions"

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_embedding_complete_url(self, brand):
        url = brand["embedding_config"]().get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="text-embedding-v3",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{brand['default_base']}/embeddings"

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_embedding_ignores_dashscope_api_base(self, monkeypatch, brand):
        monkeypatch.setenv("DASHSCOPE_API_BASE", "https://legacy.example.com/v1")
        url = brand["embedding_config"]().get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="text-embedding-v3",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{brand['default_base']}/embeddings"

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_rerank_complete_url(self, brand):
        url = brand["rerank_config"]().get_complete_url(api_base=None, model="gte-rerank-v2")
        assert url == brand["default_rerank_base"]

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_rerank_env_override(self, monkeypatch, brand):
        monkeypatch.setenv(f"{brand['base_env']}_RERANK", "https://rerank.example.com/v1/reranks")
        url = brand["rerank_config"]().get_complete_url(api_base=None, model="gte-rerank-v2")
        assert url == "https://rerank.example.com/v1/reranks"

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_rerank_remaps_chat_shaped_default_base(self, brand):
        url = brand["rerank_config"]().get_complete_url(api_base=brand["default_base"], model="gte-rerank-v2")
        assert url == brand["default_rerank_base"]

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_image_generation_complete_url(self, brand):
        url = brand["image_config"]().get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="qwen-image",
            optional_params={},
            litellm_params={},
        )
        assert url == brand["default_image_base"]

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_image_generation_ignores_chat_compatible_api_base(self, brand):
        url = brand["image_config"]().get_complete_url(
            api_base=brand["default_base"],
            api_key="sk-test",
            model="qwen-image",
            optional_params={},
            litellm_params={},
        )
        assert url == brand["default_image_base"]

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_validate_environment_requires_key(self, brand):
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            brand["embedding_config"]().validate_environment(
                headers={},
                model="text-embedding-v3",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )


class TestQwenBrandCostParity:
    @pytest.fixture(autouse=True)
    def setup_model_cost_map(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_get_model_info(self, brand):
        model_info = litellm.get_model_info(f"{brand['provider']}/qwen-max")
        dashscope_info = litellm.get_model_info("dashscope/qwen-max")
        assert model_info["litellm_provider"] == brand["provider"]
        assert model_info["input_cost_per_token"] == dashscope_info["input_cost_per_token"]
        assert model_info["output_cost_per_token"] == dashscope_info["output_cost_per_token"]

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_flat_pricing_matches_dashscope(self, brand):
        usage = Usage(prompt_tokens=1000, completion_tokens=500)
        brand_costs = dashscope_cost_per_token(model="qwen-max", usage=usage, custom_llm_provider=brand["provider"])
        dashscope_costs = dashscope_cost_per_token(model="qwen-max", usage=usage)
        assert brand_costs == dashscope_costs

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_tiered_pricing_matches_dashscope(self, brand):
        usage = Usage(prompt_tokens=300000, completion_tokens=300000)
        brand_costs = dashscope_cost_per_token(model="qwen-flash", usage=usage, custom_llm_provider=brand["provider"])
        dashscope_costs = dashscope_cost_per_token(model="qwen-flash", usage=usage)
        assert brand_costs == dashscope_costs
        tier_2 = litellm.get_model_info(f"{brand['provider']}/qwen-flash")["tiered_pricing"][1]
        assert math.isclose(brand_costs[0], 300000 * tier_2["input_cost_per_token"], rel_tol=1e-10)

    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_public_cost_per_token_routes_to_dashscope_calculator(self, brand):
        brand_costs = litellm.cost_per_token(
            model=f"{brand['provider']}/qwen-max",
            prompt_tokens=1000,
            completion_tokens=500,
            custom_llm_provider=brand["provider"],
        )
        dashscope_costs = litellm.cost_per_token(
            model="dashscope/qwen-max",
            prompt_tokens=1000,
            completion_tokens=500,
            custom_llm_provider="dashscope",
        )
        assert brand_costs == dashscope_costs


class TestQwenBrandCompletionMock:
    @pytest.mark.respx()
    @pytest.mark.parametrize("brand", BRAND_CASES)
    def test_completion_hits_brand_default_host(self, respx_mock, brand, monkeypatch):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post(f"{brand['default_base']}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "qwen-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hey from LiteLLM!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
            status_code=200,
        )

        response = completion(
            model=f"{brand['provider']}/qwen-turbo",
            messages=[{"role": "user", "content": "say hey from LiteLLM"}],
            api_key="fake-brand-key",
        )

        assert response.choices[0].message.content == "Hey from LiteLLM!"
        request = respx_mock.calls[0].request
        assert request.url == f"{brand['default_base']}/chat/completions"
        assert request.headers["Authorization"] == "Bearer fake-brand-key"
