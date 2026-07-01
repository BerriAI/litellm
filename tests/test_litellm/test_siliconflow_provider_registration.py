from typing import Callable, cast

import litellm
from pytest import MonkeyPatch

from litellm import (
    SiliconFlowChatConfig,
    SiliconFlowEmbeddingConfig,
    SiliconFlowRerankConfig,
)
from litellm.llms.siliconflow.image_generation.transformation import (
    SiliconFlowImageGenerationConfig,
)
from litellm.utils import LlmProviders, ProviderConfigManager


def test_get_provider_configs_siliconflow():
    assert isinstance(
        ProviderConfigManager.get_provider_chat_config(
            model="deepseek-ai/DeepSeek-V4-Flash",
            provider=LlmProviders.SILICONFLOW,
        ),
        SiliconFlowChatConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_embedding_config(
            model="BAAI/bge-m3",
            provider=LlmProviders.SILICONFLOW,
        ),
        SiliconFlowEmbeddingConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_rerank_config(
            model="Pro/BAAI/bge-reranker-v2-m3",
            provider=LlmProviders.SILICONFLOW,
            api_base=None,
            present_version_params=[],
        ),
        SiliconFlowRerankConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_image_generation_config(
            model="Qwen/Qwen-Image",
            provider=LlmProviders.SILICONFLOW,
        ),
        SiliconFlowImageGenerationConfig,
    )


def test_get_model_info_siliconflow_pricing(monkeypatch: MonkeyPatch):
    original_model_cost = cast(dict[str, object], litellm.model_cost)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    get_model_cost_map = cast(Callable[[str], dict[str, object]], litellm.get_model_cost_map)
    litellm.model_cost = get_model_cost_map("")
    cache_clear = cast(Callable[[], None] | None, getattr(litellm.get_model_info, "cache_clear", None))
    if cache_clear is not None:
        cache_clear()

    try:
        chat_info = litellm.get_model_info(
            model="siliconflow/deepseek-ai/DeepSeek-V4-Flash",
            custom_llm_provider="siliconflow",
        )
        assert chat_info["input_cost_per_token"] == 1.4e-07
        assert chat_info.get("cache_read_input_token_cost") == 2.8e-09

        embedding_info = litellm.get_model_info(
            model="siliconflow/Pro/BAAI/bge-m3",
            custom_llm_provider="siliconflow",
        )
        assert embedding_info["mode"] == "embedding"
        assert embedding_info["input_cost_per_token"] == 9.8e-09

        image_info = litellm.get_model_info(
            model="siliconflow/Qwen/Qwen-Image",
            custom_llm_provider="siliconflow",
        )
        assert image_info["mode"] == "image_generation"
        assert image_info.get("output_cost_per_image") == 0.042
    finally:
        litellm.model_cost = original_model_cost
        if cache_clear is not None:
            cache_clear()
