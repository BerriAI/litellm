"""
Common utilities for the DashScope LLM provider.
"""

from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
    from litellm.llms.base_llm.image_generation.transformation import (
        BaseImageGenerationConfig,
    )
    from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
    from litellm.llms.dashscope.rerank.transformation import DashScopeRerankConfig


def get_dashscope_family_embedding_config(custom_llm_provider: str) -> "BaseEmbeddingConfig":
    if custom_llm_provider == "qwencloud":
        from litellm.llms.dashscope.qwencloud import QwenCloudEmbeddingConfig

        return QwenCloudEmbeddingConfig()
    if custom_llm_provider == "qwen_ai_platform":
        from litellm.llms.dashscope.qwen_ai_platform import (
            QwenAIPlatformEmbeddingConfig,
        )

        return QwenAIPlatformEmbeddingConfig()
    from litellm.llms.dashscope.embed.transformation import DashScopeEmbeddingConfig

    return DashScopeEmbeddingConfig()


def get_dashscope_family_rerank_config(
    custom_llm_provider: str, model: str, api_base: str | None = None
) -> "BaseRerankConfig":
    provider_config: Final = _get_dashscope_family_rerank_provider_config(custom_llm_provider)
    model_cost: Final[Mapping[str, object]] = litellm.model_cost
    runtime_api: Final = next(
        (
            declared_api
            for key in (f"{custom_llm_provider}/{model}", f"dashscope/{model}", model)
            if (declared_api := _rerank_api_from_model_info(model_cost.get(key))) is not None
        ),
        None,
    )
    rerank_api: Final = runtime_api or _bundled_dashscope_rerank_apis().get(f"dashscope/{model}")
    if rerank_api == "native":
        from litellm.llms.dashscope.rerank.native_transformation import DashScopeNativeRerankConfig

        return DashScopeNativeRerankConfig(
            provider_config, api_base=api_base or get_secret_str(f"{custom_llm_provider.upper()}_API_BASE_RERANK")
        )
    return provider_config


def _rerank_api_from_model_info(raw_model_info: object) -> str | None:
    if raw_model_info is None:
        return None
    model_info: Final = TypeAdapter(Mapping[str, object]).validate_python(raw_model_info)
    provider_info: Final = model_info.get("provider_specific_entry")
    if provider_info is None:
        return None
    metadata: Final = TypeAdapter(Mapping[str, object]).validate_python(provider_info)
    rerank_api: Final[str | None] = TypeAdapter(Literal["native", "compatible"] | None).validate_python(
        metadata.get("rerank_api")
    )
    return rerank_api


@lru_cache(maxsize=1)
def _bundled_dashscope_rerank_apis() -> Mapping[str, str]:
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    model_infos: Final = TypeAdapter(Mapping[str, object]).validate_json(
        GetModelCostMap.read_local_model_cost_map_text()
    )
    return MappingProxyType(
        {
            key: rerank_api
            for key, model_info in model_infos.items()
            if key.startswith("dashscope/") and (rerank_api := _rerank_api_from_model_info(model_info)) is not None
        }
    )


def _get_dashscope_family_rerank_provider_config(custom_llm_provider: str) -> "DashScopeRerankConfig":
    if custom_llm_provider == "qwencloud":
        from litellm.llms.dashscope.qwencloud import QwenCloudRerankConfig

        return QwenCloudRerankConfig()
    if custom_llm_provider == "qwen_ai_platform":
        from litellm.llms.dashscope.qwen_ai_platform import QwenAIPlatformRerankConfig

        return QwenAIPlatformRerankConfig()
    from litellm.llms.dashscope.rerank.transformation import DashScopeRerankConfig

    return DashScopeRerankConfig()


def get_dashscope_family_image_generation_config(
    custom_llm_provider: str,
) -> "BaseImageGenerationConfig":
    if custom_llm_provider == "qwencloud":
        from litellm.llms.dashscope.qwencloud import QwenCloudImageGenerationConfig

        return QwenCloudImageGenerationConfig()
    if custom_llm_provider == "qwen_ai_platform":
        from litellm.llms.dashscope.qwen_ai_platform import (
            QwenAIPlatformImageGenerationConfig,
        )

        return QwenAIPlatformImageGenerationConfig()
    from litellm.llms.dashscope.image_generation.transformation import (
        DashScopeImageGenerationConfig,
    )

    return DashScopeImageGenerationConfig()


def resolve_dashscope_family_api_key(custom_llm_provider: str, api_key: str | None) -> str | None:
    if custom_llm_provider == "dashscope":
        return api_key or get_secret_str("DASHSCOPE_API_KEY")
    return api_key or get_secret_str(f"{custom_llm_provider.upper()}_API_KEY") or get_secret_str("DASHSCOPE_API_KEY")


def missing_dashscope_family_key_message(custom_llm_provider: str) -> str:
    if custom_llm_provider == "qwencloud":
        return (
            "Missing API key for QwenCloud. Set QWENCLOUD_API_KEY or "
            "DASHSCOPE_API_KEY environment variable or pass api_key parameter."
        )
    if custom_llm_provider == "qwen_ai_platform":
        return (
            "Missing API key for Qwen AI Platform. Set QWEN_AI_PLATFORM_API_KEY or "
            "DASHSCOPE_API_KEY environment variable or pass api_key parameter."
        )
    return "Missing API key for DashScope. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter."


class DashScopeError(BaseLLMException):
    """Exception class for DashScope provider errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: httpx.Headers | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers or httpx.Headers()
        super().__init__(
            status_code=status_code,
            message=message,
            headers=dict(self.headers),
        )
