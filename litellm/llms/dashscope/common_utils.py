"""
Common utilities for the DashScope LLM provider.
"""

from typing import TYPE_CHECKING, Final
from urllib.parse import urlparse

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
    from litellm.llms.base_llm.image_generation.transformation import (
        BaseImageGenerationConfig,
    )
    from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig

DASHSCOPE_CHAT_COMPATIBLE_PATH: Final = "/compatible-mode/v1"
DASHSCOPE_RERANK_PATH: Final = "/compatible-api/v1/reranks"


def _rerank_base_for_chat_shaped_base(api_base: str | None) -> str | None:
    if api_base is None:
        return None
    parsed: Final = urlparse(api_base)
    host: Final = parsed.hostname or ""
    on_aliyun_host: Final = host == "aliyuncs.com" or host.endswith(".aliyuncs.com")
    if not on_aliyun_host or parsed.path.rstrip("/") != DASHSCOPE_CHAT_COMPATIBLE_PATH:
        return None
    return f"{parsed.scheme}://{parsed.netloc}{DASHSCOPE_RERANK_PATH}"


def resolve_dashscope_family_rerank_api_base(api_base: str | None, env_var: str, default_rerank_base: str) -> str:
    remapped: Final = _rerank_base_for_chat_shaped_base(api_base)
    if api_base is not None and remapped is None:
        return api_base
    return get_secret_str(env_var) or remapped or default_rerank_base


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


def get_dashscope_family_rerank_config(custom_llm_provider: str) -> "BaseRerankConfig":
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
