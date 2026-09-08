from typing import Final

from litellm.secret_managers.main import get_secret_str

from .chat.transformation import DashScopeChatConfig
from .embed.transformation import DashScopeEmbeddingConfig
from .image_generation.transformation import DashScopeImageGenerationConfig
from .rerank.transformation import DashScopeRerankConfig

QWEN_AI_PLATFORM_API_BASE: Final = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_AI_PLATFORM_RERANK_API_BASE: Final = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
QWEN_AI_PLATFORM_IMAGE_API_BASE: Final = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)


def _resolve_qwen_ai_platform_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("QWEN_AI_PLATFORM_API_KEY") or get_secret_str("DASHSCOPE_API_KEY")


def _require_qwen_ai_platform_api_key(api_key: str | None) -> str:
    resolved: Final = _resolve_qwen_ai_platform_api_key(api_key)
    if resolved is None:
        raise ValueError(
            "Qwen AI Platform API key is required. Set 'QWEN_AI_PLATFORM_API_KEY' or 'DASHSCOPE_API_KEY' env var "
            "or pass api_key explicitly."
        )
    return resolved


class QwenAIPlatformChatConfig(DashScopeChatConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return self._resolve_chat_api_base(api_base), _resolve_qwen_ai_platform_api_key(api_key)

    def _resolve_chat_api_base(self, api_base: str | None) -> str:
        return api_base or get_secret_str("QWEN_AI_PLATFORM_API_BASE") or QWEN_AI_PLATFORM_API_BASE


class QwenAIPlatformEmbeddingConfig(DashScopeEmbeddingConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwen_ai_platform_api_key(api_key)

    def _resolve_embedding_api_base(self, api_base: str | None) -> str:
        return api_base or get_secret_str("QWEN_AI_PLATFORM_API_BASE") or QWEN_AI_PLATFORM_API_BASE


class QwenAIPlatformRerankConfig(DashScopeRerankConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwen_ai_platform_api_key(api_key)

    def _resolve_rerank_api_base(self, api_base: str | None) -> str:
        return api_base or get_secret_str("QWEN_AI_PLATFORM_API_BASE_RERANK") or QWEN_AI_PLATFORM_RERANK_API_BASE


class QwenAIPlatformImageGenerationConfig(DashScopeImageGenerationConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwen_ai_platform_api_key(api_key)

    def _resolve_image_api_base(self, image_api_base: str | None) -> str:
        return image_api_base or get_secret_str("QWEN_AI_PLATFORM_API_BASE_IMAGE") or QWEN_AI_PLATFORM_IMAGE_API_BASE
