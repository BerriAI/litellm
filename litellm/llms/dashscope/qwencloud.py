from typing import Final

from litellm.secret_managers.main import get_secret_str

from .chat.transformation import DashScopeChatConfig
from .embed.transformation import DashScopeEmbeddingConfig
from .image_generation.transformation import DashScopeImageGenerationConfig
from .rerank.transformation import DashScopeRerankConfig

QWENCLOUD_API_BASE: Final = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWENCLOUD_RERANK_API_BASE: Final = "https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks"
QWENCLOUD_IMAGE_API_BASE: Final = (
    "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)


def _resolve_qwencloud_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("QWENCLOUD_API_KEY") or get_secret_str("DASHSCOPE_API_KEY")


def _require_qwencloud_api_key(api_key: str | None) -> str:
    resolved: Final = _resolve_qwencloud_api_key(api_key)
    if resolved is None:
        raise ValueError(
            "QwenCloud API key is required. Set 'QWENCLOUD_API_KEY' or 'DASHSCOPE_API_KEY' env var "
            "or pass api_key explicitly."
        )
    return resolved


class QwenCloudChatConfig(DashScopeChatConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return self._resolve_chat_api_base(api_base), _resolve_qwencloud_api_key(api_key)

    def _resolve_chat_api_base(self, api_base: str | None) -> str:
        return api_base or get_secret_str("QWENCLOUD_API_BASE") or QWENCLOUD_API_BASE


class QwenCloudEmbeddingConfig(DashScopeEmbeddingConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwencloud_api_key(api_key)

    def _resolve_embedding_api_base(self, api_base: str | None) -> str:
        return api_base or get_secret_str("QWENCLOUD_API_BASE") or QWENCLOUD_API_BASE


class QwenCloudRerankConfig(DashScopeRerankConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwencloud_api_key(api_key)

    def _resolve_rerank_api_base(self, api_base: str | None) -> str:
        return api_base or get_secret_str("QWENCLOUD_API_BASE_RERANK") or QWENCLOUD_RERANK_API_BASE


class QwenCloudImageGenerationConfig(DashScopeImageGenerationConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwencloud_api_key(api_key)

    def _resolve_image_api_base(self, image_api_base: str | None) -> str:
        return image_api_base or get_secret_str("QWENCLOUD_API_BASE_IMAGE") or QWENCLOUD_IMAGE_API_BASE
