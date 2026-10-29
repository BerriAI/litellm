"""
Common utilities for the DashScope LLM provider.
"""

from dataclasses import dataclass
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
    from litellm.llms.base_llm.videos.transformation import BaseVideoConfig

DASHSCOPE_CHAT_COMPATIBLE_PATH: Final = "/compatible-mode/v1"
DASHSCOPE_ANTHROPIC_PATH: Final = "/apps/anthropic"
DASHSCOPE_NATIVE_V1_PATH: Final = "/api/v1"
DASHSCOPE_COMPATIBLE_API_PATH: Final = "/compatible-api/v1"
DASHSCOPE_RERANK_PATH: Final = f"{DASHSCOPE_COMPATIBLE_API_PATH}/reranks"
DASHSCOPE_IMAGE_PATH: Final = "/api/v1/services/aigc/multimodal-generation/generation"
DASHSCOPE_DEFAULT_ORIGIN: Final = "https://dashscope.aliyuncs.com"

_STANDARD_DASHSCOPE_HOSTS: Final = frozenset(
    {
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
        "dashscope-us.aliyuncs.com",
    }
)
_WORKSPACE_HOST_SUFFIX: Final = ".maas.aliyuncs.com"
_EXCLUDED_WORKSPACE_PREFIXES: Final = ("trial.", "token-plan.")
_PROTOCOL_PATH_PREFIXES: Final = (
    DASHSCOPE_CHAT_COMPATIBLE_PATH,
    DASHSCOPE_ANTHROPIC_PATH,
    DASHSCOPE_NATIVE_V1_PATH,
    DASHSCOPE_COMPATIBLE_API_PATH,
)


@dataclass(frozen=True, slots=True)
class DashScopeEndpoints:
    origin: str
    openai: str
    anthropic: str
    native: str


def _is_dashscope_host(host: str) -> bool:
    if host in _STANDARD_DASHSCOPE_HOSTS:
        return True
    return host.endswith(_WORKSPACE_HOST_SUFFIX) and not host.startswith(_EXCLUDED_WORKSPACE_PREFIXES)


def _dashscope_origin(api_base: str | None) -> str | None:
    if not api_base:
        return None
    parsed: Final = urlparse(api_base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    if parsed.query or parsed.fragment:
        return None
    host: Final = (parsed.hostname or "").lower()
    if not _is_dashscope_host(host):
        return None
    path: Final = parsed.path.rstrip("/")
    if path and not any(path == prefix or path.startswith(f"{prefix}/") for prefix in _PROTOCOL_PATH_PREFIXES):
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def resolve_dashscope_endpoints(api_base: str | None) -> DashScopeEndpoints | None:
    """
    Derive DashScope's protocol-specific bases from any standard DashScope base.

    Accepts the domain root, an OpenAI-compatible base, an Anthropic Messages
    base, or a native `/api/v1` base for `dashscope.aliyuncs.com`,
    `dashscope-intl.aliyuncs.com`, `dashscope-us.aliyuncs.com`, and
    `{workspace}.{region}.maas.aliyuncs.com`. Returns None for any other host
    or path so callers keep the caller-supplied base untouched.
    """
    origin: Final = _dashscope_origin(api_base)
    if origin is None:
        return None
    return DashScopeEndpoints(
        origin=origin,
        openai=f"{origin}{DASHSCOPE_CHAT_COMPATIBLE_PATH}",
        anthropic=f"{origin}{DASHSCOPE_ANTHROPIC_PATH}",
        native=f"{origin}{DASHSCOPE_NATIVE_V1_PATH}",
    )


def is_dashscope_native_endpoint_url(api_base: str) -> bool:
    return urlparse(api_base).path.rstrip("/").startswith(f"{DASHSCOPE_NATIVE_V1_PATH}/")


def resolve_dashscope_openai_api_base(
    api_base: str | None,
    env_var: str = "DASHSCOPE_API_BASE",
    default: str = f"{DASHSCOPE_DEFAULT_ORIGIN}{DASHSCOPE_CHAT_COMPATIBLE_PATH}",
) -> str:
    candidate: Final = api_base or get_secret_str(env_var)
    if candidate and is_dashscope_native_endpoint_url(candidate):
        return candidate
    endpoints: Final = resolve_dashscope_endpoints(candidate)
    if endpoints is not None:
        return endpoints.openai
    return candidate or default


def resolve_dashscope_anthropic_api_base(
    api_base: str | None,
    env_var: str = "DASHSCOPE_API_BASE",
    default: str = f"{DASHSCOPE_DEFAULT_ORIGIN}{DASHSCOPE_ANTHROPIC_PATH}",
) -> str:
    candidate: Final = api_base or get_secret_str(env_var)
    endpoints: Final = resolve_dashscope_endpoints(candidate)
    if endpoints is not None:
        return endpoints.anthropic
    return candidate or default


def dashscope_base_selects_anthropic_messages(api_base: str | None) -> bool:
    candidate: Final = api_base or get_secret_str("DASHSCOPE_API_BASE")
    if not candidate:
        return False
    path: Final = urlparse(candidate).path.rstrip("/")
    if path.endswith(DASHSCOPE_ANTHROPIC_PATH) or f"{DASHSCOPE_ANTHROPIC_PATH}/" in path:
        return True
    return not path and resolve_dashscope_endpoints(candidate) is not None


def _chat_shaped_aliyun_origin(api_base: str | None) -> str | None:
    if api_base is None:
        return None
    parsed: Final = urlparse(api_base)
    host: Final = parsed.hostname or ""
    on_aliyun_host: Final = host == "aliyuncs.com" or host.endswith(".aliyuncs.com")
    if not on_aliyun_host or parsed.path.rstrip("/") != DASHSCOPE_CHAT_COMPATIBLE_PATH:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _is_compatible_api_base(api_base: str) -> bool:
    path: Final = urlparse(api_base).path.rstrip("/")
    return path == DASHSCOPE_COMPATIBLE_API_PATH or path.startswith(f"{DASHSCOPE_COMPATIBLE_API_PATH}/")


def resolve_dashscope_family_rerank_api_base(api_base: str | None, env_var: str, default_rerank_base: str) -> str:
    if api_base is None:
        return get_secret_str(env_var) or default_rerank_base
    if _is_compatible_api_base(api_base):
        return api_base
    chat_origin: Final = _chat_shaped_aliyun_origin(api_base)
    if chat_origin is not None:
        return get_secret_str(env_var) or f"{chat_origin}{DASHSCOPE_RERANK_PATH}"
    endpoints: Final = resolve_dashscope_endpoints(api_base)
    return f"{endpoints.origin}{DASHSCOPE_RERANK_PATH}" if endpoints is not None else api_base


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


def get_dashscope_family_video_config(
    custom_llm_provider: str,
) -> "BaseVideoConfig":
    if custom_llm_provider == "qwencloud":
        from litellm.llms.dashscope.qwencloud import QwenCloudVideoConfig

        return QwenCloudVideoConfig()
    if custom_llm_provider == "qwen_ai_platform":
        from litellm.llms.dashscope.qwen_ai_platform import QwenAIPlatformVideoConfig

        return QwenAIPlatformVideoConfig()
    from litellm.llms.dashscope.videos.transformation import DashScopeVideoConfig

    return DashScopeVideoConfig()


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
            "Missing API key for Qianwen AI Platform. Set QWEN_AI_PLATFORM_API_KEY or "
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
