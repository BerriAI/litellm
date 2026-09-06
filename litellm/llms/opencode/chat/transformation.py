"""
OpenCode chat-completions config.

Routes models to {base}/v1/chat/completions with Bearer auth.
Surface ('zen' | 'go') determines the default base URL.
"""

from collections.abc import Mapping
from typing import Any, Final  # noqa: TID251  # OpenAI chat completions wire format uses Any in param/return shapes

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIGPTConfig,
)
from litellm.llms.opencode.common_utils import (
    OpenCodeException,
    resolve_opencode_api_base,
    resolve_opencode_api_key,
)
from litellm.types.llms.openai import AllMessageValues

ZEN_BASE: Final = "https://opencode.ai/zen/v1"
GO_BASE: Final = "https://opencode.ai/zen/go/v1"


class OpenCodeConfig(OpenAIGPTConfig):
    """OpenAI chat-completions config for the OpenCode gateway."""

    def __init__(self, surface: str = "zen") -> None:
        self.surface: Final = surface

    @property
    def custom_llm_provider(self) -> str:
        return f"opencode_{self.surface}"

    def _base_url(self) -> str:
        return GO_BASE if self.surface == "go" else ZEN_BASE

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature must match OpenAIGPTConfig
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: signature must match OpenAIGPTConfig
        optional_params: dict,  # mutable-ok: signature must match OpenAIGPTConfig
        litellm_params: dict,  # mutable-ok: signature must match OpenAIGPTConfig
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: signature must match OpenAIGPTConfig
        """
        Resolve api_key and api_base, inject Bearer auth into headers.

        Key resolution is shared with the other arms; see
        :func:`resolve_opencode_api_key`.
        """
        key: Final = resolve_opencode_api_key(self.surface, api_key)

        auth_header: Final = headers.get("Authorization")
        content_type: Final = headers.get("Content-Type") or headers.get("content-type")

        if key is not None and auth_header is None:
            headers["Authorization"] = f"Bearer {key}"  # rebind-ok: caller expects auth header injected
        if content_type is None:
            headers["Content-Type"] = "application/json"  # rebind-ok: caller expects content-type set

        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, Any],
        litellm_params: Mapping[str, Any],
        stream: bool | None = None,
    ) -> str:
        """Return {api_base}/v1/chat/completions."""
        base: Final = resolve_opencode_api_base(self.surface, api_base) or self._base_url()
        return f"{base.rstrip('/')}/chat/completions"

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: signature must match OpenAIGPTConfig
    ) -> BaseLLMException:
        return OpenCodeException(message=error_message, status_code=status_code, headers=headers)
