"""
Translate from OpenAI's `/v1/chat/completions` to OpenCode Zen's and OpenCode Go's.

Both surfaces are OpenAI-compatible, so only the default base URL, the credential lookup and
the required `x-opencode-session` header differ from stock OpenAI.
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar, Final

from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ..common_utils import (
    OPENCODE_SESSION_HEADER,
    has_opencode_session_header,
    resolve_opencode_session_id,
)


class OpenCodeChatConfig(OpenAILikeChatConfig):
    _provider: ClassVar[str]
    _default_api_base: ClassVar[str]
    _api_base_env_var: ClassVar[str]

    @property
    def custom_llm_provider(self) -> str | None:
        return self._provider

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = api_base or get_secret_str(self._api_base_env_var) or self._default_api_base
        resolved_api_key: Final = (
            api_key or get_secret_str("OPENCODE_API_KEY") or get_secret_str("OPENCODE_ZEN_API_KEY")
        )
        return resolved_api_base, resolved_api_key

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: return type is fixed by BaseConfig.validate_environment, whose callers mutate it
        base_headers: Final[Mapping[str, object]] = super().validate_environment(
            headers=dict(headers),  # mutable-ok: BaseConfig.validate_environment only accepts mutable dicts
            model=model,
            messages=list(messages),  # mutable-ok: BaseConfig.validate_environment only accepts a mutable list
            optional_params=dict(optional_params),  # mutable-ok: as above
            litellm_params=dict(litellm_params),  # mutable-ok: as above
            api_key=api_key,
            api_base=api_base,
        )
        session_id: Final = (
            None if has_opencode_session_header(base_headers) else resolve_opencode_session_id(litellm_params)
        )
        if session_id is None:
            return dict(base_headers)
        return {**base_headers, OPENCODE_SESSION_HEADER: session_id}


class OpenCodeZenChatConfig(OpenCodeChatConfig):
    _provider: ClassVar[str] = "opencode"
    _default_api_base: ClassVar[str] = "https://opencode.ai/zen/v1"
    _api_base_env_var: ClassVar[str] = "OPENCODE_API_BASE"


class OpenCodeGoChatConfig(OpenCodeChatConfig):
    _provider: ClassVar[str] = "opencode_go"
    _default_api_base: ClassVar[str] = "https://opencode.ai/zen/go/v1"
    _api_base_env_var: ClassVar[str] = "OPENCODE_GO_API_BASE"
