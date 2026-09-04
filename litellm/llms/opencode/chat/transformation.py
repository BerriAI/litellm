"""
Translate from OpenAI's `/v1/chat/completions` to OpenCode Zen's and OpenCode Go's.

Both surfaces are OpenAI-compatible, so only the default base URL, the credential lookup and
the required `x-opencode-session` header differ from stock OpenAI.
"""

from collections.abc import Mapping, Sequence
from itertools import chain
from typing import ClassVar, Final

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ..common_utils import resolve_opencode_api_key, with_opencode_session_header


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
        return resolved_api_base, resolve_opencode_api_key(api_key)

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
        return with_opencode_session_header(base_headers, litellm_params)


class OpenCodeZenChatConfig(OpenCodeChatConfig):
    _provider: ClassVar[str] = "opencode"
    _default_api_base: ClassVar[str] = "https://opencode.ai/zen/v1"
    _api_base_env_var: ClassVar[str] = "OPENCODE_API_BASE"


class OpenCodeGoChatConfig(OpenCodeChatConfig):
    _provider: ClassVar[str] = "opencode_go"
    _default_api_base: ClassVar[str] = "https://opencode.ai/zen/go/v1"
    _api_base_env_var: ClassVar[str] = "OPENCODE_GO_API_BASE"


class OpenCodeMessagesChatConfig(AnthropicConfig):
    """
    OpenCode serves part of its catalogue (Claude, Qwen, MiniMax) on an Anthropic-shaped
    `/messages` endpoint that authenticates with `x-api-key` rather than a bearer token.
    """

    _provider: ClassVar[str]
    _default_api_base: ClassVar[str]
    _api_base_env_var: ClassVar[str]

    @property
    def custom_llm_provider(self) -> str | None:
        return self._provider

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        resolved_api_base: Final = (
            api_base or get_secret_str(self._api_base_env_var) or self._default_api_base
        ).rstrip("/")
        if resolved_api_base.endswith("/messages"):
            return resolved_api_base
        return f"{resolved_api_base}/messages"

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: return type is fixed by BaseConfig.validate_environment
        base_headers: Final[Mapping[str, object]] = super().validate_environment(
            headers=dict(headers),  # mutable-ok: AnthropicConfig.validate_environment needs a mutable dict
            model=model,
            messages=list(messages),  # mutable-ok: as above
            optional_params=dict(optional_params),  # mutable-ok: as above
            litellm_params=dict(litellm_params),  # mutable-ok: as above
            api_key=resolve_opencode_api_key(api_key),
            api_base=api_base,
        )
        return with_opencode_session_header(base_headers, litellm_params)


class OpenCodeZenMessagesChatConfig(OpenCodeMessagesChatConfig):
    _provider: ClassVar[str] = "opencode"
    _default_api_base: ClassVar[str] = "https://opencode.ai/zen/v1"
    _api_base_env_var: ClassVar[str] = "OPENCODE_API_BASE"


class OpenCodeGoMessagesChatConfig(OpenCodeMessagesChatConfig):
    _provider: ClassVar[str] = "opencode_go"
    _default_api_base: ClassVar[str] = "https://opencode.ai/zen/go/v1"
    _api_base_env_var: ClassVar[str] = "OPENCODE_GO_API_BASE"


class OpenCodeZenGeminiChatConfig(GoogleAIStudioGeminiConfig):
    """
    OpenCode Zen serves Google models on Gemini's own `models/<id>:generateContent` shape,
    hosted under the Zen base URL and authenticated with a bearer token.
    """

    _default_api_base: ClassVar[str] = "https://opencode.ai/zen/v1"
    _api_base_env_var: ClassVar[str] = "OPENCODE_API_BASE"

    @property
    def custom_llm_provider(self) -> str | None:
        return "opencode"

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        resolved_api_base: Final = (
            api_base or get_secret_str(self._api_base_env_var) or self._default_api_base
        ).rstrip("/")
        if stream:
            return f"{resolved_api_base}/models/{model}:streamGenerateContent?alt=sse"
        return f"{resolved_api_base}/models/{model}:generateContent"

    def transform_request(
        self,
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict:  # mutable-ok: return type is fixed by BaseConfig.transform_request
        """
        VertexGeminiConfig.transform_request raises NotImplementedError because Vertex builds its
        body in a bespoke handler, so reuse the shared Gemini body builder that handler calls.
        """
        from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body

        cached_content: Final = optional_params.get("cached_content")
        return _transform_request_body(
            messages=list(messages),  # mutable-ok: the Gemini body builder takes a mutable list
            model=model,
            optional_params=dict(optional_params),  # mutable-ok: as above
            custom_llm_provider="gemini",
            litellm_params=dict(litellm_params),  # mutable-ok: as above
            cached_content=cached_content if isinstance(cached_content, str) else None,
        )

    def validate_environment(
        self,
        headers: Mapping[str, object] | None,
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | dict | None = None,  # mutable-ok: signature is fixed by VertexGeminiConfig
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: return type is fixed by BaseConfig.validate_environment
        """
        OpenCode authenticates this endpoint with Google's `x-goog-api-key`; a bearer token,
        an `x-api-key` and a `?key=` query parameter all come back as `Missing API key`.
        """
        resolved_key: Final = resolve_opencode_api_key(api_key if isinstance(api_key, str) else None)
        base_headers: Final[Mapping[str, object]] = dict(
            chain(
                (("Content-Type", "application/json"),),
                headers.items() if headers else (),
                (("x-goog-api-key", resolved_key),),
            )
        )
        return with_opencode_session_header(base_headers, litellm_params)
