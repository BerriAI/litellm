"""Direct-connect to SAP AI Core Gemini foundation-model deployments on `:generateContent`.

The Gemini sibling of `SapDeploymentAnthropicChatConfig` (Bedrock invoke) and `SapDeploymentOpenAIChatConfig`
(Azure OpenAI chat). Gemini on SAP is served by a Vertex executable that speaks Google's `generateContent`
contract at `{deployment_url}/models/{model}:generateContent` (and `:streamGenerateContent?alt=sse` for SSE),
so this config reuses `VertexGeminiConfig`'s request-body builder, response parser, and streaming iterator and
swaps auth and URL for SAP direct connect.

`VertexGeminiConfig` cannot be wrapped as cleanly as the OpenAI config: its `transform_request` raises
`NotImplementedError` because the native Vertex path splits body construction across sync and async context
caching. SAP direct connect does not do context caching, so we call the lower-level `_transform_request_body`
(the piece both Vertex paths share, no Google auth) directly with `cached_content=None`. The inherited
`transform_response` already parses `GenerateContentResponseBody` with no Google dependency, and the inherited
`ModelResponseIterator` parses the SSE frames; the generic httpx `get_model_response_iterator` hook does not
carry a `logging_obj`, and the iterator reads it only for legacy-`functions` detection that direct-connect
passthrough never uses, so a stub carrying empty `optional_params` is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator,
    VertexGeminiConfig,
)
from litellm.types.llms.openai import AllMessageValues

from ..direct_connect import (
    build_sap_auth,
    get_token_creator,
    resolve_sap_deployment_url,
    resource_group_from_params,
    without_resource_group,
)
from ..submode import split_sap_submode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.types.utils import ModelResponse

    from ..direct_connect import TokenCreatorFactory

_GENERATE_CONTENT_PATH: Final = "models/{model}:generateContent"
_STREAM_GENERATE_CONTENT_PATH: Final = "models/{model}:streamGenerateContent?alt=sse"
_VERTEX_PROVIDER: Final = "vertex_ai"


@dataclass(frozen=True, slots=True)
class _StreamLoggingStub:
    """Minimal stand-in for the logging object Gemini's `ModelResponseIterator` reads `optional_params` off.

    The generic httpx `get_model_response_iterator` hook has no `logging_obj`, and the iterator only touches
    `optional_params` to detect the deprecated OpenAI `functions` API. Direct-connect passthrough never sets it,
    so empty params give the correct modern-`tools` behavior.
    """

    optional_params: Mapping[str, object]


_STREAM_LOGGING_STUB: Final = _StreamLoggingStub(MappingProxyType({}))


class SapDeploymentGeminiChatConfig(VertexGeminiConfig):
    """Reuse the Vertex Gemini transforms and SSE parsing; swap auth and URL for SAP direct connect."""

    def __init__(
        self,
        *,
        http_client: HTTPHandler | None = None,
        token_creator_factory: TokenCreatorFactory = get_token_creator,
    ) -> None:
        super().__init__()
        self._http_client: Final = http_client
        self._token_creator_factory: Final = token_creator_factory

    @property
    def custom_llm_provider(self) -> str | None:
        return "sap"

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        return False

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: litellm base config override return
        return super().get_supported_openai_params(split_sap_submode(model)[1])

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: litellm base transform override signature
        optional_params: dict,  # mutable-ok: litellm base transform override signature
        litellm_params: dict,  # mutable-ok: litellm base transform override signature
        headers: dict,  # mutable-ok: litellm base transform override signature
    ) -> dict:  # mutable-ok: litellm base transform override signature
        _, bare_model = split_sap_submode(model)
        return dict(  # mutable-ok: litellm request payload; framework mutates it downstream
            _transform_request_body(
                messages=messages,
                model=bare_model,
                optional_params=without_resource_group(optional_params),
                custom_llm_provider=_VERTEX_PROVIDER,
                litellm_params=litellm_params,
                cached_content=None,
            )
        )

    def validate_environment(
        self,
        headers: dict | None,  # mutable-ok: litellm base transform override signature
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: litellm base transform override signature
        optional_params: dict,  # mutable-ok: litellm base transform override signature
        litellm_params: dict,  # mutable-ok: litellm base transform override signature
        api_key: str | dict | None = None,  # mutable-ok: litellm base transform override signature
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: litellm base transform override signature
        sap_headers, _, _ = build_sap_auth(
            api_key if isinstance(api_key, str) else None,
            self._token_creator_factory,
            headers or {},  # mutable-ok: SAP request headers dict
            resource_group_from_params(litellm_params),
        )
        return sap_headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: litellm base transform override signature
        litellm_params: dict,  # mutable-ok: litellm base transform override signature
        stream: bool | None = None,
    ) -> str:
        _, bare_model = split_sap_submode(model)
        if api_base:
            return self._content_url(api_base, bare_model, stream)
        sap_headers, base_url, resource_group = build_sap_auth(
            api_key, self._token_creator_factory, resource_group=resource_group_from_params(litellm_params)
        )
        deployment_url: Final = resolve_sap_deployment_url(
            base_url=base_url,
            resource_group=resource_group,
            model=bare_model,
            headers=sap_headers,
            http_client=self._http_client,
            api_base=None,
        )
        return self._content_url(deployment_url, bare_model, stream)

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> ModelResponseIterator:
        iterator: Final = ModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            logging_obj=_STREAM_LOGGING_STUB,  # pyright: ignore[reportArgumentType]  # hook carries no logging_obj; iterator reads only optional_params
        )
        # CustomStreamWrapper calls next()/anext() straight on this without priming, but the Vertex
        # iterator only binds its inner stream in __iter__/__aiter__, so prime it here.
        return iterator.__iter__() if sync_stream else iterator.__aiter__()

    @staticmethod
    def _content_url(base: str, model: str, stream: bool | None) -> str:
        suffix: Final = _STREAM_GENERATE_CONTENT_PATH if stream else _GENERATE_CONTENT_PATH
        return f"{base}/{suffix.format(model=model)}"
