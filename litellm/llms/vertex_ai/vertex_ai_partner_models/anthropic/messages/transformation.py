from collections.abc import Mapping, Sequence
from typing import Final

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params, iter_items
from litellm.llms.anthropic.common_utils import (
    AnthropicModelInfo,
    context_management_betas,
    requires_native_compaction_beta,
)
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
    messages_carry_output_config,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_BETA_HEADER_VALUES,
    ANTHROPIC_HOSTED_TOOLS,
    ANTHROPIC_MID_CONVERSATION_TOOL_CHANGES_BETA_HEADER,
)
from litellm.types.llms.anthropic_tool_search import get_tool_search_beta_header
from litellm.types.llms.vertex_ai import VertexPartnerProvider
from litellm.types.router import GenericLiteLLMParams

from ....vertex_llm_base import VertexBase
from ..output_params_utils import sanitize_vertex_anthropic_output_params

_VERSION_HEADER_VERTEX_READS_FROM_BODY: Final = "anthropic-version"
_VERTEX_DEFAULT_VERSION: Final = "vertex-2023-10-16"
_VERSIONS_SERVED_AS_VERTEX_DEFAULT: Final = frozenset({"", "2023-01-01", "2023-06-01"})
_VERSIONS_VERTEX_ACCEPTS_IN_BODY: Final = frozenset({_VERTEX_DEFAULT_VERSION, "2026-10-21"})


def _requested_version(headers: Mapping[str, object]) -> str:
    requested: Final = next(
        (value for name, value in headers.items() if name.lower() == _VERSION_HEADER_VERTEX_READS_FROM_BODY),
        "",
    )
    return requested.strip() if isinstance(requested, str) else ""


def _vertex_version(headers: Mapping[str, object], model: str, drop_params: bool) -> str | UnsupportedParamsError:
    requested: Final = _requested_version(headers)
    if requested in _VERSIONS_SERVED_AS_VERTEX_DEFAULT:
        return _VERTEX_DEFAULT_VERSION
    if requested in _VERSIONS_VERTEX_ACCEPTS_IN_BODY:
        return requested
    if drop_params:
        return _VERTEX_DEFAULT_VERSION
    return UnsupportedParamsError(
        message=(
            f"{model} does not support {_VERSION_HEADER_VERTEX_READS_FROM_BODY}={requested}. "
            "To drop unsupported params, set `litellm.drop_params = True`."
        ),
        llm_provider="vertex_ai",
        model=model,
    )


def _is_web_search(tool: Mapping[str, object]) -> bool:
    tool_type: Final = tool.get("type")
    return isinstance(tool_type, str) and tool_type.startswith(ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value)


def _without_version_header(headers: Mapping[str, str]) -> dict[str, str]:  # mutable-ok: the caller sends these headers
    return {name: value for name, value in headers.items() if name.lower() != _VERSION_HEADER_VERTEX_READS_FROM_BODY}


class VertexAIPartnerModelsAnthropicMessagesConfig(AnthropicMessagesConfig, VertexBase):
    @property
    def custom_llm_provider(self) -> str | None:
        return "vertex_ai"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def _vertex_auth_parameters(
        litellm_params: dict[str, object],  # mutable-ok: delegates to the existing Vertex parameter readers
    ) -> tuple[str | None, str | None, str | None]:
        return (
            VertexBase.safe_get_vertex_ai_project(litellm_params),
            VertexBase.safe_get_vertex_ai_location(litellm_params),
            VertexBase.safe_get_vertex_ai_credentials(litellm_params),
        )

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: Sequence[object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:
        """
        OPTIONAL

        Validate the environment for the request
        """
        vertex_ai_project, vertex_ai_location, vertex_credentials = self._vertex_auth_parameters(litellm_params)
        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_ai_project,
            custom_llm_provider="vertex_ai",
        )
        vertex_headers, vertex_api_base = self._finalize_vertex_messages_environment(
            headers,
            model,
            optional_params,
            api_base,
            access_token,
            project_id,
            vertex_ai_project,
            vertex_ai_location,
        )
        return dict(self.headers_with_anthropic_betas(vertex_headers, optional_params, messages)), vertex_api_base

    async def avalidate_anthropic_messages_environment(
        self,
        headers: dict[str, str],  # mutable-ok: mirrors the provider validation contract
        model: str,
        messages: Sequence[object],
        optional_params: dict[str, object],  # mutable-ok: mirrors the provider validation contract
        litellm_params: dict[str, object],  # mutable-ok: mirrors the provider validation contract
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: the caller consumes request headers
        vertex_ai_project, vertex_ai_location, vertex_credentials = self._vertex_auth_parameters(litellm_params)
        access_token, project_id = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_ai_project,
            custom_llm_provider="vertex_ai",
        )
        vertex_headers, vertex_api_base = self._finalize_vertex_messages_environment(
            headers,
            model,
            optional_params,
            api_base,
            access_token,
            project_id,
            vertex_ai_project,
            vertex_ai_location,
        )
        return dict(self.headers_with_anthropic_betas(vertex_headers, optional_params, messages)), vertex_api_base

    def _finalize_vertex_messages_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        optional_params: dict,  # mutable-ok: preserves the existing provider optional-parameter contract
        api_base: str | None,
        access_token: str,
        project_id: str,
        vertex_ai_project: str | None,
        vertex_ai_location: str | None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: the caller consumes request headers
        # Router shallow-copies litellm_params, so headers may be shared between requests.
        request_headers: Final = dict(headers)
        request_headers["Authorization"] = f"Bearer {access_token}"

        api_base = self.get_complete_vertex_url(
            custom_api_base=api_base,
            vertex_location=vertex_ai_location,
            vertex_project=vertex_ai_project,
            project_id=project_id or "",
            partner=VertexPartnerProvider.claude,
            stream=optional_params.get("stream", False),
            model=model,
        )

        request_headers["content-type"] = "application/json"

        return request_headers, api_base

    def headers_with_anthropic_betas(
        self,
        headers: Mapping[str, str],
        optional_params: Mapping[str, object],
        messages: Sequence[object],
    ) -> Mapping[str, str]:
        tools: Final = tuple(
            tool for tool in map(as_str_mapping, iter_items(optional_params.get("tools") or ())) if tool is not None
        )
        anthropic_model_info: Final = AnthropicModelInfo()
        existing_betas: Final = (headers.get("anthropic-beta") or "").split(",")
        beta_values: Final = frozenset(
            (
                *(beta.strip() for beta in existing_betas if beta.strip()),
                *context_management_betas(optional_params.get("context_management")),  # pyright: ignore[reportArgumentType]  # optional_params is the config contract's untyped request body
                *(
                    (ANTHROPIC_BETA_HEADER_VALUES.WEB_SEARCH_2025_03_05.value,)
                    if any(map(_is_web_search, tools))
                    else ()
                ),
                *(
                    (get_tool_search_beta_header("vertex_ai"),)
                    if anthropic_model_info.is_tool_search_used(list(tools))
                    else ()
                ),
                *(
                    (ANTHROPIC_BETA_HEADER_VALUES.DANGEROUS_TOOL_USE_2026_09_03.value,)
                    if optional_params.get("safeguards") is not None
                    else ()
                ),
                *(
                    (ANTHROPIC_BETA_HEADER_VALUES.PER_TURN_CONTROL_2026_07_01.value,)
                    if messages_carry_output_config(messages)
                    else ()
                ),
                *(
                    (ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_09_04.value,)
                    if requires_native_compaction_beta("vertex_ai", optional_params, messages)
                    else ()
                ),
                *(
                    (ANTHROPIC_MID_CONVERSATION_TOOL_CHANGES_BETA_HEADER,)
                    if anthropic_model_info.is_mid_conversation_tool_change_used(messages)
                    else ()
                ),
                *anthropic_model_info.thinking_betas(
                    optional_params.get("thinking"),  # pyright: ignore[reportArgumentType]  # optional_params is the config contract's untyped request body
                ),
            )
        )
        if not beta_values:
            return headers
        return {**headers, "anthropic-beta": ",".join(sorted(beta_values))}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if api_base is None:
            raise ValueError("api_base is required. Unable to determine the correct api_base for the request.")
        return api_base  # no transformation is needed - handled in validate_environment

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, object],  # mutable-ok: mirrors the provider transform contract
    ) -> dict:
        anthropic_messages_request: Final = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        self._normalize_system_role_messages(
            anthropic_messages_request, model=model, drop_params=litellm_params.drop_params
        )

        self._remove_scope_from_cache_control(anthropic_messages_request)

        version: Final = _vertex_version(
            headers, model=model, drop_params=drops_unsupported_params(litellm_params.drop_params)
        )
        if isinstance(version, UnsupportedParamsError):
            raise version
        anthropic_messages_request["anthropic_version"] = version

        anthropic_messages_request.pop("model", None)  # do not pass model in request body to vertex ai

        sanitize_vertex_anthropic_output_params(anthropic_messages_request, model)

        return anthropic_messages_request

    def sign_request(
        self,
        headers: Mapping[str, str],
        optional_params: Mapping[str, object],
        request_data: Mapping[str, object],
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict[str, str], bytes | None]:  # mutable-ok: the caller sends these headers
        return _without_version_header(headers), None
