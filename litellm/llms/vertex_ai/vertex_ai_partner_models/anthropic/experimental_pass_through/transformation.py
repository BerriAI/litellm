from collections.abc import Mapping, Sequence
from typing import Any, Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping, iter_items
from litellm.llms.anthropic.common_utils import AnthropicModelInfo, context_management_betas
from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
    messages_carry_output_config,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_BETA_HEADER_VALUES,
    ANTHROPIC_HOSTED_TOOLS,
)
from litellm.types.llms.anthropic_tool_search import get_tool_search_beta_header
from litellm.types.llms.vertex_ai import VertexPartnerProvider
from litellm.types.router import GenericLiteLLMParams

from ....vertex_llm_base import VertexBase
from ..output_params_utils import sanitize_vertex_anthropic_output_params


def _is_web_search(tool: Mapping[str, object]) -> bool:
    tool_type: Final = tool.get("type")
    return isinstance(tool_type, str) and tool_type.startswith(ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value)


class VertexAIPartnerModelsAnthropicMessagesConfig(AnthropicMessagesConfig, VertexBase):
    @property
    def custom_llm_provider(self) -> str | None:
        return "vertex_ai"

    def should_strip_billing_metadata(self) -> bool:
        return True

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: list[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:
        """
        OPTIONAL

        Validate the environment for the request
        """
        # Work on a local copy — router shallow-copies litellm_params so the caller's
        # headers dict may be the shared deployment extra_headers object.
        headers = dict(headers)
        vertex_ai_project: Final = VertexBase.safe_get_vertex_ai_project(litellm_params)
        vertex_ai_location: Final = VertexBase.safe_get_vertex_ai_location(litellm_params)

        vertex_credentials: Final = VertexBase.safe_get_vertex_ai_credentials(litellm_params)
        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_ai_project,
            custom_llm_provider="vertex_ai",
        )
        headers["Authorization"] = f"Bearer {access_token}"

        api_base = self.get_complete_vertex_url(
            custom_api_base=api_base,
            vertex_location=vertex_ai_location,
            vertex_project=vertex_ai_project,
            project_id=project_id or "",
            partner=VertexPartnerProvider.claude,
            stream=optional_params.get("stream", False),
            model=model,
        )

        headers["content-type"] = "application/json"

        return dict(self.headers_with_anthropic_betas(headers, optional_params, messages)), api_base

    def headers_with_anthropic_betas(
        self,
        headers: Mapping[str, str],
        optional_params: Mapping[str, object],
        messages: Sequence[object],
    ) -> Mapping[str, str]:
        tools: Final = tuple(
            tool for tool in map(as_str_mapping, iter_items(optional_params.get("tools") or ())) if tool is not None
        )
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
                    if AnthropicModelInfo().is_tool_search_used(list(tools))
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
        headers: dict,
    ) -> dict:
        anthropic_messages_request: Final = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        self._normalize_system_role_messages(anthropic_messages_request, model=model)

        self._remove_scope_from_cache_control(anthropic_messages_request)

        anthropic_messages_request["anthropic_version"] = "vertex-2023-10-16"

        anthropic_messages_request.pop("model", None)  # do not pass model in request body to vertex ai

        sanitize_vertex_anthropic_output_params(anthropic_messages_request, model)

        return anthropic_messages_request
