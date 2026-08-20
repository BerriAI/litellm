from typing import Any, Final

from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
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

        # Add beta headers for Vertex AI
        tools: Final = optional_params.get("tools", [])
        beta_values: Final[set[str]] = set()

        # Get existing beta headers if any
        existing_beta: Final = headers.get("anthropic-beta")
        if existing_beta:
            beta_values.update(b.strip() for b in existing_beta.split(","))

        # Check for context management
        context_management_param: Final = optional_params.get("context_management")
        if context_management_param is not None:
            # Check edits array for compact_20260112 type
            edits: Final = context_management_param.get("edits", [])
            has_compact = False
            has_other = False

            for edit in edits:
                edit_type = edit.get("type", "")
                if edit_type == "compact_20260112":
                    has_compact = True
                else:
                    has_other = True

            # Add compact header if any compact edits exist
            if has_compact:
                beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_01_12.value)

            # Add context management header if any other edits exist
            if has_other:
                beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value)

        # Check for web search tool
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type", "").startswith(ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value):
                beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.WEB_SEARCH_2025_03_05.value)
                break

        # Check for tool search tools - Vertex AI uses different beta header
        anthropic_model_info: Final = AnthropicModelInfo()
        if anthropic_model_info.is_tool_search_used(tools):
            beta_values.add(get_tool_search_beta_header("vertex_ai"))

        if beta_values:
            headers["anthropic-beta"] = ",".join(beta_values)

        return headers, api_base

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
