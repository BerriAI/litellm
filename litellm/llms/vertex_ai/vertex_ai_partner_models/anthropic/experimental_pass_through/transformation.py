from typing import Any, Dict, List, Optional, Tuple

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

_VERTEX_SESSION_AFFINITY_HEADER = "X-Vertex-Ai-Session-Id"


class VertexAIPartnerModelsAnthropicMessagesConfig(AnthropicMessagesConfig, VertexBase):
    def should_strip_billing_metadata(self) -> bool:
        return True

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """
        OPTIONAL

        Validate the environment for the request
        """
        # Work on a local copy — router shallow-copies litellm_params so the caller's
        # headers dict may be the shared deployment extra_headers object.
        headers = dict(headers)
        vertex_ai_project = VertexBase.safe_get_vertex_ai_project(litellm_params)
        vertex_ai_location = VertexBase.safe_get_vertex_ai_location(litellm_params)

        vertex_credentials = VertexBase.safe_get_vertex_ai_credentials(litellm_params)
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
        tools = optional_params.get("tools", [])
        beta_values: set[str] = set()

        # Get existing beta headers if any
        existing_beta = headers.get("anthropic-beta")
        if existing_beta:
            beta_values.update(b.strip() for b in existing_beta.split(","))

        # Check for context management
        context_management_param = optional_params.get("context_management")
        if context_management_param is not None:
            # Check edits array for compact_20260112 type
            edits = context_management_param.get("edits", [])
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
        anthropic_model_info = AnthropicModelInfo()
        if anthropic_model_info.is_tool_search_used(tools):
            beta_values.add(get_tool_search_beta_header("vertex_ai"))

        if beta_values:
            headers["anthropic-beta"] = ",".join(beta_values)

        # Vertex's global endpoint can route turns of one logical session to
        # different in-memory cache servers, rebuilding the Anthropic prompt
        # cache each turn. A stable X-Vertex-Ai-Session-Id pins the session to
        # one server so the cache survives across turns.
        self._ensure_session_affinity_header(headers, litellm_params)

        return headers, api_base

    @staticmethod
    def _resolve_session_affinity_id(litellm_params: Dict[str, Any]) -> Optional[str]:
        # The proxy pre-call already resolves a session/trace id from request
        # headers (see get_chain_id_from_headers) into litellm_session_id, so we
        # reuse that instead of re-scanning headers here.
        litellm_session_id = litellm_params.get("litellm_session_id")
        if litellm_session_id:
            return str(litellm_session_id)
        for metadata_key in ("litellm_metadata", "metadata"):
            metadata = litellm_params.get(metadata_key)
            if isinstance(metadata, dict) and metadata.get("session_id"):
                return str(metadata["session_id"])
        return None

    @classmethod
    def _ensure_session_affinity_header(cls, headers: Dict[str, Any], litellm_params: Dict[str, Any]) -> None:
        if any(key.lower() == _VERTEX_SESSION_AFFINITY_HEADER.lower() for key in headers):
            return
        session_id = cls._resolve_session_affinity_id(litellm_params)
        if session_id:
            headers[_VERTEX_SESSION_AFFINITY_HEADER] = session_id

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            raise ValueError("api_base is required. Unable to determine the correct api_base for the request.")
        return api_base  # no transformation is needed - handled in validate_environment

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        anthropic_messages_request = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        self._remove_scope_from_cache_control(anthropic_messages_request)

        anthropic_messages_request["anthropic_version"] = "vertex-2023-10-16"

        anthropic_messages_request.pop("model", None)  # do not pass model in request body to vertex ai

        sanitize_vertex_anthropic_output_params(anthropic_messages_request, model)

        return anthropic_messages_request
