from typing import Any, Dict, List, Optional, Tuple

from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_anthropic_image_obj,
)
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


class VertexAIPartnerModelsAnthropicMessagesConfig(AnthropicMessagesConfig, VertexBase):
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
        vertex_ai_project = VertexBase.safe_get_vertex_ai_project(litellm_params)
        vertex_ai_location = VertexBase.safe_get_vertex_ai_location(litellm_params)

        project_id: Optional[str] = None
        if "Authorization" not in headers:
            vertex_credentials = VertexBase.safe_get_vertex_ai_credentials(
                litellm_params
            )

            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_ai_project,
                custom_llm_provider="vertex_ai",
            )

            headers["Authorization"] = f"Bearer {access_token}"
        else:
            # Authorization already in headers, but we still need project_id
            project_id = vertex_ai_project

        # Always calculate api_base if not provided, regardless of Authorization header
        if api_base is None:
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
                beta_values.add(
                    ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value
                )

        # Check for web search tool
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type", "").startswith(
                ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value
            ):
                beta_values.add(
                    ANTHROPIC_BETA_HEADER_VALUES.WEB_SEARCH_2025_03_05.value
                )
                break

        # Check for tool search tools - Vertex AI uses different beta header
        anthropic_model_info = AnthropicModelInfo()
        if anthropic_model_info.is_tool_search_used(tools):
            beta_values.add(get_tool_search_beta_header("vertex_ai"))

        if beta_values:
            headers["anthropic-beta"] = ",".join(beta_values)

        return headers, api_base

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
            raise ValueError(
                "api_base is required. Unable to determine the correct api_base for the request."
            )
        return api_base  # no transformation is needed - handled in validate_environment

    @staticmethod
    def _convert_image_urls_to_base64(messages: List[Dict]) -> List[Dict]:
        """
        Convert image URL sources to base64 format for Vertex AI.

        Vertex AI Anthropic does not support URL sources for images.
        This method converts:
        {"type": "image", "source": {"type": "url", "url": "https://..."}}
        to:
        {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        """
        converted_messages = []
        for message in messages:
            if not isinstance(message, dict):
                converted_messages.append(message)
                continue

            content = message.get("content")
            if not isinstance(content, list):
                converted_messages.append(message)
                continue

            new_content = []
            for block in content:
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue

                # Check if this is an image block with URL source
                if block.get("type") == "image":
                    source = block.get("source", {})
                    if isinstance(source, dict) and source.get("type") == "url":
                        url = source.get("url")
                        if url:
                            # Convert URL to base64 using existing utility
                            image_obj = convert_to_anthropic_image_obj(
                                openai_image_url=url, format=None
                            )
                            # Preserve all original block fields (e.g., cache_control)
                            # while replacing the source
                            new_block = {
                                **block,
                                "source": {
                                    "type": image_obj["type"],
                                    "media_type": image_obj["media_type"],
                                    "data": image_obj["data"],
                                },
                            }
                            new_content.append(new_block)
                            continue

                new_content.append(block)

            new_message = {**message, "content": new_content}
            converted_messages.append(new_message)

        return converted_messages

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        # Convert image URLs to base64 for Vertex AI
        # Vertex AI Anthropic does not support URL sources for images
        converted_messages = self._convert_image_urls_to_base64(messages)

        anthropic_messages_request = super().transform_anthropic_messages_request(
            model=model,
            messages=converted_messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        anthropic_messages_request["anthropic_version"] = "vertex-2023-10-16"

        anthropic_messages_request.pop(
            "model", None
        )  # do not pass model in request body to vertex ai

        anthropic_messages_request.pop(
            "output_format", None
        )  # do not pass output_format in request body to vertex ai - vertex ai does not support output_format as yet

        anthropic_messages_request.pop(
            "output_config", None
        )  # do not pass output_config in request body to vertex ai - vertex ai does not support output_config

        return anthropic_messages_request
