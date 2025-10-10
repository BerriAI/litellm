from typing import Any, Dict, List, Optional, Tuple

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
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
        if "Authorization" not in headers:
            vertex_ai_project = VertexBase.get_vertex_ai_project(litellm_params)
            vertex_credentials = VertexBase.get_vertex_ai_credentials(litellm_params)
            vertex_ai_location = VertexBase.get_vertex_ai_location(litellm_params)

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
                project_id=project_id,
                partner=VertexPartnerProvider.claude,
                stream=optional_params.get("stream", False),
                model=model,
            )

        headers["content-type"] = "application/json"
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

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        messages = self.process_image_content(messages)
        anthropic_messages_request = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        anthropic_messages_request["anthropic_version"] = "vertex-2023-10-16"

        anthropic_messages_request.pop(
            "model", None
        )  # do not pass model in request body to vertex ai
        return anthropic_messages_request

    def process_image_content(self, messages: List[Dict]) -> List[Dict]:
        from litellm.litellm_core_utils.prompt_templates.image_handling import convert_url_to_base64
        for message in messages:
            if message.get("role") == "user" and isinstance(message.get("content"), list):
                for content in message["content"]:
                    if content.get("type") == "image":
                        source = content.get("source", {})
                        # Check if source has a URL that needs conversion
                        if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
                            try:
                                base64_data = convert_url_to_base64(source)
                                # Extract media type from the base64 data URL
                                if base64_data.startswith("data:"):
                                    media_type = base64_data.split(";")[0].split(":")[1]
                                    base64_content = base64_data.split(",")[1]
                                else:
                                    media_type = "image/jpeg"  # Default fallback
                                    base64_content = base64_data
                                
                                # Convert to base64 format
                                content["source"] = {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_content
                                }
                            except Exception:
                                # If conversion fails, keep original content
                                pass
                        elif isinstance(source, dict) and source.get("type") == "url":
                            # Handle URL object format
                            url = source.get("url", "")
                            if url.startswith("http://") or url.startswith("https://"):
                                try:
                                    base64_data = convert_url_to_base64(url)
                                    # Extract media type from the base64 data URL
                                    if base64_data.startswith("data:"):
                                        media_type = base64_data.split(";")[0].split(":")[1]
                                        base64_content = base64_data.split(",")[1]
                                    else:
                                        media_type = "image/jpeg"  # Default fallback
                                        base64_content = base64_data
                                    
                                    # Convert to base64 format
                                    content["source"] = {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_content
                                    }
                                except Exception:
                                    # If conversion fails, keep original content
                                    pass
        return messages