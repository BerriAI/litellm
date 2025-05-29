from typing import Any, List, Optional

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str

from ....vertex_llm_base import VertexBase


class VertexAIPartnerModelsAnthropicMessagesConfig(AnthropicMessagesConfig, VertexBase):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        OPTIONAL

        Validate the environment for the request
        """
        if "Authorization" not in headers:
            vertex_ai_project = (
                optional_params.get("vertex_project", None)
                or optional_params.get("vertex_ai_project", None)
                or litellm.vertex_project
                or get_secret_str("VERTEXAI_PROJECT")
            )
            vertex_credentials = (
                optional_params.pop("vertex_credentials", None)
                or optional_params.pop("vertex_ai_credentials", None)
                or get_secret_str("VERTEXAI_CREDENTIALS")
            )

            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_ai_project,
                custom_llm_provider="vertex_ai",
            )

            headers["Authorization"] = f"Bearer {access_token}"
            headers["anthropic-version"] = "vertex-2023-10-16"
        return headers
