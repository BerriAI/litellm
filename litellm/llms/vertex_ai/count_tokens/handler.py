from collections.abc import Mapping, Sequence
from typing import Any, Final

from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


class VertexAITokenCounter(GoogleAIStudioTokenCounter, VertexBase):
    def _build_count_tokens_request_body(
        self,
        model: str,
        contents: object,
        system_instruction: Mapping[str, object] | None,
        tools: Sequence[Mapping[str, object]] | None,
    ) -> dict[str, object]:
        return {
            "contents": contents,
            **({"systemInstruction": dict(system_instruction)} if system_instruction else {}),
            **({"tools": tools} if tools else {}),
        }

    async def validate_environment(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        headers: dict[str, Any] | None = None,
        model: str = "",
        litellm_params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """
        Returns a Tuple of headers and url for the Vertex AI countTokens endpoint.
        """
        resolved_litellm_params: Final = litellm_params or {}
        vertex_credentials: Final = self.get_vertex_ai_credentials(litellm_params=resolved_litellm_params)
        vertex_project: Final = self.get_vertex_ai_project(litellm_params=resolved_litellm_params)
        vertex_location: Final = self.get_vertex_ai_location(litellm_params=resolved_litellm_params)
        _auth_header, resolved_vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        auth_header, url = self._get_token_and_url(
            model=model,
            gemini_api_key=None,
            auth_header=_auth_header,
            vertex_project=resolved_vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider="vertex_ai",
            api_base=api_base,
            mode="count_tokens",
        )
        headers = {
            "Authorization": f"Bearer {auth_header}",
        }
        return headers, url
