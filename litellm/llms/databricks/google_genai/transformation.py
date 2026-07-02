"""
Databricks Unity AI Gateway — native Google Gemini generateContent config.

Routes ``litellm.google_genai.generate_content(model="databricks/<gemini-endpoint>")``
to the gateway's native Gemini surface:

    https://<workspace-host>/ai-gateway/gemini/v1beta/models/<endpoint>:generateContent

The native Gemini request/response transforms are inherited from the Google AI
Studio :class:`GoogleGenAIConfig` (the gateway speaks the native Gemini wire
format). Only auth and URL construction are overridden: auth reuses the shared
Databricks credential resolution (M2M / PAT / SDK unified) emitting
``Authorization: Bearer <token>``, and the URL targets the gateway gemini path
(this surface exists only on the AI Gateway).
"""

from typing import Optional

from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig
from litellm.types.router import GenericLiteLLMParams

from ..ai_gateway import build_gemini_generate_content_url
from ..common_utils import DatabricksBase


class DatabricksGoogleGenAIConfig(DatabricksBase, GoogleGenAIConfig):
    def __init__(self) -> None:
        GoogleGenAIConfig.__init__(self)

    def validate_environment(
        self,
        api_key: Optional[str],
        headers: Optional[dict],
        model: str,
        litellm_params: Optional[GenericLiteLLMParams | dict],
    ) -> dict:
        params = dict(litellm_params or {})
        resolved_key = api_key or params.get("api_key") or params.get("databricks_key")
        custom_user_agent = params.get("user_agent") or params.get("databricks_user_agent")

        _, resolved_headers = self.databricks_resolve_auth(
            api_key=resolved_key,
            api_base=params.get("api_base"),
            custom_endpoint=False,
            headers=dict(headers or {}),
            custom_user_agent=custom_user_agent,
            databricks_profile=self.resolve_databricks_profile(params),
        )
        resolved_headers["Content-Type"] = "application/json"
        resolved_headers = self.apply_request_tags_header(resolved_headers, litellm_params=params)
        return resolved_headers

    def sync_get_auth_token_and_url(
        self,
        api_base: Optional[str],
        model: str,
        litellm_params: dict,
        stream: bool,
    ) -> tuple[dict, str]:
        headers = self.validate_environment(
            api_key=litellm_params.get("api_key"),
            headers=None,
            model=model,
            litellm_params=litellm_params,
        )
        url = build_gemini_generate_content_url(
            self._get_api_base(api_base or litellm_params.get("api_base")),
            model=model,
            stream=stream,
        )
        return headers, url

    async def get_auth_token_and_url(
        self,
        api_base: Optional[str],
        model: str,
        litellm_params: dict,
        stream: bool,
    ) -> tuple[dict, str]:
        # Databricks auth (bearer token / SDK) is synchronous; reuse the sync path.
        return self.sync_get_auth_token_and_url(
            api_base=api_base,
            model=model,
            litellm_params=litellm_params,
            stream=stream,
        )
