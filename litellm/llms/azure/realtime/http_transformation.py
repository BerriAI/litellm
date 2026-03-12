"""Azure OpenAI realtime HTTP transformation config (client_secrets + realtime_calls)."""

from typing import Optional

import litellm
from litellm.llms.base_llm.realtime.http_transformation import BaseRealtimeHTTPConfig
from litellm.secret_managers.main import get_secret_str


class AzureRealtimeHTTPConfig(BaseRealtimeHTTPConfig):
    def get_api_base(self, api_base: Optional[str], **kwargs) -> str:
        return (
            api_base
            or litellm.api_base
            or get_secret_str("AZURE_API_BASE")
            or ""
        )

    def get_api_key(self, api_key: Optional[str], **kwargs) -> str:
        return (
            api_key
            or litellm.api_key
            or get_secret_str("AZURE_API_KEY")
            or ""
        )

    def get_complete_url(self, api_base: Optional[str], model: str, api_version: Optional[str] = None) -> str:
        base = self.get_api_base(api_base).rstrip("/")
        version = api_version or get_secret_str("AZURE_API_VERSION") or "2024-12-17"
        return f"{base}/openai/realtime/client_secrets?api-version={version}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        return {
            **headers,
            "api-key": api_key or "",
            "Content-Type": "application/json",
        }

    def get_realtime_calls_url(self, api_base: Optional[str], model: str, api_version: Optional[str] = None) -> str:
        base = self.get_api_base(api_base).rstrip("/")
        version = api_version or get_secret_str("AZURE_API_VERSION") or "2024-12-17"
        return f"{base}/openai/realtime/calls?api-version={version}"

    def get_realtime_calls_headers(self, ephemeral_key: str) -> dict:
        return {
            "api-key": ephemeral_key,
        }
