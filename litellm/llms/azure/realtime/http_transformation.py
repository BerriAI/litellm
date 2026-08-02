"""Azure OpenAI realtime HTTP transformation config (client_secrets + realtime_calls)."""

from typing import Final

import litellm
from litellm.constants import AZURE_GA_REALTIME_MODELS
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.base_llm.realtime.http_transformation import BaseRealtimeHTTPConfig
from litellm.secret_managers.main import get_secret_str


class AzureRealtimeHTTPConfig(BaseRealtimeHTTPConfig):
    @staticmethod
    def _uses_ga_api(model: str, api_version: str | None) -> bool:
        return BaseAzureLLM._is_azure_v1_api_version(api_version) or model in AZURE_GA_REALTIME_MODELS

    def get_api_base(self, api_base: str | None, **kwargs) -> str:
        return api_base or litellm.api_base or get_secret_str("AZURE_API_BASE") or ""

    def get_api_key(self, api_key: str | None, **kwargs) -> str:
        return api_key or litellm.api_key or get_secret_str("AZURE_API_KEY") or ""

    def get_complete_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        base: Final = self.get_api_base(api_base).rstrip("/")
        if self._uses_ga_api(model, api_version):
            return f"{base}/openai/v1/realtime/client_secrets"
        version: Final = api_version or get_secret_str("AZURE_API_VERSION") or "2024-12-17"
        return f"{base}/openai/realtime/client_secrets?api-version={version}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
    ) -> dict:
        validated_headers = {  # mutable-ok: provider authentication headers are extended before dispatch
            **headers,
            "Content-Type": "application/json",
        }
        if api_key:
            validated_headers["api-key"] = api_key
        return validated_headers

    def get_realtime_calls_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        base: Final = self.get_api_base(api_base).rstrip("/")
        if self._uses_ga_api(model, api_version):
            return f"{base}/openai/v1/realtime/calls"
        version: Final = api_version or get_secret_str("AZURE_API_VERSION") or "2024-12-17"
        return f"{base}/openai/realtime/calls?api-version={version}"

    def get_transcription_session_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        base: Final = self.get_api_base(api_base).rstrip("/")
        if self._uses_ga_api(model, api_version):
            return f"{base}/openai/v1/realtime/transcription_sessions"
        version: Final = api_version or get_secret_str("AZURE_API_VERSION") or "2024-12-17"
        return f"{base}/openai/realtime/transcription_sessions?api-version={version}"

    def get_translation_client_secret_url(
        self, api_base: str | None, model: str, api_version: str | None = None
    ) -> str:
        base = self.get_api_base(api_base).rstrip("/")
        return f"{base}/openai/v1/realtime/translations/client_secrets"

    def get_translation_calls_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        base = self.get_api_base(api_base).rstrip("/")
        return f"{base}/openai/v1/realtime/translations/calls"

    def get_realtime_calls_headers(self, ephemeral_key: str) -> dict:
        return {
            "api-key": ephemeral_key,
        }
