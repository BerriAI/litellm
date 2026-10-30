"""Azure OpenAI realtime HTTP transformation config (client_secrets + realtime_calls)."""

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.azure.common_utils import get_azure_ad_token
from litellm.llms.base_llm.realtime.http_transformation import BaseRealtimeHTTPConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

_CALLER_AUTH_HEADERS: Final = frozenset({"authorization", "api-key"})


def _azure_resource_base(api_base: str) -> str:
    return api_base.rstrip("/").removesuffix("/openai/v1").removesuffix("/openai")


def _azure_realtime_v1_url(api_base: str, endpoint: str) -> str:
    """Azure serves WebRTC only on the GA v1 surface, which rejects any api-version query param."""
    base: Final = api_base.rstrip("/").removesuffix("/v1").removesuffix("/openai")
    return f"{base}/openai/v1/realtime/{endpoint}"


class AzureRealtimeHTTPConfig(BaseRealtimeHTTPConfig):
    def get_api_base(self, api_base: str | None, **kwargs: object) -> str:
        return api_base or litellm.api_base or get_secret_str("AZURE_API_BASE") or ""

    def get_api_key(self, api_key: str | None, **kwargs: object) -> str:
        return api_key or litellm.api_key or get_secret_str("AZURE_API_KEY") or ""

    def get_extra_headers(
        self,
        litellm_params: GenericLiteLLMParams,
        api_key: str,
        extra_headers: Mapping[str, object] | None,
    ) -> Mapping[str, object] | None:
        if api_key or any(name.lower() in _CALLER_AUTH_HEADERS for name in extra_headers or {}):
            return extra_headers
        azure_ad_token: Final = get_azure_ad_token(litellm_params)
        if azure_ad_token is None:
            return extra_headers
        return {**(extra_headers or {}), "Authorization": f"Bearer {azure_ad_token}"}

    def get_complete_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        return _azure_realtime_v1_url(self.get_api_base(api_base), "client_secrets")

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
    ) -> dict:
        validated_headers: Final = {
            **headers,
            "Content-Type": "application/json",
        }
        if api_key:
            validated_headers["api-key"] = api_key
        return validated_headers

    def get_realtime_calls_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        return _azure_realtime_v1_url(self.get_api_base(api_base), "calls")

    def get_transcription_session_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        base: Final = _azure_resource_base(self.get_api_base(api_base))
        version: Final = api_version or get_secret_str("AZURE_API_VERSION") or "2024-12-17"
        return f"{join_api_base(base, 'openai/realtime/transcription_sessions')}?api-version={version}"

    def get_translation_client_secret_url(
        self, api_base: str | None, model: str, api_version: str | None = None
    ) -> str:
        return _azure_realtime_v1_url(self.get_api_base(api_base), "translations/client_secrets")

    def get_translation_calls_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        return _azure_realtime_v1_url(self.get_api_base(api_base), "translations/calls")

    def get_realtime_calls_headers(self, ephemeral_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {ephemeral_key}",
        }
