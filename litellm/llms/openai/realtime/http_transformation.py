"""OpenAI realtime HTTP transformation config (client_secrets + realtime_calls)."""

import litellm
from litellm.llms.base_llm.realtime.http_transformation import BaseRealtimeHTTPConfig
from litellm.secret_managers.main import get_secret_str


class OpenAIRealtimeHTTPConfig(BaseRealtimeHTTPConfig):
    def get_api_base(self, api_base: str | None, **kwargs) -> str:
        return api_base or litellm.api_base or get_secret_str("OPENAI_API_BASE") or "https://api.openai.com"

    def get_api_key(self, api_key: str | None, **kwargs) -> str:
        return api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY") or ""

    def get_complete_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        base = self.get_api_base(api_base).rstrip("/")
        base = base.removesuffix("/v1")
        return f"{base}/v1/realtime/client_secrets"

    def get_realtime_calls_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        base = self.get_api_base(api_base).rstrip("/")
        base = base.removesuffix("/v1")
        return f"{base}/v1/realtime/calls"

    def get_transcription_session_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        base = self.get_api_base(api_base).rstrip("/")
        base = base.removesuffix("/v1")
        return f"{base}/v1/realtime/transcription_sessions"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
    ) -> dict:
        return {
            **headers,
            "Authorization": f"Bearer {api_key or ''}",
            "Content-Type": "application/json",
        }
