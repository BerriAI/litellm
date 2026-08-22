from typing import Optional
from urllib.parse import urlsplit

from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str


class NearAIImageEditConfig(OpenAIImageEditConfig):
    _DEFAULT_API_BASE = "https://cloud-api.near.ai/v1"

    @staticmethod
    def _origin(api_base: str) -> tuple[str, str, Optional[int]]:
        parsed = urlsplit(api_base)
        scheme = parsed.scheme.lower()
        port = parsed.port or {"http": 80, "https": 443}.get(scheme)
        return scheme, (parsed.hostname or "").lower(), port

    def _is_trusted_api_base(self, api_base: str) -> bool:
        trusted_bases = [self._DEFAULT_API_BASE, get_secret_str("NEARAI_API_BASE")]
        candidate_origin = self._origin(api_base)
        return bool(candidate_origin[0] and candidate_origin[1]) and any(
            candidate_origin == self._origin(trusted_base) for trusted_base in trusted_bases if trusted_base
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_base and not api_key and not self._is_trusted_api_base(api_base):
            raise ValueError(
                "Refusing to send the server-configured NEARAI_API_KEY to a caller-supplied api_base. "
                "Pass an explicit api_key when overriding the NEAR AI API base."
            )
        resolved_api_key = api_key or get_secret_str("NEARAI_API_KEY")
        return {**headers, "Authorization": f"Bearer {resolved_api_key}"}

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        resolved_api_base = api_base or get_secret_str("NEARAI_API_BASE") or self._DEFAULT_API_BASE
        return f"{resolved_api_base.rstrip('/')}/images/edits"
