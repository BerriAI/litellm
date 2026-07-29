"""
GDC Gemini chat completion transformation
"""

import json
import os
import re
import threading
from typing import Any, Final
from urllib.parse import urlsplit

import litellm
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


class GDCGeminiConfig(OpenAILikeChatConfig):
    supports_vertex_params: bool = True  # Tell LiteLLM utilities not to strip vertex_ params
    _GDCH_CREDENTIAL_TYPE: Final[str] = "gdch_service_account"
    _PATH_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._creds_lock = threading.Lock()
        self._gdch_creds_cache: dict = {}

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "vertex_project",
            "vertex_location",
        ] + super().get_supported_openai_params(model)

    def _resolve_project(self, optional_params: dict, litellm_params: dict) -> str | None:
        return (
            litellm_params.get("vertex_project")
            or litellm_params.get("vertex_ai_project")
            or getattr(litellm, "vertex_project", None)
            or optional_params.get("vertex_project")
            or optional_params.get("vertex_ai_project")
        )

    def _resolve_location(self, optional_params: dict, litellm_params: dict) -> str | None:
        return (
            litellm_params.get("vertex_location")
            or litellm_params.get("vertex_ai_location")
            or getattr(litellm, "vertex_location", None)
            or optional_params.get("vertex_location")
            or optional_params.get("vertex_ai_location")
        )

    def _effective_project(self, api_base: str, optional_params: dict, litellm_params: dict) -> str | None:
        match = re.search(r"/v1/projects/([^/]+)", api_base)
        if match:
            return match.group(1)
        return self._resolve_project(optional_params, litellm_params)

    def _validate_path_id(self, value: str, field: str, model: str) -> str:
        if not self._PATH_ID_PATTERN.match(value):
            raise litellm.utils.AuthenticationError(
                message=f"{field} must be a plain identifier of letters, digits, hyphens or underscores.",
                llm_provider="gdc",
                model=model,
            )
        return value

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        api_base = api_base or litellm.gdc_api_base or litellm.api_base
        if not api_base:
            raise litellm.utils.AuthenticationError(
                message="api_base/host is required for GDC Gemini. Please set it or pass it.",
                llm_provider="gdc",
                model=model,
            )

        if not api_base.startswith("http"):
            api_base = f"https://{api_base}"

        api_base = api_base.rstrip("/")

        if "/v1/projects/" in api_base:
            return api_base

        project = self._resolve_project(optional_params, litellm_params)

        if not project:
            raise litellm.utils.AuthenticationError(
                message="project is required for GDC Gemini. Please pass vertex_project.",
                llm_provider="gdc",
                model=model,
            )

        location = self._resolve_location(optional_params, litellm_params)

        if not location:
            raise litellm.utils.AuthenticationError(
                message="location is required for GDC Gemini. Please pass vertex_location.",
                llm_provider="gdc",
                model=model,
            )

        project = self._validate_path_id(project, "vertex_project", model)
        location = self._validate_path_id(location, "vertex_location", model)

        return f"{api_base}/v1/projects/{project}/locations/{location}/chat/completions"

    def _read_env_bool(self, val: Any, env_var: str, default: bool = True) -> bool | str:
        def _parse(s: str) -> bool | str:
            cleaned = s.strip().lower()
            if cleaned in ("false", "0", "no", "off"):
                return False
            if cleaned in ("true", "1", "yes", "on"):
                return True
            return s

        if val is not None:
            if isinstance(val, str):
                return _parse(val)
            return val

        _env_val = os.getenv(env_var)
        if _env_val is None:
            return default
        return _parse(_env_val)

    def _fetch_auth(self, gdch_creds: Any, ssl_verify: bool | str) -> None:
        import requests
        from google.auth.transport import requests as auth_requests

        auth_session = requests.Session()
        auth_session.verify = ssl_verify
        auth_request = auth_requests.Request(session=auth_session)
        gdch_creds.refresh(auth_request)

    def _cached_fetch_token(self, creds: Any, audience: str, ssl_verify: bool | str, api_key: str | None = None) -> str:
        # Key cache by both audience and credential identity to prevent cross-caller contamination
        cache_key = (audience.rstrip("/"), api_key or str(id(creds)))

        with self._creds_lock:
            if cache_key not in self._gdch_creds_cache:
                self._gdch_creds_cache[cache_key] = creds.with_gdch_audience(audience.rstrip("/"))

            gdch_creds = self._gdch_creds_cache[cache_key]

            if not getattr(gdch_creds, "valid", False) or not getattr(gdch_creds, "token", None):
                self._fetch_auth(gdch_creds, ssl_verify)

            token = gdch_creds.token

        return token

    def _load_creds_from_key(self, api_key: str) -> tuple[Any, bool]:
        import google.auth

        try:
            json_obj = json.loads(api_key)
        except json.JSONDecodeError:
            return None, False
        if not isinstance(json_obj, dict) or json_obj.get("type") != self._GDCH_CREDENTIAL_TYPE:
            raise ValueError(
                "GDC only accepts a GDCH service account credential as a JSON api_key "
                '(expected "type": "gdch_service_account"). Other Google credential types are '
                "rejected so their token or external-account endpoints cannot drive server-side requests."
            )
        creds, _ = google.auth.load_credentials_from_dict(json_obj)
        return creds, True

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        import google.auth.exceptions

        api_base = api_base or litellm.gdc_api_base or litellm.api_base
        if not api_base:
            raise litellm.utils.AuthenticationError(
                message="api_base/host is required for GDC Gemini. Please set it or pass it.",
                llm_provider="gdc",
                model=model,
            )

        if not api_key:
            raise litellm.utils.AuthenticationError(
                message="api_key is required for GDC Gemini. Please pass your service account string or token as the api_key.",
                llm_provider="gdc",
                model=model,
            )

        project = self._effective_project(api_base, optional_params, litellm_params)
        if not project:
            raise litellm.utils.AuthenticationError(
                message="project is required for GDC Gemini. Please pass vertex_project.",
                llm_provider="gdc",
                model=model,
            )
        project = self._validate_path_id(project, "vertex_project", model)

        _audience_parts = urlsplit(api_base if api_base.startswith("http") else f"https://{api_base}")
        audience = f"{_audience_parts.scheme}://{_audience_parts.netloc}"

        try:
            creds, is_service_account = self._load_creds_from_key(api_key)
        except (
            google.auth.exceptions.GoogleAuthError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as e:
            raise litellm.utils.AuthenticationError(
                message=f"Failed to load service account credentials from api_key: {str(e)}",
                llm_provider="gdc",
                model=model,
            ) from e

        if creds is not None:
            ssl_verify = self._read_env_bool(litellm_params.get("ssl_verify"), "SSL_VERIFY", default=True)
            if self._read_env_bool(litellm_params.get("gdc_token_caching"), "GDC_TOKEN_CACHING", default=False):
                token = self._cached_fetch_token(creds, audience, ssl_verify, api_key)
            else:
                gdch_creds = creds.with_gdch_audience(audience)
                self._fetch_auth(gdch_creds, ssl_verify)
                token = gdch_creds.token
            headers["Authorization"] = f"Bearer {token}"

        if "Authorization" not in headers and not is_service_account:
            headers["Authorization"] = f"Bearer {api_key}"

        # Standardize necessary metadata headers
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        stale_quota_headers = tuple(h for h in headers if h.lower() == "x-goog-user-project")
        for stale in stale_quota_headers:
            headers.pop(stale, None)
        headers["x-goog-user-project"] = f"projects/{project}"

        return headers

    def transform_request(
        self,
        model: str,
        messages: list[Any],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transforms the request to the GDC provider
        """
        if model.startswith("gdc/"):
            model = model.split("/", 1)[1]

        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Remove extra params used for routing/auth
        for param in [
            "vertex_project",
            "vertex_ai_project",
            "vertex_location",
            "vertex_ai_location",
            "ssl_verify",
            "gdc_token_caching",
        ]:
            data.pop(param, None)

        return data
