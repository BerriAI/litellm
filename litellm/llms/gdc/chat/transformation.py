"""
GDC Gemini chat completion transformation
"""

import json
import os
import threading
from typing import Any

import litellm
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


class GDCGeminiConfig(OpenAILikeChatConfig):
    supports_vertex_params: bool = True  # Tell LiteLLM utilities not to strip vertex_ params

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "vertex_project",
            "vertex_location",
        ] + super().get_supported_openai_params(model)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        api_base = api_base or litellm.api_base or getattr(litellm, "gdc_api_base", None)
        if not api_base:
            raise litellm.utils.AuthenticationError(
                message="api_base/host is required for GDC Gemini. Please set it or pass it.",
                llm_provider="gdc",
                model=model,
            )

        if not api_base.startswith("http"):
            api_base = f"https://{api_base}"

        project = (
            optional_params.get("vertex_project")
            or litellm_params.get("vertex_project")
            or getattr(litellm, "vertex_project", None)
        )

        if not project:
            raise litellm.utils.AuthenticationError(
                message="project is required for GDC Gemini. Please pass vertex_project.",
                llm_provider="gdc",
                model=model,
            )

        location = (
            optional_params.get("vertex_location")
            or litellm_params.get("vertex_location")
            or getattr(litellm, "vertex_location", None)
        )

        api_base = api_base.rstrip("/")

        # If the endpoint structure is already in the api_base, don't append it again
        if "/v1/projects/" in api_base:
            return api_base

        if not location:
            raise litellm.utils.AuthenticationError(
                message="location is required for GDC Gemini. Please pass vertex_location.",
                llm_provider="gdc",
                model=model,
            )

        return f"{api_base}/v1/projects/{project}/locations/{location}/chat/completions"

    def _read_env_bool(self, val: Any, env_var: str, default: bool = True) -> bool | str:
        if val is not None:
            return val

        _env_val = os.getenv(env_var)
        if _env_val is None:
            return default
        _clean = _env_val.strip().lower()
        if _clean in ("false", "0", "no", "off"):
            return False
        if _clean in ("true", "1", "yes", "on"):
            return True
        return _env_val

    def _fetch_auth(self, gdch_creds: Any, ssl_verify: bool | str) -> None:
        import requests
        from google.auth.transport import requests as auth_requests

        auth_session = requests.Session()
        auth_session.verify = ssl_verify
        auth_request = auth_requests.Request(session=auth_session)
        gdch_creds.refresh(auth_request)

    def _cached_fetch_token(
        self,
        creds: Any,
        audience: str,
        ssl_verify: bool | str,
        api_key: str | None = None
    ) -> str:
        if not hasattr(self, "_creds_lock"):
            self._creds_lock = threading.Lock()
        if not hasattr(self, "_gdch_creds_cache"):
            self._gdch_creds_cache = {}

        # Key cache by both audience and credential identity to prevent cross-caller contamination
        cache_key = (audience.rstrip("/"), api_key or str(id(creds)))

        with self._creds_lock:
            if cache_key not in self._gdch_creds_cache:
                self._gdch_creds_cache[cache_key] = creds.with_gdch_audience(audience.rstrip("/"))

            gdch_creds = self._gdch_creds_cache[cache_key]

            if not getattr(gdch_creds, "valid", False) or not getattr(gdch_creds, "token", None):
                self._fetch_auth(gdch_creds, ssl_verify)

        return gdch_creds.token

    def _load_creds_from_key(self, api_key: str, model: str) -> tuple[Any, bool]:
        """Helper to safely parse service account JSON files or strings to reduce complexity."""
        import google.auth

        # Limit length to avoid OSError for 'File name too long'
        if len(api_key) < 2000 and os.path.exists(api_key):
            with open(api_key, "r") as f:
                json_obj = json.load(f)
            creds, _ = google.auth.load_credentials_from_dict(json_obj)
            return creds, True

        try:
            json_obj = json.loads(api_key)
            creds, _ = google.auth.load_credentials_from_dict(json_obj)
            return creds, True
        except json.JSONDecodeError:
            return None, False

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

        api_base = api_base or litellm.api_base or getattr(litellm, "gdc_api_base", None)
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

        project = (
            optional_params.get("vertex_project")
            or litellm_params.get("vertex_project")
            or getattr(litellm, "vertex_project", None)
        )
        if not project:
            raise litellm.utils.AuthenticationError(
                message="project is required for GDC Gemini. Please pass vertex_project.",
                llm_provider="gdc",
                model=model,
            )

        audience = api_base if api_base.startswith("http") else f"https://{api_base}"

        try:
            creds, is_service_account = self._load_creds_from_key(api_key, model)
        except (
            google.auth.exceptions.GoogleAuthError,
            OSError,
            UnicodeDecodeError,
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

        if "x-goog-user-project" not in headers and "X-Goog-User-Project" not in headers:
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
        for param in ["vertex_project", "vertex_location", "ssl_verify", "gdc_token_caching"]:
            data.pop(param, None)

        return data
