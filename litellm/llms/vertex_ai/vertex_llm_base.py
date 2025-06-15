"""
Base Vertex, Google AI Studio LLM Class

Handles Authentication and generating request urls for Vertex AI and Google AI Studio
"""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES, VertexPartnerProvider

from .common_utils import (
    _get_gemini_url,
    _get_vertex_url,
    all_gemini_url_modes,
    is_global_only_vertex_model,
)

if TYPE_CHECKING:
    from google.auth.credentials import Credentials as GoogleCredentialsObject
else:
    GoogleCredentialsObject = Any


class VertexBase:
    def __init__(self) -> None:
        super().__init__()
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self._credentials: Optional[GoogleCredentialsObject] = None
        self._credentials_project_mapping: Dict[
            Tuple[Optional[VERTEX_CREDENTIALS_TYPES], Optional[str]],
            GoogleCredentialsObject,
        ] = {}
        self.project_id: Optional[str] = None
        self.async_handler: Optional[AsyncHTTPHandler] = None

    def get_vertex_region(self, vertex_region: Optional[str], model: str) -> str:
        if is_global_only_vertex_model(model):
            return "global"
        return vertex_region or "us-central1"

    def load_auth(
        self, credentials: Optional[VERTEX_CREDENTIALS_TYPES], project_id: Optional[str]
    ) -> Tuple[Any, str]:
        if credentials is not None:
            if isinstance(credentials, str):
                verbose_logger.debug(
                    "Vertex: Loading vertex credentials from %s", credentials
                )
                verbose_logger.debug(
                    "Vertex: checking if credentials is a valid path, os.path.exists(%s)=%s, current dir %s",
                    credentials,
                    os.path.exists(credentials),
                    os.getcwd(),
                )

                try:
                    if os.path.exists(credentials):
                        json_obj = json.load(open(credentials))
                    else:
                        json_obj = json.loads(credentials)
                except Exception:
                    raise Exception(
                        "Unable to load vertex credentials from environment. Got={}".format(
                            credentials
                        )
                    )
            elif isinstance(credentials, dict):
                json_obj = credentials
            else:
                raise ValueError(
                    "Invalid credentials type: {}".format(type(credentials))
                )

            # Check if the JSON object contains Workload Identity Federation configuration
            if "type" in json_obj and json_obj["type"] == "external_account":
                # If environment_id key contains "aws" value it corresponds to an AWS config file
                if (
                    "credential_source" in json_obj
                    and "environment_id" in json_obj["credential_source"]
                    and "aws" in json_obj["credential_source"]["environment_id"]
                ):
                    creds = self._credentials_from_identity_pool_with_aws(json_obj)
                else:
                    creds = self._credentials_from_identity_pool(json_obj)
            # Check if the JSON object contains Authorized User configuration (via gcloud auth application-default login)
            elif "type" in json_obj and json_obj["type"] == "authorized_user":
                creds = self._credentials_from_authorized_user(
                    json_obj,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                if project_id is None:
                    project_id = (
                        creds.quota_project_id
                    )  # authorized user credentials don't have a project_id, only quota_project_id
            else:
                creds = self._credentials_from_service_account(
                    json_obj,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

            if project_id is None:
                project_id = getattr(creds, "project_id", None)
        else:
            creds, creds_project_id = self._credentials_from_default_auth(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            if project_id is None:
                project_id = creds_project_id

        self.refresh_auth(creds)

        if not project_id:
            raise ValueError("Could not resolve project_id")

        if not isinstance(project_id, str):
            raise TypeError(
                f"Expected project_id to be a str but got {type(project_id)}"
            )

        return creds, project_id

    # Google Auth Helpers -- extracted for mocking purposes in tests
    def _credentials_from_identity_pool(self, json_obj):
        from google.auth import identity_pool

        return identity_pool.Credentials.from_info(json_obj)
    
    def _credentials_from_identity_pool_with_aws(self, json_obj):
        from google.auth import aws

        return aws.Credentials.from_info(json_obj)

    def _credentials_from_authorized_user(self, json_obj, scopes):
        import google.oauth2.credentials

        return google.oauth2.credentials.Credentials.from_authorized_user_info(
            json_obj, scopes=scopes
        )

    def _credentials_from_service_account(self, json_obj, scopes):
        import google.oauth2.service_account

        return google.oauth2.service_account.Credentials.from_service_account_info(
            json_obj, scopes=scopes
        )

    def _credentials_from_default_auth(self, scopes):
        import google.auth as google_auth

        return google_auth.default(scopes=scopes)

    def get_default_vertex_location(self) -> str:
        return "us-central1"

    def get_api_base(
        self, api_base: Optional[str], vertex_location: Optional[str]
    ) -> str:
        if api_base:
            return api_base
        elif vertex_location == "global":
            return "https://aiplatform.googleapis.com"
        elif vertex_location:
            return f"https://{vertex_location}-aiplatform.googleapis.com"
        else:
            return f"https://{self.get_default_vertex_location()}-aiplatform.googleapis.com"

    @staticmethod
    def create_vertex_url(
        vertex_location: str,
        vertex_project: str,
        partner: VertexPartnerProvider,
        stream: Optional[bool],
        model: str,
        api_base: Optional[str] = None,
    ) -> str:
        """Return the base url for the vertex partner models"""

        api_base = api_base or f"https://{vertex_location}-aiplatform.googleapis.com"
        if partner == VertexPartnerProvider.llama:
            return f"{api_base}/v1beta1/projects/{vertex_project}/locations/{vertex_location}/endpoints/openapi/chat/completions"
        elif partner == VertexPartnerProvider.mistralai:
            if stream:
                return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/mistralai/models/{model}:streamRawPredict"
            else:
                return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/mistralai/models/{model}:rawPredict"
        elif partner == VertexPartnerProvider.ai21:
            if stream:
                return f"{api_base}/v1beta1/projects/{vertex_project}/locations/{vertex_location}/publishers/ai21/models/{model}:streamRawPredict"
            else:
                return f"{api_base}/v1beta1/projects/{vertex_project}/locations/{vertex_location}/publishers/ai21/models/{model}:rawPredict"
        elif partner == VertexPartnerProvider.claude:
            if stream:
                return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/anthropic/models/{model}:streamRawPredict"
            else:
                return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/anthropic/models/{model}:rawPredict"

    def get_complete_vertex_url(
        self,
        custom_api_base: Optional[str],
        vertex_location: Optional[str],
        vertex_project: Optional[str],
        project_id: str,
        partner: VertexPartnerProvider,
        stream: Optional[bool],
        model: str,
    ) -> str:
        api_base = self.get_api_base(
            api_base=custom_api_base, vertex_location=vertex_location
        )
        default_api_base = VertexBase.create_vertex_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
            partner=partner,
            stream=stream,
            model=model,
            api_base=api_base,
        )

        if len(default_api_base.split(":")) > 1:
            endpoint = default_api_base.split(":")[-1]
        else:
            endpoint = ""

        _, api_base = self._check_custom_proxy(
            api_base=custom_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=stream,
            auth_header=None,
            url=default_api_base,
        )
        return api_base

    def refresh_auth(self, credentials: Any) -> None:
        from google.auth.transport.requests import (
            Request,  # type: ignore[import-untyped]
        )

        credentials.refresh(Request())

    def _ensure_access_token(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
    ) -> Tuple[str, str]:
        """
        Returns auth token and project id
        """
        if custom_llm_provider == "gemini":
            return "", ""
        else:
            return self.get_access_token(
                credentials=credentials,
                project_id=project_id,
            )

    def is_using_v1beta1_features(self, optional_params: dict) -> bool:
        """
        VertexAI only supports ContextCaching on v1beta1

        use this helper to decide if request should be sent to v1 or v1beta1

        Returns v1beta1 if context caching is enabled
        Returns v1 in all other cases
        """
        if "cached_content" in optional_params:
            return True
        if "CachedContent" in optional_params:
            return True
        return False

    def _check_custom_proxy(
        self,
        api_base: Optional[str],
        custom_llm_provider: str,
        gemini_api_key: Optional[str],
        endpoint: str,
        stream: Optional[bool],
        auth_header: Optional[str],
        url: str,
    ) -> Tuple[Optional[str], str]:
        """
        for cloudflare ai gateway - https://github.com/BerriAI/litellm/issues/4317

        ## Returns
        - (auth_header, url) - Tuple[Optional[str], str]
        """
        if api_base:
            if custom_llm_provider == "gemini":
                url = "{}:{}".format(api_base, endpoint)
                if gemini_api_key is None:
                    raise ValueError(
                        "Missing gemini_api_key, please set `GEMINI_API_KEY`"
                    )
                auth_header = (
                    gemini_api_key  # cloudflare expects api key as bearer token
                )
            else:
                url = "{}:{}".format(api_base, endpoint)

            if stream is True:
                url = url + "?alt=sse"
        return auth_header, url

    def _get_token_and_url(
        self,
        model: str,
        auth_header: Optional[str],
        gemini_api_key: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        stream: Optional[bool],
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        api_base: Optional[str],
        should_use_v1beta1_features: Optional[bool] = False,
        mode: all_gemini_url_modes = "chat",
    ) -> Tuple[Optional[str], str]:
        """
        Internal function. Returns the token and url for the call.

        Handles logic if it's google ai studio vs. vertex ai.

        Returns
            token, url
        """
        if custom_llm_provider == "gemini":
            url, endpoint = _get_gemini_url(
                mode=mode,
                model=model,
                stream=stream,
                gemini_api_key=gemini_api_key,
            )
            auth_header = None  # this field is not used for gemin
        else:
            vertex_location = self.get_vertex_region(
                vertex_region=vertex_location,
                model=model,
            )

            ### SET RUNTIME ENDPOINT ###
            version: Literal["v1beta1", "v1"] = (
                "v1beta1" if should_use_v1beta1_features is True else "v1"
            )
            url, endpoint = _get_vertex_url(
                mode=mode,
                model=model,
                stream=stream,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                vertex_api_version=version,
            )

        return self._check_custom_proxy(
            api_base=api_base,
            auth_header=auth_header,
            custom_llm_provider=custom_llm_provider,
            gemini_api_key=gemini_api_key,
            endpoint=endpoint,
            stream=stream,
            url=url,
        )

    def get_access_token(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
    ) -> Tuple[str, str]:
        """
        Get access token and project id

        1. Check if credentials are already in self._credentials_project_mapping
        2. If not, load credentials and add to self._credentials_project_mapping
        3. Check if loaded credentials have expired
        4. If expired, refresh credentials
        5. Return access token and project id
        """

        # Convert dict credentials to string for caching
        cache_credentials = (
            json.dumps(credentials) if isinstance(credentials, dict) else credentials
        )
        credential_cache_key = (cache_credentials, project_id)
        _credentials: Optional[GoogleCredentialsObject] = None

        verbose_logger.debug(
            f"Checking cached credentials for project_id: {project_id}"
        )

        if credential_cache_key in self._credentials_project_mapping:
            verbose_logger.debug(
                f"Cached credentials found for project_id: {project_id}."
            )
            _credentials = self._credentials_project_mapping[credential_cache_key]
            verbose_logger.debug("Using cached credentials")
            credential_project_id = _credentials.quota_project_id or getattr(
                _credentials, "project_id", None
            )

        else:
            verbose_logger.debug(
                f"Credential cache key not found for project_id: {project_id}, loading new credentials"
            )

            try:
                _credentials, credential_project_id = self.load_auth(
                    credentials=credentials, project_id=project_id
                )
            except Exception as e:
                verbose_logger.exception(
                    f"Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: {str(e)}"
                )
                raise e

            if _credentials is None:
                raise ValueError(
                    "Could not resolve credentials - either dynamically or from environment, for project_id: {}".format(
                        project_id
                    )
                )

            self._credentials_project_mapping[credential_cache_key] = _credentials

        ## VALIDATE CREDENTIALS
        verbose_logger.debug(f"Validating credentials for project_id: {project_id}")
        if (
            project_id is None
            and credential_project_id is not None
            and isinstance(credential_project_id, str)
        ):
            project_id = credential_project_id

        if _credentials.expired:
            self.refresh_auth(_credentials)

        ## VALIDATION STEP
        if _credentials.token is None or not isinstance(_credentials.token, str):
            raise ValueError(
                "Could not resolve credentials token. Got None or non-string token - {}".format(
                    _credentials.token
                )
            )

        if project_id is None:
            raise ValueError("Could not resolve project_id")

        return _credentials.token, project_id

    async def _ensure_access_token_async(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
    ) -> Tuple[str, str]:
        """
        Async version of _ensure_access_token
        """
        if custom_llm_provider == "gemini":
            return "", ""
        else:
            try:
                return await asyncify(self.get_access_token)(
                    credentials=credentials,
                    project_id=project_id,
                )
            except Exception as e:
                raise e

    def set_headers(
        self, auth_header: Optional[str], extra_headers: Optional[dict]
    ) -> dict:
        headers = {
            "Content-Type": "application/json",
        }
        if auth_header is not None:
            headers["Authorization"] = f"Bearer {auth_header}"
        if extra_headers is not None:
            headers.update(extra_headers)

        return headers
