"""
Base Vertex, Google AI Studio LLM Class

Handles Authentication and generating request urls for Vertex AI and Google AI Studio
"""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES, VertexPartnerProvider

from .common_utils import (
    _get_gemini_url,
    _get_vertex_url,
    all_gemini_url_modes,
    get_vertex_base_model_name,
    get_vertex_base_url,
    is_global_only_vertex_model,
)

GOOGLE_IMPORT_ERROR_MESSAGE = (
    "Google Cloud SDK not found. Install it with: pip install 'litellm[google]' "
    "or pip install google-cloud-aiplatform"
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
            Tuple[GoogleCredentialsObject, str],
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
                credential_source = json_obj.get("credential_source", {})
                environment_id = (
                    credential_source.get("environment_id", "")
                    if isinstance(credential_source, dict)
                    else ""
                )
                if isinstance(environment_id, str) and "aws" in environment_id:
                    creds = self._credentials_from_identity_pool_with_aws(
                        json_obj,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                else:
                    creds = self._credentials_from_identity_pool(
                        json_obj,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
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
    def _credentials_from_identity_pool(self, json_obj, scopes):
        try:
            from google.auth import identity_pool
        except ImportError:
            raise ImportError(GOOGLE_IMPORT_ERROR_MESSAGE)

        creds = identity_pool.Credentials.from_info(json_obj)
        if scopes and hasattr(creds, "requires_scopes") and creds.requires_scopes:
            creds = creds.with_scopes(scopes)
        return creds

    def _credentials_from_identity_pool_with_aws(self, json_obj, scopes):
        try:
            from google.auth import aws
        except ImportError:
            raise ImportError(GOOGLE_IMPORT_ERROR_MESSAGE)

        creds = aws.Credentials.from_info(json_obj)
        if scopes and hasattr(creds, "requires_scopes") and creds.requires_scopes:
            creds = creds.with_scopes(scopes)
        return creds

    def _credentials_from_authorized_user(self, json_obj, scopes):
        try:
            import google.oauth2.credentials
        except ImportError:
            raise ImportError(GOOGLE_IMPORT_ERROR_MESSAGE)

        return google.oauth2.credentials.Credentials.from_authorized_user_info(
            json_obj, scopes=scopes
        )

    def _credentials_from_service_account(self, json_obj, scopes):
        try:
            import google.oauth2.service_account
        except ImportError:
            raise ImportError(GOOGLE_IMPORT_ERROR_MESSAGE)

        return google.oauth2.service_account.Credentials.from_service_account_info(
            json_obj, scopes=scopes
        )

    def _credentials_from_default_auth(self, scopes):
        try:
            import google.auth as google_auth
        except ImportError:
            raise ImportError(GOOGLE_IMPORT_ERROR_MESSAGE)

        return google_auth.default(scopes=scopes)

    def get_default_vertex_location(self) -> str:
        return "us-central1"

    def get_api_base(
        self, api_base: Optional[str], vertex_location: Optional[str]
    ) -> str:
        if api_base:
            return api_base
        return get_vertex_base_url(vertex_location or self.get_default_vertex_location())

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

        if api_base is None:
            api_base = get_vertex_base_url(vertex_location)
        if partner == VertexPartnerProvider.llama:
            return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/endpoints/openapi/chat/completions"
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
        # Use get_vertex_region to handle global-only models
        resolved_location = self.get_vertex_region(vertex_location, model)
        api_base = self.get_api_base(
            api_base=custom_api_base, vertex_location=resolved_location
        )
        default_api_base = VertexBase.create_vertex_url(
            vertex_location=resolved_location,
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
            model=model,
            vertex_project=vertex_project or project_id,
            vertex_location=resolved_location,
            vertex_api_version="v1",  # Partner models typically use v1
        )
        return api_base

    def refresh_auth(self, credentials: Any) -> None:
        try:
            from google.auth.transport.requests import (
                Request,  # type: ignore[import-untyped]
            )
        except ImportError:
            raise ImportError(GOOGLE_IMPORT_ERROR_MESSAGE)

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
        use this helper to decide if request should be sent to v1 or v1beta1

        Returns true if any beta feature is enabled
        Returns false in all other cases
        """
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
        model: Optional[str] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_api_version: Optional[Literal["v1", "v1beta1"]] = None,
        use_psc_endpoint_format: bool = False,
    ) -> Tuple[Optional[str], str]:
        """
        for cloudflare ai gateway - https://github.com/BerriAI/litellm/issues/4317

        Handles custom api_base for:
        1. Gemini (Google AI Studio) - constructs /models/{model}:{endpoint}
        2. Vertex AI with standard proxies - constructs {api_base}:{endpoint}
        3. Vertex AI with PSC endpoints - constructs full path structure
           {api_base}/v1/projects/{project}/locations/{location}/endpoints/{model}:{endpoint}
           (only when use_psc_endpoint_format=True)

        Args:
            use_psc_endpoint_format: If True, constructs PSC endpoint URL format.
                                     If False (default), uses api_base as-is and appends :{endpoint}

        ## Returns
        - (auth_header, url) - Tuple[Optional[str], str]
        """
        if api_base:
            if custom_llm_provider == "gemini":
                # For Gemini (Google AI Studio), construct the full path like other providers
                if model is None:
                    raise ValueError(
                        "Model parameter is required for Gemini custom API base URLs"
                    )
                url = "{}/models/{}:{}".format(api_base, model, endpoint)
                if gemini_api_key is None:
                    raise ValueError(
                        "Missing gemini_api_key, please set `GEMINI_API_KEY`"
                    )
                if gemini_api_key is not None:
                    auth_header = {"x-goog-api-key": gemini_api_key}  # type: ignore[assignment]
            else:
                # For Vertex AI
                if use_psc_endpoint_format:
                    # User explicitly specified PSC endpoint format
                    # Construct full PSC/custom endpoint URL
                    if not (vertex_project and vertex_location and model):
                        raise ValueError(
                            "vertex_project, vertex_location, and model are required when use_psc_endpoint_format=True"
                        )
                    # Strip routing prefixes (bge/, gemma/, etc.) for endpoint URL construction
                    model_for_url = get_vertex_base_model_name(model=model)
                    # Format: {api_base}/v1/projects/{project}/locations/{location}/endpoints/{model}:{endpoint}
                    version = vertex_api_version or "v1"
                    url = "{}/{}/projects/{}/locations/{}/endpoints/{}:{}".format(
                        api_base.rstrip("/"),
                        version,
                        vertex_project,
                        vertex_location,
                        model_for_url,
                        endpoint,
                    )
                else:
                    # Fallback to simple format if we don't have all parameters
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
        use_psc_endpoint_format: bool = False,
    ) -> Tuple[Optional[str], str]:
        """
        Internal function. Returns the token and url for the call.

        Handles logic if it's google ai studio vs. vertex ai.

        Returns
            token, url
        """
        version: Optional[Literal["v1beta1", "v1"]] = None
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
            version = "v1beta1" if should_use_v1beta1_features is True else "v1"
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
            model=model,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_api_version=version,
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

    def _handle_reauthentication(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
        credential_cache_key: Tuple,
        error: Exception,
    ) -> Tuple[str, str]:
        """
        Handle reauthentication when credentials refresh fails.

        This method clears the cached credentials and attempts to reload them once.
        It should only be called when "Reauthentication is needed" error occurs.

        Args:
            credentials: The original credentials
            project_id: The project ID
            credential_cache_key: The cache key to clear
            error: The original error that triggered reauthentication

        Returns:
            Tuple of (access_token, project_id)

        Raises:
            The original error if reauthentication fails
        """
        verbose_logger.debug(
            f"Handling reauthentication for project_id: {project_id}. "
            f"Clearing cache and retrying once."
        )

        # Clear the cached credentials
        if credential_cache_key in self._credentials_project_mapping:
            del self._credentials_project_mapping[credential_cache_key]

        # Retry once with _retry_reauth=True to prevent infinite recursion
        try:
            return self.get_access_token(
                credentials=credentials,
                project_id=project_id,
                _retry_reauth=True,
            )
        except Exception as retry_error:
            verbose_logger.error(
                f"Reauthentication retry failed for project_id: {project_id}. "
                f"Original error: {str(error)}. Retry error: {str(retry_error)}"
            )
            # Re-raise the original error for better context
            raise error

    def get_access_token(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
        _retry_reauth: bool = False,
    ) -> Tuple[str, str]:
        """
        Get access token and project id

        1. Check if credentials are already in self._credentials_project_mapping
        2. If not, load credentials and add to self._credentials_project_mapping
        3. Check if loaded credentials have expired
        4. If expired, refresh credentials
        5. Return access token and project id

        Args:
            credentials: The credentials to use for authentication
            project_id: The Google Cloud project ID
            _retry_reauth: Internal flag to prevent infinite recursion during reauthentication

        Returns:
            Tuple of (access_token, project_id)
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
            # Retrieve both credentials and cached project_id
            cached_entry = self._credentials_project_mapping[credential_cache_key]
            verbose_logger.debug("cached_entry: %s", cached_entry)
            if isinstance(cached_entry, tuple):
                _credentials, credential_project_id = cached_entry
            else:
                # Backward compatibility with old cache format
                _credentials = cached_entry
                credential_project_id = _credentials.quota_project_id or getattr(
                    _credentials, "project_id", None
                )
            verbose_logger.debug(
                "Using cached credentials for project_id: %s",
                credential_project_id,
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
            # Cache the project_id and credentials from load_auth result (resolved project_id)
            self._credentials_project_mapping[credential_cache_key] = (
                _credentials,
                credential_project_id,
            )

        ## VALIDATE CREDENTIALS
        verbose_logger.debug(f"Validating credentials for project_id: {project_id}")
        if (
            project_id is None
            and credential_project_id is not None
            and isinstance(credential_project_id, str)
        ):
            project_id = credential_project_id
            # Update cache with resolved project_id for future lookups
            resolved_cache_key = (cache_credentials, project_id)
            if resolved_cache_key not in self._credentials_project_mapping:
                self._credentials_project_mapping[resolved_cache_key] = (
                    _credentials,
                    credential_project_id,
                )

        # Check if credentials are None before accessing attributes
        if _credentials is None:
            raise ValueError("Credentials are None after loading")

        if _credentials.expired:
            try:
                verbose_logger.debug(
                    f"Credentials expired, refreshing for project_id: {project_id}"
                )
                self.refresh_auth(_credentials)
                self._credentials_project_mapping[credential_cache_key] = (
                    _credentials,
                    credential_project_id,
                )
            except Exception as e:
                # if refresh fails, it's possible the user has re-authenticated via `gcloud auth application-default login`
                # in this case, we should try to reload the credentials by clearing the cache and retrying
                if "Reauthentication is needed" in str(e) and not _retry_reauth:
                    return self._handle_reauthentication(
                        credentials=credentials,
                        project_id=project_id,
                        credential_cache_key=credential_cache_key,
                        error=e,
                    )
                raise e

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

    @staticmethod
    def get_vertex_ai_project(litellm_params: dict) -> Optional[str]:
        return (
            litellm_params.pop("vertex_project", None)
            or litellm_params.pop("vertex_ai_project", None)
            or litellm.vertex_project
            or get_secret_str("VERTEXAI_PROJECT")
        )

    @staticmethod
    def get_vertex_ai_credentials(litellm_params: dict) -> Optional[str]:
        return (
            litellm_params.pop("vertex_credentials", None)
            or litellm_params.pop("vertex_ai_credentials", None)
            or get_secret_str("VERTEXAI_CREDENTIALS")
        )

    @staticmethod
    def get_vertex_ai_location(litellm_params: dict) -> Optional[str]:
        return (
            litellm_params.pop("vertex_location", None)
            or litellm_params.pop("vertex_ai_location", None)
            or litellm.vertex_location
            or get_secret_str("VERTEXAI_LOCATION")
            or get_secret_str("VERTEX_LOCATION")
        )

    @staticmethod
    def safe_get_vertex_ai_project(litellm_params: dict) -> Optional[str]:
        """
        Safely get Vertex AI project without mutating the litellm_params dict.

        Unlike get_vertex_ai_project(), this does NOT pop values from the dict,
        making it safe to call multiple times with the same litellm_params.

        Args:
            litellm_params: Dictionary containing Vertex AI parameters

        Returns:
            Vertex AI project ID or None
        """
        return (
            litellm_params.get("vertex_project")
            or litellm_params.get("vertex_ai_project")
            or litellm.vertex_project
            or get_secret_str("VERTEXAI_PROJECT")
        )

    @staticmethod
    def safe_get_vertex_ai_credentials(litellm_params: dict) -> Optional[str]:
        """
        Safely get Vertex AI credentials without mutating the litellm_params dict.

        Unlike get_vertex_ai_credentials(), this does NOT pop values from the dict,
        making it safe to call multiple times with the same litellm_params.

        Args:
            litellm_params: Dictionary containing Vertex AI parameters

        Returns:
            Vertex AI credentials or None
        """
        return (
            litellm_params.get("vertex_credentials")
            or litellm_params.get("vertex_ai_credentials")
            or get_secret_str("VERTEXAI_CREDENTIALS")
        )

    @staticmethod
    def safe_get_vertex_ai_location(litellm_params: dict) -> Optional[str]:
        """
        Safely get Vertex AI location without mutating the litellm_params dict.

        Unlike get_vertex_ai_location(), this does NOT pop values from the dict,
        making it safe to call multiple times with the same litellm_params.

        Args:
            litellm_params: Dictionary containing Vertex AI parameters

        Returns:
            Vertex AI location/region or None
        """
        return (
            litellm_params.get("vertex_location")
            or litellm_params.get("vertex_ai_location")
            or litellm.vertex_location
            or get_secret_str("VERTEXAI_LOCATION")
            or get_secret_str("VERTEX_LOCATION")
        )
