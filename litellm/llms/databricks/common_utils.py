"""
Databricks integration utilities for LiteLLM.

This module provides authentication, telemetry, and security utilities
for the Databricks LLM provider integration.

Authentication priority:
1. OAuth M2M (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET) - Recommended for production
2. PAT (DATABRICKS_API_KEY) - Supported for development
3. Databricks SDK automatic auth - Fallback (uses unified auth)
"""

import hashlib
import os
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

_DEFAULT_M2M_TOKEN_LIFETIME_SECONDS: Final = 3600
_TOKEN_EXPIRY_BUFFER_SECONDS: Final = 60


class _TokenCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: str, *, ttl: int) -> None: ...


_OAUTH_TOKENS: Final[_TokenCache] = InMemoryCache(max_size_in_memory=64)


class DatabricksException(BaseLLMException):
    pass


DATABRICKS_ANTHROPIC_MESSAGES_PATH: Final = "/anthropic/v1/messages"
DATABRICKS_SERVING_ENDPOINTS_MESSAGES_PATH: Final = f"/serving-endpoints{DATABRICKS_ANTHROPIC_MESSAGES_PATH}"
DATABRICKS_AI_GATEWAY_ANTHROPIC_PATH: Final = "/ai-gateway/anthropic"
DATABRICKS_AI_GATEWAY_MESSAGES_PATH: Final = f"{DATABRICKS_AI_GATEWAY_ANTHROPIC_PATH}/v1/messages"
DATABRICKS_WORKSPACE_ROUTE_PREFIXES: Final = (
    "/serving-endpoints",
    "/serving-endpoints/anthropic",
    DATABRICKS_SERVING_ENDPOINTS_MESSAGES_PATH,
    "/ai-gateway/mlflow/v1",
    DATABRICKS_AI_GATEWAY_ANTHROPIC_PATH,
    DATABRICKS_AI_GATEWAY_MESSAGES_PATH,
)
DATABRICKS_NATIVE_MESSAGES_MODEL_PREFIXES: Final = ("databricks-claude-", "system.ai.claude-")
DATABRICKS_HOSTED_MODEL_PREFIXES: Final = ("databricks-", "system.ai.")
DATABRICKS_ENDPOINT_PATHS: Final = MappingProxyType(
    {
        "chat_completions": "/chat/completions",
        "embeddings": "/embeddings",
        "anthropic_messages": DATABRICKS_ANTHROPIC_MESSAGES_PATH,
    }
)


class OAuthM2MToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    access_token: str
    expires_in: int = _DEFAULT_M2M_TOKEN_LIFETIME_SECONDS


def _resolve_param_secret(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    val: Final = value.strip()
    if not val:
        return None
    if val.startswith("os.environ/"):
        return get_secret_str(val)
    return val


def _get_credential_param(params: Mapping[str, object] | object | None, *keys: str) -> object | None:
    if params is None:
        return None
    if isinstance(params, Mapping):
        for key in keys:
            val = params.get(key)
            if val is not None:
                return val
        nested = params.get("litellm_params")
        if isinstance(nested, Mapping):
            for key in keys:
                val = nested.get(key)
                if val is not None:
                    return val
        return None
    for key in keys:
        if hasattr(params, key):
            val = getattr(params, key)
            if val is not None:
                return val
    nested = getattr(params, "litellm_params", None)
    if isinstance(nested, Mapping):
        for key in keys:
            val = nested.get(key)
            if val is not None:
                return val
    return None


def resolve_databricks_oauth_credentials(
    litellm_params: Mapping[str, object] | object | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> tuple[str, str] | None:
    """
    Resolve Databricks OAuth M2M credentials.

    Follows atomic pair semantics:
    - If deployment credentials are supplied via `litellm_params` or explicit arguments,
      both `client_id` and `client_secret` must be provided and resolve to non-empty values.
      Missing, empty, or unresolvable deployment credentials raise a DatabricksException.
    - Deployment credentials override global environment variables.
    - If no deployment credentials are supplied, falls back to `DATABRICKS_CLIENT_ID`
      and `DATABRICKS_CLIENT_SECRET`. If neither or only one is set globally, returns None.

    Returns:
        (client_id, client_secret) tuple if configured, or None.
    Raises:
        DatabricksException: If incomplete deployment credentials are provided.
    """
    raw_id: Final = (
        client_id
        if client_id is not None
        else _get_credential_param(litellm_params, "client_id", "databricks_client_id")
    )
    raw_secret: Final = (
        client_secret
        if client_secret is not None
        else _get_credential_param(litellm_params, "client_secret", "databricks_client_secret")
    )

    deployment_credentials_provided: Final = raw_id is not None or raw_secret is not None

    if deployment_credentials_provided:
        resolved_id: Final = _resolve_param_secret(raw_id)
        resolved_secret: Final = _resolve_param_secret(raw_secret)
        if not resolved_id or not resolved_secret:
            missing: Final = []
            if not resolved_id:
                missing.append("client_id")
            if not resolved_secret:
                missing.append("client_secret")
            raise DatabricksException(
                status_code=400,
                message=(
                    f"Incomplete Databricks deployment OAuth credentials: missing or unresolved {', '.join(missing)}. "
                    "Both client_id and client_secret must be set and non-empty."
                ),
            )
        return resolved_id, resolved_secret

    # Global environment fallback
    global_id: Final = get_secret_str("DATABRICKS_CLIENT_ID")
    global_secret: Final = get_secret_str("DATABRICKS_CLIENT_SECRET")
    if global_id and global_secret:
        return global_id, global_secret

    return None


class DatabricksBase:
    """
    Base class for Databricks integration with authentication,
    telemetry, and security utilities.
    """

    # Patterns to redact in logs
    SENSITIVE_PATTERNS = [
        (re.compile(r"(Bearer\s+)[A-Za-z0-9\-_\.]+", re.IGNORECASE), r"\1[REDACTED]"),
        (re.compile(r"(Authorization:\s*)[^\s,}]+", re.IGNORECASE), r"\1[REDACTED]"),
        (
            re.compile(r'(api[_-]?key["\s:=]+)[^\s,}"\']+', re.IGNORECASE),
            r"\1[REDACTED]",
        ),
        (
            re.compile(r'(client[_-]?secret["\s:=]+)[^\s,}"\']+', re.IGNORECASE),
            r"\1[REDACTED]",
        ),
        (re.compile(r"(dapi[a-zA-Z0-9]{32,})", re.IGNORECASE), r"[REDACTED_PAT]"),
        (
            re.compile(r'(access[_-]?token["\s:=]+)[^\s,}"\']+', re.IGNORECASE),
            r"\1[REDACTED]",
        ),
    ]

    @classmethod
    def redact_sensitive_data(cls, data: object) -> object:
        """
        Redact sensitive information (tokens, secrets) from data before logging.

        Handles strings, dicts, and lists recursively. Keys containing sensitive
        terms (authorization, api_key, token, secret, password, credential) are
        fully redacted.

        Args:
            data: String, dict, or other data structure to redact

        Returns:
            Redacted version of the data safe for logging
        """
        if data is None:
            return None

        if isinstance(data, str):
            result = data
            for pattern, replacement in cls.SENSITIVE_PATTERNS:
                result = pattern.sub(replacement, result)
            return result

        if isinstance(data, dict):
            redacted: Final = {}
            for key, value in data.items():
                lower_key = key.lower()
                if any(
                    sensitive in lower_key
                    for sensitive in [
                        "authorization",
                        "api_key",
                        "apikey",
                        "token",
                        "secret",
                        "password",
                        "credential",
                    ]
                ):
                    redacted[key] = "[REDACTED]"
                else:
                    redacted[key] = cls.redact_sensitive_data(value)
            return redacted

        if isinstance(data, list):
            return [cls.redact_sensitive_data(item) for item in data]

        return data

    @classmethod
    def redact_headers_for_logging(cls, headers: dict[str, str]) -> dict[str, str]:
        """
        Create a copy of headers with sensitive values redacted for safe logging.

        Shows first 8 characters of sensitive values for debugging purposes,
        with the rest redacted.

        Args:
            headers: HTTP headers dictionary

        Returns:
            New dictionary with sensitive headers redacted
        """
        if not headers:
            return {}

        redacted: Final = {}
        sensitive_headers: Final = {
            "authorization",
            "x-api-key",
            "api-key",
            "x-databricks-token",
        }

        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                if len(value) > 10:
                    redacted[key] = f"{value[:8]}...[REDACTED]"
                else:
                    redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value

        return redacted

    @staticmethod
    def resolve_api_key(api_key: str | None) -> str | None:
        return api_key or litellm.api_key or litellm.databricks_key or get_secret_str("DATABRICKS_API_KEY")

    @staticmethod
    def resolve_api_base(api_base: str | None) -> str | None:
        return api_base or litellm.api_base or os.getenv("DATABRICKS_API_BASE")

    @staticmethod
    def resolve_oauth_credentials(
        litellm_params: Mapping[str, object] | object | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> tuple[str, str] | None:
        return resolve_databricks_oauth_credentials(litellm_params, client_id, client_secret)

    @staticmethod
    def serves_native_anthropic_messages(model: str) -> bool:
        return model.removeprefix("databricks/").lower().startswith(DATABRICKS_NATIVE_MESSAGES_MODEL_PREFIXES)

    @staticmethod
    def uses_ai_gateway(model: str) -> bool:
        return model.removeprefix("databricks/").count(".") >= 2

    @staticmethod
    def recognizes_workspace_route(api_base: str, model: str) -> bool:
        route: Final = urlsplit(api_base).path.rstrip("/")
        return route.endswith(DATABRICKS_WORKSPACE_ROUTE_PREFIXES) or (
            route == "" and DatabricksBase.uses_ai_gateway(model)
        )

    @staticmethod
    def workspace_root(api_base: str) -> str:
        normalized: Final = api_base.rstrip("/")
        route_prefix: Final = next(
            (prefix for prefix in DATABRICKS_WORKSPACE_ROUTE_PREFIXES if normalized.endswith(prefix)), ""
        )
        return normalized.removesuffix(route_prefix)

    @staticmethod
    def configured_user_agent(litellm_params: Mapping[str, object]) -> str | None:
        configured: Final = (
            litellm_params.get("user_agent")
            or litellm_params.get("databricks_user_agent")
            or os.getenv("LITELLM_USER_AGENT")
            or os.getenv("DATABRICKS_USER_AGENT")
        )
        return configured if isinstance(configured, str) else None

    @staticmethod
    def _build_user_agent(custom_user_agent: str | None = None) -> str:
        """
        Build the User-Agent string for Databricks API calls.

        If a custom user agent is provided, the partner name (part before /)
        is extracted and prefixed to the litellm user agent with an underscore.
        The custom version is ignored; LiteLLM's version is always used.

        Args:
            custom_user_agent: Optional custom user agent string (e.g., "mycompany/1.0.0")

        Returns:
            User-Agent string in format:
            - Default: "litellm/{version}"
            - With custom: "{partner}_litellm/{version}"

        Examples:
            - None -> "litellm/1.79.1"
            - "mycompany/1.0.0" -> "mycompany_litellm/1.79.1"
            - "partner_product/2.0.0" -> "partner_product_litellm/1.79.1"
            - "acme" -> "acme_litellm/1.79.1"
        """
        try:
            from litellm._version import version
        except Exception:
            version = "0.0.0"

        if custom_user_agent:
            custom_user_agent = custom_user_agent.strip()

            # Extract partner name (part before / if present)
            if "/" in custom_user_agent:
                partner_name = custom_user_agent.split("/")[0].strip()
            else:
                partner_name = custom_user_agent

            # Validate partner name: alphanumeric, underscore, hyphen only
            if partner_name and partner_name.replace("_", "").replace("-", "").isalnum():
                return f"{partner_name}_litellm/{version}"

        # Default: just litellm
        return f"litellm/{version}"

    def _get_api_base(self, api_base: str | None, use_ai_gateway: bool = False) -> str:
        if api_base is None:
            try:
                from databricks.sdk import WorkspaceClient

                databricks_client: Final = WorkspaceClient()
                api_base = f"{databricks_client.config.host}/serving-endpoints"
            except ImportError:
                raise DatabricksException(
                    status_code=400,
                    message=(
                        "Either set the DATABRICKS_API_BASE and DATABRICKS_API_KEY environment variables, "
                        "or install the databricks-sdk Python library."
                    ),
                )

        if not use_ai_gateway:
            return api_base

        normalized_api_base: Final = api_base.rstrip("/")
        if normalized_api_base.endswith("/ai-gateway/mlflow/v1"):
            return normalized_api_base
        if normalized_api_base.endswith("/serving-endpoints"):
            return join_api_base(normalized_api_base.removesuffix("/serving-endpoints"), "ai-gateway/mlflow/v1")
        api_base_parts: Final = urlsplit(normalized_api_base)
        if api_base_parts.path in ("", "/"):
            return join_api_base(normalized_api_base, "ai-gateway/mlflow/v1")
        return api_base

    def get_oauth_m2m_token(
        self,
        api_base: str,
        client_id: str,
        client_secret: str,
    ) -> str:
        return self.get_oauth_m2m_token_and_lifetime(api_base, client_id, client_secret).access_token

    def get_cached_oauth_m2m_token(
        self,
        api_base: str,
        client_id: str,
        client_secret: str,
        *,
        token_cache: _TokenCache = _OAUTH_TOKENS,
    ) -> str:
        workspace: Final = self.workspace_root(api_base)
        secret_digest: Final = hashlib.sha256(client_secret.encode()).hexdigest()
        cache_key: Final = f"{workspace}|{client_id}|{secret_digest}"
        cached: Final = token_cache.get_cache(cache_key)
        if isinstance(cached, str):
            return cached
        token: Final = self.get_oauth_m2m_token_and_lifetime(workspace, client_id, client_secret)
        ttl: Final = token.expires_in - _TOKEN_EXPIRY_BUFFER_SECONDS
        if ttl > 0:
            token_cache.set_cache(cache_key, token.access_token, ttl=ttl)
        return token.access_token

    def get_oauth_m2m_token_and_lifetime(
        self,
        api_base: str,
        client_id: str,
        client_secret: str,
    ) -> OAuthM2MToken:
        """
        Obtain an OAuth M2M access token using client credentials flow.

        This is the recommended authentication method for production integrations
        per Databricks Partner requirements.

        Args:
            api_base: Databricks workspace URL
            client_id: OAuth client ID (Service Principal application ID)
            client_secret: OAuth client secret

        Returns:
            The access token and its lifetime in seconds

        Raises:
            DatabricksException: If token request fails
        """
        import requests

        api_base_parts: Final = urlsplit(api_base)
        workspace_url: Final = urlunsplit((api_base_parts.scheme, api_base_parts.netloc, "", "", ""))
        token_url: Final = join_api_base(workspace_url, "oidc/v1/token")

        try:
            response: Final = requests.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "scope": "all-apis",
                },
                auth=(client_id, client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
        except requests.RequestException as e:
            raise DatabricksException(
                status_code=500,
                message=f"OAuth M2M token request failed: {e}",
            )

        if response.status_code != 200:
            raise DatabricksException(
                status_code=response.status_code,
                message=f"OAuth M2M token request failed: {response.text}",
            )

        return OAuthM2MToken.model_validate(response.json())

    def _get_databricks_credentials(
        self, api_key: str | None, api_base: str | None, headers: dict | None
    ) -> tuple[str, dict]:
        """
        Get Databricks credentials using the Databricks SDK.

        Also registers LiteLLM as a partner for proper telemetry attribution
        in Databricks system.access.audit table.

        Args:
            api_key: Optional API key (PAT)
            api_base: Optional API base URL
            headers: Optional existing headers

        Returns:
            Tuple of (api_base, headers)
        """
        headers = headers or {"Content-Type": "application/json"}
        try:
            from databricks.sdk import WorkspaceClient, useragent

            # Register LiteLLM as partner for Databricks telemetry attribution
            useragent.with_partner("litellm")

            databricks_client: Final = WorkspaceClient()

            api_base = api_base or f"{databricks_client.config.host}/serving-endpoints"

            if api_key is None:
                databricks_auth_headers: Final[dict[str, str]] = databricks_client.config.authenticate()
                headers = {**databricks_auth_headers, **headers}

            return api_base, headers
        except ImportError:
            raise DatabricksException(
                status_code=400,
                message=(
                    "If the Databricks base URL and API key are not set, the databricks-sdk "
                    "Python library must be installed. Please install the databricks-sdk, set "
                    "{LLM_PROVIDER}_API_BASE and {LLM_PROVIDER}_API_KEY environment variables, "
                    "or provide the base URL and API key as arguments."
                ),
            )

    def databricks_validate_environment(
        self,
        api_key: str | None,
        api_base: str | None,
        endpoint_type: Literal["chat_completions", "embeddings", "anthropic_messages"],
        custom_endpoint: bool | None,
        headers: dict | None,
        custom_user_agent: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        litellm_params: Mapping[str, object] | object | None = None,
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: becomes the outbound headers
        """
        Validate and configure the Databricks environment.

        Authentication priority:
        1. Deployment static API key (api_key)
        2. OAuth M2M (deployment or global DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET)
        3. PAT (DATABRICKS_API_KEY) / Databricks SDK automatic auth - Fallback (uses unified auth)

        Args:
            api_key: Personal access token (PAT)
            api_base: Databricks workspace URL with /serving-endpoints
            endpoint_type: Type of endpoint (chat_completions, embeddings or anthropic_messages)
            custom_endpoint: Whether using a custom endpoint URL
            headers: Existing headers dict
            custom_user_agent: Optional custom user agent to prefix
            client_id: Optional OAuth client ID
            client_secret: Optional OAuth client secret
            litellm_params: Optional deployment litellm_params

        Returns:
            Tuple of (api_base, headers) with authentication configured
        """
        from litellm._logging import verbose_logger

        # Determine api_base first
        if api_base is None:
            api_base = os.getenv("DATABRICKS_API_BASE")

        deployment_api_key: Final = (
            _get_credential_param(litellm_params, "api_key") if litellm_params is not None else None
        )
        has_deployment_oauth: Final = bool(
            client_id is not None
            or client_secret is not None
            or _get_credential_param(litellm_params, "client_id", "databricks_client_id") is not None
            or _get_credential_param(litellm_params, "client_secret", "databricks_client_secret") is not None
        )
        global_key: Final = get_secret_str("DATABRICKS_API_KEY") or get_secret_str("DATABRICKS_TOKEN")
        is_ambient_global_key: Final = bool(
            has_deployment_oauth
            and deployment_api_key is None
            and api_key is not None
            and (api_key == global_key or api_key == litellm.api_key or api_key == litellm.databricks_key)
        )
        effective_static_key: Final = None if is_ambient_global_key else (deployment_api_key or api_key)

        if effective_static_key is not None:
            # Deployment static API key takes precedence over OAuth M2M
            if api_base is None:
                if custom_endpoint:
                    raise DatabricksException(
                        status_code=400,
                        message="Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                    )
                else:
                    api_base, headers = self._get_databricks_credentials(
                        api_base=api_base, api_key=effective_static_key, headers=headers
                    )
            headers = headers or {}
            headers["Authorization"] = f"Bearer {effective_static_key}"
            headers["Content-Type"] = "application/json"
        else:
            oauth_credentials: Final = resolve_databricks_oauth_credentials(
                litellm_params=litellm_params,
                client_id=client_id,
                client_secret=client_secret,
            )
            if oauth_credentials is not None:
                oauth_id, oauth_secret = oauth_credentials
                if api_base is None:
                    if custom_endpoint:
                        raise DatabricksException(
                            status_code=400,
                            message="Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                        )
                    else:
                        api_base, _ = self._get_databricks_credentials(api_base=api_base, api_key=None, headers=headers)
                verbose_logger.debug("Using OAuth M2M authentication for Databricks")
                access_token: Final = self.get_oauth_m2m_token(api_base, oauth_id, oauth_secret)
                headers = headers or {}
                headers["Authorization"] = f"Bearer {access_token}"
                headers["Content-Type"] = "application/json"
            elif api_key is not None:
                # Ambient global DATABRICKS_API_KEY fallback
                if api_base is None:
                    if custom_endpoint:
                        raise DatabricksException(
                            status_code=400,
                            message="Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                        )
                    else:
                        api_base, headers = self._get_databricks_credentials(
                            api_base=api_base, api_key=api_key, headers=headers
                        )
                headers = headers or {}
                headers["Authorization"] = f"Bearer {api_key}"
                headers["Content-Type"] = "application/json"
            elif not headers:
                if custom_endpoint is True:
                    raise DatabricksException(
                        status_code=400,
                        message="Missing API Key - A call is being made to LLM Provider but no key is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                    )
                else:
                    # Fallback to Databricks SDK (registers partner telemetry)
                    verbose_logger.debug("Using Databricks SDK for authentication")
                    api_base, headers = self._get_databricks_credentials(
                        api_base=api_base, api_key=None, headers=headers
                    )

        if api_base is None:
            if custom_endpoint:
                raise DatabricksException(
                    status_code=400,
                    message="Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                )
            else:
                api_base, headers = self._get_databricks_credentials(
                    api_base=api_base, api_key=api_key, headers=headers
                )

        if headers is None:
            headers = {
                "Content-Type": "application/json",
            }
            if effective_static_key is not None:
                headers["Authorization"] = f"Bearer {effective_static_key}"
            elif api_key is not None:
                headers["Authorization"] = f"Bearer {api_key}"
        elif "Authorization" not in headers:
            if effective_static_key is not None:
                headers["Authorization"] = f"Bearer {effective_static_key}"
            elif api_key is not None:
                headers["Authorization"] = f"Bearer {api_key}"

        # Set User-Agent with optional custom prefix
        headers["User-Agent"] = self._build_user_agent(custom_user_agent)

        # Debug logging with redaction (never log actual tokens)
        verbose_logger.debug("Databricks request headers: %s", self.redact_headers_for_logging(headers))

        if custom_endpoint is True:
            return api_base, headers
        return join_api_base(api_base, DATABRICKS_ENDPOINT_PATHS[endpoint_type]), headers
