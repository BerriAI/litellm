"""
Databricks integration utilities for LiteLLM.

This module provides authentication, telemetry, and security utilities
for the Databricks LLM provider integration.

Authentication priority:
1. PAT (api_key, else DATABRICKS_API_KEY when the deployment sets no OAuth credentials)
2. OAuth M2M (client_id + client_secret on the deployment, else DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET)
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
from litellm.litellm_core_utils.credential_rejection import credential_rejection_in
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


class DatabricksOAuthRejectionError(DatabricksException):
    pass


def caused_by_databricks_oauth_rejection(error: BaseException) -> bool:
    return credential_rejection_in(error, DatabricksOAuthRejectionError) is not None


def databricks_oauth_environment_defaults() -> Mapping[str, object]:
    return MappingProxyType(
        {"client_id": os.getenv("DATABRICKS_CLIENT_ID"), "client_secret": os.getenv("DATABRICKS_CLIENT_SECRET")}
    )


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


def deployment_oauth_credentials(client_id: object, client_secret: object) -> tuple[str, str] | None:
    if not client_id and not client_secret:
        return None
    if isinstance(client_id, str) and client_id and isinstance(client_secret, str) and client_secret:
        return client_id, client_secret
    missing: Final = "client_secret" if isinstance(client_id, str) and client_id else "client_id"
    raise DatabricksException(
        status_code=400,
        message=(
            f"Incomplete Databricks OAuth credentials: missing {missing}. "
            "Set both client_id and client_secret, or neither"
        ),
    )


class DeploymentOAuthParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    client_id: object = None
    client_secret: object = None


class OAuthM2MToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    access_token: str
    expires_in: int = _DEFAULT_M2M_TOKEN_LIFETIME_SECONDS


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
    def resolve_api_key(api_key: str | None, client_id: object = None, client_secret: object = None) -> str | None:
        if api_key or deployment_oauth_credentials(client_id, client_secret):
            return api_key or None
        return litellm.api_key or litellm.databricks_key or get_secret_str("DATABRICKS_API_KEY")

    @staticmethod
    def resolve_api_base(api_base: str | None) -> str | None:
        return api_base or litellm.api_base or os.getenv("DATABRICKS_API_BASE")

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
            raise DatabricksOAuthRejectionError(
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

    @staticmethod
    def _oauth_credentials(client_id: object, client_secret: object) -> tuple[str, str] | None:
        deployment_credentials: Final = deployment_oauth_credentials(client_id, client_secret)
        if deployment_credentials:
            return deployment_credentials
        env_client_id: Final = os.getenv("DATABRICKS_CLIENT_ID")
        env_client_secret: Final = os.getenv("DATABRICKS_CLIENT_SECRET")
        return (env_client_id, env_client_secret) if env_client_id and env_client_secret else None

    def databricks_validate_environment(
        self,
        api_key: str | None,
        api_base: str | None,
        endpoint_type: Literal["chat_completions", "embeddings", "anthropic_messages"],
        custom_endpoint: bool | None,
        headers: dict | None,
        custom_user_agent: str | None = None,
        client_id: object = None,
        client_secret: object = None,
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: becomes the outbound headers
        """
        Validate and configure the Databricks environment.

        Authentication priority:
        1. PAT (api_key)
        2. OAuth M2M (client_id + client_secret, else DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET)
        3. Databricks SDK automatic auth - Fallback (uses unified auth)

        Args:
            api_key: Personal access token (PAT)
            api_base: Databricks workspace URL with /serving-endpoints
            endpoint_type: Type of endpoint (chat_completions, embeddings or anthropic_messages)
            custom_endpoint: Whether using a custom endpoint URL
            headers: Existing headers dict
            custom_user_agent: Optional custom user agent to prefix
            client_id: The deployment's OAuth M2M client ID
            client_secret: The deployment's OAuth M2M client secret

        Returns:
            Tuple of (api_base, headers) with authentication configured
        """
        from litellm._logging import verbose_logger

        oauth_credentials: Final = None if api_key is not None else self._oauth_credentials(client_id, client_secret)

        # Determine api_base first
        if api_base is None:
            api_base = os.getenv("DATABRICKS_API_BASE")

        if api_base is None and api_key is None and deployment_oauth_credentials(client_id, client_secret):
            raise DatabricksException(
                status_code=400,
                message=(
                    "Missing api_base for a deployment that sets client_id and client_secret, "
                    "so set api_base or DATABRICKS_API_BASE to the workspace's /serving-endpoints URL"
                ),
            )

        if oauth_credentials and api_base:
            # Use OAuth M2M flow (preferred for production)
            verbose_logger.debug("Using OAuth M2M authentication for Databricks")
            access_token: Final = self.get_oauth_m2m_token(api_base, *oauth_credentials)
            headers = headers or {}
            headers["Authorization"] = f"Bearer {access_token}"
            headers["Content-Type"] = "application/json"
        elif api_key is None and not headers:
            if custom_endpoint is True:
                raise DatabricksException(
                    status_code=400,
                    message="Missing API Key - A call is being made to LLM Provider but no key is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                )
            else:
                # Fallback to Databricks SDK (registers partner telemetry)
                verbose_logger.debug("Using Databricks SDK for authentication")
                api_base, headers = self._get_databricks_credentials(
                    api_base=api_base, api_key=api_key, headers=headers
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
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        else:
            if api_key is not None:
                headers.update({"Authorization": f"Bearer {api_key}"})

        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        # Set User-Agent with optional custom prefix
        headers["User-Agent"] = self._build_user_agent(custom_user_agent)

        # Debug logging with redaction (never log actual tokens)
        verbose_logger.debug("Databricks request headers: %s", self.redact_headers_for_logging(headers))

        if custom_endpoint is True:
            return api_base, headers
        return join_api_base(api_base, DATABRICKS_ENDPOINT_PATHS[endpoint_type]), headers
