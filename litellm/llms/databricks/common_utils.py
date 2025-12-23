"""
Databricks integration utilities for LiteLLM.

This module provides authentication, telemetry, and security utilities
for the Databricks LLM provider integration.

Authentication priority:
1. OAuth M2M (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET) - Recommended for production
2. PAT (DATABRICKS_API_KEY) - Supported for development
3. Databricks SDK automatic auth - Fallback (uses unified auth)
"""

import os
import re
from typing import Any, Dict, Literal, Optional, Tuple

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class DatabricksException(BaseLLMException):
    pass


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
    def redact_sensitive_data(cls, data: Any) -> Any:
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
            redacted = {}
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
    def redact_headers_for_logging(cls, headers: Dict[str, str]) -> Dict[str, str]:
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

        redacted = {}
        sensitive_headers = {
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
    def _build_user_agent(custom_user_agent: Optional[str] = None) -> str:
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
            if (
                partner_name
                and partner_name.replace("_", "").replace("-", "").isalnum()
            ):
                return f"{partner_name}_litellm/{version}"

        # Default: just litellm
        return f"litellm/{version}"

    def _get_api_base(self, api_base: Optional[str]) -> str:
        """
        Get the Databricks API base URL.

        If not provided, attempts to get it from the Databricks SDK.
        """
        if api_base is None:
            try:
                from databricks.sdk import WorkspaceClient

                databricks_client = WorkspaceClient()
                api_base = f"{databricks_client.config.host}/serving-endpoints"
                return api_base
            except ImportError:
                raise DatabricksException(
                    status_code=400,
                    message=(
                        "Either set the DATABRICKS_API_BASE and DATABRICKS_API_KEY environment variables, "
                        "or install the databricks-sdk Python library."
                    ),
                )
        return api_base

    def _get_oauth_m2m_token(
        self,
        api_base: str,
        client_id: str,
        client_secret: str,
    ) -> str:
        """
        Obtain an OAuth M2M access token using client credentials flow.

        This is the recommended authentication method for production integrations
        per Databricks Partner requirements.

        Args:
            api_base: Databricks workspace URL
            client_id: OAuth client ID (Service Principal application ID)
            client_secret: OAuth client secret

        Returns:
            Access token string

        Raises:
            DatabricksException: If token request fails
        """
        import requests

        # Extract workspace URL from api_base
        workspace_url = api_base.rstrip("/")
        if "/serving-endpoints" in workspace_url:
            workspace_url = workspace_url.replace("/serving-endpoints", "")

        token_url = f"{workspace_url}/oidc/v1/token"

        try:
            response = requests.post(
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
                message=f"OAuth M2M token request failed: {str(e)}",
            )

        if response.status_code != 200:
            raise DatabricksException(
                status_code=response.status_code,
                message=f"OAuth M2M token request failed: {response.text}",
            )

        token_data = response.json()
        return token_data["access_token"]

    def _get_databricks_credentials(
        self, api_key: Optional[str], api_base: Optional[str], headers: Optional[dict]
    ) -> Tuple[str, dict]:
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

            databricks_client = WorkspaceClient()

            api_base = api_base or f"{databricks_client.config.host}/serving-endpoints"

            if api_key is None:
                databricks_auth_headers: dict[
                    str, str
                ] = databricks_client.config.authenticate()
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
        api_key: Optional[str],
        api_base: Optional[str],
        endpoint_type: Literal["chat_completions", "embeddings"],
        custom_endpoint: Optional[bool],
        headers: Optional[dict],
        custom_user_agent: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Validate and configure the Databricks environment.

        Authentication priority:
        1. OAuth M2M (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET) - Recommended
        2. PAT (DATABRICKS_API_KEY) - Supported for development
        3. Databricks SDK automatic auth - Fallback (uses unified auth)

        Args:
            api_key: Personal access token (PAT)
            api_base: Databricks workspace URL with /serving-endpoints
            endpoint_type: Type of endpoint (chat_completions or embeddings)
            custom_endpoint: Whether using a custom endpoint URL
            headers: Existing headers dict
            custom_user_agent: Optional custom user agent to prefix

        Returns:
            Tuple of (api_base, headers) with authentication configured
        """
        from litellm._logging import verbose_logger

        # Check for OAuth M2M credentials (recommended for production)
        client_id = os.getenv("DATABRICKS_CLIENT_ID")
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")

        # Determine api_base first
        if api_base is None:
            api_base = os.getenv("DATABRICKS_API_BASE")

        if client_id and client_secret and api_base:
            # Use OAuth M2M flow (preferred for production)
            verbose_logger.debug("Using OAuth M2M authentication for Databricks")
            access_token = self._get_oauth_m2m_token(api_base, client_id, client_secret)
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
                "Authorization": "Bearer {}".format(api_key),
                "Content-Type": "application/json",
            }
        else:
            if api_key is not None:
                headers.update({"Authorization": "Bearer {}".format(api_key)})

        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        # Set User-Agent with optional custom prefix
        headers["User-Agent"] = self._build_user_agent(custom_user_agent)

        # Debug logging with redaction (never log actual tokens)
        verbose_logger.debug(
            f"Databricks request headers: {self.redact_headers_for_logging(headers)}"
        )

        if endpoint_type == "chat_completions" and custom_endpoint is not True:
            api_base = "{}/chat/completions".format(api_base)
        elif endpoint_type == "embeddings" and custom_endpoint is not True:
            api_base = "{}/embeddings".format(api_base)

        return api_base, headers
