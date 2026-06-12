"""
OAuth M2M (client_credentials) support for A2A agents that target Databricks
App endpoints.

Databricks Apps reject static bearer tokens; they require a short-lived OAuth
access token minted from the workspace OIDC token endpoint. When an agent is
registered with a ``databricks_oauth`` block in its ``litellm_params``, LiteLLM
fetches that token via the client_credentials grant, caches it until shortly
before expiry, and attaches it as the outbound ``Authorization`` header on every
call the proxy makes to the agent.

Config example::

    agents:
      - agent_name: my-databricks-app
        agent_card_params:
          url: https://my-app-1234.aws.databricksapps.com
        litellm_params:
          databricks_oauth:
            client_id: os.environ/DATABRICKS_CLIENT_ID
            client_secret: os.environ/DATABRICKS_CLIENT_SECRET
            workspace_url: https://dbc-abc123.cloud.databricks.com
"""

import asyncio
import base64
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider

DATABRICKS_OAUTH_PARAM = "databricks_oauth"

_DEFAULT_SCOPE = "all-apis"
_TOKEN_EXPIRY_BUFFER_SECONDS = 60
_DEFAULT_TTL_SECONDS = 3600


def _resolve_secret(value: Any) -> Optional[str]:
    """Resolve a config value, expanding ``os.environ/`` references."""
    if not isinstance(value, str):
        return None
    if value.startswith("os.environ/"):
        return get_secret_str(value)
    return value


def _token_url_from_workspace(workspace_url: str) -> str:
    """Build the workspace OIDC token endpoint from a workspace URL."""
    base = workspace_url.strip().rstrip("/")
    if base.endswith("/serving-endpoints"):
        base = base[: -len("/serving-endpoints")]
    return f"{base}/oidc/v1/token"


@dataclass(frozen=True)
class DatabricksAppOAuthConfig:
    client_id: str
    client_secret: str
    token_url: str
    scope: str

    @property
    def cache_key(self) -> str:
        # Include a digest of the secret so a rotated client_secret yields a new
        # key and forces a fresh token instead of serving the stale one.
        secret_digest = hashlib.sha256(self.client_secret.encode()).hexdigest()[:16]
        return f"{self.token_url}|{self.client_id}|{self.scope}|{secret_digest}"


def parse_databricks_oauth_config(
    litellm_params: Optional[Dict[str, Any]],
) -> Optional[DatabricksAppOAuthConfig]:
    """Build a Databricks App OAuth config from an agent's ``litellm_params``.

    Returns ``None`` when the agent has no ``databricks_oauth`` block. Raises
    ``ValueError`` when the block is present but incomplete, so misconfiguration
    surfaces loudly instead of silently sending an unauthenticated request.
    """
    if not litellm_params:
        return None

    raw = litellm_params.get(DATABRICKS_OAUTH_PARAM)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"'{DATABRICKS_OAUTH_PARAM}' must be a mapping of OAuth settings, "
            f"got {type(raw).__name__}"
        )

    client_id = _resolve_secret(raw.get("client_id"))
    client_secret = _resolve_secret(raw.get("client_secret"))
    workspace_url = _resolve_secret(raw.get("workspace_url"))

    missing = [
        name
        for name, value in (
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("workspace_url", workspace_url),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Databricks App OAuth config is missing required field(s): "
            f"{', '.join(missing)}"
        )

    scope = _resolve_secret(raw.get("scope")) or _DEFAULT_SCOPE

    return DatabricksAppOAuthConfig(
        client_id=client_id,  # type: ignore[arg-type]
        client_secret=client_secret,  # type: ignore[arg-type]
        token_url=_token_url_from_workspace(workspace_url),  # type: ignore[arg-type]
        scope=scope,
    )


class DatabricksAppOAuthTokenCache(InMemoryCache):
    """In-memory cache for Databricks App OAuth client_credentials tokens.

    Keyed by token endpoint + client_id + scope so distinct agents and service
    principals never share a token. A per-key ``asyncio.Lock`` collapses
    concurrent fetches into a single token request.
    """

    def __init__(self) -> None:
        super().__init__(default_ttl=_DEFAULT_TTL_SECONDS)
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, cache_key: str) -> asyncio.Lock:
        return self._locks.setdefault(cache_key, asyncio.Lock())

    def _remove_key(self, key: str) -> None:
        # Drop the per-key lock alongside the cached token so ``_locks`` stays
        # bounded by the live key set rather than growing for every key ever seen.
        super()._remove_key(key)
        self._locks.pop(key, None)

    def flush_cache(self) -> None:
        super().flush_cache()
        self._locks.clear()

    async def async_get_token(self, config: DatabricksAppOAuthConfig) -> str:
        cache_key = config.cache_key

        cached = self.get_cache(cache_key)
        if cached is not None:
            return cached

        async with self._get_lock(cache_key):
            cached = self.get_cache(cache_key)
            if cached is not None:
                return cached

            token, ttl = await self._fetch_token(config)
            # ttl == 0 means the token's own lifetime is shorter than the
            # refresh buffer; skip caching so we never hand out a stale token,
            # and drop the lock we just created since no cached entry will ever
            # trigger _remove_key to clean it up.
            if ttl > 0:
                self.set_cache(cache_key, token, ttl=ttl)
            else:
                self._locks.pop(cache_key, None)
            return token

    async def _fetch_token(self, config: DatabricksAppOAuthConfig) -> Tuple[str, int]:
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.A2A)

        verbose_logger.debug(
            "Fetching Databricks App OAuth token from %s", config.token_url
        )

        basic_auth = base64.b64encode(
            f"{config.client_id}:{config.client_secret}".encode()
        ).decode()
        try:
            response = await client.post(
                config.token_url,
                data={
                    "grant_type": "client_credentials",
                    "scope": config.scope,
                },
                headers={
                    "Authorization": f"Basic {basic_auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        except httpx.HTTPStatusError as exc:
            raise ValueError(
                "Databricks App OAuth token request failed with status "
                f"{exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ValueError(
                f"Databricks App OAuth token request failed: {exc}"
            ) from exc

        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(
                "Databricks App OAuth token response returned non-object JSON "
                f"(got {type(body).__name__})"
            )

        access_token = body.get("access_token")
        if not access_token:
            raise ValueError(
                "Databricks App OAuth token response missing 'access_token'"
            )

        raw_expires_in = body.get("expires_in")
        try:
            expires_in = (
                int(raw_expires_in)
                if raw_expires_in is not None
                else _DEFAULT_TTL_SECONDS
            )
        except (TypeError, ValueError):
            expires_in = _DEFAULT_TTL_SECONDS

        ttl = max(expires_in - _TOKEN_EXPIRY_BUFFER_SECONDS, 0)
        return access_token, ttl


databricks_app_oauth_token_cache = DatabricksAppOAuthTokenCache()


async def resolve_databricks_app_auth_header(
    litellm_params: Optional[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    """Return ``{"Authorization": "Bearer <token>"}`` for a Databricks App agent.

    Returns ``None`` when the agent is not configured for Databricks App OAuth.
    """
    config = parse_databricks_oauth_config(litellm_params)
    if config is None:
        return None

    token = await databricks_app_oauth_token_cache.async_get_token(config)
    return {"Authorization": f"Bearer {token}"}
