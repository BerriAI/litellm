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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider

DATABRICKS_OAUTH_PARAM: Final = "databricks_oauth"
DATABRICKS_AGENT_PROVIDER: Final = "databricks_agent"
_FLAT_OAUTH_FIELDS: Final = ("client_id", "client_secret", "workspace_url", "scope")

_DEFAULT_SCOPE: Final = "all-apis"
_TOKEN_EXPIRY_BUFFER_SECONDS: Final = 60
_DEFAULT_TTL_SECONDS: Final = 3600
_OAUTH_BLOCK: Final = TypeAdapter(Mapping[str, object])


def _resolve_secret(value: object) -> str | None:
    """Resolve a config value, expanding ``os.environ/`` references."""
    if not isinstance(value, str):
        return None
    if value.startswith("os.environ/"):
        return get_secret_str(value)
    return value


def _token_url_from_workspace(workspace_url: str) -> str:
    """Build the workspace OIDC token endpoint from a workspace URL."""
    base = workspace_url.strip().rstrip("/")
    base = base.removesuffix("/serving-endpoints")
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
        secret_digest: Final = hashlib.sha256(self.client_secret.encode()).hexdigest()[:16]
        return f"{self.token_url}|{self.client_id}|{self.scope}|{secret_digest}"


def _workspace_origin(url: object) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    parts: Final = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}"


def _is_databricks_agent(litellm_params: Mapping[str, object]) -> bool:
    return litellm_params.get("custom_llm_provider") == DATABRICKS_AGENT_PROVIDER


def _flat_databricks_agent_oauth(litellm_params: Mapping[str, object]) -> Mapping[str, object] | None:
    """The Admin UI catalog cannot nest fields, so a ``databricks_agent`` spells its OAuth block as flat
    ``client_id`` / ``client_secret`` / ``workspace_url`` / ``scope``. ``workspace_url`` falls back to the
    ``api_base`` origin, which holds for Model Serving but not for a Databricks App host."""
    if not _is_databricks_agent(litellm_params):
        return None
    if not litellm_params.get("client_id") and not litellm_params.get("client_secret"):
        return None
    return {
        "client_id": litellm_params.get("client_id"),
        "client_secret": litellm_params.get("client_secret"),
        "workspace_url": litellm_params.get("workspace_url")
        or _workspace_origin(_resolve_secret(litellm_params.get("api_base"))),
        "scope": litellm_params.get("scope"),
    }


def _raw_oauth_block(litellm_params: Mapping[str, object]) -> object:
    return litellm_params.get(DATABRICKS_OAUTH_PARAM) or _flat_databricks_agent_oauth(litellm_params)


def has_databricks_oauth(litellm_params: Mapping[str, object]) -> bool:
    return bool(_raw_oauth_block(litellm_params))


def without_databricks_oauth_params(
    litellm_params: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: asend_message takes litellm_params as a dict
    consumed: Final = (
        frozenset({DATABRICKS_OAUTH_PARAM, *_FLAT_OAUTH_FIELDS})
        if _is_databricks_agent(litellm_params)
        else frozenset({DATABRICKS_OAUTH_PARAM})
    )
    return {key: value for key, value in litellm_params.items() if key not in consumed}


def parse_databricks_oauth_config(
    litellm_params: Mapping[str, object] | None,
) -> DatabricksAppOAuthConfig | None:
    """Build a Databricks App OAuth config from an agent's ``litellm_params``.

    Returns ``None`` when the agent has no ``databricks_oauth`` block. Raises
    ``ValueError`` when the block is present but incomplete, so misconfiguration
    surfaces loudly instead of silently sending an unauthenticated request.
    """
    if not litellm_params:
        return None

    raw: Final = _raw_oauth_block(litellm_params)
    if not raw:
        return None
    try:
        block: Final = _OAUTH_BLOCK.validate_python(raw)
    except ValidationError as e:
        raise ValueError(
            f"'{DATABRICKS_OAUTH_PARAM}' must be a mapping of OAuth settings, got {type(raw).__name__}"
        ) from e

    client_id: Final = _resolve_secret(block.get("client_id"))
    client_secret: Final = _resolve_secret(block.get("client_secret"))
    workspace_url: Final = _resolve_secret(block.get("workspace_url"))

    if not (client_id and client_secret and workspace_url):
        missing: Final = tuple(
            name
            for name, value in (
                ("client_id", client_id),
                ("client_secret", client_secret),
                ("workspace_url", workspace_url),
            )
            if not value
        )
        raise ValueError(f"Databricks App OAuth config is missing required field(s): {', '.join(missing)}")

    scope: Final = _resolve_secret(block.get("scope")) or _DEFAULT_SCOPE

    return DatabricksAppOAuthConfig(
        client_id=client_id,
        client_secret=client_secret,
        token_url=_token_url_from_workspace(workspace_url),
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
        self._locks: dict[str, asyncio.Lock] = {}

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
        cache_key: Final = config.cache_key

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

    async def _fetch_token(self, config: DatabricksAppOAuthConfig) -> tuple[str, int]:
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.A2A)

        verbose_logger.debug("Fetching Databricks App OAuth token from %s", config.token_url)

        basic_auth: Final = base64.b64encode(f"{config.client_id}:{config.client_secret}".encode()).decode()
        try:
            response: Final = await client.post(
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
                f"Databricks App OAuth token request failed with status {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ValueError(f"Databricks App OAuth token request failed: {exc}") from exc

        body: Final[object] = response.json()
        if not isinstance(body, dict):
            raise ValueError(
                f"Databricks App OAuth token response returned non-object JSON (got {type(body).__name__})"
            )

        access_token: Final = body.get("access_token")
        if not access_token:
            raise ValueError("Databricks App OAuth token response missing 'access_token'")

        raw_expires_in: Final = body.get("expires_in")
        try:
            expires_in = int(raw_expires_in) if raw_expires_in is not None else _DEFAULT_TTL_SECONDS
        except (TypeError, ValueError):
            expires_in = _DEFAULT_TTL_SECONDS

        ttl: Final = max(expires_in - _TOKEN_EXPIRY_BUFFER_SECONDS, 0)
        return access_token, ttl


databricks_app_oauth_token_cache: Final = DatabricksAppOAuthTokenCache()


async def resolve_databricks_app_auth_header(
    litellm_params: Mapping[str, object] | None,
) -> dict[str, str] | None:
    """Return ``{"Authorization": "Bearer <token>"}`` for a Databricks App agent.

    Returns ``None`` when the agent is not configured for Databricks App OAuth.
    """
    config: Final = parse_databricks_oauth_config(litellm_params)
    if config is None:
        return None

    token: Final = await databricks_app_oauth_token_cache.async_get_token(config)
    return {"Authorization": f"Bearer {token}"}
