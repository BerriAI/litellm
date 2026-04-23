"""
DB-backed GitHub Copilot authenticator.

Mirrors the filesystem-backed :class:`Authenticator` but reads the long-lived
GitHub OAuth access token from ``litellm.credential_list`` (the in-memory
decrypted cache) and persists it to ``LiteLLM_CredentialsTable`` via a
fire-and-forget background thread.

The short-lived Copilot API key (obtained from
``/copilot_internal/v2/token``) is cached in-memory per-process keyed by
credential name — it rotates frequently (~30 min), is cheap to refresh, and
cross-replica coherence is unnecessary.
"""

import asyncio
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.types.utils import CredentialItem

from .authenticator import Authenticator
from .common_utils import GetAccessTokenError, GetAPIKeyError

CREDENTIAL_TYPE = "copilot_oauth"


class DBAuthenticator(Authenticator):
    """
    Uses ``LiteLLM_CredentialsTable`` as the backing store for the GitHub
    access token. The Copilot API key lives only in a per-process cache
    (``_api_key_cache``).
    """

    # Shared across instances so multiple requests for the same credential
    # reuse the same cached API key payload without re-hitting GitHub.
    _api_key_cache: Dict[str, Dict[str, Any]] = {}
    _api_key_cache_lock = threading.Lock()

    def __init__(self, credential_name: str) -> None:
        self.credential_name = credential_name
        # Parent fields that aren't used by this subclass — kept non-None so
        # accidental references fail loudly rather than silently hitting disk.
        self.token_dir = ""
        self.access_token_file = ""
        self.api_key_file = ""

    def _ensure_token_dir(self) -> None:
        return

    def get_access_token(self) -> str:
        values = CredentialAccessor.get_credential_values(self.credential_name)
        token = values.get("access_token") if values else None
        if not token:
            raise GetAccessTokenError(
                message=(
                    f"No GitHub access token stored for credential "
                    f"'{self.credential_name}'. Sign in via the UI first."
                ),
                status_code=401,
            )
        return token

    def get_api_key(self) -> str:
        cached = self._get_cached_api_key()
        if cached is not None:
            return cached
        info = self._refresh_api_key()
        self._cache_api_key(info)
        token = info.get("token")
        if not token:
            raise GetAPIKeyError(
                message="API key response missing token",
                status_code=401,
            )
        return token

    def get_api_base(self) -> Optional[str]:
        with self._api_key_cache_lock:
            info = self._api_key_cache.get(self.credential_name)
        if info is None:
            return None
        endpoints = info.get("endpoints") or {}
        return endpoints.get("api")

    def force_refresh_api_key(self) -> Dict[str, Any]:
        """
        Force a call to ``/copilot_internal/v2/token`` even if the cached
        key is still valid. Used by the UI's "Refresh" button.
        """
        info = self._refresh_api_key()
        self._cache_api_key(info)
        return info

    def store_access_token(self, access_token: str) -> None:
        """
        Called by the OAuth login flow to persist a freshly-obtained GitHub
        access token. Writes to the in-memory credential cache and schedules
        a DB write.
        """
        item = CredentialItem(
            credential_name=self.credential_name,
            credential_values={"access_token": access_token},
            credential_info={
                "type": CREDENTIAL_TYPE,
                "custom_llm_provider": "github_copilot",
            },
        )
        CredentialAccessor.upsert_credentials([item])
        # Invalidate any cached API key tied to an old access token.
        with self._api_key_cache_lock:
            self._api_key_cache.pop(self.credential_name, None)
        _schedule_db_persist(item)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_cached_api_key(self) -> Optional[str]:
        with self._api_key_cache_lock:
            info = self._api_key_cache.get(self.credential_name)
        if info is None:
            return None
        if info.get("expires_at", 0) <= datetime.now().timestamp():
            return None
        return info.get("token")

    def _cache_api_key(self, info: Dict[str, Any]) -> None:
        with self._api_key_cache_lock:
            self._api_key_cache[self.credential_name] = info


# ---------------------------------------------------------------------------
# Fire-and-forget DB persistence
# ---------------------------------------------------------------------------

# See chatgpt/db_authenticator.py for the full rationale: prisma_client's
# internal asyncio primitives are bound to the proxy's main loop, so we
# must schedule persist coroutines back onto that loop rather than spin
# up a fresh one in a worker thread.
_proxy_main_loop: Optional["asyncio.AbstractEventLoop"] = None


def _register_proxy_main_loop(loop: "asyncio.AbstractEventLoop") -> None:
    global _proxy_main_loop
    _proxy_main_loop = loop


def _schedule_db_persist(item: CredentialItem) -> None:
    loop = _proxy_main_loop
    if loop is not None and loop.is_running():
        asyncio.run_coroutine_threadsafe(persist_credential_to_db(item), loop)
        return
    thread = threading.Thread(
        target=_persist_item_sync,
        args=(item,),
        daemon=True,
        name="copilot-oauth-persist",
    )
    thread.start()


def _persist_item_sync(item: CredentialItem) -> None:
    try:
        asyncio.run(persist_credential_to_db(item))
    except Exception as exc:
        verbose_logger.error(
            "Failed to persist Copilot OAuth credential %s: %s",
            item.credential_name,
            exc,
        )


async def persist_credential_to_db(item: CredentialItem) -> None:
    # Inline imports: proxy_server transitively imports this module via
    # the OAuth router (avoids circular), and prisma_client is a
    # module-level global mutated at proxy startup — a top-level import
    # would bind the stale None reference rather than the live client.
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
    from litellm.proxy.proxy_server import prisma_client
    from litellm.proxy.utils import jsonify_object

    if prisma_client is None:
        verbose_logger.debug(
            "prisma_client unavailable; skipping DB persist for %s",
            item.credential_name,
        )
        return

    encrypted_values = {
        k: encrypt_value_helper(v) for k, v in item.credential_values.items()
    }
    credential_info = item.credential_info or {
        "type": CREDENTIAL_TYPE,
        "custom_llm_provider": "github_copilot",
    }
    # Prisma's Json columns want JSON-serialized strings, not raw dicts.
    jsonified = jsonify_object(
        {
            "credential_values": encrypted_values,
            "credential_info": credential_info,
        }
    )
    await prisma_client.db.litellm_credentialstable.upsert(
        where={"credential_name": item.credential_name},
        data={
            "create": {
                "credential_name": item.credential_name,
                **jsonified,
                "created_by": "copilot_oauth_flow",
                "updated_by": "copilot_oauth_flow",
            },
            "update": {
                **jsonified,
                "updated_by": "copilot_oauth_flow",
            },
        },
    )


# ---------------------------------------------------------------------------
# api_key-prefix dispatch helper
# ---------------------------------------------------------------------------

OAUTH_CREDENTIAL_API_KEY_PREFIX = "oauth:"


def resolve_authenticator(
    api_key: Optional[str],
    litellm_params: Any,
    fallback: Authenticator,
) -> Authenticator:
    """
    If ``api_key`` (or ``litellm_params.api_key``) starts with ``oauth:``,
    the suffix names a credential in ``LiteLLM_CredentialsTable`` and this
    returns a :class:`DBAuthenticator` for it. Otherwise returns the given
    fallback (typically the filesystem-backed :class:`Authenticator`).

    Two sources are checked because upstream may rewrite ``api_key`` to the
    resolved Copilot token before ``validate_environment`` runs, while the
    raw marker survives on ``litellm_params``.
    """
    candidates = [api_key]
    if litellm_params is not None:
        if isinstance(litellm_params, dict):
            candidates.append(litellm_params.get("api_key"))
        else:
            candidates.append(getattr(litellm_params, "api_key", None))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.startswith(
            OAUTH_CREDENTIAL_API_KEY_PREFIX
        ):
            return DBAuthenticator(
                credential_name=candidate[len(OAUTH_CREDENTIAL_API_KEY_PREFIX) :]
            )
    return fallback
