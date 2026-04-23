"""
DB-backed ChatGPT / Codex OAuth authenticator.

Mirrors the filesystem-backed :class:`Authenticator` but reads tokens from
``litellm.credential_list`` (the in-memory decrypted cache) and writes them
back to ``LiteLLM_CredentialsTable`` via a fire-and-forget background thread.

Kept in its own module so the upstream-facing ``authenticator.py`` (which
houses the device-code flow shared with the CLI) stays nearly untouched.
"""

import asyncio
import threading
from typing import Any, Dict, Optional

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.types.utils import CredentialItem

from .authenticator import Authenticator

CREDENTIAL_TYPE = "chatgpt_oauth"

# When ``api_key`` begins with this prefix, the suffix is a credential_name
# in ``LiteLLM_CredentialsTable`` and a DB-backed authenticator is used in
# place of the filesystem one.
OAUTH_CREDENTIAL_API_KEY_PREFIX = "oauth:"


class DBAuthenticator(Authenticator):
    """
    Uses ``LiteLLM_CredentialsTable`` as the backing store instead of
    ``~/.config/litellm/chatgpt/auth.json``. The in-memory
    ``litellm.credential_list`` serves reads (sync, called from
    :meth:`validate_environment`); writes fan out to both the in-memory cache
    (sync) and the database (async, via a worker thread so the sync refresh
    path need not await).
    """

    def __init__(self, credential_name: str) -> None:
        self.credential_name = credential_name
        # Parent fields that aren't used by this subclass — kept non-None so
        # any accidental reference fails loudly instead of silently reading
        # or writing the wrong file.
        self.token_dir = ""
        self.auth_file = ""

    def _ensure_token_dir(self) -> None:
        return

    def _read_auth_file(self) -> Optional[Dict[str, Any]]:
        values = CredentialAccessor.get_credential_values(self.credential_name)
        if not values:
            return None
        return _unpack_auth_record(values)

    def _write_auth_file(self, data: Dict[str, Any]) -> None:
        credential_values = _pack_auth_record(data)
        item = CredentialItem(
            credential_name=self.credential_name,
            credential_values=credential_values,
            credential_info={
                "type": CREDENTIAL_TYPE,
                "custom_llm_provider": "chatgpt",
            },
        )
        CredentialAccessor.upsert_credentials([item])
        _schedule_db_persist(item)


# ---------------------------------------------------------------------------
# Auth-record (de)serialization
# ---------------------------------------------------------------------------
#
# ``credential_values`` is a flat ``dict[str, str]`` — the encryption helper
# only handles strings, so ``expires_at`` (an int) is stored as a string and
# parsed back here.


def _pack_auth_record(data: Dict[str, Any]) -> Dict[str, str]:
    packed: Dict[str, str] = {}
    for key in ("access_token", "refresh_token", "id_token", "account_id"):
        value = data.get(key)
        if value is not None:
            packed[key] = str(value)
    expires_at = data.get("expires_at")
    if expires_at is not None:
        packed["expires_at"] = str(int(expires_at))
    return packed


def _unpack_auth_record(values: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for key in ("access_token", "refresh_token", "id_token", "account_id"):
        value = values.get(key)
        if value:
            record[key] = value
    expires_at = values.get("expires_at")
    if expires_at is not None:
        try:
            record["expires_at"] = int(expires_at)
        except (TypeError, ValueError):
            pass
    return record


# ---------------------------------------------------------------------------
# Fire-and-forget DB persistence
# ---------------------------------------------------------------------------

# The proxy's main asyncio event loop, captured by the OAuth start endpoint.
# ``prisma_client``'s internal primitives (locks, futures) are bound to this
# loop; awaiting them from a new loop spun up by ``asyncio.run`` in a worker
# thread raises "bound to a different event loop". We schedule the persist
# coroutine back onto this loop via ``run_coroutine_threadsafe`` from the
# refresh path (which runs in a FastAPI threadpool thread, not on any loop).
_proxy_main_loop: Optional["asyncio.AbstractEventLoop"] = None


def _register_proxy_main_loop(loop: "asyncio.AbstractEventLoop") -> None:
    """Called from the OAuth ``/start`` endpoint (guaranteed to be on the
    proxy's main loop) so the refresh-path persist can find the loop."""
    global _proxy_main_loop
    _proxy_main_loop = loop


def _schedule_db_persist(item: CredentialItem) -> None:
    """
    Persist the credential to ``LiteLLM_CredentialsTable`` on a worker thread.

    This is fire-and-forget: the caller (typically ``_refresh_tokens``) runs
    in sync context and has already updated the in-memory cache, so the
    request-in-flight can proceed with fresh tokens even if the DB write
    lags or fails. Failures are logged; the cache will be reconciled on the
    next successful write.
    """
    # If the proxy's main loop is known and running, schedule directly on it
    # (cross-loop-safe for prisma_client's bound primitives). Otherwise fall
    # back to spawning a thread + asyncio.run for non-proxy contexts (CLI,
    # tests) where no prisma_client is set anyway.
    loop = _proxy_main_loop
    if loop is not None and loop.is_running():
        asyncio.run_coroutine_threadsafe(persist_credential_to_db(item), loop)
        return
    thread = threading.Thread(
        target=_persist_item_sync,
        args=(item,),
        daemon=True,
        name="chatgpt-oauth-persist",
    )
    thread.start()


def _persist_item_sync(item: CredentialItem) -> None:
    try:
        asyncio.run(persist_credential_to_db(item))
    except Exception as exc:
        verbose_logger.error(
            "Failed to persist refreshed ChatGPT OAuth credential %s: %s",
            item.credential_name,
            exc,
        )


async def persist_credential_to_db(item: CredentialItem) -> None:
    """
    Encrypt and upsert the credential row.

    Intended for both the fire-and-forget refresh path and the interactive
    login endpoint (which can ``await`` it directly on the request event
    loop).
    """
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
        "custom_llm_provider": "chatgpt",
    }
    # Prisma's Json columns want JSON-serialized strings, not raw dicts —
    # see credential_endpoints/endpoints.py for the canonical pattern.
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
                "created_by": "chatgpt_oauth_flow",
                "updated_by": "chatgpt_oauth_flow",
            },
            "update": {
                **jsonified,
                "updated_by": "chatgpt_oauth_flow",
            },
        },
    )


def resolve_authenticator(
    api_key: Optional[str],
    litellm_params: Any,
    fallback: Authenticator,
) -> Authenticator:
    """
    If ``api_key`` (or ``litellm_params.api_key``) starts with ``oauth:``,
    returns a :class:`DBAuthenticator` for the named credential. Otherwise
    returns the given fallback (typically the filesystem
    :class:`Authenticator`).

    Two sources are checked because the chat transformation's
    ``_get_openai_compatible_provider_info`` call-site has ``api_key`` but
    not ``litellm_params``, while ``validate_environment`` on both chat
    and responses has one or both.
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
