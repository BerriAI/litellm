"""Shared helpers for MongoDB Atlas integrations.

pymongo ships in the optional ``mongodb`` extra, so every import of it is
deferred to call time and raises an actionable error when it is absent.

Clients are cached per connection because building one costs an SRV lookup, a
TLS handshake and topology discovery: measured at ~890ms against Atlas versus
~80ms on a warm client, so a client per search would dominate query latency.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pymongo import AsyncMongoClient, MongoClient

PYMONGO_INSTALL_HINT: Final = (
    "The MongoDB vector store requires the 'pymongo' package. "
    "Run 'pip install litellm[mongodb]' (or 'pip install pymongo') to install it."
)

DEFAULT_CONNECT_TIMEOUT_MS: Final = 10_000
DEFAULT_SOCKET_TIMEOUT_MS: Final = 30_000
DEFAULT_SERVER_SELECTION_TIMEOUT_MS: Final = 10_000

_MAX_CACHED_CLIENTS: Final = 32

_APP_NAME: Final = "litellm"


@dataclass(frozen=True, slots=True)
class MongoClientKey:
    connection_string: str
    connect_timeout_ms: int
    socket_timeout_ms: int
    server_selection_timeout_ms: int


_sync_clients: dict[MongoClientKey, "MongoClient"] = {}  # mutable-ok: process-level connection cache, see module docstring
_async_clients: dict[tuple[MongoClientKey, int], "AsyncMongoClient"] = {}  # mutable-ok: same cache, keyed per event loop


def import_sync_mongo_client() -> "type[MongoClient]":
    try:
        from pymongo import MongoClient as SyncMongoClient
    except ImportError as e:
        raise ValueError(PYMONGO_INSTALL_HINT) from e
    return SyncMongoClient


def import_async_mongo_client() -> "type[AsyncMongoClient]":
    try:
        from pymongo import AsyncMongoClient as AsyncMongoClientClass
    except ImportError as e:
        raise ValueError(PYMONGO_INSTALL_HINT) from e
    return AsyncMongoClientClass


def _client_kwargs(key: MongoClientKey) -> dict[str, object]:
    return {  # mutable-ok: pymongo's client constructor takes keyword arguments
        "connectTimeoutMS": key.connect_timeout_ms,
        "socketTimeoutMS": key.socket_timeout_ms,
        "serverSelectionTimeoutMS": key.server_selection_timeout_ms,
        "appname": _APP_NAME,
    }


def get_sync_client(key: MongoClientKey) -> "MongoClient":
    cached: Final = _sync_clients.get(key)
    if cached is not None:
        return cached
    client: Final = import_sync_mongo_client()(key.connection_string, **_client_kwargs(key))
    if len(_sync_clients) < _MAX_CACHED_CLIENTS:
        _sync_clients[key] = client
    return client


def get_async_client(key: MongoClientKey) -> "AsyncMongoClient":
    """Async clients bind to the loop that created them, so the cache is keyed per loop."""
    loop_key: Final = (key, id(asyncio.get_running_loop()))
    cached: Final = _async_clients.get(loop_key)
    if cached is not None:
        return cached
    client: Final = import_async_mongo_client()(key.connection_string, **_client_kwargs(key))
    if len(_async_clients) < _MAX_CACHED_CLIENTS:
        _async_clients[loop_key] = client
    return client


def reset_client_cache() -> None:
    _sync_clients.clear()
    _async_clients.clear()


_AUTHENTICATION_FAILED_CODE: Final = 18
_UNAUTHORIZED_CODE: Final = 13


def _index_hint(index_name: str, database: str, collection: str) -> str:
    return (
        f"No queryable Atlas Vector Search index named '{index_name}' was found on "
        f"'{database}.{collection}'. Confirm the index exists on that exact collection, that its "
        "status is READY rather than still building, and that the vector store id matches the index name."
    )


def translate_mongo_error(error: Exception, index_name: str, database: str, collection: str) -> Exception:
    """Turn a driver failure into a message that names the misconfiguration, never a silent empty result.

    Returns the exception to raise so callers keep the original as ``__cause__``.
    """
    try:
        from pymongo.errors import (
            ConfigurationError,
            ExecutionTimeout,
            InvalidOperation,
            NetworkTimeout,
            OperationFailure,
            ServerSelectionTimeoutError,
        )
    except ImportError:
        return error

    if isinstance(error, ServerSelectionTimeoutError):
        return ValueError(
            "Could not reach the MongoDB deployment before the timeout. On Atlas this is usually the "
            "project's IP access list not containing this host, or a paused cluster; it can also be an "
            f"unresolvable hostname. Driver detail: {error}"
        )
    if isinstance(error, OperationFailure):
        code: Final = error.code
        if code in (_AUTHENTICATION_FAILED_CODE, _UNAUTHORIZED_CODE):
            return ValueError(
                "MongoDB rejected the credentials in mongodb_connection_string, or the database user "
                f"lacks read access to '{database}.{collection}'. Driver detail: {error.details}"
            )
        detail: Final = str(error).lower()
        if "index" in detail and ("not found" in detail or "does not exist" in detail or "unknown" in detail):
            return ValueError(f"{_index_hint(index_name, database, collection)} Driver detail: {error}")
        if "dimension" in detail or "numdimensions" in detail or "queryvector" in detail:
            return ValueError(
                "The query embedding does not match the vector dimensions the Atlas index was built for. "
                "litellm_embedding_model must be the same model that produced the stored vectors. "
                f"Driver detail: {error}"
            )
        return ValueError(
            f"MongoDB rejected the vector search against '{database}.{collection}' using index "
            f"'{index_name}'. Driver detail: {error}"
        )
    if isinstance(error, (NetworkTimeout, ExecutionTimeout)):
        return ValueError(
            f"The MongoDB vector search against '{database}.{collection}' timed out before returning. "
            f"Driver detail: {error}"
        )
    if isinstance(error, ConfigurationError):
        return ValueError(
            "mongodb_connection_string is not a usable MongoDB connection string. "
            f"Driver detail: {error}"
        )
    if isinstance(error, InvalidOperation):
        return ValueError(f"The MongoDB client was already closed or is unusable. Driver detail: {error}")
    return error
