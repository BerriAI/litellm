"""Shared helpers for MongoDB Atlas integrations.

pymongo ships in the optional ``mongodb`` extra, so every import of it is
deferred to call time and raises an actionable error when it is absent.

Clients are cached per connection because building one costs an SRV lookup, a
TLS handshake and topology discovery: measured at ~890ms against Atlas versus
~80ms on a warm client, so a client per search would dominate query latency.
"""

import asyncio
import weakref
from asyncio import AbstractEventLoop
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.exceptions import BadRequestError, Timeout

if TYPE_CHECKING:
    from pymongo import AsyncMongoClient, MongoClient

PYMONGO_INSTALL_HINT: Final = (
    "The MongoDB vector store requires the 'pymongo' package. "
    "Run 'pip install litellm[mongodb]' (or 'pip install pymongo') to install it."
)

MONGODB_PROVIDER: Final = "mongodb"


def config_error(message: str) -> BadRequestError:
    """Misconfiguration is the caller's to fix, so it maps to 400 rather than the 500
    a bare ValueError would become once litellm.exception_type wraps it."""
    return BadRequestError(message=message, model=None, llm_provider=MONGODB_PROVIDER)


def timeout_error(message: str) -> Timeout:
    return Timeout(message=message, model=None, llm_provider=MONGODB_PROVIDER)


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
# The value carries a weak reference to the loop the client was built on: CPython recycles
# id() aggressively (measured: 200 of 200 fresh loops landed on an id already in this cache),
# so the id alone would hand a new loop a client bound to a closed one.
_async_clients: dict[  # mutable-ok: same cache, keyed per event loop
    tuple[MongoClientKey, int], tuple["weakref.ref[AbstractEventLoop]", "AsyncMongoClient"]
] = {}


def import_sync_mongo_client() -> "type[MongoClient]":
    try:
        from pymongo import MongoClient as SyncMongoClient
    except ImportError as e:
        raise config_error(PYMONGO_INSTALL_HINT) from e
    return SyncMongoClient


def import_async_mongo_client() -> "type[AsyncMongoClient]":
    try:
        from pymongo import AsyncMongoClient as AsyncMongoClientClass
    except ImportError as e:
        raise config_error(PYMONGO_INSTALL_HINT) from e
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
    loop: Final = asyncio.get_running_loop()
    loop_key: Final = (key, id(loop))
    cached: Final = _async_clients.get(loop_key)
    if cached is not None and cached[0]() is loop:
        return cached[1]
    client: Final = import_async_mongo_client()(key.connection_string, **_client_kwargs(key))
    if len(_async_clients) < _MAX_CACHED_CLIENTS or loop_key in _async_clients:
        _async_clients[loop_key] = (weakref.ref(loop), client)
    return client


def reset_client_cache() -> None:
    _sync_clients.clear()
    _async_clients.clear()


_AUTHENTICATION_FAILED_CODE: Final = 18
_UNAUTHORIZED_CODE: Final = 13
# Atlas reports a rejected user as code 8000 "AtlasError" rather than 18, so the
# message is the only reliable signal for a serverless or shared-tier deployment.
_AUTHENTICATION_MESSAGE_MARKERS: Final = ("bad auth", "authentication failed", "not authorized")


def _index_hint(index_name: str, database: str, collection: str) -> str:
    return (
        f"No queryable Atlas Vector Search index named '{index_name}' was found on "
        f"'{database}.{collection}'. Confirm the index exists on that exact collection, that its "
        "status is READY rather than still building, and that the vector store id matches the index name."
    )


def missing_index_error(index_name: str, database: str, collection: str) -> BadRequestError:
    """$vectorSearch against a missing index, database or collection returns zero documents
    instead of failing, so an empty result set is checked against the index catalogue and
    turned into this rather than being reported as 'no matches'."""
    return config_error(
        f"{_index_hint(index_name, database, collection)} A vector search against a database, "
        "collection or index that does not exist returns no results rather than an error, so this "
        "was reported as an empty result set by MongoDB."
    )


def index_not_ready_error(index_name: str, database: str, collection: str, status: str) -> BadRequestError:
    return config_error(
        f"The Atlas Vector Search index '{index_name}' on '{database}.{collection}' is not queryable "
        f"yet; its status is {status}. Searches against it return no results until the build finishes."
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
        return timeout_error(
            "Could not reach the MongoDB deployment before the timeout. On Atlas this is usually the "
            "project's IP access list not containing this host, or a paused cluster; it can also be an "
            f"unresolvable hostname. Driver detail: {error}"
        )
    # ExecutionTimeout subclasses OperationFailure, so it has to be matched before it
    if isinstance(error, (NetworkTimeout, ExecutionTimeout)):
        return timeout_error(
            f"The MongoDB vector search against '{database}.{collection}' timed out before returning. "
            f"Driver detail: {error}"
        )
    if isinstance(error, OperationFailure):
        code: Final = error.code
        detail: Final = str(error).lower()
        if code in (_AUTHENTICATION_FAILED_CODE, _UNAUTHORIZED_CODE) or any(
            marker in detail for marker in _AUTHENTICATION_MESSAGE_MARKERS
        ):
            return config_error(
                "MongoDB rejected the credentials in mongodb_connection_string, or the database user "
                f"lacks read access to '{database}.{collection}'. Driver detail: {error.details}"
            )
        if "dimension" in detail:
            return config_error(
                "The query embedding does not match the vector dimensions the Atlas index was built for. "
                "litellm_embedding_model must be the same model that produced the stored vectors. "
                f"Driver detail: {error}"
            )
        if "is not indexed as vector" in detail:
            return config_error(
                "mongodb_embedding_field names a field the Atlas Vector Search index does not cover. "
                f"It must match the 'path' the index '{index_name}' was created on. Driver detail: {error}"
            )
        if "index" in detail and ("not found" in detail or "does not exist" in detail or "unknown" in detail):
            return config_error(f"{_index_hint(index_name, database, collection)} Driver detail: {error}")
        return config_error(
            f"MongoDB rejected the vector search against '{database}.{collection}' using index "
            f"'{index_name}'. Driver detail: {error}"
        )
    if isinstance(error, ConfigurationError):
        return config_error(
            "mongodb_connection_string is not a usable MongoDB connection string. "
            f"Driver detail: {error}"
        )
    if isinstance(error, InvalidOperation):
        return config_error(f"The MongoDB client was already closed or is unusable. Driver detail: {error}")
    return error
