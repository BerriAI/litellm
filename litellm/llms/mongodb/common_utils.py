"""Shared helpers for the MongoDB integrations. pymongo lives in the optional ``mongodb`` extra,
so every import of it is deferred to call time."""

import asyncio
import threading
import weakref
from asyncio import AbstractEventLoop
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias, TypeVar

from litellm.exceptions import BadRequestError, ServiceUnavailableError, Timeout

if TYPE_CHECKING:
    from pymongo import AsyncMongoClient, MongoClient

PYMONGO_INSTALL_HINT: Final = (
    "The MongoDB vector store requires the 'pymongo' package. "
    "Run 'pip install litellm[mongodb]' (or 'pip install pymongo') to install it."
)

MONGODB_PROVIDER: Final = "mongodb"


def config_error(message: str) -> BadRequestError:
    """400 rather than the 500 a bare ValueError becomes once litellm.exception_type wraps it."""
    return BadRequestError(message=message, model=None, llm_provider=MONGODB_PROVIDER)


def timeout_error(message: str) -> Timeout:
    return Timeout(message=message, model=None, llm_provider=MONGODB_PROVIDER)


def unavailable_error(message: str) -> ServiceUnavailableError:
    """litellm only retries 408, 409, 429 and 5xx, so a 400 here would make a failover permanent."""
    return ServiceUnavailableError(message=message, model=None, llm_provider=MONGODB_PROVIDER)


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


SyncClientFactory: TypeAlias = Callable[..., "MongoClient"]
AsyncClientFactory: TypeAlias = Callable[..., "AsyncMongoClient"]

_K = TypeVar("_K")
_V = TypeVar("_V")

_AsyncClientCacheKey: TypeAlias = tuple[MongoClientKey, int]
# CPython recycles id() aggressively, so the id alone would hand a new loop a closed loop's client
_AsyncClientEntry: TypeAlias = tuple["weakref.ref[AbstractEventLoop]", "AsyncMongoClient"]

_SyncClientCache: TypeAlias = "OrderedDict[MongoClientKey, MongoClient]"
_AsyncClientCache: TypeAlias = "OrderedDict[_AsyncClientCacheKey, _AsyncClientEntry]"

_sync_clients: Final[_SyncClientCache] = OrderedDict()  # mutable-ok: process-level client cache
_async_clients: Final[_AsyncClientCache] = OrderedDict()  # mutable-ok: same cache, per loop
# async searches reach the sync client through executor threads, so both caches are shared state
_cache_lock: Final = threading.Lock()


def _store_bounded(cache: "OrderedDict[_K, _V]", cache_key: "_K", value: "_V") -> None:
    """Eviction only drops this cache's reference; an in-flight search keeps its client alive."""
    with _cache_lock:
        cache[cache_key] = value  # mutable-ok: an LRU cache is mutable state by definition
        cache.move_to_end(cache_key)
        while len(cache) > _MAX_CACHED_CLIENTS:
            cache.popitem(last=False)


def _mark_used(cache: "OrderedDict[_K, _V]", cache_key: "_K") -> None:
    with _cache_lock:
        if cache_key in cache:
            cache.move_to_end(cache_key)


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


def _client_kwargs(key: MongoClientKey) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "connectTimeoutMS": key.connect_timeout_ms,
            "socketTimeoutMS": key.socket_timeout_ms,
            "serverSelectionTimeoutMS": key.server_selection_timeout_ms,
            "appname": _APP_NAME,
        }
    )


def get_sync_client(key: MongoClientKey, client_class: SyncClientFactory | None = None) -> "MongoClient":
    cached: Final = _sync_clients.get(key)
    if cached is not None:
        _mark_used(_sync_clients, key)
        return cached
    build: Final = client_class if client_class is not None else import_sync_mongo_client()
    client: Final = build(key.connection_string, **_client_kwargs(key))
    _store_bounded(_sync_clients, key, client)
    return client


def _purge_dead_loops() -> None:
    """A cached client holds its loop alive, so a closed loop's entry would pin that client and its
    sockets for the life of the process."""
    with _cache_lock:
        for stale in tuple(
            cache_key
            for cache_key, (loop_ref, _) in _async_clients.items()
            if (cached_loop := loop_ref()) is None or cached_loop.is_closed()
        ):
            del _async_clients[stale]


def get_async_client(key: MongoClientKey, client_class: AsyncClientFactory | None = None) -> "AsyncMongoClient":
    """Async clients bind to the loop that created them, so the cache is keyed per loop."""
    loop: Final = asyncio.get_running_loop()
    loop_key: Final = (key, id(loop))
    cached: Final = _async_clients.get(loop_key)
    if cached is not None and cached[0]() is loop:
        _mark_used(_async_clients, loop_key)
        return cached[1]
    _purge_dead_loops()
    build: Final = client_class if client_class is not None else import_async_mongo_client()
    client: Final = build(key.connection_string, **_client_kwargs(key))
    _store_bounded(_async_clients, loop_key, (weakref.ref(loop), client))
    return client


def reset_client_cache() -> None:
    with _cache_lock:
        _sync_clients.clear()
        _async_clients.clear()


_AUTHENTICATION_FAILED_CODE: Final = 18
_UNAUTHORIZED_CODE: Final = 13
# Atlas reports a rejected user as code 8000 "AtlasError" where a self-managed mongod reports 18
_AUTHENTICATION_MESSAGE_MARKERS: Final = ("bad auth", "authentication failed", "not authorized")
_RESOLUTION_TIMEOUT_MARKERS: Final = ("resolution lifetime expired", "dns operation timed out")
_UNKNOWN_HOSTNAME_MARKERS: Final = ("dns query name does not exist", "name or service not known")
_CREDENTIAL_ESCAPING_MARKERS: Final = ("must be escaped according to rfc 3986", "bad database name")


def _index_hint(index_name: str, database: str, collection: str) -> str:
    return (
        f"No queryable MongoDB Vector Search index named '{index_name}' was found on "
        f"'{database}.{collection}'. Confirm the index exists on that exact collection, that its "
        "status is READY rather than still building, and that the vector store id matches the index name."
    )


def missing_index_error(index_name: str, database: str, collection: str) -> BadRequestError:
    """$vectorSearch against a missing index, database or collection returns zero documents rather
    than failing, so an empty result set is checked against the catalogue and reported as this."""
    return config_error(
        f"{_index_hint(index_name, database, collection)} A vector search against a database, "
        "collection or index that does not exist returns no results rather than an error, so this "
        "was reported as an empty result set by MongoDB."
    )


def index_not_ready_error(index_name: str, database: str, collection: str, status: str) -> BadRequestError:
    return config_error(
        f"The MongoDB Vector Search index '{index_name}' on '{database}.{collection}' is not queryable "
        f"yet; its status is {status}. Searches against it return no results until the build finishes."
    )


def translate_mongo_error(error: Exception, index_name: str, database: str, collection: str) -> Exception:
    """Returns the exception to raise, so callers keep the driver error as ``__cause__``."""
    try:
        from pymongo.errors import (
            ConfigurationError,
            ConnectionFailure,
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
            "project's IP access list not containing this host, or a paused cluster. On a self-managed "
            "deployment it is usually the host or port in the URI, or a firewall between this process "
            f"and mongod. Either way it can also be an unresolvable hostname. Driver detail: {error}"
        )
    # ExecutionTimeout subclasses OperationFailure, so it has to be matched before it
    if isinstance(error, (NetworkTimeout, ExecutionTimeout)):
        return timeout_error(
            f"The MongoDB vector search against '{database}.{collection}' timed out before returning. "
            f"Driver detail: {error}"
        )
    # ServerSelectionTimeoutError and NetworkTimeout also subclass ConnectionFailure, so this only
    # sees what those branches left
    if isinstance(error, ConnectionFailure):
        return unavailable_error(
            f"The connection to '{database}.{collection}' was dropped or refused. That is usually a "
            "replica set failover or a restarted node, so the search is worth retrying. If it keeps "
            "happening: on Atlas the usual cause is a connection string with no username and password, "
            "or a TLS failure, so confirm the URI is the one Atlas shows under Connect, Drivers; on a "
            "self-managed deployment, check that mongod is listening on the host and port in the URI. "
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
                "The query embedding does not match the vector dimensions the index was built for. "
                "litellm_embedding_model must be the same model that produced the stored vectors. "
                f"Driver detail: {error}"
            )
        if "is not indexed as vector" in detail:
            return config_error(
                "mongodb_embedding_field names a field the MongoDB Vector Search index does not cover. "
                f"It must match the 'path' the index '{index_name}' was created on. Driver detail: {error}"
            )
        if "index" in detail and ("not found" in detail or "does not exist" in detail or "unknown" in detail):
            return config_error(f"{_index_hint(index_name, database, collection)} Driver detail: {error}")
        return config_error(
            f"MongoDB rejected the vector search against '{database}.{collection}' using index "
            f"'{index_name}'. Driver detail: {error}"
        )
    if isinstance(error, ConfigurationError):
        configuration_detail: Final = str(error).lower()
        if any(marker in configuration_detail for marker in _RESOLUTION_TIMEOUT_MARKERS):
            return timeout_error(
                "The DNS lookup for the cluster in mongodb_connection_string did not finish in time. "
                "A mongodb+srv:// URI needs an SRV lookup before any connection is attempted, so this "
                f"is DNS or the configured timeout, not MongoDB. Driver detail: {error}"
            )
        if any(marker in configuration_detail for marker in _UNKNOWN_HOSTNAME_MARKERS):
            return config_error(
                "The hostname in mongodb_connection_string does not exist in DNS. On Atlas, check the "
                "cluster name against the URI shown under Connect, Drivers. On a self-managed deployment, "
                f"check that the hostname resolves from this process. Driver detail: {error}"
            )
        if any(marker in configuration_detail for marker in _CREDENTIAL_ESCAPING_MARKERS):
            return config_error(
                "mongodb_connection_string could not be parsed. A username or password containing "
                "'@', '/', ':' or '%' has to be percent-encoded per RFC 3986, so 'p@ss/word' becomes "
                "'p%40ss%2Fword'. If the credentials are already encoded, check the database name in "
                f"the URI path instead. Driver detail: {error}"
            )
        return config_error(
            f"mongodb_connection_string is not a usable MongoDB connection string. Driver detail: {error}"
        )
    if isinstance(error, InvalidOperation):
        return config_error(f"The MongoDB client was already closed or is unusable. Driver detail: {error}")
    # An unreadable tlsCAFile or tlsCertificateKeyFile raises OSError, not a PyMongoError
    if isinstance(error, OSError) and error.filename:
        return config_error(
            f"'{error.filename}', named by a TLS option in mongodb_connection_string, could not be read. "
            "Check that tlsCAFile and tlsCertificateKeyFile point at files this process can open; inside "
            f"a container that is the path in the container, not on the host. Driver detail: {error}"
        )
    # pymongo raises a plain ValueError, not a PyMongoError, for an unusable port
    if isinstance(error, ValueError):
        return config_error(
            "The host and port in mongodb_connection_string could not be parsed. If the port is a "
            "number between 0 and 65535, the cause is usually an unescaped ':' in the password, which "
            f"has to be percent-encoded per RFC 3986 as '%3A'. Driver detail: {error}"
        )
    return error
