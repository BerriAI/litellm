"""OTel ``db.*`` / ``server.*`` attributes naming the database litellm talks to.

Prisma reaches PostgreSQL through a query engine listening on loopback, so
transport-level instrumentation attributes the work to ``localhost`` and an
operator cannot tell it is a PostgreSQL call or correlate it with the database's
own metrics. These attributes name the real server on litellm's DB spans.

Only the host, port, database and schema of the DSN are read. The user,
password, IAM token and every other query parameter are never parsed out, so no
credential can reach an exporter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, unquote, urlparse

from litellm.integrations.otel.model.semconv import DB, Server
from litellm.integrations.otel.model.spans import POSTGRESQL, db_system
from litellm.secret_managers.main import get_secret_str

_DEFAULT_POSTGRES_PORT: Final = 5432
# Prisma's own default. Emitting it would split ``db.namespace`` across two
# deployments of the same database that differ only in spelling the default out.
_DEFAULT_POSTGRES_SCHEMA: Final = "public"
_POSTGRES_SCHEMES: Final = frozenset({"postgres", "postgresql"})
_EMPTY_ATTRIBUTES: Final[Mapping[str, str | int]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class DatabaseEndpoint:
    """The non-sensitive identity of a PostgreSQL server, parsed from a DSN."""

    address: str
    port: int
    namespace: str | None


def parse_database_endpoint(url: str | None) -> DatabaseEndpoint | None:
    """Parse a PostgreSQL DSN into its exportable endpoint identity.

    Returns ``None`` for an absent, malformed or non-PostgreSQL URL rather than
    raising: an unparseable DSN must degrade to a span without endpoint
    attributes, never break the request that emitted it.
    """
    if not url:
        return None
    try:
        parsed: Final = urlparse(url)
        if parsed.scheme not in _POSTGRES_SCHEMES:
            return None
        query: Final = parse_qs(parsed.query)
        # A ``host=`` parameter overrides the netloc host: it is how libpq
        # addresses a Unix socket directory, and how the Cloud SQL connector is
        # configured behind a ``localhost`` netloc. Trusting the netloc there
        # would report the same ``localhost`` this module exists to replace.
        address: Final = _first(query.get("host")) or parsed.hostname
        if not address:
            return None
        port: Final = parsed.port or _DEFAULT_POSTGRES_PORT
        database: Final = unquote((parsed.path or "").lstrip("/"))
        schema: Final = _first(query.get("schema"))
    except ValueError:
        return None
    return DatabaseEndpoint(address=address, port=port, namespace=_namespace(database, schema))


def _first(values: Sequence[str] | None) -> str:
    return values[0] if values else ""


def _namespace(database: str, schema: str) -> str | None:
    """``{database}|{schema}`` per the PostgreSQL semconv, dropping absent halves."""
    qualifier: Final = "" if schema == _DEFAULT_POSTGRES_SCHEMA else schema
    return "|".join(part for part in (database, qualifier) if part) or None


@lru_cache(maxsize=1)
def postgres_endpoint() -> DatabaseEndpoint | None:
    """The configured PostgreSQL endpoint, or ``None`` when it cannot be named.

    ``DATABASE_URL`` is deployment-static, and ``get_secret_str`` can reach a
    configured secret manager, so this must not be re-resolved per DB span. The
    empty default keeps a misconfigured secret manager from raising into span
    emission. Tests that change the environment call
    ``postgres_endpoint.cache_clear()``.

    A configured read replica yields ``None``: ``RoutingPrismaWrapper`` picks
    reader or writer per Prisma call, underneath the span, so no single endpoint
    is true for every span and naming the writer would attribute replica reads to
    the primary.
    """
    if get_secret_str("DATABASE_URL_READ_REPLICA", default_value=""):
        return None
    return parse_database_endpoint(get_secret_str("DATABASE_URL", default_value=""))


def db_span_attributes(service_name: str, call_type: str | None = None) -> Mapping[str, str | int]:
    """The ``db.*``/``server.*`` attributes for a datastore service call.

    Empty for services that are not outbound datastore calls. Endpoint
    attributes are PostgreSQL-only: ``DATABASE_URL`` says nothing about where
    the redis-backed services point.

    ``db.system`` is emitted alongside the current ``db.system.name`` because
    Datadog's OTLP intake still infers a span's database type from the older key.
    """
    system: Final = db_system(service_name)
    if system is None:
        return _EMPTY_ATTRIBUTES
    endpoint: Final = postgres_endpoint() if system == POSTGRESQL else None
    pairs: Final[tuple[tuple[str, str | int | None], ...]] = (
        (DB.SYSTEM_NAME, system),
        (DB.SYSTEM_LEGACY, system),
        (DB.OPERATION_NAME, call_type),
        (Server.ADDRESS, endpoint.address if endpoint is not None else None),
        (Server.PORT, endpoint.port if endpoint is not None else None),
        (DB.NAMESPACE, endpoint.namespace if endpoint is not None else None),
    )
    return MappingProxyType({key: value for key, value in pairs if value})
