"""OTel ``db.*`` / ``server.*`` attributes naming the database litellm talks to.

Prisma reaches PostgreSQL through a query engine listening on loopback, so
transport-level instrumentation attributes the work to ``localhost`` and an
operator cannot tell it is a PostgreSQL call or correlate it with the database's
own metrics. These attributes name the real server on litellm's DB spans.

Only the host, port, database and schema of the DSN are read, so no credential
can reach an exporter.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from urllib.parse import ParseResult, parse_qs, unquote, urlparse

from litellm.integrations.otel.model.semconv import DB, Server
from litellm.integrations.otel.model.spans import POSTGRESQL, db_system

_DATABASE_URL_ENV: Final = "DATABASE_URL"
_READ_REPLICA_ENV: Final = "DATABASE_URL_READ_REPLICA"
_DEFAULT_POSTGRES_PORT: Final = 5432
_DEFAULT_POSTGRES_SCHEMA: Final = "public"
_POSTGRES_SCHEMES: Final = frozenset({"postgres", "postgresql"})
_MISPARSED_AUTHORITY_MARKERS: Final = frozenset("@/")
_EMPTY_ATTRIBUTES: Final[Mapping[str, str | int]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class DatabaseEndpoint:
    """The non-sensitive identity of a PostgreSQL server, parsed from a DSN."""

    address: str | None
    port: int | None
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
        raw_database: Final = (parsed.path or "").lstrip("/")
        if _is_misparsed_authority(parsed, raw_database, query):
            return None
        # ``host=`` beats the netloc: it is how libpq names a Unix socket
        # directory and how the Cloud SQL connector sits behind a localhost
        # netloc, where the netloc is the very answer this module replaces.
        address: Final = _first(query.get("host")) or parsed.hostname
        port: Final = (parsed.port or _DEFAULT_POSTGRES_PORT) if address else None
        namespace: Final = _namespace(unquote(raw_database), _first(query.get("schema")))
    except ValueError:
        return None
    if address is None and namespace is None:
        return None
    return DatabaseEndpoint(address=address, port=port, namespace=namespace)


def _is_misparsed_authority(parsed: ParseResult, raw_database: str, query: Mapping[str, Sequence[str]]) -> bool:
    """Whether an unencoded character in the password truncated the authority.

    ``/``, ``#`` or ``?`` in a password ends the netloc early, so urlparse hands
    back the username as the host and strands the real userinfo ``@`` in the
    path, fragment or query. A PostgreSQL DSN never carries a fragment, and its
    database name cannot hold an unencoded ``@`` or ``/``. An ``@`` in the query
    is legitimate only when the query parsed as parameters AND a database path
    preceded it, as in ``postgres://host/db?application_name=svc@prod``. A ``?``
    in a password strands the tail in the query with no path left behind, and it
    can still parse as parameters, so neither test alone is enough.
    """
    if parsed.fragment:
        return True
    if any(marker in raw_database for marker in _MISPARSED_AUTHORITY_MARKERS):
        return True
    return "@" in parsed.query and (not query or not parsed.path)


def _first(values: Sequence[str] | None) -> str:
    return values[0] if values else ""


def _namespace(database: str, schema: str) -> str | None:
    """``{database}|{schema}`` per the PostgreSQL semconv, dropping absent halves.

    Only Prisma's literal default schema stays implicit. The match is
    case-sensitive because Prisma quotes the name, so ``?schema=PUBLIC`` builds
    a second schema alongside ``public`` and the two must not collapse to one
    namespace.
    """
    qualifier: Final = "" if schema == _DEFAULT_POSTGRES_SCHEMA else schema
    return "|".join(part for part in (database, qualifier) if part) or None


def postgres_endpoint() -> DatabaseEndpoint | None:
    """The PostgreSQL endpoint the process is currently connected to.

    Read from ``os.environ`` on every span, deliberately, on both counts.

    The environment is what Prisma itself connects with, so the span cannot
    disagree with the connection; ``get_secret_str`` would consult a configured
    secret manager first and could name a different server than the one serving
    the query. And the value is not static: the RDS IAM refresh rebuilds the URL
    from ``DATABASE_HOST``/``PORT``/``NAME``/``SCHEMA`` every rotation, the
    reconnect path re-reads ``DATABASE_URL``, and the DB-backed
    ``environment_variables`` config overlay can rewrite any of them after
    startup, so a value cached for the process lifetime goes stale against a
    connection that has genuinely moved. Only the parse is memoized, keyed on the
    URL, which keeps the per-span cost to a dict lookup.

    A configured read replica yields ``None``: ``RoutingPrismaWrapper`` picks
    reader or writer per Prisma call, underneath the span, so naming the writer
    would attribute replica reads to the primary.
    """
    if os.environ.get(_READ_REPLICA_ENV):
        return None
    return parse_database_endpoint(os.environ.get(_DATABASE_URL_ENV, ""))


def db_span_attributes(service_name: str, call_type: str | None = None) -> Mapping[str, str | int]:
    """The ``db.*``/``server.*`` attributes for a datastore service call.

    Empty for services that are not outbound datastore calls. Endpoint
    attributes are PostgreSQL-only: ``DATABASE_URL`` says nothing about where
    the redis-backed services point. ``db.system`` rides alongside the current
    ``db.system.name`` because Datadog's OTLP intake still types a database span
    from the older key.
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
