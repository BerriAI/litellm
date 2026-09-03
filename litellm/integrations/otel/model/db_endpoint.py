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
        if _is_misparsed_authority(parsed, url, raw_database):
            return None
        # ``host=`` beats the netloc: it is how libpq names a Unix socket
        # directory and how the Cloud SQL connector sits behind a localhost
        # netloc, where the netloc is the very answer this module replaces.
        address: Final = _first(query.get("host")) or parsed.hostname
        # ``port=`` accompanies ``host=`` in a libpq URI, so honour it the same way.
        port: Final = _port(_first(query.get("port")), parsed.port) if address else None
        namespace: Final = _namespace(unquote(raw_database), _first(query.get("schema")))
    except ValueError:
        return None
    if address is None and namespace is None:
        return None
    return DatabaseEndpoint(address=address, port=port, namespace=namespace)


def _is_misparsed_authority(parsed: ParseResult, url: str, raw_database: str) -> bool:
    """Whether the URL authority may have been truncated by an unencoded character.

    ``/``, ``#`` or ``?`` in a password ends the netloc early, so urlparse hands
    back the username as the host, the leading digits of the password as the
    port, and the rest of the credential as the path, query or fragment. The
    stranded userinfo ``@`` is the only surviving evidence.

    A database name cannot hold an unencoded slash either, so a second path
    segment is the same evidence.

    A DSN that carries the at-sign in a query parameter instead, such as
    ``?application_name=svc@prod``, is indistinguishable from a mis-split by any
    property of the parse: both leave no userinfo, a host, a port and a path.
    Since guessing wrong publishes a credential fragment to a tracing backend,
    that ambiguity resolves to refusing the endpoint. Such a DSN loses
    ``server.address`` and ``db.namespace`` and keeps the rest of the span,
    which is the cheaper error of the two. Percent-encode the at-sign to keep
    them.
    """
    if "/" in raw_database:
        return True
    return "@" in url and "@" not in parsed.netloc


def _first(values: Sequence[str] | None) -> str:
    return values[0] if values else ""


def _port(from_query: str, from_netloc: int | None) -> int:
    return int(from_query) if from_query.isdigit() else (from_netloc or _DEFAULT_POSTGRES_PORT)


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
    connection that has genuinely moved. Nothing is memoized either: a cache
    keyed on the URL would hold a rotated credential past its rotation, and the
    parse is a single ``urlparse`` on a short string.

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
