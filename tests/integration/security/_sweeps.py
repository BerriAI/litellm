"""Sweeps: every place a canary must NOT appear, searched with ``find_canary``.

Each sweep returns ``Hit(sweep, location, slot, encoding)`` records; ``assert_no_hits`` fails
with a table that names the slot, the sweep and the exact location, so the code path that copied it is
usually obvious from the failure alone. The sweeps are generic on purpose: a new table, a new
GET route or a new copy of the request body is covered without editing this module.

API:

- ``sweep_database(canaries, *, database_url=None) -> tuple[Hit, ...]`` (S1): every base table
  of every non-system schema from ``information_schema.tables``, read as
  ``SELECT to_jsonb(t)::text FROM "<schema>"."<table>" t``. Location is ``table.column``
  (``schema.table.column`` outside ``public``); a table dropped mid-sweep is skipped. With
  ``since``, the append-only log tables in ``TIME_SCOPED_TABLES`` are read from ``since`` on
  (minus ``SCOPE_SLACK``), so the sweep stays fast on a database shared by many tests.
- ``get_routes() -> tuple[str, ...]`` and ``sweep_routes(gateway, canaries, ids, *, callers)``
  (S2): every GET route registered on the proxy app (``app.routes``, which includes the
  routes hidden from the OpenAPI spec and every lazily registered feature router), enumerated
  once per session by importing the app in a child interpreter. Path parameters are filled from
  ``ids`` (parameter name -> value), then from ``DEFAULT_IDS``; any other parameter gets
  ``PLACEHOLDER_ID`` so the route is still called and its (usually 404) response still searched.
  A parameter in ``REAL_ID_REQUIRED`` is never given a placeholder (the proxy would call a public
  provider); such a route is skipped unless ``ids`` supplies it. Routes called with a placeholder
  or skipped for want of a real id are listed in ``RouteSweep.unfilled``; pass real ids to make
  them return data. ``route_denied(route)`` names why a route is skipped: ``ROUTE_DENY_LIST``
  holds the routes that stream forever, redirect into an external flow or contact an external
  service, and ``PROVIDER_PASSTHROUGH`` matches the ``/<provider>/{endpoint:path}`` routes that
  forward to the provider (swept by the pass-through slots, not by S2). Every response is searched
  whatever its status; responses with status >= 500 are also listed in ``RouteSweep.errors``.
  A call that got no response at all (timeout, reset) is listed in ``RouteSweep.unreachable``,
  and ``sweep_all`` fails on it, since that route went unchecked. ``ADMIN_ONLY_ALLOWANCES``
  names exact ``(route, caller label)`` pairs allowed to return a credential by design; those
  hits land in ``RouteSweep.allowed`` instead of ``hits``, and every other caller of that route
  is still swept. ``record_route_sweep(routes, node)`` appends the report to
  ``$INTEGRATION_RESULTS_DIR/security-route-sweep.jsonl`` (a CI artifact). With ``since``,
  unpaginated list routes (``SCENARIO_SCOPED_LIST_ROUTES``, today ``/spend/logs``) are called
  with this scenario's request id, user id and a date window (row by row and summarized) instead
  of unfiltered.
- ``sweep_responses(responses, canaries) -> tuple[Hit, ...]`` (S3): body and headers of every
  client-facing response the scenario received.
- ``sweep_sink(name, requests, canaries, *, own_header=None) -> tuple[Hit, ...]`` (S4): every
  byte a sink double received (gzip bodies are inflated by ``find_canary``). ``own_header`` is
  the ``(header name, slot)`` pair the sink legitimately authenticates with; that one header may
  carry that one canary.
- ``sweep_redis(canaries, *, host, port) -> tuple[Hit, ...]`` (S5): ``SCAN`` of every key, with
  strings, hashes, lists, sets and sorted sets dumped and searched along with the key name.
- ``sweep_all(gateway, canaries, *, responses, sinks, ids, callers=None, own_headers=None,
  since=None) -> SweepReport``: S1 to S5 in one pass for a finished scenario. Search the scenario's marker
  and its credential canaries together; ``SweepReport.credential_hits()`` is every hit that is not the
  marker, and ``assert_marker_seen(report, expected)`` is the per-test sensitivity control
  (``expected`` maps a sweep id to a location substring the marker must be reported at).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import quote, urlencode

import httpx
import psycopg
from integration._support.client import Gateway
from integration._support.wire import Request
from integration.security._canary import MARKER, Canary, find_canary
from psycopg import sql
from redis import Redis

_PATH_PARAMETER: Final = re.compile(r"{([^}:]+)(?::[^}]+)?}")
_ROUTE_TIMEOUT: Final = 20.0

ROUTE_DENY_LIST: Final = MappingProxyType(
    {
        "/mcp": "streamable HTTP GET opens a server-sent event stream that never ends",
        "/mcp/proxy": "MCP transport endpoint, not a JSON read",
        "/{mcp_server_name}/mcp": "MCP transport endpoint, not a JSON read",
        "/toolset/{toolset_name}/mcp": "MCP transport endpoint, not a JSON read",
        "/sso/key/generate": "starts an external SSO redirect flow",
        "/sso/callback": "external SSO redirect target",
        "/sso/saml/login": "starts an external SAML redirect flow",
        "/sso/debug/login": "starts an external SSO redirect flow",
        "/sso/debug/callback": "external SSO redirect target",
        "/fallback/login": "HTML login page",
        "/plugin-proxy/{plugin_name}/{path:path}": "reverse proxy to a plugin process",
        "/openai_passthrough/{endpoint:path}": "forwards to a provider, not a proxy read",
        "/get/latest_release_info": "fetches the latest release from api.github.com",
    }
)

PROVIDER_PASSTHROUGH: Final = re.compile(r"^(/[^/{}]+)+/\{endpoint:path\}$")
PROVIDER_PASSTHROUGH_REASON: Final = "provider pass-through: forwards to the provider, not a proxy read"


def route_denied(route: str) -> str | None:
    """Why S2 skips ``route``, or None when it is swept."""
    if route in ROUTE_DENY_LIST:
        return ROUTE_DENY_LIST[route]
    return PROVIDER_PASSTHROUGH_REASON if PROVIDER_PASSTHROUGH.match(route) else None


DEFAULT_IDS: Final = MappingProxyType({"provider": "openai"})
PLACEHOLDER_ID: Final = "canary-placeholder-id"
REAL_ID_REQUIRED: Final = MappingProxyType(
    {
        "video_id": "a video id encodes its provider; an unknown id falls back to the public OpenAI API",
        "character_id": "a character id encodes its provider; an unknown id falls back to the public OpenAI API",
    }
)

ADMIN_ONLY_ALLOWANCES: Final = MappingProxyType(
    {
        ("/get/config/callbacks", "admin"): (
            "proxy admin holds the master key and edits these env values in the config UI"
        ),
    }
)


def route_allowance(route: str, caller: str) -> str | None:
    """The documented reason ``caller`` may read a credential from ``route``, or None."""
    return ADMIN_ONLY_ALLOWANCES.get((route, caller))


@dataclass(frozen=True, slots=True)
class Hit:
    sweep: str
    location: str
    slot: str
    encoding: str


@dataclass(frozen=True, slots=True)
class RouteSweep:
    hits: tuple[Hit, ...]
    called: tuple[str, ...]
    unfilled: tuple[str, ...]
    errors: tuple[str, ...] = field(default=())
    unreachable: tuple[str, ...] = field(default=())
    allowed: tuple[Hit, ...] = field(default=())


def format_hits(hits: Iterable[Hit]) -> str:
    rows: Final = tuple((hit.slot, hit.sweep, hit.encoding, hit.location) for hit in hits)
    header: Final = ("slot", "sweep", "encoding", "location")
    widths: Final = tuple(max(len(row[index]) for row in (header, *rows)) for index in range(3))
    return "\n".join(
        f"{slot:<{widths[0]}}  {sweep:<{widths[1]}}  {encoding:<{widths[2]}}  {location}"
        for slot, sweep, encoding, location in (header, *rows)
    )


def assert_no_hits(hits: Sequence[Hit], context: str) -> None:
    assert not hits, f"Credential canary found outside its destination ({context}):\n{format_hits(hits)}"


def _hits(sweep: str, location: str, blob: bytes | str, canaries: Sequence[Canary]) -> tuple[Hit, ...]:
    return tuple(Hit(sweep, location, match.slot, match.encoding) for match in find_canary(blob, canaries))


def sweep_database(
    canaries: Sequence[Canary], *, database_url: str | None = None, since: datetime | None = None
) -> tuple[Hit, ...]:
    """S1: every row of every base table, as ``to_jsonb``, attributed to the column that holds it.

    With ``since``, the append-only log tables in ``TIME_SCOPED_TABLES`` are read only for rows
    written or changed at or after it; every other table is still read in full.
    """
    found: Final[list[Hit]] = []  # mutable-ok: accumulated across tables
    with psycopg.connect(database_url or os.environ["DATABASE_URL"], autocommit=True) as connection:
        tables: Final = connection.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema') "
            "ORDER BY table_schema, table_name"
        ).fetchall()
        for schema, table in tables:
            query = sql.SQL("SELECT to_jsonb(t)::text FROM {}.{} t").format(
                sql.Identifier(schema), sql.Identifier(table)
            )
            scoped = TIME_SCOPED_TABLES.get(table) if since is not None else None
            if scoped is not None:
                query = sql.SQL("{} WHERE {}").format(
                    query,
                    sql.SQL(" OR ").join(
                        sql.SQL("t.{} >= {}").format(sql.Identifier(column), sql.Literal(_naive_utc(since)))
                        for column in scoped
                    ),
                )
            where = table if schema == "public" else f"{schema}.{table}"
            try:
                rows = connection.execute(query).fetchall()
            except psycopg.errors.UndefinedTable:
                continue
            for (row,) in rows:
                if not find_canary(row, canaries):
                    continue
                for column, value in json.loads(row).items():
                    found.extend(_hits("S1", f"{where}.{column}", json.dumps(value), canaries))
    return tuple(found)


TIME_SCOPED_TABLES: Final = MappingProxyType(
    {
        "LiteLLM_SpendLogs": ("startTime", "updated_at"),
        "LiteLLM_ErrorLogs": ("startTime", "endTime"),
        "LiteLLM_AuditLog": ("updated_at",),
        "LiteLLM_DeletedTeamTable": ("deleted_at",),
        "LiteLLM_DeletedVerificationToken": ("deleted_at",),
    }
)
SCOPE_SLACK: Final = timedelta(seconds=5)


def _naive_utc(moment: datetime) -> datetime:
    """Prisma writes these columns as naive UTC; compare with a little slack for clock skew."""
    aware: Final = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return (aware - SCOPE_SLACK).astimezone(UTC).replace(tzinfo=None)


def _route_queries(route: str, ids: Mapping[str, str], since: datetime | None) -> tuple[str, ...]:
    """Query strings a route is called with; unbounded list routes are narrowed to this scenario."""
    if route not in SCENARIO_SCOPED_LIST_ROUTES or since is None:
        return ("",)
    return tuple(
        "?" + urlencode(query) for query in SCENARIO_SCOPED_LIST_ROUTES[route](ids, since.astimezone(UTC).date())
    )


def _spend_logs_queries(ids: Mapping[str, str], day: date) -> tuple[Mapping[str, str], ...]:
    window: Final = {
        "start_date": day.isoformat(),
        "end_date": (datetime.now(UTC).date() + timedelta(days=1)).isoformat(),
    }
    return (
        *(({"request_id": ids["request_id"]},) if "request_id" in ids else ()),
        *(({"user_id": ids["user_id"]},) if "user_id" in ids else ()),
        {"summarize": "false", **window},
        window,
    )


SCENARIO_SCOPED_LIST_ROUTES: Final[Mapping[str, Callable[[Mapping[str, str], date], tuple[Mapping[str, str], ...]]]] = (
    MappingProxyType({"/spend/logs": _spend_logs_queries})
)


@cache
def get_routes() -> tuple[str, ...]:
    """Every GET route path on the proxy app, including routes hidden from the OpenAPI spec.

    The child imports the same source tree the owned proxy runs from (``INTEGRATION_PROXY_ROOT``
    or this checkout), without reading the database. Lazily registered feature routers
    (``LAZY_FEATURES``) are loaded first, so their GET routes are enumerated too; on the running
    proxy the first request to such a path registers the router before it is served. Mounted
    ASGI sub-apps (the MCP server) have no methods and are out of scope for S2.
    """
    script: Final = (
        "import asyncio, json\n"
        "from litellm.proxy._lazy_features import LAZY_FEATURES, _force_load\n"
        "from litellm.proxy.proxy_server import app\n"
        "async def load():\n"
        "    for feature in LAZY_FEATURES:\n"
        "        await _force_load(app, feature)\n"
        "asyncio.run(load())\n"
        "paths = [getattr(r, 'path', '') for r in app.routes]\n"
        "missing = sorted(f.name for f in LAZY_FEATURES if not any(f.matches(p) for p in paths))\n"
        "print('MISSING=' + json.dumps(missing))\n"
        "print('ROUTES=' + json.dumps(sorted({r.path for r in app.routes "
        "if 'GET' in (getattr(r, 'methods', None) or ())})))\n"
    )
    root: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
    inherited: Final = {name: value for name, value in os.environ.items() if name != "DATABASE_URL"}
    completed: Final = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=root,
        env={**inherited, "PYTHONPATH": os.pathsep.join((str(root), inherited.get("PYTHONPATH", "")))},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    lines: Final = completed.stdout.splitlines()
    missing: Final = json.loads(next(line for line in lines if line.startswith("MISSING=")).removeprefix("MISSING="))
    routes: Final = tuple(
        json.loads(next(line for line in lines if line.startswith("ROUTES=")).removeprefix("ROUTES="))
    )
    assert missing == [], f"Lazy features registered no route, so S2 cannot sweep them: {missing}"
    assert "/spend/logs/ui/{request_id}" in routes, "Route enumeration missed hidden routes"
    assert "/guardrails/list" in routes, "Route enumeration missed lazily registered feature routes"
    return routes


def _filled(route: str, ids: Mapping[str, str]) -> tuple[str, bool]:
    """The concrete path, and whether any parameter fell back to ``PLACEHOLDER_ID``."""
    known: Final = {**DEFAULT_IDS, **ids}
    names: Final = _PATH_PARAMETER.findall(route)
    path: Final = _PATH_PARAMETER.sub(lambda match: quote(known.get(match.group(1), PLACEHOLDER_ID), safe=""), route)
    return path, any(name not in known for name in names)


@dataclass(frozen=True, slots=True)
class _RouteCall:
    hits: tuple[Hit, ...]
    allowed: tuple[Hit, ...]
    error: str | None
    unreachable: str | None


def sweep_routes(
    gateway: Gateway,
    canaries: Sequence[Canary],
    ids: Mapping[str, str],
    *,
    callers: Mapping[str, str] | None = None,
    since: datetime | None = None,
) -> RouteSweep:
    """S2: call every GET route as each caller (label -> bearer key; default the master key).

    With ``since``, the unpaginated list routes in ``SCENARIO_SCOPED_LIST_ROUTES`` are called with
    this scenario's filters (its request id, its user, and a date window from ``since``, row by
    row and summarized) instead of unfiltered, which on a shared database returns every row ever written.
    """
    routes: Final = tuple(route for route in get_routes() if route_denied(route) is None)
    targets: Final = tuple(
        (route, *_filled(route, ids))
        for route in routes
        if all(name in ids for name in _PATH_PARAMETER.findall(route) if name in REAL_ID_REQUIRED)
    )
    who: Final = callers if callers is not None else {"admin": gateway.key}
    base_url: Final = str(gateway.client.base_url)

    def call(route: str, label: str, key: str, path: str) -> _RouteCall:
        location: Final = f"GET {path} as {label}"
        try:
            with httpx.Client(base_url=base_url, timeout=_ROUTE_TIMEOUT, trust_env=False) as client:
                response = client.get(path, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError as error:
            return _RouteCall((), (), None, f"{location}: {type(error).__name__}")
        headers = "\n".join(f"{name}: {value}" for name, value in response.headers.items())
        found = _hits(
            "S2", f"{location} -> {response.status_code}", response.content + b"\n" + headers.encode(), canaries
        )
        allowed = route_allowance(route, label) is not None
        return _RouteCall(
            () if allowed else found,
            found if allowed else (),
            f"{location}: {response.status_code}" if response.status_code >= 500 else None,
            None,
        )

    jobs: Final = tuple(
        (route, label, key, path + query)
        for label, key in who.items()
        for route, path, _ in targets
        for query in _route_queries(route, ids, since)
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results: Final = tuple(pool.map(lambda job: call(*job), jobs))
    return RouteSweep(
        hits=tuple(hit for result in results for hit in result.hits),
        called=tuple(f"{label} {path}" for _, label, _, path in jobs),
        unfilled=(
            *(route for route, _, placeholder in targets if placeholder),
            *(route for route in routes if route not in {target for target, _, _ in targets}),
        ),
        errors=tuple(result.error for result in results if result.error is not None),
        unreachable=tuple(result.unreachable for result in results if result.unreachable is not None),
        allowed=tuple(hit for result in results for hit in result.allowed),
    )


def record_route_sweep(routes: RouteSweep, node: str) -> None:
    """Append the route sweep's errors and unfilled routes to the results directory, when set."""
    destination: Final = os.environ.get("INTEGRATION_RESULTS_DIR")
    if not destination:
        return
    entry: Final = {
        "node": node,
        "called": len(routes.called),
        "errors": routes.errors,
        "unreachable": routes.unreachable,
        "unfilled": routes.unfilled,
        "allowed": [f"{hit.slot} {hit.location}" for hit in routes.allowed],
    }
    with (Path(destination) / "security-route-sweep.jsonl").open("a") as report:
        report.write(json.dumps(entry) + "\n")


def sweep_responses(responses: Sequence[httpx.Response], canaries: Sequence[Canary]) -> tuple[Hit, ...]:
    """S3: body and headers of each client-facing response."""
    found: Final[list[Hit]] = []  # mutable-ok: accumulated across responses
    for index, response in enumerate(responses):
        where = f"response[{index}] {response.request.method} {response.request.url.path} -> {response.status_code}"
        found.extend(_hits("S3", where + " body", response.content, canaries))
        for name, value in response.headers.items():
            found.extend(_hits("S3", f"{where} header {name}", value, canaries))
    return tuple(found)


def sweep_sink(
    name: str,
    requests: Sequence[Request],
    canaries: Sequence[Canary],
    *,
    own_header: tuple[str, str] | None = None,
) -> tuple[Hit, ...]:
    """S4: every request a sink double received; ``own_header`` may carry its own canary only."""
    found: Final[list[Hit]] = []  # mutable-ok: accumulated across requests
    for index, request in enumerate(requests):
        where = f"{name}[{index}] {request.method} {request.target}"
        found.extend(_hits("S4", where + " body", request.body, canaries))
        for header, value in request.headers.items():
            found.extend(
                hit
                for hit in _hits("S4", f"{where} header {header}", value, canaries)
                if own_header is None or (header, hit.slot) != own_header
            )
    return tuple(found)


def _redis_values(cache: Redis, key: bytes) -> Iterable[bytes]:
    kind: Final = cache.type(key)
    readers: Final[Mapping[bytes, Callable[[], Iterable[bytes]]]] = {
        b"string": lambda: (cache.get(key) or b"",),
        b"hash": lambda: (part for pair in cache.hgetall(key).items() for part in pair),
        b"list": lambda: cache.lrange(key, 0, -1),
        b"set": lambda: cache.smembers(key),
        b"zset": lambda: cache.zrange(key, 0, -1),
    }
    reader: Final = readers.get(kind)
    return reader() if reader is not None else ()


def sweep_redis(canaries: Sequence[Canary], *, host: str | None = None, port: int | None = None) -> tuple[Hit, ...]:
    """S5: every key name and value in the Redis database the proxy uses."""
    found: Final[list[Hit]] = []  # mutable-ok: accumulated across keys
    with Redis(
        host=host or os.environ["REDIS_HOST"], port=port or int(os.environ["REDIS_PORT"]), decode_responses=False
    ) as cache:
        for key in cache.scan_iter(count=500):
            found.extend(_hits("S5", f"redis key {key!r}", key, canaries))
            for value in _redis_values(cache, key):
                found.extend(_hits("S5", f"redis value {key!r}", value, canaries))
    return tuple(found)


@dataclass(frozen=True, slots=True)
class SweepReport:
    hits: tuple[Hit, ...]
    routes: RouteSweep

    def credential_hits(self) -> tuple[Hit, ...]:
        return tuple(hit for hit in self.hits if hit.slot != MARKER)

    def marker_locations(self) -> tuple[tuple[str, str], ...]:
        return tuple((hit.sweep, hit.location) for hit in self.hits if hit.slot == MARKER)


def sweep_all(
    gateway: Gateway,
    canaries: Sequence[Canary],
    *,
    responses: Sequence[httpx.Response],
    sinks: Mapping[str, Sequence[Request]],
    ids: Mapping[str, str],
    callers: Mapping[str, str] | None = None,
    own_headers: Mapping[str, tuple[str, str]] | None = None,
    since: datetime | None = None,
) -> SweepReport:
    """S1 to S5 for one finished scenario; fails if any GET route returned no response.

    Redis goes first: it holds entries with a TTL, and the route walk is the slow sweep. Pass
    ``since`` (taken before the scenario's first request) to scope the append-only log tables
    and the unpaginated log list routes to this scenario; the sensitivity marker's own spend-log
    row must then still be found, which ``assert_marker_seen`` checks.
    """
    redis: Final = sweep_redis(canaries)
    routes: Final = sweep_routes(gateway, canaries, ids, callers=callers, since=since)
    assert not routes.unreachable, f"GET routes returned no response, so S2 did not check them: {routes.unreachable}"
    hits: Final = (
        *sweep_database(canaries, since=since),
        *routes.hits,
        *sweep_responses(responses, canaries),
        *(
            hit
            for name, received in sinks.items()
            for hit in sweep_sink(name, received, canaries, own_header=(own_headers or {}).get(name))
        ),
        *redis,
    )
    return SweepReport(hits, routes)


def assert_marker_seen(report: SweepReport, expected: Mapping[str, str]) -> None:
    """Sensitivity control: the marker must be reported by each sweep at the expected location."""
    seen: Final = report.marker_locations()
    missing: Final = tuple(
        f"{sweep} at *{where}*"
        for sweep, where in expected.items()
        if not any(found_sweep == sweep and where in location for found_sweep, location in seen)
    )
    assert not missing, f"Sweep could not see its surface, missing marker {missing}; marker seen at:\n" + "\n".join(
        f"  {sweep} {location}" for sweep, location in seen
    )
