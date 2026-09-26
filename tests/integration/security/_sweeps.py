"""Sweeps: every place a canary must NOT appear, searched with ``find_canary``.

Each sweep returns ``Hit(sweep, location, slot, encoding)`` records; ``assert_no_hits`` fails
with a table that names the slot, the sweep and the exact location, so the code path that copied it is
usually obvious from the failure alone. The sweeps are generic on purpose: a new table, a new
GET route or a new copy of the request body is covered without editing this module.

API:

- ``sweep_database(canaries, *, database_url=None) -> tuple[Hit, ...]`` (S1): every base table
  of every non-system schema from ``information_schema.tables``, read as
  ``SELECT to_jsonb(t)::text FROM "<schema>"."<table>" t``. Location is ``table.column``
  (``schema.table.column`` outside ``public``); a table dropped mid-sweep is skipped.
- ``get_routes() -> tuple[str, ...]`` and ``sweep_routes(gateway, canaries, ids, *, callers)``
  (S2): every GET ``APIRoute`` registered on the proxy app (``app.routes``, which includes the
  routes hidden from the OpenAPI spec), enumerated once per session by importing the app in a
  child interpreter. Path parameters are filled from ``ids`` (parameter name -> value); a route
  whose parameters are not all known is listed in ``RouteSweep.unfilled``. ``ROUTE_DENY_LIST``
  names the routes skipped because they stream forever or redirect into an external flow.
  Responses with status >= 500 and transport errors are reported in ``RouteSweep.errors`` so a
  broken route is visible instead of silently passing. ``record_route_sweep(routes, node)``
  appends that report to ``$INTEGRATION_RESULTS_DIR/security-route-sweep.jsonl`` (a CI artifact).
- ``sweep_responses(responses, canaries) -> tuple[Hit, ...]`` (S3): body and headers of every
  client-facing response the scenario received.
- ``sweep_sink(name, requests, canaries, *, own_header=None) -> tuple[Hit, ...]`` (S4): every
  byte a sink double received (gzip bodies are inflated by ``find_canary``). ``own_header`` is
  the ``(header name, slot)`` pair the sink legitimately authenticates with; that one header may
  carry that one canary.
- ``sweep_redis(canaries, *, host, port) -> tuple[Hit, ...]`` (S5): ``SCAN`` of every key, with
  strings, hashes, lists, sets and sorted sets dumped and searched along with the key name.
- ``sweep_all(gateway, canaries, *, responses, sinks, ids, callers=None, own_headers=None)
  -> SweepReport``: S1 to S5 in one pass for a finished scenario. Search the scenario's marker
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
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import quote

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
    }
)


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


def sweep_database(canaries: Sequence[Canary], *, database_url: str | None = None) -> tuple[Hit, ...]:
    """S1: every row of every base table, as ``to_jsonb``, attributed to the column that holds it."""
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


@cache
def get_routes() -> tuple[str, ...]:
    """Every GET APIRoute path on the proxy app, including routes hidden from the OpenAPI spec.

    The child imports the same source tree the owned proxy runs from (``INTEGRATION_PROXY_ROOT``
    or this checkout), without reading the database.
    """
    script: Final = (
        "import json\n"
        "from fastapi.routing import APIRoute\n"
        "from litellm.proxy.proxy_server import app\n"
        "print('ROUTES=' + json.dumps(sorted({r.path for r in app.routes "
        "if isinstance(r, APIRoute) and 'GET' in r.methods})))\n"
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
    line: Final = next(line for line in completed.stdout.splitlines() if line.startswith("ROUTES="))
    routes: Final = tuple(json.loads(line.removeprefix("ROUTES=")))
    assert "/spend/logs/ui/{request_id}" in routes, "Route enumeration missed hidden routes"
    return routes


def _filled(route: str, ids: Mapping[str, str]) -> str | None:
    names: Final = _PATH_PARAMETER.findall(route)
    if any(name not in ids for name in names):
        return None
    return _PATH_PARAMETER.sub(lambda match: quote(ids[match.group(1)], safe=""), route)


def sweep_routes(
    gateway: Gateway,
    canaries: Sequence[Canary],
    ids: Mapping[str, str],
    *,
    callers: Mapping[str, str] | None = None,
) -> RouteSweep:
    """S2: call every GET route as each caller (label -> bearer key; default the master key)."""
    routes: Final = tuple(route for route in get_routes() if route not in ROUTE_DENY_LIST)
    targets: Final = tuple((route, _filled(route, ids)) for route in routes)
    unfilled: Final = tuple(route for route, path in targets if path is None)
    paths: Final = tuple(path for _, path in targets if path is not None)
    who: Final = callers if callers is not None else {"admin": gateway.key}
    base_url: Final = str(gateway.client.base_url)

    def call(label: str, key: str, path: str) -> tuple[tuple[Hit, ...], str | None]:
        location: Final = f"GET {path} as {label}"
        try:
            with httpx.Client(base_url=base_url, timeout=_ROUTE_TIMEOUT, trust_env=False) as client:
                response = client.get(path, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError as error:
            return (), f"{location}: {type(error).__name__}"
        headers = "\n".join(f"{name}: {value}" for name, value in response.headers.items())
        found = _hits(
            "S2", f"{location} -> {response.status_code}", response.content + b"\n" + headers.encode(), canaries
        )
        return found, (f"{location}: {response.status_code}" if response.status_code >= 500 else None)

    jobs: Final = tuple((label, key, path) for label, key in who.items() for path in paths)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results: Final = tuple(pool.map(lambda job: call(*job), jobs))
    return RouteSweep(
        hits=tuple(hit for found, _ in results for hit in found),
        called=tuple(f"{label} {path}" for label, _, path in jobs),
        unfilled=unfilled,
        errors=tuple(error for _, error in results if error is not None),
    )


def record_route_sweep(routes: RouteSweep, node: str) -> None:
    """Append the route sweep's errors and unfilled routes to the results directory, when set."""
    destination: Final = os.environ.get("INTEGRATION_RESULTS_DIR")
    if not destination:
        return
    entry: Final = {"node": node, "called": len(routes.called), "errors": routes.errors, "unfilled": routes.unfilled}
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
) -> SweepReport:
    """S1 to S5 for one finished scenario."""
    routes: Final = sweep_routes(gateway, canaries, ids, callers=callers)
    hits: Final = (
        *sweep_database(canaries),
        *routes.hits,
        *sweep_responses(responses, canaries),
        *(
            hit
            for name, received in sinks.items()
            for hit in sweep_sink(name, received, canaries, own_header=(own_headers or {}).get(name))
        ),
        *sweep_redis(canaries),
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
