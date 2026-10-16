"""Event-loop holds, read from the gateway's own asyncio debug output. A gateway started with
`PYTHONASYNCIODEBUG=1` on the stdlib loop (`LITELLM_EVENT_LOOP=asyncio`, uvloop never reports them) logs every
callback or task step over `slow_callback_duration` (0.1 s) as
`Executing <Task ... coro=<name() running at file:line> ...> took X seconds`. A case with `event_loop` set reads
the gateway log for its traffic window, groups those lines by coroutine and location, and fails on a location that
is not on its allowlist of known holds or shows up more often than the allowlist gives it. asyncio names the task,
not the frame that held the loop, so every request handler lands under `RequestResponseCycle.run_asgi`; the
allowlist caps how often a known location may appear instead of ignoring it. The log comes from the gateway pods
(`pods/log` with the runner's service account, pods picked by `E2E_PERF_GATEWAY_POD_SELECTOR`) or, on a local run,
from the whole file named by `E2E_PERF_GATEWAY_LOG_FILE` (no time window, so startup holds are in it too)."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Final
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from setup.podcpu import SELECTOR_ENV, in_cluster

OPT_IN_ENV: Final = "E2E_PERF_EVENT_LOOP"
LOG_FILE_ENV: Final = "E2E_PERF_GATEWAY_LOG_FILE"
_API: Final = "https://kubernetes.default.svc/api/v1/namespaces"
_LINE: Final = re.compile(r"Executing <(?P<desc>.*?)> took (?P<seconds>[0-9.]+) seconds")
_CORO: Final = re.compile(r"coro=<(?P<name>[^ ]+)")
_AT: Final = re.compile(r" at (?P<where>[^ >]+:\d+)")


@dataclass(frozen=True, slots=True)
class KnownStall:
    """A hold the gateway has today: `where` is a substring of the coroutine name or its file:line, `max_count` the
    most times it may show up in one run before the case fails."""

    where: str
    max_count: int


@dataclass(frozen=True, slots=True)
class EventLoopStalls:
    allowlist: tuple[KnownStall, ...]
    max_hold_seconds: float = 0.1


@dataclass(frozen=True, slots=True)
class Stall:
    coroutine: str
    where: str
    seconds: float


@dataclass(frozen=True, slots=True)
class StallRow:
    coroutine: str
    where: str
    count: int
    total_seconds: float
    max_seconds: float
    allowed_by: KnownStall | None

    @property
    def offending(self) -> bool:
        return self.allowed_by is None or self.count > self.allowed_by.max_count


def _short(path: str) -> str:
    if "site-packages/" in path:
        return path.rsplit("site-packages/", 1)[1]
    if "/litellm/" in path:
        return "litellm/" + path.rsplit("/litellm/", 1)[1]
    return path


def _parse_line(line: str) -> Stall | None:
    match: Final = _LINE.search(line)
    if match is None:
        return None
    desc: Final = match.group("desc")
    coro: Final = _CORO.search(desc)
    at: Final = _AT.search(desc)
    return Stall(
        coroutine=coro.group("name") if coro else desc[:80],
        where=_short(at.group("where")) if at else "",
        seconds=float(match.group("seconds")),
    )


def parse_stalls(lines: Iterable[str]) -> tuple[Stall, ...]:
    return tuple(stall for stall in map(_parse_line, lines) if stall is not None)


def _allowed_by(spec: EventLoopStalls, coroutine: str, where: str) -> KnownStall | None:
    return next((known for known in spec.allowlist if known.where in coroutine or known.where in where), None)


def summarize(spec: EventLoopStalls, stalls: Iterable[Stall]) -> tuple[StallRow, ...]:
    """One row per (coroutine, location) over the hold threshold, worst total first."""
    over: Final = sorted(
        (stall for stall in stalls if stall.seconds >= spec.max_hold_seconds), key=lambda s: (s.coroutine, s.where)
    )
    rows: Final = tuple(
        StallRow(
            coroutine=coroutine,
            where=where,
            count=len(group),
            total_seconds=sum(stall.seconds for stall in group),
            max_seconds=max(stall.seconds for stall in group),
            allowed_by=_allowed_by(spec, coroutine, where),
        )
        for (coroutine, where), group in (
            (key, tuple(items)) for key, items in groupby(over, key=lambda s: (s.coroutine, s.where))
        )
    )
    return tuple(sorted(rows, key=lambda row: -row.total_seconds))


def offenders(rows: Iterable[StallRow]) -> tuple[str, ...]:
    return tuple(
        f"{row.coroutine} {row.where}: {row.count}x, max {row.max_seconds:.2f} s"
        + (
            f" (allowlist {row.allowed_by.where} permits {row.allowed_by.max_count})"
            if row.allowed_by
            else " (not on the allowlist)"
        )
        for row in rows
        if row.offending
    )


def render_stalls(spec: EventLoopStalls, rows: Iterable[StallRow]) -> str:
    header: Final = (
        f"  event-loop holds >= {spec.max_hold_seconds:.2f} s in the gateway log (count  total_s  max_s  where)"
    )
    body: Final = tuple(
        f"    {'FAIL' if row.offending else 'ok  '} {row.count:5d} {row.total_seconds:8.2f} {row.max_seconds:6.2f}  "
        f"{row.coroutine} {row.where}"
        for row in rows
    )
    return "\n".join((header, *body)) if body else header + "\n    none"


class _PodName(BaseModel):
    name: str


class _Pod(BaseModel):
    metadata: _PodName


class _PodList(BaseModel):
    items: list[_Pod]


def _get(url: str, token: str, ca_bundle: str) -> str:
    response: Final = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, verify=ca_bundle, timeout=60.0)
    return response.raise_for_status().text


def _pod_logs(since: datetime) -> tuple[str, ...] | None:
    cluster: Final = in_cluster()
    selector: Final = os.environ.get(SELECTOR_ENV, "")
    if cluster is None or not selector:
        return None
    token, ca_bundle, namespace = cluster
    pods: Final = _PodList.model_validate(
        json.loads(_get(f"{_API}/{namespace}/pods?labelSelector={quote(selector)}", token, ca_bundle))
    )
    since_param: Final = quote(since.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return tuple(
        line
        for pod in pods.items
        for line in _get(
            f"{_API}/{namespace}/pods/{pod.metadata.name}/log?sinceTime={since_param}", token, ca_bundle
        ).splitlines()
    )


def read_gateway_log(since: datetime) -> tuple[str, ...] | None:
    """Gateway log lines for the traffic window, or None when neither a pod nor a local file is reachable."""
    local: Final = os.environ.get(LOG_FILE_ENV, "")
    if local:
        return tuple(Path(local).read_text(errors="replace").splitlines())
    return _pod_logs(since)
