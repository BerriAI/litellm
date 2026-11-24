"""Event-loop holds, read from the gateway's own asyncio debug output. A gateway started with
`PYTHONASYNCIODEBUG=1` on the stdlib loop (`LITELLM_EVENT_LOOP=asyncio`, uvloop never reports them) logs every
callback or task step over `slow_callback_duration` (0.1 s) as
`Executing <Task ... coro=<name() running at file:line> ...> took X seconds`. A case with `event_loop` set reads
the gateway log for its traffic window, groups those lines by coroutine and location, and fails on a location that
is not on its allowlist of known holds or shows up more often than the allowlist gives it. asyncio names the task,
not the frame that held the loop, so every request handler lands under `RequestResponseCycle.run_asgi`; the
allowlist caps how often a known location may appear instead of ignoring it. The log comes from the gateway pods
(`pods/log` with the runner's service account, pods picked by `E2E_PERF_GATEWAY_POD_SELECTOR`) or, on a local run,
from the whole file named by `E2E_PERF_GATEWAY_LOG_FILE` (no time window, so startup holds are in it too).
A garbage collection pass pauses the loop too, and asyncio charges it to whatever task was running, so the gateway
logs every pass over the same threshold (`Garbage collection gen=2 collected=N took X seconds`, debug mode only);
a hold that such a pass accounts for is reported under `garbage collection` instead of the task it landed on."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Final
from urllib.parse import quote

from e2e_http import AuthHeaders, NetworkError, Success, get_external, get_text_external
from pydantic import BaseModel

from setup.podcpu import SELECTOR_ENV, in_cluster

OPT_IN_ENV: Final = "E2E_PERF_EVENT_LOOP"
LOG_FILE_ENV: Final = "E2E_PERF_GATEWAY_LOG_FILE"
_API: Final = "https://kubernetes.default.svc/api/v1/namespaces"
_LINE: Final = re.compile(r"Executing <(?P<desc>.*?)> took (?P<seconds>[0-9.]+) seconds")
_CORO: Final = re.compile(r"coro=<(?P<name>[^ ]+)")
_AT: Final = re.compile(r" at (?P<where>[^ >]+:\d+)")
_GC_LINE: Final = re.compile(
    r"Garbage collection gen=(?P<generation>\d+) collected=\d+ took (?P<seconds>[0-9.]+) seconds"
)
GC_BUCKET: Final = "garbage collection"


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
class GcPass:
    generation: int
    seconds: float


@dataclass(frozen=True, slots=True)
class Stall:
    coroutine: str
    where: str
    seconds: float
    gc_seconds: float = 0.0
    gc_generation: int | None = None


@dataclass(frozen=True, slots=True)
class StallRow:
    coroutine: str
    where: str
    count: int
    total_seconds: float
    max_seconds: float
    allowed_by: KnownStall | None
    """Every hold in the row, longest first."""
    holds: tuple[float, ...] = ()

    @property
    def offending(self) -> bool:
        return self.allowed_by is None or self.count > self.allowed_by.max_count

    @property
    def over_allowance(self) -> int:
        return self.count if self.allowed_by is None else max(0, self.count - self.allowed_by.max_count)

    @property
    def note(self) -> str:
        allowed: Final = (
            f"{self.count} of {self.allowed_by.max_count} allowed" if self.allowed_by else "not on the allowlist"
        )
        return f"{allowed}, max {self.max_seconds:.2f} s"


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


def _parse_event(line: str) -> Stall | GcPass | None:
    match: Final = _GC_LINE.search(line)
    if match is None:
        return _parse_line(line)
    return GcPass(generation=int(match.group("generation")), seconds=float(match.group("seconds")))


def _with_passes(stall: Stall, passes: tuple[GcPass, ...]) -> Stall:
    fitting: Final = tuple(gc_pass for gc_pass in passes if gc_pass.seconds <= stall.seconds)
    if not fitting:
        return stall
    longest: Final = max(fitting, key=lambda gc_pass: gc_pass.seconds)
    return replace(stall, gc_seconds=longest.seconds, gc_generation=longest.generation)


def parse_stalls(lines: Iterable[str]) -> tuple[Stall, ...]:
    """Holds in log order, each carrying the collector passes logged since the previous hold: a pass runs inside the
    task asyncio then blames and is only logged when it is itself over the threshold, so its hold line follows it."""
    events: Final = tuple(event for event in map(_parse_event, lines) if event is not None)
    holds: Final = tuple((index, event) for index, event in enumerate(events) if isinstance(event, Stall))
    return tuple(
        _with_passes(stall, tuple(event for event in events[previous + 1 : index] if isinstance(event, GcPass)))
        for previous, (index, stall) in zip((-1, *(index for index, _ in holds)), holds)
    )


def _reported(spec: EventLoopStalls, stall: Stall) -> Stall:
    """A hold the collector accounts for, what is left under the threshold, is the collector's, not the task's."""
    if stall.gc_generation is None:
        return stall
    if stall.seconds - stall.gc_seconds >= spec.max_hold_seconds:
        return stall
    return Stall(coroutine=GC_BUCKET, where=f"gen={stall.gc_generation}", seconds=stall.seconds)


def _allowed_by(spec: EventLoopStalls, coroutine: str, where: str) -> KnownStall | None:
    return next((known for known in spec.allowlist if known.where in coroutine or known.where in where), None)


def summarize(spec: EventLoopStalls, stalls: Iterable[Stall]) -> tuple[StallRow, ...]:
    """One row per (coroutine, location) over the hold threshold, worst total first."""
    reported: Final = tuple(_reported(spec, stall) for stall in stalls)
    over: Final = sorted(
        (stall for stall in reported if stall.seconds >= spec.max_hold_seconds), key=lambda s: (s.coroutine, s.where)
    )
    rows: Final = tuple(
        StallRow(
            coroutine=coroutine,
            where=where,
            count=len(group),
            total_seconds=sum(stall.seconds for stall in group),
            max_seconds=max(stall.seconds for stall in group),
            allowed_by=_allowed_by(spec, coroutine, where),
            holds=tuple(sorted((stall.seconds for stall in group), reverse=True)),
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


TOP_HOLDS: Final = 10


@dataclass(frozen=True, slots=True)
class Slice:
    """One `_breakdown` row in the shape the other metrics use (label, value, share) plus the allowlist verdict the
    dashboard shows under the label and colours by. `share` is None where the rows are not parts of a total."""

    label: str
    value: float
    share: float | None
    note: str | None
    over: bool


def _label(row: StallRow) -> str:
    return f"{row.coroutine} {row.where}".strip()


def _slice(row: StallRow, value: float, total: float) -> Slice:
    return Slice(label=_label(row), value=value, share=value / total, note=row.note, over=row.offending)


def _breakdown(name: str, slices: Iterable[Slice]) -> tuple[tuple[str, str], ...]:
    rows: Final = tuple(asdict(row) for row in slices)
    return ((f"perf_event_loop_{name}_breakdown", json.dumps(rows)),) if rows else ()


def _longest_holds(table: tuple[StallRow, ...]) -> tuple[Slice, ...]:
    holds: Final = sorted(((seconds, row) for row in table for seconds in row.holds), key=lambda hold: -hold[0])
    return tuple(
        Slice(label=_label(row), value=seconds, share=None, note=None, over=row.offending)
        for seconds, row in holds[:TOP_HOLDS]
    )


def junit_properties(spec: EventLoopStalls, rows: Iterable[StallRow]) -> tuple[tuple[str, float | str], ...]:
    """The stall table as `perf_event_loop_*` properties for the dashboard: every hold over the threshold with the
    allowlist's total as its limit, the holds past their allowance (limit 0, the case's one assertion), the longest
    and the summed hold, each with a `_breakdown` the dashboard opens on click: holds and seconds per location, the
    locations past their cap by how far over they are, and the ten longest single holds with where they landed."""
    table: Final = tuple(rows)
    holds: Final = sum(row.count for row in table)
    offending: Final = sum(row.over_allowance for row in table)
    total_seconds: Final = sum(row.total_seconds for row in table)
    return (
        ("perf_event_loop_holds", holds),
        ("perf_event_loop_holds_limit", sum(known.max_count for known in spec.allowlist)),
        ("perf_event_loop_offending_holds", offending),
        ("perf_event_loop_offending_holds_limit", 0),
        ("perf_event_loop_hold_seconds_max", max((row.max_seconds for row in table), default=0.0)),
        ("perf_event_loop_hold_seconds_total", total_seconds),
        *_breakdown("holds", (_slice(row, row.count, holds) for row in table)),
        *_breakdown(
            "offending_holds",
            (_slice(row, row.over_allowance, offending) for row in table if row.over_allowance),
        ),
        *_breakdown("hold_seconds_total", (_slice(row, row.total_seconds, total_seconds) for row in table)),
        *_breakdown("hold_seconds_max", _longest_holds(table)),
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


def _pod_log_lines(url: str, headers: AuthHeaders, ca_bundle: str) -> tuple[str, ...]:
    text: Final = get_text_external(url, headers=headers, timeout=60.0, ca_bundle=ca_bundle)
    if isinstance(text, NetworkError):
        raise RuntimeError(f"gateway pod log unreadable: {text.message}")
    return tuple(text.splitlines())


def _pod_logs(since: datetime) -> tuple[str, ...] | None:
    cluster: Final = in_cluster()
    selector: Final = os.environ.get(SELECTOR_ENV, "")
    if cluster is None or not selector:
        return None
    token, ca_bundle, namespace = cluster
    headers: Final = AuthHeaders(authorization=f"Bearer {token}")
    pods: Final = get_external(
        f"{_API}/{namespace}/pods?labelSelector={quote(selector)}",
        response_type=_PodList,
        headers=headers,
        timeout=60.0,
        ca_bundle=ca_bundle,
    )
    if not isinstance(pods, Success):
        raise RuntimeError(f"gateway pod list unreadable: {pods}")
    since_param: Final = quote(since.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return tuple(
        line
        for pod in pods.data.items
        for line in _pod_log_lines(
            f"{_API}/{namespace}/pods/{pod.metadata.name}/log?sinceTime={since_param}", headers, ca_bundle
        )
    )


def read_gateway_log(since: datetime) -> tuple[str, ...] | None:
    """Gateway log lines for the traffic window, or None when neither a pod nor a local file is reachable."""
    local: Final = os.environ.get(LOG_FILE_ENV, "")
    if local:
        return tuple(Path(local).read_text(errors="replace").splitlines())
    return _pod_logs(since)
