"""Jaeger read-back for the OTEL trace-completeness tests: typed models over the
Jaeger query API (the destination's own API - completeness is judged on what the
backend actually holds, never on "export succeeded" proxy-side).

Traces are fetched server-side by the ``litellm.call_id`` tag the gen-AI span
carries (the request's x-litellm-call-id response header), so read-back is
immune to the query page filling up with unrelated traffic (background jobs,
other suites sharing the stack). Jaeger returns every span of a matching trace,
so the completeness assertions see the whole tree. A failed query is a hard
failure, never an empty result - an unreachable destination must not read as
"the trace never arrived".

Service spans that end after the response (the cost write is one) carry no
call id and land as the root of their own trace with a link back to the
request span, so they are fetched by operation name and matched by that link
to the request root rather than by the tag query.

External reads go through ``e2e_http`` (the only module allowed to call
``requests.*``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, Field

from e2e_config import OTEL_QUERY_URL, POLL_INTERVAL, POLL_TIMEOUT
from e2e_http import URL, NetworkError, NoBody, Result, Success, get

#: OTEL resource service.name the proxy exports under (OTEL_SERVICE_NAME default).
JAEGER_SERVICE = "litellm"
#: Span tag carrying the request's x-litellm-call-id (stamped on the gen-AI span).
CALL_ID_TAG = "litellm.call_id"


class JaegerTag(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    value: str | int | float | bool | None = None


class JaegerReference(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ref_type: str = Field(alias="refType")
    trace_id: str = Field(alias="traceID")
    span_id: str = Field(alias="spanID")


class JaegerSpan(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    span_id: str = Field(alias="spanID")
    operation_name: str = Field(alias="operationName")
    start_time: int = Field(default=0, alias="startTime")
    #: Span duration in microseconds, as reported by the Jaeger query API.
    duration: int = 0
    references: list[JaegerReference] = []
    tags: list[JaegerTag] = []

    @property
    def kind(self) -> str:
        for tag in self.tags:
            if tag.key == "span.kind":
                return str(tag.value)
        return ""


class JaegerTrace(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    trace_id: str = Field(alias="traceID")
    spans: list[JaegerSpan] = []

    def span_names(self) -> list[str]:
        return sorted(span.operation_name for span in self.spans)


class JaegerTracesPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[JaegerTrace] = []


class _TracesQuery(BaseModel):
    service: str
    tags: str | None = None
    operation: str | None = None
    limit: int = 20
    lookback: str = "1h"
    start: int | None = None
    end: int | None = None


def _settled(trace: JaegerTrace, names: set[str], prefixes: set[str]) -> bool:
    present = set(trace.span_names())
    return names.issubset(present) and all(any(name.startswith(prefix) for name in present) for prefix in prefixes)


def root_span(trace: JaegerTrace) -> JaegerSpan | None:
    """The single span whose references all point outside the trace (a span
    with no references qualifies). None when there is not exactly one."""
    in_trace = {span.span_id for span in trace.spans}
    roots = [span for span in trace.spans if all(ref.span_id not in in_trace for ref in span.references)]
    return roots[0] if len(roots) == 1 else None


def _follows(trace: JaegerTrace, parent_trace_id: str, parent_span_id: str) -> bool:
    root = root_span(trace)
    return root is not None and any(
        ref.trace_id == parent_trace_id and ref.span_id == parent_span_id for ref in root.references
    )


@dataclass(frozen=True, slots=True)
class CallTraces:
    hits: tuple[JaegerTrace, ...]
    linked: tuple[JaegerTrace, ...]


@dataclass(frozen=True, slots=True)
class _Observation:
    traces: CallTraces
    missing: tuple[str, ...]
    unreachable: NetworkError | None

    def settled(self, names: set[str], prefixes: set[str]) -> bool:
        hits: Final = self.traces.hits
        return self.unreachable is None and len(hits) == 1 and not self.missing and _settled(hits[0], names, prefixes)


@dataclass(frozen=True, slots=True)
class OtelReader:
    query_url: str

    def _query_traces(self, call_id: str) -> Result[JaegerTracesPage]:
        return get(
            URL(f"{self.query_url}/api/traces"),
            headers=NoBody(),
            params=_TracesQuery(service=JAEGER_SERVICE, tags=json.dumps({CALL_ID_TAG: call_id})),
            response_type=JaegerTracesPage,
            timeout=30.0,
        )

    def traces_for_call(self, call_id: str) -> list[JaegerTrace]:
        """Every trace holding a span tagged with this call id. Jaeger matches
        spans server-side and returns their full traces; more than one hit for
        one call IS the split-trace bug, so this never collapses to one."""
        match self._query_traces(call_id):
            case Success(data=page):
                return page.data
            case failure:
                pytest.fail(f"Jaeger query API at {self.query_url} failed: {failure}")

    def _query_operation(self, operation: str, *, start: int) -> Result[JaegerTracesPage]:
        return get(
            URL(f"{self.query_url}/api/traces"),
            headers=NoBody(),
            params=_TracesQuery(
                service=JAEGER_SERVICE,
                operation=operation,
                limit=200,
                start=start,
                end=int(time.time() * 1_000_000),
            ),
            response_type=JaegerTracesPage,
            timeout=30.0,
        )

    def linked_traces(self, *, operation: str, parent: JaegerTrace) -> tuple[JaegerTrace, ...] | NetworkError:
        """Traces whose root span references the parent trace's root span.
        Detached post-response work lands as the root of its own trace with a
        link back to the request span instead of the call-id tag, so it is
        found by operation name, windowed to start at the parent root's start
        time (the detached span always starts after it), and matched on that
        link. A NetworkError is handed back so the polling caller can tell an
        unreachable read-back endpoint from a span that never arrived."""
        parent_root: Final = root_span(parent)
        if parent_root is None:
            return ()
        match self._query_operation(operation, start=parent_root.start_time):
            case Success(data=page):
                return tuple(t for t in page.data if _follows(t, parent.trace_id, parent_root.span_id))
            case NetworkError() as failure:
                return failure
            case failure:
                pytest.fail(f"Jaeger query API at {self.query_url} failed: {failure}")

    def poll_traces_for_call(
        self,
        *,
        call_id: str,
        settled_names: set[str],
        settled_prefixes: set[str],
        linked_names: frozenset[str] = frozenset(),
    ) -> CallTraces:
        """Poll until exactly one trace holds the call, it carries every span
        name in ``settled_names`` plus at least one name per prefix in
        ``settled_prefixes``, and every name in ``linked_names`` is either in
        that trace or is the root of its own trace referencing the request
        root (post-response work detaches per #42826). At the deadline the
        last observed state is returned as-is so the caller's assertions
        report the real final state - on a split trace this never settles and
        the orphan comes back. A read-back endpoint still failing at the
        deadline (either query) is a hard failure, not a missing span."""
        deadline: Final = time.monotonic() + POLL_TIMEOUT
        last: Final = self._poll(call_id, settled_names, settled_prefixes, linked_names, deadline)
        if last.unreachable is not None:
            pytest.fail(
                f"Jaeger query API at {self.query_url} stayed unreachable until the "
                f"{POLL_TIMEOUT}s poll deadline: {last.unreachable}"
            )
        return last.traces

    def _observe(self, call_id: str, linked_names: frozenset[str]) -> _Observation:
        match self._query_traces(call_id):
            case NetworkError() as failure:
                return _Observation(CallTraces((), ()), tuple(linked_names), failure)
            case Success(data=page):
                if len(page.data) != 1:
                    return _Observation(CallTraces(tuple(page.data), ()), tuple(linked_names), None)
                hit: Final = page.data[0]
                present: Final = frozenset(hit.span_names())
                results: Final = {
                    name: self.linked_traces(operation=name, parent=hit) for name in linked_names if name not in present
                }
                unreachable: Final = next((r for r in results.values() if isinstance(r, NetworkError)), None)
                linked: Final = tuple(t for r in results.values() if not isinstance(r, NetworkError) for t in r)
                missing: Final = tuple(name for name, r in results.items() if isinstance(r, NetworkError) or not r)
                return _Observation(CallTraces((hit,), linked), missing, unreachable)
            case failure:
                pytest.fail(f"Jaeger query API at {self.query_url} failed: {failure}")

    def _poll(
        self,
        call_id: str,
        names: set[str],
        prefixes: set[str],
        linked_names: frozenset[str],
        deadline: float,
    ) -> _Observation:
        observed: Final = self._observe(call_id, linked_names)
        if observed.settled(names, prefixes) or time.monotonic() >= deadline:
            return observed
        time.sleep(POLL_INTERVAL)
        return self._poll(call_id, names, prefixes, linked_names, deadline)


def build_otel_reader() -> OtelReader:
    return OtelReader(query_url=OTEL_QUERY_URL)
