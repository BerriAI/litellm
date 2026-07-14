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

External reads go through ``e2e_http`` (the only module allowed to call
``requests.*``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict, Field

from e2e_config import OTEL_QUERY_URL, POLL_INTERVAL, POLL_TIMEOUT
from e2e_http import URL, NoBody, Success, get

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
    tags: str
    limit: int = 20
    lookback: str = "1h"


def _settled(trace: JaegerTrace, names: set[str], prefixes: set[str]) -> bool:
    present = set(trace.span_names())
    return names.issubset(present) and all(
        any(name.startswith(prefix) for name in present) for prefix in prefixes
    )


@dataclass(frozen=True, slots=True)
class OtelReader:
    query_url: str

    def traces_for_call(self, call_id: str) -> list[JaegerTrace]:
        """Every trace holding a span tagged with this call id. Jaeger matches
        spans server-side and returns their full traces; more than one hit for
        one call IS the split-trace bug, so this never collapses to one."""
        result = get(
            URL(f"{self.query_url}/api/traces"),
            headers=NoBody(),
            params=_TracesQuery(service=JAEGER_SERVICE, tags=json.dumps({CALL_ID_TAG: call_id})),
            response_type=JaegerTracesPage,
            timeout=30.0,
        )
        match result:
            case Success(data=page):
                return page.data
            case failure:
                pytest.fail(f"Jaeger query API at {self.query_url} failed: {failure}")

    def poll_traces_for_call(
        self, *, call_id: str, settled_names: set[str], settled_prefixes: set[str]
    ) -> list[JaegerTrace]:
        """Poll until exactly one trace holds the call and it carries every span
        name in ``settled_names`` plus at least one name per prefix in
        ``settled_prefixes`` (spans flush in batches, the cost write lands after
        the response), then return the hits. At the deadline the last hits are
        returned as-is so the caller's assertions report the real final state -
        on a split trace this never settles and the orphan comes back."""
        deadline = time.monotonic() + POLL_TIMEOUT
        hits: list[JaegerTrace] = []
        while time.monotonic() < deadline:
            hits = self.traces_for_call(call_id)
            if len(hits) == 1 and _settled(hits[0], settled_names, settled_prefixes):
                return hits
            time.sleep(POLL_INTERVAL)
        return hits


def build_otel_reader() -> OtelReader:
    return OtelReader(query_url=OTEL_QUERY_URL)
