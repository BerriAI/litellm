"""Jaeger read-back for the OTEL trace-completeness tests: typed models over the
Jaeger query API (the destination's own API - completeness is judged on what the
backend actually holds, never on "export succeeded" proxy-side).

A trace matches a call when the request's x-litellm-call-id or the unique prompt
marker appears in any span tag. External reads go through ``e2e_http`` (the only
module allowed to call ``requests.*``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from e2e_config import OTEL_QUERY_URL, POLL_INTERVAL, POLL_TIMEOUT
from e2e_http import URL, NoBody, Success, get

#: OTEL resource service.name the proxy exports under (OTEL_SERVICE_NAME default).
JAEGER_SERVICE = "litellm"


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

    def tag_blob(self) -> str:
        return " ".join(f"{tag.key}={tag.value}" for tag in self.tags)


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
    limit: int = 200
    lookback: str = "1h"


@dataclass(frozen=True, slots=True)
class OtelReader:
    query_url: str

    def _recent_traces(self) -> list[JaegerTrace]:
        result = get(
            URL(f"{self.query_url}/api/traces"),
            headers=NoBody(),
            params=_TracesQuery(service=JAEGER_SERVICE),
            response_type=JaegerTracesPage,
            timeout=30.0,
        )
        match result:
            case Success(data=page):
                return page.data
            case _:
                return []

    def traces_for_call(self, *, call_id: str, marker: str) -> list[JaegerTrace]:
        """Every trace holding the call: matched by the x-litellm-call-id or the
        unique prompt marker in any span tag. More than one hit for one call IS
        the split-trace bug, so this never collapses to a single trace."""
        hits: list[JaegerTrace] = []
        for trace in self._recent_traces():
            blob = " ".join(span.tag_blob() for span in trace.spans)
            if call_id in blob or marker in blob:
                hits.append(trace)
        return hits

    def poll_traces_for_call(
        self, *, call_id: str, marker: str, settled_names: set[str]
    ) -> list[JaegerTrace]:
        """Poll until exactly one trace holds the call and it carries every span
        name in ``settled_names`` (spans flush in batches, the cost write lands
        after the response), then return the hits. At the deadline the last hits
        are returned as-is so the caller's assertions report the real final
        state - on a split trace this never settles and the orphan comes back."""
        deadline = time.monotonic() + POLL_TIMEOUT
        hits: list[JaegerTrace] = []
        while time.monotonic() < deadline:
            hits = self.traces_for_call(call_id=call_id, marker=marker)
            if len(hits) == 1 and settled_names.issubset(set(hits[0].span_names())):
                return hits
            time.sleep(POLL_INTERVAL)
        return hits


def build_otel_reader() -> OtelReader:
    return OtelReader(query_url=OTEL_QUERY_URL)
