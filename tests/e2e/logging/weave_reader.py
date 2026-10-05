"""Read-back for the Weave (Weights & Biases) logging tests against the real
Weave project.

The proxy ships OTEL spans to https://trace.wandb.ai/otel/v1/traces with the
``weave_otel`` callback, and the tests read the ingested calls back through
Weave's own query API (``POST /calls/stream_query``), which answers JSON Lines:
one JSON object per call, so the body is parsed line by line rather than as one
document.

The project is shared with other traffic, so the read never relies on the target
being among the newest N calls: the query is scoped server-side to the
``litellm_request`` op and to calls that started after the test's own request,
and pages with ``offset`` until the window is exhausted.

Weave's own ``summary.weave.status`` is a rollup that reads "success" even for a
span the exporter marked failed, so status comes from the OTEL span itself
(``attributes.otel_span.status.code``), and the shipped cost from
``attributes.otel_span.attributes.llm.response.cost`` - the StandardLogging
``response_cost``, which is what makes this a spend assertion rather than a
delivery ping.

Missing configuration is a hard failure, never a skip.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from itertools import count, takewhile
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, Field

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT
from e2e_http import URL, AuthHeaders, send

_WEAVE_TRACE_API: Final = "https://trace.wandb.ai"

#: The op every litellm LLM call lands under. The proxy also exports a root
#: server span ("Received Proxy Server Request") and management spans; only the
#: LLM call carries the usage and cost this suite asserts on.
LITELLM_REQUEST_OP: Final = "litellm_request"

#: How long to keep re-reading after the first matching call before trusting the
#: exactly-one assertion. The OTEL batch exporter flushes on its own schedule, so
#: a duplicate export can surface well after the first one, and a duplicate IS
#: the bug being guarded against.
WEAVE_SETTLE_SECONDS: Final = 45.0

#: Rows per page. The query is already scoped to this run's time window, so this
#: only bounds one round trip, not what the read can see.
_PAGE_SIZE: Final = 500


class _WeaveSortBy(BaseModel):
    field: str
    direction: str


class _WeaveOpFilter(BaseModel):
    op_names: list[str]


class _WeaveGetField(BaseModel):
    get_field: str = Field(serialization_alias="$getField")


class _WeaveLiteral(BaseModel):
    literal: float = Field(serialization_alias="$literal")


class _WeaveGreaterThan(BaseModel):
    gt: tuple[_WeaveGetField, _WeaveLiteral] = Field(serialization_alias="$gt")


class _WeaveQuery(BaseModel):
    expr: _WeaveGreaterThan = Field(serialization_alias="$expr")


class _WeaveQueryBody(BaseModel):
    project_id: str
    filter: _WeaveOpFilter
    query: _WeaveQuery
    limit: int = _PAGE_SIZE
    offset: int = 0
    sort_by: list[_WeaveSortBy] = [_WeaveSortBy(field="started_at", direction="asc")]


class _OtelStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str | None = None
    message: str | None = None


class _OtelError(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str | None = None
    type: str | None = None
    message: str | None = None


class _LlmResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cost: float | None = None


class _LlmAttributes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response: _LlmResponse | None = None


class _OtelSpanAttributes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm: _LlmAttributes | None = None
    error: _OtelError | None = None


class _OtelSpan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    status: _OtelStatus | None = None
    attributes: _OtelSpanAttributes | None = None


class _CallAttributes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    otel_span: _OtelSpan | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_tokens: int | None = None


class _WeaveSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    usage: dict[str, _Usage] = {}


class WeaveCall(BaseModel):
    """One ingested Weave call, reduced to what the scenarios assert on."""

    model_config = ConfigDict(extra="ignore")

    id: str
    op_name: str
    started_at: str | None = None
    inputs: dict[str, object] = {}
    attributes: _CallAttributes | None = None
    summary: _WeaveSummary | None = Field(default=None)

    @property
    def op(self) -> str:
        """The bare op name out of ``weave:///<entity>/<project>/op/<op>:<digest>``."""
        return self.op_name.split("/op/")[-1].split(":")[0]

    @property
    def status_code(self) -> str | None:
        """The OTEL span status, not Weave's own rollup (which reads "success"
        even for a span the exporter marked ERROR)."""
        span = self.attributes.otel_span if self.attributes else None
        return span.status.code if span and span.status else None

    @property
    def error(self) -> _OtelError | None:
        span = self.attributes.otel_span if self.attributes else None
        return span.attributes.error if span and span.attributes else None

    @property
    def response_cost(self) -> float | None:
        span = self.attributes.otel_span if self.attributes else None
        llm = span.attributes.llm if span and span.attributes else None
        return llm.response.cost if llm and llm.response else None

    @property
    def total_tokens(self) -> int | None:
        """Weave keys usage by model, so the total is summed across whatever
        models the call reported."""
        if not self.summary or not self.summary.usage:
            return None
        totals = [usage.total_tokens for usage in self.summary.usage.values() if usage.total_tokens is not None]
        return sum(totals) if totals else None

    def mentions(self, needle: str) -> bool:
        return needle in json.dumps(self.inputs, default=str)


@dataclass(frozen=True, slots=True)
class WeaveReader:
    project_id: str
    api_key: str

    @property
    def _headers(self) -> AuthHeaders:
        """Weave authenticates with HTTP Basic as the fixed user ``api``."""
        token = base64.b64encode(f"api:{self.api_key}".encode()).decode()
        return AuthHeaders(authorization=f"Basic {token}")

    def _query_body(self, *, since: float, offset: int, op: str) -> _WeaveQueryBody:
        return _WeaveQueryBody(
            project_id=self.project_id,
            filter=_WeaveOpFilter(op_names=[f"weave:///{self.project_id}/op/{op}:*"]),
            query=_WeaveQuery(
                expr=_WeaveGreaterThan(gt=(_WeaveGetField(get_field="started_at"), _WeaveLiteral(literal=since)))
            ),
            offset=offset,
        )

    def _page(self, *, since: float, offset: int, op: str) -> tuple[WeaveCall, ...]:
        outcome = send(
            URL(f"{_WEAVE_TRACE_API}/calls/stream_query"),
            headers=self._headers,
            json=self._query_body(since=since, offset=offset, op=op),
        )
        if not outcome.ok:
            pytest.fail(
                f"Weave calls query for project {self.project_id!r} failed "
                f"({outcome.status_code}): {outcome.body[:300]}"
            )
        return tuple(WeaveCall.model_validate_json(line) for line in outcome.body.splitlines() if line.strip())

    def calls_matching(self, marker: str, *, since: float, op: str = LITELLM_REQUEST_OP) -> tuple[WeaveCall, ...]:
        """Every call under ``op`` started after ``since`` whose inputs carry
        ``marker``, paging until the window is exhausted.

        More than one is the duplicate-delivery bug, so this never collapses to a
        single call.
        """
        pages = tuple(
            takewhile(
                bool,
                (self._page(since=since, offset=offset, op=op) for offset in count(0, _PAGE_SIZE)),
            )
        )
        return tuple(call for page in pages for call in page if call.mentions(marker))

    def poll_calls_matching(self, marker: str, *, since: float, op: str = LITELLM_REQUEST_OP) -> tuple[WeaveCall, ...]:
        """Poll until the call is readable, then keep re-reading for
        WEAVE_SETTLE_SECONDS so a duplicate exported by a later batch flush
        cannot hide from the exactly-one assertion. A duplicate ends the settle
        early, because more waiting cannot clear it."""
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            calls = self.calls_matching(marker, since=since, op=op)
            if calls:
                return self._settled(marker, since=since, op=op, first=calls)
            time.sleep(POLL_INTERVAL)
        return ()

    def _settled(self, marker: str, *, since: float, op: str, first: tuple[WeaveCall, ...]) -> tuple[WeaveCall, ...]:
        """A transiently empty re-read never downgrades what was already seen."""
        settle_deadline = time.monotonic() + WEAVE_SETTLE_SECONDS
        latest = first  # rebind-ok: one settle window, re-read per poll interval
        while time.monotonic() < settle_deadline and len(latest) <= 1:
            time.sleep(POLL_INTERVAL)
            latest = self.calls_matching(marker, since=since, op=op) or latest
        return latest


def build_weave_reader() -> WeaveReader:
    project_id = (os.environ.get("WEAVE_PROJECT_ID") or os.environ.get("WANDB_PROJECT_ID") or "").strip()
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not project_id or not api_key:
        pytest.fail(
            "Weave e2e requires WANDB_API_KEY and WEAVE_PROJECT_ID (or WANDB_PROJECT_ID, "
            "format <entity>/<project>): the test reads the proxy's weave_otel delivery "
            "back from the real Weave project; missing credentials is a hard failure, not a skip"
        )
    return WeaveReader(project_id=project_id, api_key=api_key)
