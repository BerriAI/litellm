"""Read-back for the Weave (Weights & Biases) logging tests against the real
Weave project.

The proxy ships OTEL spans to https://trace.wandb.ai/otel/v1/traces with the
``weave_otel`` callback, and the tests read the ingested calls back through
Weave's own query API (``POST /calls/stream_query``), which answers JSON Lines:
one JSON object per call, so the body is parsed line by line rather than as one
document.

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

#: Rows fetched per query. The project is shared with other traffic, so the
#: window has to be wide enough to still hold this run's call once concurrent
#: calls land alongside it.
_QUERY_LIMIT: Final = 200


class _WeaveSortBy(BaseModel):
    field: str
    direction: str


class _WeaveQueryBody(BaseModel):
    project_id: str
    limit: int = _QUERY_LIMIT
    sort_by: list[_WeaveSortBy] = [_WeaveSortBy(field="started_at", direction="desc")]


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

    def calls_matching(self, marker: str, *, op: str = LITELLM_REQUEST_OP) -> list[WeaveCall]:
        """Every ingested call under ``op`` whose inputs carry ``marker``.

        More than one is the duplicate-delivery bug, so this never collapses to a
        single call.
        """
        outcome = send(
            URL(f"{_WEAVE_TRACE_API}/calls/stream_query"),
            headers=self._headers,
            json=_WeaveQueryBody(project_id=self.project_id),
        )
        if not outcome.ok:
            pytest.fail(
                f"Weave calls query for project {self.project_id!r} failed "
                f"({outcome.status_code}): {outcome.body[:300]}"
            )
        matches: list[WeaveCall] = []
        for line in outcome.body.splitlines():
            if not line.strip():
                continue
            call = WeaveCall.model_validate_json(line)
            if call.op == op and call.mentions(marker):
                matches.append(call)
        return matches

    def poll_calls_matching(self, marker: str, *, op: str = LITELLM_REQUEST_OP) -> list[WeaveCall]:
        """Poll until the call is readable, then keep re-reading for
        WEAVE_SETTLE_SECONDS so a duplicate exported by a later batch flush
        cannot hide from the exactly-one assertion. A duplicate ends the settle
        early, because more waiting cannot clear it."""
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            calls = self.calls_matching(marker, op=op)
            if calls:
                return self._settled(marker, op=op, first=calls)
            time.sleep(POLL_INTERVAL)
        return []

    def _settled(self, marker: str, *, op: str, first: list[WeaveCall]) -> list[WeaveCall]:
        """A transiently empty re-read never downgrades what was already seen."""
        settle_deadline = time.monotonic() + WEAVE_SETTLE_SECONDS
        latest = first
        while time.monotonic() < settle_deadline and len(latest) <= 1:
            time.sleep(POLL_INTERVAL)
            latest = self.calls_matching(marker, op=op) or latest
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
