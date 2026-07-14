"""Live e2e: OTEL trace completeness on the admin-owned destination (LIT-3787).

Covers logging.otel.success.exports_metric: a successful non-streaming call must
land at the OTEL destination as ONE connected trace - a single root SERVER span
with the auth phase, db lookups, and cost write under it, and the gen-AI CLIENT
span parented into the same tree. The regression this pins: the proxy publishing
the global TracerProvider before callbacks init made server spans export through
a different provider than the preset's gen-AI spans, so the destination received
the gen-AI span alone, dangling (fixed in #30590; verified failing at its parent
commit 1bd603d1ac).

Both halves of the contract are asserted: the recorded state (the proxy reports
the OTEL v2 logger active via /health/readiness/details) and the enforced
behavior (the complete span tree at the destination, read back through the
destination's own query API - never proxy-side "export succeeded" logs).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from pydantic import BaseModel, ConfigDict

from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from e2e_http import NoBody, StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from logging_client import LoggingClient
from otel_client import JaegerTrace, OtelReader

pytestmark = pytest.mark.e2e

MODEL = CHEAP_ANTHROPIC_MODEL
COST_SPAN = "batch_write_to_db _PROXY_track_cost_callback"
DB_SPAN_PREFIX = "postgres "
#: The active OTEL v2 logger's name in /health/readiness/details success_callbacks.
OTEL_V2_LOGGER_NAME = "OpenTelemetryV2"


class _ReadinessDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success_callbacks: list[str] = []


def _assert_otel_destination_configured(client: LoggingClient) -> None:
    """Recorded state: the proxy reports the OTEL v2 logger among its active
    callbacks, so a missing/failed destination config fails here, before any
    traffic-based assertion can time out confusingly."""
    result = client.gateway.probe("/health/readiness/details", params=NoBody())
    assert result.status_code == 200, (
        f"/health/readiness/details must answer 200, got {result.status_code}: {result.body[:300]}"
    )
    details = _ReadinessDetails.model_validate_json(result.body)
    assert OTEL_V2_LOGGER_NAME in details.success_callbacks, (
        f"the proxy must report the {OTEL_V2_LOGGER_NAME} callback active "
        f"(LITELLM_OTEL_V2 + arize_phoenix preset in the compose config); got: {details.success_callbacks}"
    )


def _first_ok(client: LoggingClient, send: Callable[[], StreamingResponse]) -> StreamingResponse:
    """First successful call on a fresh key. A fresh key may briefly 401 until
    the data plane's auth cache picks it up, so retry on 401 to a deadline; a
    401 is rejected before the LLM call so it exports no gen-AI span and cannot
    contaminate the trace assertions. Any other failure is behavior under test
    and fails hard."""
    deadline = time.monotonic() + client.gateway.poll_timeout
    while True:
        outcome = send()
        if outcome.ok:
            return outcome
        if outcome.status_code != 401 or time.monotonic() >= deadline:
            require_successful_call(outcome)
        time.sleep(client.gateway.poll_interval)


def _parent_ids(span_id: str, trace: JaegerTrace) -> list[str]:
    span = next(s for s in trace.spans if s.span_id == span_id)
    return [ref.span_id for ref in span.references if ref.ref_type == "CHILD_OF"]


def _chain_reaches(span_id: str, root_id: str, trace: JaegerTrace) -> bool:
    """Walk parent references (within the trace) from span_id up to root_id."""
    seen: set[str] = set()
    in_trace = {s.span_id for s in trace.spans}
    current = span_id
    while current not in seen:
        if current == root_id:
            return True
        seen.add(current)
        parents = [p for p in _parent_ids(current, trace) if p in in_trace]
        if not parents:
            return False
        current = parents[0]
    return False


def _assert_complete_trace(hits: list[JaegerTrace], *, route: str, genai_span: str) -> None:
    """The enforced behavior: the destination holds exactly one trace for the
    call, rooted at the SERVER span, with auth/db/cost children and the gen-AI
    span all connected into that one tree - no dangling parent references."""
    assert hits, (
        "no trace for this call arrived at the destination within the deadline "
        "(nothing tagged with its call id was found)"
    )
    assert len(hits) == 1, (
        f"expected exactly ONE trace for the call, got {len(hits)}: "
        f"{[(t.trace_id, t.span_names()) for t in hits]} - more than one trace for "
        "one call is the split-trace bug (gen-AI span exported away from its root)"
    )
    trace = hits[0]
    names = trace.span_names()
    in_trace = {span.span_id for span in trace.spans}

    dangling = [
        span.operation_name
        for span in trace.spans
        if span.references and not any(ref.span_id in in_trace for ref in span.references)
    ]
    assert not dangling, (
        f"span(s) {dangling} reference a parent that never reached the destination "
        f"(orphaned trace); spans present: {names}"
    )

    roots = [span for span in trace.spans if not span.references]
    assert len(roots) == 1, f"expected exactly one root span, got {[s.operation_name for s in roots]}; spans: {names}"
    root = roots[0]
    assert root.operation_name == f"POST {route}", (
        f"the root must be the SERVER span 'POST {route}', got {root.operation_name!r}"
    )
    assert root.kind == "server", f"the root span must have kind=server, got {root.kind!r}"

    assert f"auth {route}" in names, f"auth phase span 'auth {route}' missing; spans: {names}"
    assert any(name.startswith(DB_SPAN_PREFIX) for name in names), (
        f"no db ('{DB_SPAN_PREFIX}*') span in the trace; spans: {names}"
    )
    assert COST_SPAN in names, f"cost write span {COST_SPAN!r} missing; spans: {names}"

    genai = next((span for span in trace.spans if span.operation_name == genai_span), None)
    assert genai is not None, f"gen-AI span {genai_span!r} missing; spans: {names}"
    assert genai.kind == "client", f"gen-AI span must have kind=client, got {genai.kind!r}"
    assert _chain_reaches(genai.span_id, root.span_id, trace), (
        f"gen-AI span {genai_span!r} is in the trace but its parent chain does not "
        f"reach the root SERVER span; spans: {names}"
    )


def _settled_names(*, route: str, genai_span: str) -> set[str]:
    return {f"POST {route}", f"auth {route}", COST_SPAN, genai_span}


class TestOtelTraceCompleteness:
    @pytest.mark.covers("logging.otel.success.exports_metric")
    def test_chat_completions_exports_complete_trace(
        self, client: LoggingClient, otel_reader: OtelReader, resources: ResourceManager
    ) -> None:
        """This test verifies that a successful non-streaming
        /chat/completions request produces one complete OTEL trace.

        The trace should have a single server root span for the incoming request, with
        the authentication, database, and cost-recording work beneath it. The span for
        the actual model call must also belong to that same trace, rather than being
        exported separately with a missing parent.

        This matters because a split trace is easy to miss: all of the spans may still
        arrive, but the model call appears without the surrounding request context.
        That makes it difficult to understand where time was spent, connect the model
        cost to the original request, or investigate a slow or failed call.

        /chat/completions is the main OpenAI-compatible route used by most customers,
        so it is important that trace parenting works correctly on this path.
        """
        route = "/chat/completions"
        _assert_otel_destination_configured(client)

        key = client.key_with_alias(f"otel-trace-chat-{unique_marker()}", models=[MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = _first_ok(
            client, lambda: client.chat_raw(key, MODEL, f"reply with one word {marker}", max_tokens=16)
        )
        assert outcome.call_id is not None, "success response must carry x-litellm-call-id"

        hits = otel_reader.poll_traces_for_call(
            call_id=outcome.call_id,
            settled_names=_settled_names(route=route, genai_span=f"chat {MODEL}"),
            settled_prefixes={DB_SPAN_PREFIX},
        )
        _assert_complete_trace(hits, route=route, genai_span=f"chat {MODEL}")
