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
from pydantic import BaseModel, ConfigDict, ValidationError

from e2e_config import CHEAP_ANTHROPIC_MODEL, CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import NoBody, StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from logging_client import INVALID_UPSTREAM_API_KEY, LoggingClient
from models import LiteLLMParamsBody
from otel_client import JaegerSpan, JaegerTrace, OtelReader

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


def _assert_complete_trace(
    hits: list[JaegerTrace], *, route: str, genai_span: str, require_cost_span: bool = True
) -> None:
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
    if require_cost_span:
        assert COST_SPAN in names, f"cost write span {COST_SPAN!r} missing; spans: {names}"

    genai = next((span for span in trace.spans if span.operation_name == genai_span), None)
    assert genai is not None, f"gen-AI span {genai_span!r} missing; spans: {names}"
    assert genai.kind == "client", f"gen-AI span must have kind=client, got {genai.kind!r}"
    assert _chain_reaches(genai.span_id, root.span_id, trace), (
        f"gen-AI span {genai_span!r} is in the trace but its parent chain does not "
        f"reach the root SERVER span; spans: {names}"
    )


def _settled_names(*, route: str, genai_span: str, require_cost_span: bool = True) -> set[str]:
    names = {f"POST {route}", f"auth {route}", genai_span}
    return (names | {COST_SPAN}) if require_cost_span else names


def _tag(span: JaegerSpan, key: str) -> str | int | float | bool | None:
    for tag in span.tags:
        if tag.key == key:
            return tag.value
    return None


#: The attribute contract a failed call's gen-AI span must carry (LIT-4179), as
#: one reviewable payload. Exact-match values; error.message is additionally
#: proven untruncated by _assert_error_span_contract, which parses the provider
#: error JSON embedded in it - a truncated message stops parsing.
EXPECTED_ERROR_SPAN_ATTRIBUTES: dict[str, str] = {
    "error": "True",
    "error.type": "AuthenticationError",
    "otel.status_code": "ERROR",
    "litellm.provider.error.code": "401",
    "litellm.provider.error.llm_provider": "anthropic",
}


class _ProviderErrorDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    message: str


class _ProviderError(BaseModel):
    """The provider error JSON embedded in error.message; validating it proves
    the attribute survived untruncated (a cut-off message stops parsing)."""

    model_config = ConfigDict(extra="ignore")

    error: _ProviderErrorDetail


def _assert_error_span_contract(span: JaegerSpan) -> None:
    """The failed call's gen-AI span carries the LIT-4179 error contract: the
    exact attributes in EXPECTED_ERROR_SPAN_ATTRIBUTES, plus an untruncated
    error.message whose embedded provider error JSON still parses and whose
    text also rides the span status description."""
    for key, expected in EXPECTED_ERROR_SPAN_ATTRIBUTES.items():
        actual = _tag(span, key)
        assert str(actual) == expected, f"error span attribute {key!r} must be {expected!r}, got {actual!r}"

    message = _tag(span, "error.message")
    assert isinstance(message, str) and message, "error span must carry a non-empty error.message"
    assert "AnthropicException" in message, (
        f"error.message must carry the upstream provider exception, got: {message[:200]}"
    )
    start, end = message.find("{"), message.rfind("}")
    assert start != -1 and end > start, (
        f"error.message carries no parseable provider error JSON (truncated?): {message[:200]}"
    )
    try:
        provider_error = _ProviderError.model_validate_json(message[start : end + 1])
    except ValidationError:
        pytest.fail(f"the embedded provider error JSON does not parse (truncated?): {message[:300]}")
    assert provider_error.error.message == "invalid x-api-key", (
        f"the embedded provider error must survive untruncated; parsed: {provider_error}"
    )
    assert _tag(span, "otel.status_description") == message, (
        "the span status description must carry the same untruncated message as error.message"
    )
    stack = _tag(span, "litellm.provider.error.stack_trace")
    assert isinstance(stack, str) and stack, (
        "the error span must carry a non-empty litellm.provider.error.stack_trace"
    )


class TestOtelTraceCompleteness:
    @pytest.mark.covers("logging.otel.success.exports_metric", exercised_on=["chat_completions"])
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

    @pytest.mark.covers("logging.otel.success.exports_metric", exercised_on=["messages"])
    def test_messages_exports_complete_trace(
        self, client: LoggingClient, otel_reader: OtelReader, resources: ResourceManager
    ) -> None:
        """This test verifies that one successful non-streaming /v1/messages request
        produces exactly one complete OTEL trace.

        The trace must have a single root span named "POST /v1/messages". The
        authentication, database, cost-writing, and model-call spans must all belong to
        the same trace and have valid parent relationships leading back to that root.

        The model-call span is expected to be named "chat <model>". The test fails if
        the request is split across multiple traces, if any span references a missing
        parent, or if the model-call span cannot be connected back to the root."""
        route = "/v1/messages"
        _assert_otel_destination_configured(client)

        key = client.key_with_alias(f"otel-trace-messages-{unique_marker()}", models=[MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = _first_ok(
            client, lambda: client.messages_raw(key, MODEL, f"reply with one word {marker}", max_tokens=16)
        )
        assert outcome.call_id is not None, "success response must carry x-litellm-call-id"

        hits = otel_reader.poll_traces_for_call(
            call_id=outcome.call_id,
            settled_names=_settled_names(route=route, genai_span=f"chat {MODEL}"),
            settled_prefixes={DB_SPAN_PREFIX},
        )
        _assert_complete_trace(hits, route=route, genai_span=f"chat {MODEL}")

    @pytest.mark.covers("logging.otel.success.exports_metric", exercised_on=["responses"])
    def test_responses_exports_complete_trace(
        self, client: LoggingClient, otel_reader: OtelReader, resources: ResourceManager
    ) -> None:
        """This test verifies that one successful non-streaming /v1/responses request
        produces exactly one complete OTEL trace.

        The trace must have a single root span named "POST /v1/responses". The
        authentication, database, cost-writing, and model-call spans must all belong to
        the same trace and have valid parent relationships leading back to that root.

        The model-call span is expected to be named "chat <model>". The test fails if
        the request is split across multiple traces, if any span references a missing
        parent, or if the model-call span cannot be connected back to the root."""
        route = "/v1/responses"
        _assert_otel_destination_configured(client)

        key = client.key_with_alias(f"otel-trace-responses-{unique_marker()}", models=[CHEAP_OPENAI_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = _first_ok(
            client,
            lambda: client.responses_raw(key, CHEAP_OPENAI_MODEL, f"reply with one word {marker}"),
        )
        assert outcome.call_id is not None, "success response must carry x-litellm-call-id"

        genai_span = f"chat {CHEAP_OPENAI_MODEL}"
        hits = otel_reader.poll_traces_for_call(
            call_id=outcome.call_id,
            settled_names=_settled_names(route=route, genai_span=genai_span),
            settled_prefixes={DB_SPAN_PREFIX},
        )
        _assert_complete_trace(hits, route=route, genai_span=genai_span)

    @pytest.mark.covers("logging.otel.stream.exports_metric", exercised_on=["chat_completions"])
    def test_chat_completions_stream_exports_complete_trace(
        self, client: LoggingClient, otel_reader: OtelReader, resources: ResourceManager
    ) -> None:
        """A successful streamed `/chat/completions` request should export one
        complete OTEL trace. The trace must contain a single root `SERVER`
        span, with the auth, database, cost, and gen-AI `CLIENT` spans all
        connected back to that root.

        Streaming has an additional lifecycle risk because the gen-AI span is
        closed by the stream-consumption path after the final chunk has
        arrived and usage has been aggregated. Historically, this has caused
        duplicate or orphaned spans.

        The test therefore confirms that:

        * The response actually streams.
        * Exactly one gen-AI span is created for the request.
        * The gen-AI span contains `litellm.request.streaming=true`.
        """
        route = "/chat/completions"
        _assert_otel_destination_configured(client)

        key = client.key_with_alias(f"otel-stream-chat-{unique_marker()}", models=[MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = _first_ok(
            client,
            lambda: client.chat_raw(key, MODEL, f"reply with one word {marker}", stream=True, max_tokens=16),
        )
        assert outcome.call_id is not None, "success response must carry x-litellm-call-id"
        assert outcome.is_streaming, f"response must be an event stream, got content-type {outcome.content_type!r}"
        assert outcome.chunks > 0, "the stream must deliver at least one event"
        assert outcome.stream_error is None, (
            f"the stream carried an upstream error event despite the 200: {outcome.stream_error}"
        )

        genai_span = f"chat {MODEL}"
        hits = otel_reader.poll_traces_for_call(
            call_id=outcome.call_id,
            settled_names=_settled_names(route=route, genai_span=genai_span),
            settled_prefixes={DB_SPAN_PREFIX},
        )
        _assert_complete_trace(hits, route=route, genai_span=genai_span)

        genai_spans = [span for span in hits[0].spans if span.operation_name == genai_span]
        assert len(genai_spans) == 1, (
            f"a streamed call must produce exactly ONE gen-AI span, got {len(genai_spans)}; "
            f"spans: {hits[0].span_names()}"
        )
        assert _tag(genai_spans[0], "litellm.request.streaming") is True, (
            "the gen-AI span must record litellm.request.streaming=true; its absence means "
            "the stream flag was dropped before the model call"
        )

    @pytest.mark.covers("logging.otel.stream.exports_metric", exercised_on=["messages"])
    def test_messages_stream_exports_complete_trace(
        self, client: LoggingClient, otel_reader: OtelReader, resources: ResourceManager
    ) -> None:
        """A successful streamed `/v1/messages` request should export one
        complete OTEL trace. The trace must contain a single root `SERVER`
        span, with the auth, database, cost, and gen-AI `CLIENT` spans all
        connected back to that root.

        This endpoint has the same streaming lifecycle risk as
        `/chat/completions`: the gen-AI span is closed by the
        stream-consumption path after the final chunk has arrived and usage
        has been aggregated.

        The test therefore confirms that:

        * The response actually streams.
        * Exactly one gen-AI span is created for the request.
        * The gen-AI span contains `litellm.request.streaming=true`.
        """
        route = "/v1/messages"
        _assert_otel_destination_configured(client)

        key = client.key_with_alias(f"otel-stream-messages-{unique_marker()}", models=[MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = _first_ok(
            client,
            lambda: client.messages_raw(key, MODEL, f"reply with one word {marker}", max_tokens=16, stream=True),
        )
        assert outcome.call_id is not None, "success response must carry x-litellm-call-id"
        assert outcome.is_streaming, f"response must be an event stream, got content-type {outcome.content_type!r}"
        assert outcome.chunks > 0, "the stream must deliver at least one event"
        assert outcome.stream_error is None, (
            f"the stream carried an upstream error event despite the 200: {outcome.stream_error}"
        )

        genai_span = f"chat {MODEL}"
        hits = otel_reader.poll_traces_for_call(
            call_id=outcome.call_id,
            settled_names=_settled_names(route=route, genai_span=genai_span),
            settled_prefixes={DB_SPAN_PREFIX},
        )
        _assert_complete_trace(hits, route=route, genai_span=genai_span)

        genai_spans = [span for span in hits[0].spans if span.operation_name == genai_span]
        assert len(genai_spans) == 1, (
            f"a streamed call must produce exactly ONE gen-AI span, got {len(genai_spans)}; "
            f"spans: {hits[0].span_names()}"
        )
        assert _tag(genai_spans[0], "litellm.request.streaming") is True, (
            "the gen-AI span must record litellm.request.streaming=true; its absence means "
            "the stream flag was dropped before the model call"
        )

    @pytest.mark.covers("logging.otel.stream.exports_metric", exercised_on=["responses"])
    def test_responses_stream_exports_complete_trace(
        self, client: LoggingClient, otel_reader: OtelReader, resources: ResourceManager
    ) -> None:
        """A successful streamed /v1/responses request should export one complete
        OTEL trace. The trace must contain a single root SERVER span, with the
        auth, database, and gen-AI CLIENT spans all connected back to that
        root.

        This endpoint has the same streaming lifecycle risk as the other
        streaming surfaces: the gen-AI span is closed by the
        stream-consumption path after the final event has arrived and usage
        has been aggregated.

        The test therefore confirms that:

        * The response actually streams.
        * Exactly one gen-AI span is created for the request.
        * Spend is recorded correctly.
        """
        route = "/v1/responses"
        _assert_otel_destination_configured(client)

        key = client.key_with_alias(
            f"otel-stream-responses-{unique_marker()}", models=[CHEAP_OPENAI_MODEL]
        )
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = _first_ok(
            client,
            lambda: client.responses_raw(key, CHEAP_OPENAI_MODEL, f"reply with one word {marker}", stream=True),
        )
        assert outcome.call_id is not None, "success response must carry x-litellm-call-id"
        assert outcome.is_streaming, f"response must be an event stream, got content-type {outcome.content_type!r}"
        assert outcome.chunks > 0, "the stream must deliver at least one event"
        assert outcome.stream_error is None, (
            f"the stream carried an upstream error event despite the 200: {outcome.stream_error}"
        )

        genai_span = f"chat {CHEAP_OPENAI_MODEL}"
        hits = otel_reader.poll_traces_for_call(
            call_id=outcome.call_id,
            settled_names=_settled_names(route=route, genai_span=genai_span, require_cost_span=False),
            settled_prefixes={DB_SPAN_PREFIX},
        )
        _assert_complete_trace(hits, route=route, genai_span=genai_span, require_cost_span=False)

        genai_spans = [span for span in hits[0].spans if span.operation_name == genai_span]
        assert len(genai_spans) == 1, (
            f"a streamed call must produce exactly ONE gen-AI span, got {len(genai_spans)}; "
            f"spans: {hits[0].span_names()}"
        )

        spend_row = client.poll_proxy_spend_for_key(key)
        assert spend_row is not None and spend_row.spend is not None and spend_row.spend > 0, (
            "a successful streamed responses call must record a positive-spend row in /spend/logs "
            "(the cost-write SPAN is knowingly absent on this surface, LIT-4428, but the spend "
            f"itself must land); got {spend_row!r}"
        )
        assert spend_row.call_type == "aresponses", (
            f"the spend row must be attributed to the responses call type, got {spend_row.call_type!r}"
        )

    @pytest.mark.covers("logging.otel.failure.exports_metric", exercised_on=["chat_completions"])
    def test_failed_chat_completions_error_span_attributes(
        self, client: LoggingClient, otel_reader: OtelReader, resources: ResourceManager
    ) -> None:
        """This test checks that a failed `/chat/completions` request produces a
        single, complete OTEL trace. The model-call span should include all
        expected error attributes, a non-empty stack trace, and the full,
        untruncated `error.message`. The provider error embedded in that
        message must also remain valid JSON. The root server span should
        record the same 401 response returned to the client.

        The test uses a deployment with an invalid upstream API key. This
        allows the request to pass LiteLLM’s proxy authentication and fail at
        the provider, which is necessary to generate a model-call error span.
        There should be no cost-write span because failed requests are not
        billed."""
        route = "/chat/completions"
        _assert_otel_destination_configured(client)

        model_name = f"otel-err-{unique_marker()}"
        model_id = client.create_model(
            model_name,
            LiteLLMParamsBody(model="anthropic/claude-haiku-4-5", api_key=INVALID_UPSTREAM_API_KEY),
        )
        resources.defer(lambda: client.delete_model(model_id))
        key = client.key_with_alias(f"otel-err-{unique_marker()}", models=[model_name])
        resources.defer(lambda: client.delete_key(key))

        deadline = time.monotonic() + client.gateway.poll_timeout
        while True:
            outcome = client.chat_raw(key, model_name, "trigger an upstream auth failure", max_tokens=16)
            assert not outcome.ok, "the call must fail; the deployment's upstream key is invalid"
            if "AnthropicException" in outcome.body or time.monotonic() >= deadline:
                break
            time.sleep(client.gateway.poll_interval)
        assert "AnthropicException" in outcome.body, (
            "never saw the upstream provider failure before the deadline; the key may still be "
            f"propagating - last outcome {outcome.status_code}: {outcome.body[:200]}"
        )
        assert outcome.status_code == 401, (
            f"an upstream auth failure must map to 401, got {outcome.status_code}: {outcome.body[:200]}"
        )
        assert outcome.call_id is not None, "failed responses must still carry x-litellm-call-id"

        genai_span = f"chat {model_name}"
        hits = otel_reader.poll_traces_for_call(
            call_id=outcome.call_id,
            settled_names=_settled_names(route=route, genai_span=genai_span, require_cost_span=False),
            settled_prefixes={DB_SPAN_PREFIX},
        )
        _assert_complete_trace(hits, route=route, genai_span=genai_span, require_cost_span=False)

        root = next(span for span in hits[0].spans if not span.references)
        assert str(_tag(root, "http.status_code")) == "401", (
            f"the SERVER span must record the 401 the client received, got {_tag(root, 'http.status_code')!r}"
        )
        genai = next(span for span in hits[0].spans if span.operation_name == genai_span)
        _assert_error_span_contract(genai)
