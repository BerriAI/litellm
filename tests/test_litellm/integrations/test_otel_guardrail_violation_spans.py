"""
Tests for guardrail OTEL spans on violation.

Two distinct gaps surface together when a pre-call guardrail blocks the
request before it reaches the LLM provider:

  1. ``async_post_call_failure_hook`` (the OTEL hook that actually runs on
     the proxy failure path) only stamps attributes on the proxy parent
     span. It never creates the child ``guardrail`` span, even though
     ``request_data["metadata"]["standard_logging_guardrail_information"]``
     is populated by the time the hook runs.

  2. ``_create_guardrail_span`` records ``guardrail_name`` / ``guardrail_mode``
     / ``guardrail_response`` but does not surface ``guardrail_status``
     (success / guardrail_intervened / guardrail_failed_to_respond /
     not_run) or the violation categories (Bedrock topic policy names,
     content filter types, etc.) as queryable span attributes — the data
     is buried inside the serialised ``guardrail_response`` blob and cannot
     be filtered on in the trace backend.

The tests below use real OTEL SDK objects (TracerProvider +
InMemorySpanExporter + a real BatchSpanProcessor-equivalent) and the
real ``OpenTelemetry`` integration. No monkey patching of the integration
under test — only the OTEL exporter is in-memory.
"""

import os
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.opentelemetry import (
    LITELLM_REQUEST_SPAN_NAME,
    OpenTelemetry,
)
from litellm.proxy._types import UserAPIKeyAuth


GUARDRAIL_SPAN_NAME = "guardrail"
PROXY_SPAN_NAME = "Received Proxy Server Request"


def _bedrock_block_response():
    """Realistic Bedrock ApplyGuardrail response when a topic policy fires.

    Mirrors the shape in ``litellm/types/proxy/guardrails/guardrail_hooks/
    bedrock_guardrails.py`` so the violation-category extraction can be
    tested against the exact payload Bedrock returns.
    """
    return {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "topicPolicy": {
                    "topics": [
                        {
                            "name": "Fiduciary Advice",
                            "type": "DENY",
                            "action": "BLOCKED",
                        }
                    ]
                },
                "contentPolicy": {
                    "filters": [
                        {
                            "type": "VIOLENCE",
                            "confidence": "HIGH",
                            "action": "BLOCKED",
                        }
                    ]
                },
                "wordPolicy": {
                    "customWords": [{"match": "secret-codeword", "action": "BLOCKED"}],
                    "managedWordLists": [
                        {"match": "fuck", "type": "PROFANITY", "action": "BLOCKED"}
                    ],
                },
            }
        ],
        "outputs": [{"text": "Sorry, the model cannot respond to this request."}],
    }


def _slg_entry(
    guardrail_status,
    guardrail_response,
    *,
    name="bedrock-test",
    mode="pre_call",
    provider="bedrock",
    start=1.0,
    end=2.0,
    violation_categories=None,
    guardrail_action=None,
):
    """Build a StandardLoggingGuardrailInformation entry the way
    ``add_standard_logging_guardrail_information_to_request_data`` does."""
    entry = {
        "guardrail_name": name,
        "guardrail_provider": provider,
        "guardrail_mode": mode,
        "guardrail_response": guardrail_response,
        "guardrail_status": guardrail_status,
        "start_time": start,
        "end_time": end,
        "duration": end - start,
    }
    if violation_categories is not None:
        entry["violation_categories"] = violation_categories
    if guardrail_action is not None:
        entry["guardrail_action"] = guardrail_action
    return entry


def _kwargs_with_guardrail(
    *,
    entries,
    parent_span=None,
    include_exception=False,
):
    """Build the kwargs / model_call_details shape that the OTEL integration
    consumes. ``litellm_params.metadata`` is the SAME dict that the proxy's
    ``request_data["metadata"]`` becomes after ``update_environment_variables``,
    so ``_otel_internal`` dedupe state lives there too."""
    metadata = {"standard_logging_guardrail_information": list(entries)}
    if parent_span is not None:
        metadata["litellm_parent_otel_span"] = parent_span
    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "optional_params": {},
        "litellm_params": {
            "custom_llm_provider": "openai",
            "metadata": metadata,
        },
        "standard_logging_object": {
            "id": "test-call-id",
            "call_type": "completion",
            "metadata": metadata,
            "hidden_params": {},
            "guardrail_information": list(entries),
        },
    }
    if include_exception:
        kwargs["exception"] = Exception("guardrail blocked the request")
    return kwargs


def _make_otel():
    """Spin up a real OTEL pipeline backed by an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel = OpenTelemetry(tracer_provider=provider)
    otel.tracer = provider.get_tracer(__name__)
    return otel, provider, exporter


def _run(coro):
    """Run a coroutine on a fresh event loop and close it — prevents the
    "unclosed event loop" / ResourceWarning that you get from
    asyncio.new_event_loop().run_until_complete() with no cleanup."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _attr(span, key):
    return (span.attributes or {}).get(key)


class TestGuardrailSpanOnViolation(unittest.TestCase):
    """Bug 1: when a pre-call guardrail blocks, the guardrail span and the
    litellm_request span must both appear with the correct status."""

    def test_handle_failure_creates_litellm_request_and_guardrail_spans(self):
        """Driving ``_handle_failure`` with a populated
        ``standard_logging_object['guardrail_information']`` entry must
        emit both spans, parented correctly, with ERROR on the parent."""
        otel, _, exporter = _make_otel()

        kwargs = _kwargs_with_guardrail(
            entries=[
                _slg_entry("guardrail_intervened", _bedrock_block_response()),
            ],
            include_exception=True,
        )

        start = datetime.now(timezone.utc)
        end = start + timedelta(milliseconds=20)
        otel._handle_failure(kwargs, response_obj=None, start_time=start, end_time=end)

        spans = exporter.get_finished_spans()
        litellm_spans = [s for s in spans if s.name == LITELLM_REQUEST_SPAN_NAME]
        guardrail_spans = [s for s in spans if s.name == GUARDRAIL_SPAN_NAME]

        self.assertEqual(
            len(litellm_spans),
            1,
            "Expected exactly one litellm_request span on guardrail block",
        )
        self.assertEqual(litellm_spans[0].status.status_code, StatusCode.ERROR)

        self.assertEqual(
            len(guardrail_spans),
            1,
            "Expected exactly one guardrail span on guardrail block",
        )

        # Guardrail span must be a child of the litellm_request span
        self.assertIsNotNone(
            guardrail_spans[0].parent,
            "Guardrail span must be parented (not a root span)",
        )
        self.assertEqual(
            guardrail_spans[0].parent.span_id,
            litellm_spans[0].context.span_id,
        )

    def test_async_post_call_failure_hook_emits_guardrail_span(self):
        """The production failure path on the proxy calls
        ``async_post_call_failure_hook`` with the (still-populated)
        ``request_data``. The hook currently only stamps attrs on the proxy
        span; it must also emit the guardrail span so the violation is
        visible in the trace."""
        otel, provider, exporter = _make_otel()
        parent_span = provider.get_tracer(__name__).start_span(PROXY_SPAN_NAME)

        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            parent_otel_span=parent_span,
            request_route="/chat/completions",
        )

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {
                "standard_logging_guardrail_information": [
                    _slg_entry("guardrail_intervened", _bedrock_block_response())
                ],
            },
        }

        _run(
            otel.async_post_call_failure_hook(
                request_data=request_data,
                original_exception=Exception("guardrail blocked"),
                user_api_key_dict=user_api_key_dict,
            )
        )

        spans = exporter.get_finished_spans()
        guardrail_spans = [s for s in spans if s.name == GUARDRAIL_SPAN_NAME]
        self.assertEqual(
            len(guardrail_spans),
            1,
            "async_post_call_failure_hook must emit the guardrail span when "
            "request_data['metadata'] carries standard_logging_guardrail_information",
        )

        # The guardrail span must be parented to the proxy request span so
        # backends correlate it with the rest of the trace.
        self.assertIsNotNone(guardrail_spans[0].parent)
        self.assertEqual(
            guardrail_spans[0].parent.span_id,
            parent_span.context.span_id,
        )

    def test_handle_failure_and_post_call_failure_hook_dedupe(self):
        """When _handle_failure and async_post_call_failure_hook BOTH fire
        for the same request (the production flow on a guardrail block),
        exactly one guardrail span must be emitted. The dedupe relies on
        request_data['metadata'] and kwargs['litellm_params']['metadata']
        referencing the SAME dict so _emit_once sees its earlier marker."""
        otel, provider, exporter = _make_otel()
        parent_span = provider.get_tracer(__name__).start_span(PROXY_SPAN_NAME)

        # Shared metadata dict — same identity, mirroring how
        # update_environment_variables wires them in the proxy.
        shared_metadata = {
            "standard_logging_guardrail_information": [
                _slg_entry(
                    "guardrail_intervened",
                    _bedrock_block_response(),
                    violation_categories=["Fiduciary Advice"],
                )
            ],
        }

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {
                "custom_llm_provider": "openai",
                "metadata": shared_metadata,
            },
            "standard_logging_object": {
                "id": "test-call-id",
                "call_type": "completion",
                "metadata": shared_metadata,
                "hidden_params": {},
                "guardrail_information": shared_metadata[
                    "standard_logging_guardrail_information"
                ],
            },
            "exception": Exception("guardrail blocked"),
        }
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": shared_metadata,
        }
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            parent_otel_span=parent_span,
            request_route="/chat/completions",
        )

        start = datetime.now(timezone.utc)
        end = start + timedelta(milliseconds=20)
        otel._handle_failure(kwargs, response_obj=None, start_time=start, end_time=end)
        _run(
            otel.async_post_call_failure_hook(
                request_data=request_data,
                original_exception=Exception("guardrail blocked"),
                user_api_key_dict=user_api_key_dict,
            )
        )

        guardrail_spans = [
            s for s in exporter.get_finished_spans() if s.name == GUARDRAIL_SPAN_NAME
        ]
        self.assertEqual(
            len(guardrail_spans),
            1,
            "Dedupe must collapse the two emit calls into one span when the "
            "metadata dict identity is shared between kwargs and request_data",
        )


class TestGuardrailSpanAttributesOnViolation(unittest.TestCase):
    """Bug 2: the guardrail span must surface the violation status and
    violation categories as queryable span attributes, not bury them inside
    ``guardrail_response`` (which is logged as a single serialised blob)."""

    def _emit_and_get_guardrail_span(self, entry):
        otel, _, exporter = _make_otel()
        kwargs = _kwargs_with_guardrail(entries=[entry])
        otel._create_guardrail_span(kwargs=kwargs, context=None)

        guardrail_spans = [
            s for s in exporter.get_finished_spans() if s.name == GUARDRAIL_SPAN_NAME
        ]
        self.assertEqual(len(guardrail_spans), 1)
        return guardrail_spans[0]

    def test_status_attribute_present_for_intervened(self):
        entry = _slg_entry("guardrail_intervened", _bedrock_block_response())
        span = self._emit_and_get_guardrail_span(entry)
        self.assertEqual(
            _attr(span, "guardrail_status"),
            "guardrail_intervened",
            "guardrail_status must be exposed as a top-level span attribute",
        )

    def test_status_attribute_present_for_success(self):
        entry = _slg_entry(
            "success",
            {"action": "NONE", "assessments": []},
        )
        span = self._emit_and_get_guardrail_span(entry)
        self.assertEqual(_attr(span, "guardrail_status"), "success")

    def test_status_attribute_present_for_failed_to_respond(self):
        entry = _slg_entry(
            "guardrail_failed_to_respond",
            {"error": "endpoint unreachable"},
        )
        span = self._emit_and_get_guardrail_span(entry)
        self.assertEqual(_attr(span, "guardrail_status"), "guardrail_failed_to_respond")

    def test_violation_categories_surfaced_when_provider_populates_them(self):
        """The provider hook (e.g. Bedrock) extracts violation categories
        from the raw response BEFORE redaction and stamps them onto the
        StandardLoggingGuardrailInformation entry. OTEL must surface that
        list as a queryable span attribute so dashboards can group by
        violation type without parsing the redacted guardrail_response."""
        entry = _slg_entry(
            "guardrail_intervened",
            _bedrock_block_response(),
            violation_categories=["Fiduciary Advice", "VIOLENCE", "PROFANITY"],
        )
        span = self._emit_and_get_guardrail_span(entry)

        categories = _attr(span, "guardrail_violation_categories")
        self.assertIsNotNone(
            categories,
            "guardrail_violation_categories must be set when the entry "
            "carries violation_categories",
        )
        # Serialised as JSON to keep set_attribute typing simple.
        as_str = categories if isinstance(categories, str) else repr(list(categories))
        self.assertIn("Fiduciary Advice", as_str)
        self.assertIn("VIOLENCE", as_str)
        self.assertIn("PROFANITY", as_str)

    def test_no_violation_categories_when_field_absent(self):
        """When the provider didn't populate violation_categories (success
        path, or provider didn't extract them), don't pollute the trace
        with an empty attribute."""
        entry = _slg_entry("success", {"action": "NONE", "assessments": []})
        span = self._emit_and_get_guardrail_span(entry)
        self.assertIsNone(_attr(span, "guardrail_violation_categories"))

    def test_no_violation_categories_when_field_is_empty(self):
        """Empty list must not produce a span attribute either."""
        entry = _slg_entry(
            "guardrail_intervened",
            _bedrock_block_response(),
            violation_categories=[],
        )
        span = self._emit_and_get_guardrail_span(entry)
        self.assertIsNone(_attr(span, "guardrail_violation_categories"))

    def test_guardrail_action_surfaced_when_provider_populates_it(self):
        """The provider hook (e.g. Bedrock) writes its raw top-level
        ``action`` string onto StandardLoggingGuardrailInformation as
        ``guardrail_action``. OTEL must expose it as a queryable span
        attribute so dashboards can pivot on the raw provider verdict
        (Bedrock ``GUARDRAIL_INTERVENED`` / ``NONE``) without parsing
        the redacted guardrail_response blob."""
        entry = _slg_entry(
            "guardrail_intervened",
            _bedrock_block_response(),
            guardrail_action="GUARDRAIL_INTERVENED",
        )
        span = self._emit_and_get_guardrail_span(entry)
        self.assertEqual(
            _attr(span, "guardrail_action"),
            "GUARDRAIL_INTERVENED",
            "guardrail_action must be exposed as a top-level span attribute",
        )

    def test_guardrail_action_surfaced_for_allowed_request(self):
        """Even on the success path, the provider's raw action (e.g.
        Bedrock ``NONE``) should be queryable so dashboards can group
        allowed-vs-blocked counts off the same attribute."""
        entry = _slg_entry(
            "success",
            {"action": "NONE", "assessments": []},
            guardrail_action="NONE",
        )
        span = self._emit_and_get_guardrail_span(entry)
        self.assertEqual(_attr(span, "guardrail_action"), "NONE")

    def test_no_guardrail_action_when_field_absent(self):
        """If the provider didn't populate the field (older payloads,
        non-Bedrock providers without a top-level action), don't emit
        an empty attribute."""
        entry = _slg_entry("success", {"action": "NONE", "assessments": []})
        span = self._emit_and_get_guardrail_span(entry)
        self.assertIsNone(_attr(span, "guardrail_action"))


class TestMultipleGuardrailsOneBlocks(unittest.TestCase):
    """When several guardrails run sequentially and only the last one
    intervenes, every guardrail span must appear with its own status —
    losing the early "allowed" spans would mask which checks ran."""

    def test_all_guardrail_spans_emitted_with_per_entry_status(self):
        otel, _, exporter = _make_otel()

        entries = [
            _slg_entry(
                "success",
                {"action": "NONE", "assessments": []},
                name="pii-mask",
                start=1.0,
                end=1.5,
            ),
            _slg_entry(
                "success",
                {"action": "NONE", "assessments": []},
                name="prompt-injection",
                start=2.0,
                end=2.2,
            ),
            _slg_entry(
                "guardrail_intervened",
                _bedrock_block_response(),
                name="bedrock-policy",
                start=3.0,
                end=3.4,
            ),
        ]
        kwargs = _kwargs_with_guardrail(
            entries=entries,
            include_exception=True,
        )

        start = datetime.now(timezone.utc)
        end = start + timedelta(milliseconds=50)
        otel._handle_failure(kwargs, response_obj=None, start_time=start, end_time=end)

        spans = exporter.get_finished_spans()
        guardrail_spans = sorted(
            (s for s in spans if s.name == GUARDRAIL_SPAN_NAME),
            key=lambda s: (s.attributes or {}).get("guardrail_name", ""),
        )
        self.assertEqual(
            len(guardrail_spans),
            3,
            "Every guardrail invocation must emit a span — even the ones "
            "that allowed the request through before the blocker fired",
        )

        statuses = {
            _attr(s, "guardrail_name"): _attr(s, "guardrail_status")
            for s in guardrail_spans
        }
        self.assertEqual(statuses["pii-mask"], "success")
        self.assertEqual(statuses["prompt-injection"], "success")
        self.assertEqual(statuses["bedrock-policy"], "guardrail_intervened")


class TestCustomGuardrailEndToEnd(unittest.TestCase):
    """End-to-end: a real ``CustomGuardrail`` subclass calls
    ``add_standard_logging_guardrail_information_to_request_data`` and then
    raises. We then drive ``_handle_failure`` with the resulting kwargs
    (matching the shape ``async_failure_handler`` would build) and verify
    the guardrail span carries the recorded information."""

    def test_real_custom_guardrail_violation_path(self):
        # Deliberately not importing fastapi here — the real Bedrock guardrail
        # raises HTTPException, but the OTEL span flow is exception-type
        # agnostic. Using a plain Exception keeps this test runnable in
        # SDK-only installs that don't ship fastapi.
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        class BlockingViolation(Exception):
            pass

        class BlockingGuardrail(CustomGuardrail):
            async def async_pre_call_hook(
                self,
                user_api_key_dict,
                cache,
                data,
                call_type,
            ):
                start_ts = time.time()
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider="bedrock",
                    guardrail_json_response=_bedrock_block_response(),
                    request_data=data,
                    guardrail_status="guardrail_intervened",
                    start_time=start_ts,
                    end_time=start_ts + 0.01,
                    duration=0.01,
                    event_type=GuardrailEventHooks.pre_call,
                    tracing_detail={
                        "violation_categories": ["Fiduciary Advice", "VIOLENCE"]
                    },
                )
                raise BlockingViolation("violation")

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "metadata": {},
        }
        guardrail = BlockingGuardrail(
            guardrail_name="blocking-test",
            event_hook=GuardrailEventHooks.pre_call,
        )

        with self.assertRaises(BlockingViolation):
            _run(
                guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
                    cache=None,
                    data=request_data,
                    call_type="completion",
                )
            )

        slg_info = request_data["metadata"].get(
            "standard_logging_guardrail_information"
        )
        self.assertTrue(
            slg_info,
            "Guardrail must have recorded its information to request_data "
            "BEFORE raising — otherwise the OTEL hook sees nothing",
        )

        # Now simulate the OTEL failure handler picking up this metadata
        otel, _, exporter = _make_otel()
        kwargs = _kwargs_with_guardrail(
            entries=slg_info,
            include_exception=True,
        )
        start = datetime.now(timezone.utc)
        end = start + timedelta(milliseconds=15)
        otel._handle_failure(kwargs, response_obj=None, start_time=start, end_time=end)

        spans = exporter.get_finished_spans()
        guardrail_spans = [s for s in spans if s.name == GUARDRAIL_SPAN_NAME]
        self.assertEqual(len(guardrail_spans), 1)
        self.assertEqual(
            _attr(guardrail_spans[0], "guardrail_status"),
            "guardrail_intervened",
        )
        self.assertEqual(
            _attr(guardrail_spans[0], "guardrail_name"),
            "blocking-test",
        )
        # End-to-end: the violation_categories the guardrail passed through
        # tracing_detail must arrive as a queryable span attribute.
        categories = _attr(guardrail_spans[0], "guardrail_violation_categories")
        self.assertIsNotNone(categories)
        self.assertIn("Fiduciary Advice", str(categories))
        self.assertIn("VIOLENCE", str(categories))


if __name__ == "__main__":
    unittest.main()
