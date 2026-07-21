"""Live e2e: the built-in Presidio PII guardrail masks PII on the request, on the
model output, and in what the proxy logs.

Presidio replaces detected PII with `<ENTITY_TYPE>` placeholders (e.g.
`<EMAIL_ADDRESS>`) via a real analyzer + anonymizer. Three modes are checked
independently, each opted into per request (default_on=False) so it never touches
unrelated traffic:

- pre_call: the prompt is anonymized before it reaches the model, so a
  repeat-verbatim request comes back with the placeholder, never the raw email
- post_call (apply_to_output): PII the model itself emits is masked on the way
  out, so the caller never receives the raw value the model produced
- logging_only: the call is not blocked, and the request the proxy records is
  masked. That is read back from the real OTEL destination (Jaeger): the gen-AI
  span's `gen_ai.input.messages` attribute carries the masked placeholder, never
  the raw email

Analyzer/anonymizer bases come from PRESIDIO_ANALYZER_API_BASE /
PRESIDIO_ANONYMIZER_API_BASE (compose provides the in-network hosts; point them at
locally published container ports for a host run). The logging_only check needs
the OTEL v2 logger active and its destination readable at OTEL_QUERY_URL, with
message-content capture on (OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT).
The chat backend is a gemini deployment created for the test.
"""

from __future__ import annotations

import time

import pytest

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, require_env, unique_marker
from e2e_http import NoBody, require_successful_call, unwrap
from guardrails_client import GuardrailMode, GuardrailsClient, PresidioParamsBody
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, ReadinessDetailsResponse
from otel_client import JaegerSpan, OtelReader, build_otel_reader

pytestmark = pytest.mark.e2e

RAW_EMAIL = "alice.example.person@example.com"
PLACEHOLDER = "<EMAIL_ADDRESS>"

ECHO_REQUEST = f"Repeat the following text back exactly, verbatim, with no changes: My email is {RAW_EMAIL}"
EMIT_REQUEST = f"Output exactly this one line and nothing else: Please contact {RAW_EMAIL} today"
LOG_REQUEST = f"Say hello and include this email once verbatim: {RAW_EMAIL}"

OTEL_V2_LOGGER = "OpenTelemetryV2"
INPUT_MESSAGES_TAG = "gen_ai.input.messages"


def _content(response: ChatResponse) -> str:
    if not response.choices:
        return ""
    message = response.choices[0].message
    return (message.content if message else None) or ""


def _span_tag(span: JaegerSpan, key: str) -> str | None:
    for tag in span.tags:
        if tag.key == key and isinstance(tag.value, str):
            return tag.value
    return None


def _poll_logged_prompt(reader: OtelReader, *, call_id: str, genai_span: str) -> str | None:
    """Poll the OTEL destination until the call's gen-AI span carries a masked
    logged prompt, and return it. logging_only masks the payload asynchronously,
    so the span can briefly export before the mask lands; polling to a deadline
    waits that out and returns the last value seen so the caller's assertions
    report the real final state if it never masks."""
    deadline = time.monotonic() + POLL_TIMEOUT
    last: str | None = None
    while time.monotonic() < deadline:
        for trace in reader.traces_for_call(call_id):
            for span in trace.spans:
                if span.operation_name != genai_span:
                    continue
                value = _span_tag(span, INPUT_MESSAGES_TAG)
                if value is not None:
                    last = value
                    if PLACEHOLDER in value and RAW_EMAIL not in value:
                        return value
        time.sleep(POLL_INTERVAL)
    return last


def _presidio_params(
    mode: GuardrailMode, *, apply_to_output: bool = False, logging_only: bool = False
) -> PresidioParamsBody:
    analyzer, anonymizer = require_env(
        "PRESIDIO_ANALYZER_API_BASE", "PRESIDIO_ANONYMIZER_API_BASE"
    )
    return PresidioParamsBody(
        mode=mode,
        default_on=False,
        presidio_analyzer_api_base=analyzer,
        presidio_anonymizer_api_base=anonymizer,
        apply_to_output=apply_to_output,
        logging_only=logging_only,
    )


def _require_otel_v2_active(client: GuardrailsClient) -> None:
    details = unwrap(
        client.proxy.transport.get(
            "/health/readiness/details",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=ReadinessDetailsResponse,
        )
    )
    assert OTEL_V2_LOGGER in details.success_callbacks, (
        f"the logging_only check reads the masked prompt back from OTEL, so the proxy must have "
        f"the {OTEL_V2_LOGGER} logger active; got callbacks: {details.success_callbacks}"
    )


class TestPresidioGuardrail:
    @pytest.mark.covers(
        "guardrail.presidio.pre_call.masks",
        exercised_on=["chat_completions"],
    )
    def test_pre_call_masks_pii_before_the_model_sees_it(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        require_env("GEMINI_API_KEY")
        model = client.create_backend_model(resources, prefix="e2e-presidio-pre")
        name = f"e2e-presidio-pre-{unique_marker()}"
        guardrail_id = client.register(name, _presidio_params("pre_call"))
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        echoed = _content(
            unwrap(client.chat(scoped_key, model, ECHO_REQUEST, guardrails=[name], max_tokens=128))
        )
        assert RAW_EMAIL not in echoed, (
            "pre_call masking must strip the raw email before the model sees it, but the "
            f"model echoed it back: {echoed[:300]!r}"
        )
        assert PLACEHOLDER in echoed, (
            "the model should have echoed the masked placeholder the guardrail substituted, "
            f"got: {echoed[:300]!r}"
        )

    @pytest.mark.covers(
        "guardrail.presidio.post_call.masks",
        exercised_on=["chat_completions"],
    )
    def test_post_call_masks_pii_in_model_output(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        require_env("GEMINI_API_KEY")
        model = client.create_backend_model(resources, prefix="e2e-presidio-post")
        name = f"e2e-presidio-post-{unique_marker()}"
        guardrail_id = client.register(name, _presidio_params("post_call", apply_to_output=True))
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        out = _content(
            unwrap(client.chat(scoped_key, model, EMIT_REQUEST, guardrails=[name], max_tokens=128))
        )
        assert RAW_EMAIL not in out, (
            "post_call masking must strip PII the model emitted, but the raw email reached the "
            f"caller: {out[:300]!r}"
        )
        assert PLACEHOLDER in out, (
            f"the masked placeholder should replace the model's PII output, got: {out[:300]!r}"
        )

    @pytest.mark.covers(
        "guardrail.presidio.logging_only.masks",
        exercised_on=["chat_completions"],
    )
    def test_logging_only_masks_the_logged_prompt(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        require_env("GEMINI_API_KEY")
        _require_otel_v2_active(client)
        reader = build_otel_reader()

        model = client.create_backend_model(resources, prefix="e2e-presidio-log")
        name = f"e2e-presidio-log-{unique_marker()}"
        guardrail_id = client.register(name, _presidio_params("logging_only", logging_only=True))
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        outcome = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=LOG_REQUEST)],
                max_tokens=64,
                guardrails=[name],
            ),
        )
        require_successful_call(outcome)  # logging_only must not block
        assert outcome.call_id is not None, "the response must carry x-litellm-call-id to find its trace"

        genai_span = f"chat {model}"
        logged_prompt = _poll_logged_prompt(reader, call_id=outcome.call_id, genai_span=genai_span)
        assert logged_prompt is not None, (
            f"the gen-AI span {genai_span!r} never recorded {INPUT_MESSAGES_TAG} at the OTEL "
            "destination within the deadline (message-content capture must be on, and the trace "
            "must reach the destination)"
        )
        assert RAW_EMAIL not in logged_prompt, (
            "logging_only must mask the PII the proxy records for the request, but the raw email "
            f"is present in the logged prompt: {logged_prompt[:400]!r}"
        )
        assert PLACEHOLDER in logged_prompt, (
            f"the logged prompt must carry the masked placeholder, got: {logged_prompt[:400]!r}"
        )
