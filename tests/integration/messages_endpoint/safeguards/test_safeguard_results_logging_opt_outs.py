"""The classifier call is logged the way the caller asked for its own turn. A callback the caller disabled by header
skips both calls, one disabled by key metadata logs the classifier call exactly when it logs the turn, a redaction
opt-in by header or body field keeps the conversation out of both payloads, and a turn without an opt-out logs both
calls with their content."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import pytest
from integration._support.client import eventually, held
from integration._support.otlp_sink import Span, SpanSinks, owned_sinks, recorded_spans
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    CHAT_MODEL,
    DANGEROUS_TOOL_USE_BETA,
    Rig,
    auto_mode_body,
    chat_backend,
    flagged,
    json_body,
    key_hash,
    nonce,
    safeguards_rig,
    tool_id,
    tool_use_ids,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(graceful_stop_seconds() + 240)

CALL_ID: Final = "litellm.call_id"
KEY_HASH: Final = "metadata.user_api_key_hash"
SPAN_SECONDS: Final = 30
QUIET_SECONDS: Final = 3


@pytest.fixture(scope="module")
def sinks(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SpanSinks]:
    with owned_sinks(tmp_path_factory.mktemp("safeguards-otlp")) as value:
        yield value


@pytest.fixture(scope="module")
def rig(sinks: SpanSinks, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    def otel_destination(wire_url: str) -> Mapping[str, JsonValue]:
        return {
            "callback_settings": {
                "otel": {"exporter": "http/json", "endpoint": sinks.operator, "use_simple_processor": True}
            }
        }

    def otel_exporter(wire_url: str) -> Mapping[str, str]:
        return {"OTEL_EXPORTER": "http/json", "OTEL_ENDPOINT": sinks.operator}

    with safeguards_rig(
        tmp_path_factory.mktemp("safeguards-otel"),
        environment=otel_exporter,
        general_settings={"allow_client_side_credentials": True},
        litellm_settings={"callbacks": ["otel"]},
        top_level=otel_destination,
    ) as value:
        yield value


def turn(rig: Rig, key: str, headers: Mapping[str, str] | None = None, **extra: JsonValue) -> str:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    response: Final = rig.gateway.client.post(
        "/v1/messages",
        json=dict(auto_mode_body(CHAT_MODEL, request_nonce, **extra)),
        headers={**rig.headers(key, DANGEROUS_TOOL_USE_BETA), **(headers or {})},
        timeout=120,
    )
    body: Final = json_body(response)
    assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
    return request_nonce


def key_spans(sinks: SpanSinks, key: str) -> tuple[Span, ...]:
    _, spans = recorded_spans(sinks.operator)
    return tuple(span for span in spans if span["attributes"].get(KEY_HASH) == key_hash(key))


def call_ids(spans: Sequence[Span]) -> frozenset[JsonValue]:
    return frozenset(span["attributes"][CALL_ID] for span in spans if CALL_ID in span["attributes"])


def calls_carrying(spans: Sequence[Span], text: str) -> frozenset[JsonValue]:
    return call_ids(tuple(span for span in spans if text in json.dumps(span["attributes"])))


def assert_both_calls_logged_with_content(rig: Rig, sinks: SpanSinks, key: str) -> None:
    shown: Final = turn(rig, key)
    eventually(
        lambda: key_spans(sinks, key), lambda spans: len(calls_carrying(spans, shown)) == 2, seconds=SPAN_SECONDS
    )


def assert_skipped_by_the_callback(rig: Rig, sinks: SpanSinks, hidden_key: str, shown_key: str) -> None:
    assert_both_calls_logged_with_content(rig, sinks, shown_key)
    held(lambda: key_spans(sinks, hidden_key), lambda spans: spans == (), holding=QUIET_SECONDS, seconds=SPAN_SECONDS)


def assert_logged_without_content(sinks: SpanSinks, key: str, redacted: str) -> None:
    spans: Final = held(
        lambda: key_spans(sinks, key),
        lambda spans: len(call_ids(spans)) == 2,
        holding=QUIET_SECONDS,
        seconds=SPAN_SECONDS,
    )
    assert calls_carrying(spans, redacted) == frozenset(), [span["attributes"] for span in spans]


def test_disable_callbacks_header_keeps_the_classifier_call_out_of_that_callback(rig: Rig, sinks: SpanSinks) -> None:
    with rig.gateway.scenario() as scenario:
        hidden_key: Final = scenario.key()
        turn(rig, hidden_key, headers={"x-litellm-disable-callbacks": "opentelemetry"})
        assert_skipped_by_the_callback(rig, sinks, hidden_key, scenario.key())


def test_key_metadata_callback_opt_out_logs_the_classifier_call_exactly_like_its_turn(
    rig: Rig, sinks: SpanSinks
) -> None:
    with rig.gateway.scenario() as scenario:
        opted_out: Final = scenario.key(metadata={"litellm_disabled_callbacks": ["opentelemetry"]})
        opted_out_turn: Final = turn(rig, opted_out)
        assert_both_calls_logged_with_content(rig, sinks, scenario.key())
        held(
            lambda: calls_carrying(key_spans(sinks, opted_out), opted_out_turn),
            lambda calls: len(calls) != 1,
            holding=QUIET_SECONDS,
            seconds=SPAN_SECONDS,
        )


def test_redaction_header_keeps_the_conversation_out_of_the_classifier_calls_payload(
    rig: Rig, sinks: SpanSinks
) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        redacted: Final = turn(rig, key, headers={"x-litellm-enable-message-redaction": "true"})
        assert_logged_without_content(sinks, key, redacted)
        assert_both_calls_logged_with_content(rig, sinks, key)


def test_turn_off_message_logging_in_the_body_keeps_the_conversation_out_of_the_classifier_calls_payload(
    rig: Rig, sinks: SpanSinks
) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        redacted: Final = turn(rig, key, turn_off_message_logging=True)
        assert_logged_without_content(sinks, key, redacted)
        assert_both_calls_logged_with_content(rig, sinks, key)
