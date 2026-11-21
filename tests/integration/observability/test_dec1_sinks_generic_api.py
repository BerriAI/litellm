"""Generic API (webhook) sink contract: named `generic_api` callbacks from `callback_settings`, body formats,
event filters, retries, the v1 payload, and outage isolation.

Siblings already pin the default `generic_api` callback (a JSON array of StandardLoggingPayloads with the
`GENERIC_LOGGER_HEADERS` headers) in `test_dec1_callbacks_scoping.py` and the payload fields in
`test_dec1_callbacks_payload_*.py`. This file covers the operator configured webhooks: an entry under
`callback_settings` with `callback_type: generic_api` turns any callback name listed in `callbacks` into its own
GenericAPILogger with its own endpoint, headers, `event_types`, `log_format` (`json_array`, `ndjson`, `single`),
`max_retries` and `retry_delay` (litellm/litellm_core_utils/logging_callback_manager.py
`add_custom_callback_generic_api_str`, litellm/integrations/generic_api/generic_api_callback.py). A second rig sets
`generic_api_use_v1: true`, the legacy body where every value is stringified.

Each named webhook posts to its own path on the scripted peer, so a test reads exactly what one webhook sent. These
are the formats SIEM and log shippers (Sumo Logic uses ndjson) ingest, so the Rust port must keep them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Final

import pytest
from integration._support.client import eventually, held
from integration.observability.dec1_callbacks_support import CHAT_COST, CHAT_REJECTED
from integration.observability.dec1_sinks_support import (
    SinkRig,
    assert_not_held,
    chat,
    healthy_seconds,
    marker,
    new_call_id,
    sinks_rig,
    timed_chat,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

WEBHOOK_SECRET: Final = "synthetic-dec1-webhook-secret"
NAMED: Final = ("dec1_ndjson", "dec1_single", "dec1_array", "dec1_failures", "dec1_retry_500", "dec1_retry_400")


def _webhooks(urls: Mapping[str, str]) -> Mapping[str, Mapping[str, JsonValue]]:
    base: Final = urls["generic"]
    headers: Final = {"Authorization": f"Bearer {WEBHOOK_SECRET}", "X-Dec1-Sink": "named"}

    def entry(path: str, **fields: JsonValue) -> dict[str, JsonValue]:
        return {"callback_type": "generic_api", "endpoint": base + path, "headers": headers, **fields}

    return {
        "sections": {
            "callback_settings": {
                "dec1_ndjson": entry("/ndjson", log_format="ndjson"),
                "dec1_single": entry("/single", log_format="single"),
                "dec1_array": entry("/array"),
                "dec1_failures": entry("/failures", event_types=["llm_api_failure"]),
                "dec1_retry_500": entry("/always-500", max_retries=2, retry_delay=0),
                "dec1_retry_400": entry("/always-400", max_retries=2, retry_delay=0),
            }
        }
    }


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SinkRig]:
    with sinks_rig(
        tmp_path_factory.mktemp("dec1-sinks-generic"), {"callbacks": list(NAMED)}, late=_webhooks
    ) as created:
        yield created


@pytest.fixture(scope="module")
def legacy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SinkRig]:
    with sinks_rig(
        tmp_path_factory.mktemp("dec1-sinks-generic-v1"),
        {"callbacks": ["generic_api"], "generic_api_use_v1": True},
    ) as created:
        yield created


def _posts(rig: SinkRig, path: str, needle: str) -> tuple[bytes, ...]:
    return tuple(request.body for request in rig.peer("generic").posts(path) if needle.encode() in request.body)


def _await_posts(rig: SinkRig, path: str, needle: str, count: int = 1) -> tuple[bytes, ...]:
    return eventually(lambda: _posts(rig, path, needle), lambda bodies: len(bodies) >= count, seconds=30)


def test_named_webhook_posts_to_its_own_endpoint_with_its_own_headers(rig: SinkRig) -> None:
    """A success must reach the `dec1_array` endpoint (/array) with the configured `Authorization: Bearer <secret>`
    and `X-Dec1-Sink: named` headers and `Content-Type: application/json`."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    _await_posts(rig, "/array", call_id)
    for request in rig.peer("generic").posts("/array"):
        assert request.headers["authorization"] == f"Bearer {WEBHOOK_SECRET}"
        assert request.headers["x-dec1-sink"] == "named"
        assert request.headers["content-type"] == "application/json"


def test_default_log_format_is_a_json_array_of_payloads(rig: SinkRig) -> None:
    """With no `log_format`, the body must parse as a JSON array whose element for the call has litellm_call_id
    equal to the header and status "success"."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    body: Final = json.loads(_await_posts(rig, "/array", call_id)[0])
    assert isinstance(body, list)
    assert [entry["status"] for entry in body if entry["litellm_call_id"] == call_id] == ["success"]


def test_ndjson_log_format_posts_one_payload_per_line(rig: SinkRig) -> None:
    """`log_format: ndjson` must post newline separated JSON objects, never a JSON array: every non empty line of
    the body parses to an object, and one of them has the call id."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    body: Final = _await_posts(rig, "/ndjson", call_id)[0].decode()
    assert not body.lstrip().startswith("[")
    lines: Final = [json.loads(line) for line in body.split("\n") if line.strip()]
    assert all(isinstance(line, dict) for line in lines)
    assert [line["litellm_call_id"] for line in lines].count(call_id) == 1


def test_ndjson_batches_several_calls_into_one_post(rig: SinkRig) -> None:
    """Three calls made back to back within one flush interval must arrive in fewer than three ndjson posts, each
    post holding one line per payload. UNVERIFIED: relies on DEFAULT_FLUSH_INTERVAL_SECONDS=1 grouping the burst."""
    calls: Final = tuple(new_call_id() for _ in range(3))
    for call_id in calls:
        assert chat(rig, call_id, marker()).status_code == 200
    for call_id in calls:
        _await_posts(rig, "/ndjson", call_id)
    posts: Final = {body for call_id in calls for body in _posts(rig, "/ndjson", call_id)}
    assert len(posts) < 3


def test_single_log_format_posts_each_payload_as_its_own_object(rig: SinkRig) -> None:
    """`log_format: single` must post one request per payload whose body is a single JSON object (not an array)
    with the call id."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    body: Final = json.loads(_await_posts(rig, "/single", call_id)[0])
    assert isinstance(body, dict)
    assert body["litellm_call_id"] == call_id


def test_failure_only_event_types_skip_successes(rig: SinkRig) -> None:
    """`event_types: [llm_api_failure]` must deliver the failure of a provider 400 to /failures and never deliver a
    successful call there."""
    ok, failed = new_call_id(), new_call_id()
    assert chat(rig, ok, marker()).status_code == 200
    assert chat(rig, failed, marker(), model=CHAT_REJECTED).status_code == 400
    body: Final = json.loads(_await_posts(rig, "/failures", failed)[0])
    assert [entry["status"] for entry in body if entry["litellm_call_id"] == failed] == ["failure"]
    _await_posts(rig, "/array", ok)
    held(lambda: _posts(rig, "/failures", ok), lambda bodies: not bodies, holding=3, seconds=10)


def test_a_5xx_endpoint_is_retried_max_retries_times(rig: SinkRig) -> None:
    """An endpoint answering 500 with `max_retries: 2` and `retry_delay: 0` must receive the same batch body three
    times (one try plus two retries) and then no more (`_post_with_retries`)."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    bodies: Final = held(
        lambda: _posts(rig, "/always-500", call_id), lambda found: len(found) == 3, holding=3, seconds=30
    )
    assert len(set(bodies)) == 1


def test_a_4xx_endpoint_is_never_retried(rig: SinkRig) -> None:
    """An endpoint answering 400 with `max_retries: 2` must receive the batch exactly once, since only timeouts,
    transport errors and 5xx are retried."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    held(lambda: _posts(rig, "/always-400", call_id), lambda found: len(found) == 1, holding=3, seconds=30)


def test_failing_webhooks_never_fail_the_request(rig: SinkRig) -> None:
    """With two of the six webhooks permanently failing (500 retried twice, 400), a request must still return 200
    and take no longer than a healthy control plus the 30 second stall bound. The control is measured on the same
    rig, so it already includes the failing webhooks; what is pinned is that the request never waits on them."""
    control: Final = healthy_seconds(rig)
    response, elapsed = timed_chat(rig, new_call_id(), marker())
    assert response.status_code == 200
    assert_not_held(elapsed, control)


def test_stalled_webhooks_never_hold_the_response(rig: SinkRig) -> None:
    """With the webhook peer holding every post open, three chats must each return 200 within the healthy control time plus the 30 second stall, and the
    held payloads must be delivered once the peer recovers."""
    control: Final = healthy_seconds(rig)
    peer: Final = rig.peer("generic")
    peer.switch.stall()
    calls: Final = tuple(new_call_id() for _ in range(3))
    try:
        for call_id in calls:
            response, elapsed = timed_chat(rig, call_id, marker())
            assert response.status_code == 200
            assert_not_held(elapsed, control)
    finally:
        peer.switch.release()
    for call_id in calls:
        _await_posts(rig, "/array", call_id)


def test_v1_payload_stringifies_every_field(legacy: SinkRig) -> None:
    """With `generic_api_use_v1: true` the batch element for a call must have exactly the keys id, call_type,
    cache_hit, startTime, endTime, model, user, modelParameters, messages, response, usage, metadata and cost, every
    value a string, with call_type "acompletion" and cost "0.019" (`_get_v1_logging_payload`)."""
    call_id, text = new_call_id(), marker()
    assert chat(legacy, call_id, text).status_code == 200
    entries: Final = eventually(
        lambda: tuple(
            entry
            for body in legacy.generic_bodies()
            for entry in json.loads(body)
            if text in str(entry.get("messages"))
        ),
        lambda found: len(found) >= 1,
        seconds=30,
    )
    entry: Final = entries[0]
    assert set(entry) == {
        "id", "call_type", "cache_hit", "startTime", "endTime", "model", "user",
        "modelParameters", "messages", "response", "usage", "metadata", "cost",
    }  # fmt: skip
    assert all(isinstance(value, str) for value in entry.values())
    assert entry["call_type"] == "acompletion"
    assert float(entry["cost"]) == pytest.approx(CHAT_COST)


def test_v1_payload_is_not_sent_for_failures_without_a_response(legacy: SinkRig) -> None:
    """A provider 400 has no response object, and the v1 builder reads `response_obj["usage"]`, so the failure is
    dropped with a logged error instead of being delivered. The request still returns 400. UNVERIFIED: whether a
    rewrite should deliver it; this pins that nothing reaches the webhook for the failed call."""
    text: Final = marker()
    assert chat(legacy, new_call_id(), text, model=CHAT_REJECTED).status_code == 400
    held(
        lambda: tuple(body for body in legacy.generic_bodies() if text.encode() in body),
        lambda found: not found,
        holding=5,
        seconds=10,
    )
