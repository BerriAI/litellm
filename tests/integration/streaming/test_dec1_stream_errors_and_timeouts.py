"""How a streaming request fails.

There are two regimes and the boundary between them is the first byte. Before the proxy has written
anything it can still answer with a real HTTP status and a JSON error body, exactly like a non streaming
call; that covers auth failures, limits, provider errors on stream open and a provider that never sends
a first chunk. After the first chunk the status line (200) is already on the wire, so a failure becomes
an in-band frame: `data: {"error": {message, type, param, code}}` followed by `data: [DONE]`, and the
call is logged as a failure. Clients branch on this difference, so a rewrite must keep both the boundary
and the shapes.
"""

from __future__ import annotations

import json
import threading
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply
from integration.streaming.dec1_stream_support import (
    chunks_of,
    content_of,
    delta,
    finish_reasons,
    marker,
    payloads,
    scripted,
    serving,
    sse,
    text_frames,
)


def _provider_error(status: int) -> Reply:
    body: Final = {"error": {"message": f"scripted provider failure {status}", "type": "scripted_error", "code": None}}
    return Reply(status=status, body=json.dumps(body).encode())


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
def test_provider_error_on_stream_open_is_an_http_error_with_a_json_body(gateway: Gateway, status: int) -> None:
    """The provider refuses the streaming request with an HTTP error before sending any event. The caller
    receives that same status code, an `application/json` body shaped
    `{"error": {"message", "type", "param", "code"}}`, the provider's message inside `message`, and no
    `data:` frame. A caller's retry policy keys off the status code, which a 200 event stream with an
    error frame would hide."""
    with scripted(gateway, lambda request: _provider_error(status)) as rig:
        response: Final = rig.chat()
        assert response.status_code == status, response.text
        assert response.headers["content-type"].split(";")[0] == "application/json"
        assert not response.text.startswith("data:")
        error: Final = object_value(response.json()["error"])
        assert {"message", "type", "param", "code"} <= set(error)
        assert f"scripted provider failure {status}" in str(error["message"])
        assert str(error["code"]) == str(status)


def test_error_on_stream_open_still_carries_the_call_id_header(gateway: Gateway) -> None:
    """A stream that fails on open with a provider 500 answers with an `x-litellm-call-id` header, so the
    failed call can be found in logs. UNVERIFIED: read from `handle_llm_api_exception` building headers
    with `get_custom_headers` in `litellm/proxy/common_request_processing.py`."""
    with scripted(gateway, lambda request: _provider_error(500)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 500
        assert response.headers.get("x-litellm-call-id")


def test_failed_stream_open_writes_one_failure_row_and_no_spend(gateway: Gateway) -> None:
    """A stream fails on open with a provider 400. One SpendLogs row lands for the key with
    `status == "failure"`, `spend == 0` and zero completion tokens. Failed calls are visible in the
    logs UI but must never cost anything."""
    with scripted(
        gateway, lambda request: _provider_error(400), input_cost_per_token=0.001, output_cost_per_token=0.002
    ) as rig:
        assert rig.chat().status_code == 400
        row: Final = rig.spend_rows()[0]
        assert row["status"] == "failure"
        assert float(str(row["spend"])) == 0.0
        assert row["completion_tokens"] == 0


def test_unknown_key_on_a_streaming_request_is_a_json_401_and_never_reaches_the_provider(gateway: Gateway) -> None:
    """The caller streams with a key the proxy has never issued. The answer is a 401 JSON error, not an
    event stream, and the provider peer receives nothing."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("never",)))) as rig:
        response: Final = rig.post("/v1/chat/completions", rig.chat_body(), key="sk-dec1-not-a-real-key")
        assert response.status_code == 401, response.text
        assert response.headers["content-type"].split(";")[0] == "application/json"
        assert "error" in response.json()
        assert rig.upstream_bodies() == []


def test_key_without_access_to_the_model_gets_a_json_403_before_any_stream(gateway: Gateway) -> None:
    """A key scoped to one deployment streams against a different deployment. The proxy answers 403 with
    a JSON error whose type is `key_model_access_denied` and neither provider is called. The key is valid,
    so this is an authorization refusal (403), not an authentication one (401):
    `_can_object_call_model` in `litellm/proxy/auth/auth_checks.py` raises
    `ModelAccessDeniedProxyException` with `HTTP_403_FORBIDDEN`."""
    with (
        scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("mine",)))) as allowed,
        scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("theirs",)))) as other,
    ):
        response: Final = allowed.post("/v1/chat/completions", other.chat_body())
        assert response.status_code == 403, response.text
        assert object_value(response.json()["error"])["type"] == "key_model_access_denied"
        assert other.upstream_bodies() == [] and allowed.upstream_bodies() == []


def test_unknown_model_on_a_streaming_request_is_a_json_400(gateway: Gateway) -> None:
    """The master key streams against a model name no deployment serves. The answer is a 400 JSON error
    and not an event stream. UNVERIFIED on the status: `Invalid model name passed in model=...` is raised
    as a 400 `ProxyException` by the route's model resolution in `litellm/proxy/proxy_server.py`."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("never",)))) as rig:
        body: Final = {**rig.chat_body(), "model": f"dec1-no-such-model-{marker()}"}
        response: Final = rig.post("/v1/chat/completions", body, key=gateway.key)
        assert response.status_code == 400, response.text
        assert response.headers["content-type"].split(";")[0] == "application/json"
        assert rig.upstream_bodies() == []


def test_second_stream_over_the_key_rpm_limit_is_a_json_429(gateway: Gateway) -> None:
    """A key with `rpm_limit: 1` streams twice in the same minute. The first stream completes. The second
    is refused with a 429 JSON error before the provider is called, so the provider sees exactly one
    request."""
    with scripted(
        gateway, lambda request: sse(text_frames(f"chatcmpl-{marker()}", ("first",))), key_fields={"rpm_limit": 1}
    ) as rig:
        assert rig.chat().status_code == 200
        second: Final = rig.chat()
        assert second.status_code == 429, second.text
        assert response_is_json(second.headers["content-type"])
        assert len(rig.upstream_bodies()) == 1


def response_is_json(content_type: str) -> bool:
    return content_type.split(";")[0] == "application/json"


def test_connection_drop_after_content_ends_the_stream_with_an_error_frame_then_done(gateway: Gateway) -> None:
    """The provider sends two content chunks and then the TCP connection drops without a finish reason.
    The caller's HTTP status is 200 (already sent), the two text pieces arrive, then one frame
    `data: {"error": {...}}` with `message`, `type`, `param` and `code`, then `data: [DONE]`. No chunk
    claims a `finish_reason`, so a client cannot mistake the truncated text for a complete answer."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = text_frames(identity, ("partial ", "answer", "never sent"))
    with scripted(gateway, lambda request: sse(frames, abort_after=2)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200
        seen: Final = payloads(response.text)
        assert seen[-1] == "[DONE]"
        error_frame: Final = json.loads(seen[-2])
        assert {"message", "type", "param", "code"} <= set(error_frame["error"])
        chunks: Final = [json.loads(item) for item in seen[:-2]]
        assert content_of(chunks) == "partial answer"
        assert finish_reasons(chunks) == []


def test_mid_stream_failure_is_logged_as_a_failure_row(gateway: Gateway) -> None:
    """A stream is cut by the provider after two content chunks. Exactly one SpendLogs row lands for the
    key and its `status` is `failure`. A cut stream must not be recorded as a successful, fully billed
    call, and must not vanish from the logs either."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = text_frames(identity, ("cut ", "here", "unsent"))
    with scripted(gateway, lambda request: sse(frames, abort_after=2)) as rig:
        assert rig.chat().status_code == 200
        rows: Final = rig.spend_rows()
        assert [row["status"] for row in rows] == ["failure"]


def test_provider_error_object_mid_stream_becomes_the_callers_error_frame(gateway: Gateway) -> None:
    """After one content chunk the provider sends `data: {"error": {"message": "scripted overload",
    "type": "server_error", "code": "overloaded"}}`. The caller gets the content chunk, then an error
    frame whose message contains `scripted overload`, then `[DONE]`. The provider's error text has to
    survive so users can tell an overload from a content filter."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "content": "before "}),
        b'data: {"error": {"message": "scripted overload", "type": "server_error", "code": "overloaded"}}\n\n',
    )
    with scripted(gateway, lambda request: sse(frames)) as rig:
        seen: Final = payloads(rig.chat().text)
        assert seen[-1] == "[DONE]"
        errors: Final = [json.loads(item) for item in seen[:-1] if '"error"' in item]
        assert len(errors) == 1
        assert "scripted overload" in errors[0]["error"]["message"]
        assert content_of(chunks_of("\n".join(f"data: {item}" for item in seen if '"error"' not in item))) == "before "


def test_stream_after_a_failed_stream_on_the_same_deployment_succeeds(gateway: Gateway) -> None:
    """A deployment's first stream is cut mid answer and its second stream is healthy. The second caller
    gets the complete text and a `stop` finish. The shared proxy disables cooldowns, so this pins that a
    broken stream leaves no poisoned connection or iterator behind."""
    calls: Final = []

    def respond(request):  # type: ignore[no-untyped-def]
        calls.append(request.target)
        frames: Final = text_frames(f"chatcmpl-{marker()}", ("healthy ", "again", "tail"))
        return sse(frames, abort_after=1) if len(calls) == 1 else sse(frames)

    with scripted(gateway, respond) as rig:
        first: Final = payloads(rig.chat().text)
        assert any('"error"' in item for item in first)
        chunks: Final = chunks_of(rig.chat().text)
        assert content_of(chunks) == "healthy againtail"
        assert finish_reasons(chunks) == ["stop"]


def test_provider_silent_past_stream_timeout_before_the_first_chunk_is_an_http_timeout_error(gateway: Gateway) -> None:
    """A deployment has `stream_timeout: 1` and its provider accepts the request, sends response headers,
    and then never sends an event. The caller gets a JSON error with HTTP status 408 instead of a hung
    connection. UNVERIFIED on 408: `litellm.Timeout` carries `status_code = 408`
    (`litellm/exceptions.py`); the router may retry first, so the test allows 30 seconds."""
    never: Final = threading.Event()
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("too ", "late"))
    with scripted(gateway, lambda request: sse(frames, gate_before_first=never), stream_timeout=1) as rig:
        response: Final = rig.chat()
        assert response.status_code == 408, response.text
        assert response_is_json(response.headers["content-type"])
        assert "error" in response.json()


def test_provider_stalling_mid_stream_past_stream_timeout_ends_with_an_error_frame(gateway: Gateway) -> None:
    """A deployment has `stream_timeout: 1`. Its provider sends one content chunk and then stalls forever.
    The caller receives that chunk, then within seconds an in-band error frame and `[DONE]`, with HTTP
    status 200. `stream_timeout` is a gap timeout between chunks, not only a time to first byte."""
    never: Final = threading.Event()
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("first ", "then silence"))
    with scripted(gateway, lambda request: sse(frames, gate_after_first=never), stream_timeout=1) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200
        seen: Final = payloads(response.text)
        assert seen[-1] == "[DONE]"
        assert "error" in json.loads(seen[-2])
        assert content_of([json.loads(item) for item in seen[:-2]]) == "first "


def test_stream_timeout_header_from_the_caller_bounds_the_wait_for_the_first_chunk(gateway: Gateway) -> None:
    """The deployment has no stream timeout. The caller sends `x-litellm-stream-timeout: 1` and the
    provider never sends an event. The caller gets an error response (408) within seconds rather than
    waiting for the default request timeout. Read from the header handling at
    `litellm/proxy/litellm_pre_call_utils.py` (`x-litellm-stream-timeout`)."""
    never: Final = threading.Event()
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("too ", "late"))
    with scripted(gateway, lambda request: sse(frames, gate_before_first=never)) as rig:
        response: Final = rig.chat(headers={"x-litellm-stream-timeout": "1"})
        assert response.status_code == 408, response.text


def test_request_timeout_alone_does_not_cut_a_slow_but_live_stream(gateway: Gateway) -> None:
    """A deployment has `timeout: 1` and `stream_timeout: 10`. Its provider streams five chunks 0.4
    seconds apart, 1.6 seconds in total. The caller receives the full text and a `stop` finish: for a
    stream, `stream_timeout` replaces `timeout`, and it bounds the gap between chunks rather than the
    whole answer. UNVERIFIED: the precedence is read from `stream_timeout` handling in
    `litellm/router.py` and `litellm/main.py`."""
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("s1 ", "s2 ", "s3 ", "s4"), usage=None)
    with scripted(gateway, lambda request: sse(frames, pause_between_chunks=0.4), timeout=1, stream_timeout=10) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert content_of(chunks) == "s1 s2 s3 s4"
        assert finish_reasons(chunks) == ["stop"]


def test_provider_answering_a_stream_request_with_plain_json_is_still_delivered_or_fails_cleanly(
    gateway: Gateway,
) -> None:
    """A misbehaving provider ignores `stream: true` and returns one `application/json` chat completion.
    The proxy must not hang: the caller gets a response that either ends with `[DONE]` or is a JSON
    error. UNVERIFIED which of the two; the OpenAI SDK's stream reader yields nothing for a body with no
    `data:` lines, which the clean EOF check turns into an error frame. This pins only termination."""
    from integration.streaming.dec1_stream_support import completion_json

    body: Final = completion_json(f"chatcmpl-{marker()}", "not a stream")
    with scripted(gateway, lambda request: Reply(body=body)) as rig:
        response: Final = rig.chat()
        assert response.status_code >= 400 or payloads(response.text)[-1] == "[DONE]", response.text
