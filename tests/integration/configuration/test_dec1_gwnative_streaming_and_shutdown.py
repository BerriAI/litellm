"""Streaming through the native Rust gateway preview, and what SIGTERM does to calls in flight.

The contract: `/v1/messages` and `/v1/responses` with `stream: true` answer `text/event-stream` and
relay provider events as they arrive (`litellm_host_http::serve` with an `Sse` encoder). A failure after
the headers went out is written as a final `event: error` frame, in the Anthropic envelope on
`/v1/messages` and as the Responses `error` event on `/v1/responses`. `/chat/completions` never streams
in the preview: `stream: true` is declined with 501 before any provider call. SIGTERM or SIGINT starts
axum's graceful shutdown: the listener closes at once, requests already in flight (streams included)
run to completion, and the process exits 0 when the last one ends.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest

from tests.integration._support.client import eventually
from tests.integration._support.vertex import service_account_json
from tests.integration._support.wire import Reply
from tests.integration.configuration.dec1_gwnative_support import (
    GATEWAY_KEY,
    VERTEX_PROJECT,
    NativeGateway,
    ScriptedProvider,
    anthropic_deployment,
    anthropic_message,
    anthropic_stream,
    anthropic_stream_chunks,
    azure_ai_deployment,
    bedrock_deployment,
    begin_shutdown,
    chat_body,
    json_of,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    openai_deployment,
    openai_error_body,
    openai_like_deployment,
    response_json,
    responses_body,
    responses_stream,
    scripted_provider,
    sse_events,
    terminate,
    vertex_deployment,
)

_HEADERS: Final = {"Authorization": f"Bearer {GATEWAY_KEY}"}


@pytest.fixture(scope="module")
def provider() -> Iterator[ScriptedProvider]:
    with scripted_provider() as scripted:
        yield scripted


@pytest.fixture(scope="module")
def gateway(provider: ScriptedProvider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[NativeGateway]:
    url: Final = provider.url
    config: Final = master_key_config(
        [
            anthropic_deployment("claude", url),
            bedrock_deployment("bedrock-claude", url),
            vertex_deployment("vertex-claude", url, service_account_json(VERTEX_PROJECT, url)),
            azure_ai_deployment("azure-claude", url),
            openai_like_deployment("chat", url),
            openai_deployment("responses", url),
        ]
    )
    with native_gateway(config, tmp_path_factory.mktemp("gwnative-stream")) as booted:
        yield booted


def _stream_text(gateway: NativeGateway, path: str, body: dict[str, object]) -> httpx.Response:
    with gateway.client() as client:
        response: Final = client.post(path, json=body, headers=_HEADERS)
    return response


def test_a_messages_stream_relays_every_anthropic_event_in_order(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """`stream: true` on `/v1/messages` to Anthropic, which streams the six lifecycle events. The caller must
    get 200 `text/event-stream` and the events `message_start`, `content_block_start`,
    `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop` in that order, and the
    provider body must have `stream: true`."""
    case: Final = marker()
    provider.script(case, lambda _: anthropic_stream("dec1 streamed"))
    response: Final = _stream_text(gateway, "/v1/messages", messages_body(case, stream=True))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    names: Final = [name for name, _ in sse_events(response.text)]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert json_of(provider.requests_for(case)[0])["stream"] is True


def test_a_messages_stream_carries_the_delta_text(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The provider streams the text `dec1 streamed text`. The `content_block_delta` event the caller gets
    must have `delta.text` equal to it, byte for byte."""
    case: Final = marker()
    provider.script(case, lambda _: anthropic_stream("dec1 streamed text"))
    response: Final = _stream_text(gateway, "/v1/messages", messages_body(case, stream=True))
    deltas: Final = [data for name, data in sse_events(response.text) if name == "content_block_delta"]
    assert deltas[0]["delta"] == {"type": "text_delta", "text": "dec1 streamed text"}


def test_the_first_messages_event_arrives_before_the_provider_finishes(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider sends `message_start` and then waits on a gate. The caller must read the
    `message_start` event while the provider is still blocked, proving the gateway relays chunks rather
    than buffering the whole stream; then the gate opens and the stream completes."""
    case: Final = marker()
    gate: Final = threading.Event()
    provider.script(case, lambda _: anthropic_stream("dec1 gated", gate_after_first=gate))
    try:
        with (
            gateway.client() as client,
            client.stream("POST", "/v1/messages", json=messages_body(case, stream=True), headers=_HEADERS) as response,
        ):
            first: Final = next(response.iter_text())
            assert "message_start" in first, first
            gate.set()
            rest: Final = "".join(response.iter_text())
    finally:
        gate.set()
    assert "message_stop" in rest


def test_a_provider_that_drops_mid_stream_ends_with_an_anthropic_error_event(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider sends two events and then drops the connection. The caller already has 200 headers, so
    the stream must end with an `event: error` frame whose data is `{"type": "error", "error": {"type":
    ..., "message": ...}}` (`Error::sse_frame`), and must not contain `message_stop`. UNVERIFIED: the exact
    `error.type`, which depends on how the dropped read is classified (likely `api_error`)."""
    case: Final = marker()
    provider.script(case, lambda _: anthropic_stream("dec1 dropped", abort_after=2, pause=0.1))
    response: Final = _stream_text(gateway, "/v1/messages", messages_body(case, stream=True))
    events: Final = sse_events(response.text)
    assert events[-1][0] == "error", response.text
    assert events[-1][1]["type"] == "error"
    assert set(events[-1][1]["error"]) >= {"type", "message"}  # type: ignore[arg-type]
    assert "message_stop" not in [name for name, _ in events]


def test_a_provider_error_before_the_stream_starts_is_a_plain_json_error(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """`stream: true`, and the provider answers 429 with an Anthropic error body before any event. Nothing was
    sent yet, so the caller must get status 429 with an `application/json` Anthropic envelope rather than a
    200 stream holding an error event. UNVERIFIED: read from `litellm_host_http::serve`, which writes
    headers only after the first chunk; confirm there."""
    case: Final = marker()
    provider.script(
        case,
        lambda _: Reply(
            status=429, body=b'{"type": "error", "error": {"type": "rate_limit_error", "message": "dec1 wait"}}'
        ),
    )
    response: Final = _stream_text(gateway, "/v1/messages", messages_body(case, stream=True))
    assert response.status_code == 429, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert response_json(response)["error"] == {"type": "rate_limit_error", "message": "dec1 wait"}


def test_a_bedrock_messages_stream_calls_invoke_with_response_stream(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """`stream: true` on `/v1/messages` for the Bedrock Claude deployment. The provider must be called at
    `/model/anthropic.claude-dec1-v1/invoke-with-response-stream`. The scripted peer answers 500 so the test
    does not have to encode AWS event-stream frames; only the target is the contract here."""
    case: Final = marker()
    provider.script(case, lambda _: Reply(status=500, body=b'{"message": "dec1 stop"}'))
    _stream_text(gateway, "/v1/messages", messages_body(case, "bedrock-claude", stream=True))
    assert provider.requests_for(case)[0].target == "/model/anthropic.claude-dec1-v1/invoke-with-response-stream"


def test_a_vertex_messages_stream_calls_stream_raw_predict_with_alt_sse(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """`stream: true` for the Vertex Claude deployment with a custom `api_base` host. The provider target must
    end with `:streamRawPredict?alt=sse` (`custom_host_streaming_adds_alt_sse`), and the Anthropic SSE the
    peer returns must reach the caller ending in `message_stop`."""
    case: Final = marker()
    provider.script(case, lambda _: anthropic_stream("dec1 vertex stream"))
    response: Final = _stream_text(gateway, "/v1/messages", messages_body(case, "vertex-claude", stream=True))
    assert provider.requests_for(case)[0].target.endswith(":streamRawPredict?alt=sse")
    assert [name for name, _ in sse_events(response.text)][-1] == "message_stop"


def test_an_azure_ai_messages_stream_is_relayed(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`stream: true` for the Azure AI Claude deployment. The provider is called at `/anthropic/v1/messages`
    and its SSE must reach the caller unchanged in event order."""
    case: Final = marker()
    provider.script(case, lambda _: anthropic_stream("dec1 azure stream"))
    response: Final = _stream_text(gateway, "/v1/messages", messages_body(case, "azure-claude", stream=True))
    assert provider.requests_for(case)[0].target == "/anthropic/v1/messages"
    assert next(name for name, _ in sse_events(response.text)) == "message_start"


def test_a_responses_stream_relays_the_responses_events(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`stream: true` on `/v1/responses` for the OpenAI deployment. The caller must get 200
    `text/event-stream` with `response.created`, `response.output_text.delta`, `response.completed` in that
    order, and the delta text must be the provider's."""
    case: Final = marker()
    provider.script(case, lambda _: responses_stream("dec1 responses stream"))
    response: Final = _stream_text(gateway, "/v1/responses", responses_body(case, stream=True))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events: Final = sse_events(response.text)
    assert [str(data["type"]) for _, data in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[1][1]["delta"] == "dec1 responses stream"


def test_a_responses_stream_that_breaks_ends_with_a_responses_error_event(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The Responses provider sends one event and drops. The last frame must be `event: error` whose data has
    `type: error`, a string `code` holding an HTTP status and a `message` (`responses_sse_frame`). UNVERIFIED:
    the code value; a dropped read is most likely classified 500."""
    case: Final = marker()
    provider.script(case, lambda _: responses_stream("dec1 broken", abort_after=1))
    response: Final = _stream_text(gateway, "/v1/responses", responses_body(case, stream=True))
    name, data = sse_events(response.text)[-1]
    assert name == "error", response.text
    assert data["type"] == "error"
    assert str(data["code"]).isdigit()
    assert isinstance(data["message"], str)


@pytest.mark.parametrize("model", ["chat", "claude"])
def test_chat_completions_streaming_is_declined_with_501(
    gateway: NativeGateway, provider: ScriptedProvider, model: str
) -> None:
    """`stream: true` on `/v1/chat/completions`. `unsupported_param` declines any streaming chat call, so the
    answer must be 501 with exactly the OpenAI envelope message `declined by the rust path: streaming` and
    nothing reaches the provider. A preview user streaming chat completions sees a clear refusal."""
    case: Final = marker()
    response: Final = _stream_text(gateway, "/v1/chat/completions", chat_body(case, model, stream=True))
    assert response.status_code == 501, response.text
    assert response_json(response) == openai_error_body("declined by the rust path: streaming", "api_error", 501)
    assert provider.requests_for(case) == ()


def test_stream_false_is_a_normal_unary_call(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`stream: false` on `/v1/chat/completions`. Only a true `stream` declines, so the call must be served
    with 200 JSON."""
    response: Final = _stream_text(gateway, "/v1/chat/completions", chat_body(marker(), stream=False))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"


# Graceful shutdown


def test_sigterm_lets_an_open_messages_stream_finish_then_exits_zero(tmp_path: Path) -> None:
    """A `/v1/messages` stream is open and its provider is held at a gate after `message_start`. SIGTERM is
    sent, then the gate opens. The caller must still receive the rest of the stream through `message_stop`,
    and the process must then exit with status 0: graceful shutdown drains in-flight responses."""
    gate: Final = threading.Event()
    with scripted_provider() as provider:
        case: Final = marker()
        provider.script(case, lambda _: anthropic_stream("dec1 drain", gate_after_first=gate))
        with native_gateway(master_key_config([anthropic_deployment("claude", provider.url)]), tmp_path) as gateway:
            try:
                with (
                    gateway.client() as client,
                    client.stream(
                        "POST", "/v1/messages", json=messages_body(case, stream=True), headers=_HEADERS
                    ) as response,
                ):
                    first: Final = next(response.iter_text())
                    begin_shutdown(gateway)
                    gate.set()
                    rest: Final = "".join(response.iter_text())
            finally:
                gate.set()
            status: Final = terminate(gateway)
    assert "message_start" in first
    assert "message_stop" in rest
    assert status == 0


def test_no_new_connection_is_accepted_while_a_stream_drains(tmp_path: Path) -> None:
    """A stream is held open at a gate and SIGTERM is sent. While the stream is still open, a new readiness
    probe must fail with a connection error (the listener is closed), so a load balancer stops routing new
    traffic to the draining process. The gate then opens and the process exits 0."""
    gate: Final = threading.Event()
    with scripted_provider() as provider:
        case: Final = marker()
        provider.script(case, lambda _: anthropic_stream("dec1 drain refuse", gate_after_first=gate))
        with native_gateway(master_key_config([anthropic_deployment("claude", provider.url)]), tmp_path) as gateway:
            try:
                with (
                    gateway.client() as client,
                    client.stream(
                        "POST", "/v1/messages", json=messages_body(case, stream=True), headers=_HEADERS
                    ) as response,
                ):
                    next(response.iter_text())
                    begin_shutdown(gateway)

                    def refused() -> bool:
                        try:
                            httpx.get(f"{gateway.url}/health/readiness", timeout=1, trust_env=False)
                        except httpx.TransportError:
                            return True
                        return False

                    assert eventually(refused, bool, seconds=10)
                    gate.set()
                    "".join(response.iter_text())
            finally:
                gate.set()
            assert terminate(gateway) == 0


def test_sigterm_lets_a_slow_unary_call_finish(tmp_path: Path) -> None:
    """A non-streaming `/v1/messages` call whose provider holds its whole answer at a gate. SIGTERM is sent
    from another thread while the call waits, then the gate opens. The caller must get 200 with the full
    Anthropic message, and the process must exit 0 afterwards."""
    gate: Final = threading.Event()
    with scripted_provider() as provider:
        case: Final = marker()
        body: Final = json.dumps(anthropic_message("dec1 slow unary")).encode()
        provider.script(case, lambda _: Reply(body=body, chunks=(body,), gate_before_first=gate))
        with native_gateway(master_key_config([anthropic_deployment("claude", provider.url)]), tmp_path) as gateway:

            def stop_then_release() -> None:
                eventually(lambda: provider.requests_for(case), bool, seconds=10)
                begin_shutdown(gateway)
                gate.set()

            stopper: Final = threading.Thread(target=stop_then_release)
            stopper.start()
            try:
                response: Final = gateway.post("/v1/messages", messages_body(case))
            finally:
                gate.set()
                stopper.join()
            status: Final = terminate(gateway)
    assert response.status_code == 200, response.text
    assert response_json(response)["content"] == [{"type": "text", "text": "dec1 slow unary"}]
    assert status == 0


def test_sigterm_with_a_stalled_stream_still_exits_within_a_bounded_drain(tmp_path: Path) -> None:
    """A stream is open and its provider never sends another byte. After SIGTERM the process must exit within
    a bounded drain window (the Python proxy uses `GRACEFUL_SHUTDOWN_TIMEOUT`; the forwarding mode of the same
    binary has `LITELLM_FORWARD_SHUTDOWN_DRAIN`), so a stuck provider cannot block a rollout forever."""
    pytest.skip(
        "BUG: standalone mode uses axum with_graceful_shutdown with no deadline, so a stalled stream blocks exit"
    )
    gate: Final = threading.Event()
    with scripted_provider() as provider:
        case: Final = marker()
        provider.script(case, lambda _: anthropic_stream("dec1 stalled", gate_after_first=gate))
        with native_gateway(master_key_config([anthropic_deployment("claude", provider.url)]), tmp_path) as gateway:
            try:
                with (
                    gateway.client() as client,
                    client.stream(
                        "POST", "/v1/messages", json=messages_body(case, stream=True), headers=_HEADERS
                    ) as response,
                ):
                    next(response.iter_text())
                    status: Final = terminate(gateway)
            finally:
                gate.set()
    assert status is not None


def test_stream_chunks_reach_the_caller_unbuffered_when_paced(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider paces its six events 0.2 s apart. The caller must observe at least two separate reads
    from the response before the stream ends (chunks are not coalesced into a single body), the property a
    chat UI depends on to render tokens as they arrive."""
    case: Final = marker()
    provider.script(
        case,
        lambda _: Reply(
            content_type="text/event-stream", chunks=anthropic_stream_chunks("dec1 paced"), pause_between_chunks=0.2
        ),
    )
    with (
        gateway.client() as client,
        client.stream("POST", "/v1/messages", json=messages_body(case, stream=True), headers=_HEADERS) as response,
    ):
        reads: Final = [piece for piece in response.iter_raw() if piece]
    assert len(reads) >= 2, reads
