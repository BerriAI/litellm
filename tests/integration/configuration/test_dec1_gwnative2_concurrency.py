"""Concurrent calls and callers that hang up, on the native Rust gateway preview.

The contract, read from `litellm/rust/crates/gateway/src/main.rs` (one `axum::serve` listener on a tokio
runtime), `inference-messages/src/handler.rs` (`streaming_response`, `draining_after_detach`) and
`host-http/src/driver.rs` (`serve`, the body is a lazily polled stream):

Every call is independent: concurrent unary calls and concurrent streams each get exactly their own provider
answer, and a stream that is held open does not delay other calls. An Anthropic stream has no stream decoder,
so it also has no detached drain (`decoder.and(detached_drain)` is None): when the caller hangs up, the body
stream is dropped and the provider connection is closed instead of being read to the end. The Python proxy's
Anthropic native path also stops reading; only its Bedrock path drains in the background for billing, which is
why that case is not pinned here.

One gateway and one scripted provider serve the module.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Final

import httpx
import pytest

from tests.integration._support.client import eventually
from tests.integration._support.wire import Reply
from tests.integration.configuration.dec1_gwnative2_support import (
    disconnected_targets,
    first_block,
    in_threads,
    open_stream,
)
from tests.integration.configuration.dec1_gwnative_support import (
    GATEWAY_KEY,
    NativeGateway,
    ScriptedProvider,
    anthropic_deployment,
    anthropic_reply,
    anthropic_stream,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    response_json,
    scripted_provider,
    sse_events,
)


@pytest.fixture(scope="module")
def provider() -> Iterator[ScriptedProvider]:
    with scripted_provider() as scripted:
        yield scripted


@pytest.fixture(scope="module")
def gateway(provider: ScriptedProvider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[NativeGateway]:
    config: Final = master_key_config([anthropic_deployment("claude", provider.url)])
    with native_gateway(config, tmp_path_factory.mktemp("gwnative2-concurrency")) as booted:
        yield booted


def test_concurrent_unary_calls_each_get_their_own_answer(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """Sixteen messages calls run at once, each with its own marker, and the provider echoes the marker back in
    the text. Every caller must get 200 with its own marker in `content[0].text`, and the provider must see
    exactly one request per marker (no shared state between calls, no retries)."""
    cases: Final = tuple(marker() for _ in range(16))
    for case in cases:
        provider.script(case, lambda _, text=case: anthropic_reply(f"echo {text}"))
    responses: Final = in_threads(
        tuple(lambda case=case: gateway.post("/v1/messages", messages_body(case)) for case in cases)
    )
    for case, response in zip(cases, responses, strict=True):
        assert response.status_code == 200, response.text
        content: Final = response_json(response)["content"]
        assert content == [{"type": "text", "text": f"echo {case}"}]
        assert len(provider.requests_for(case)) == 1


def test_two_open_streams_do_not_mix_their_events(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """Two streaming calls are held open after their first event, then the second is released before the first.
    Each caller's `content_block_delta` text must be its own marker only, and the second must finish while the
    first is still held."""
    first_case: Final = marker()
    second_case: Final = marker()
    first_gate: Final = threading.Event()
    second_gate: Final = threading.Event()
    provider.script(first_case, lambda _: anthropic_stream(first_case, gate_after_first=first_gate))
    provider.script(second_case, lambda _: anthropic_stream(second_case, gate_after_first=second_gate))
    first_client, first_response = open_stream(gateway, "/v1/messages", messages_body(first_case, stream=True))
    second_client, second_response = open_stream(gateway, "/v1/messages", messages_body(second_case, stream=True))
    with first_client, second_client, first_response, second_response:
        first_head: Final = first_block(first_response)
        second_head: Final = first_block(second_response)
        second_gate.set()
        second_rest: Final = second_response.read().decode()
        assert not first_gate.is_set()
        first_gate.set()
        first_rest: Final = first_response.read().decode()
    first_events: Final = dict(sse_events(first_head + first_rest))
    second_events: Final = dict(sse_events(second_head + second_rest))
    assert first_events["content_block_delta"]["delta"] == {"type": "text_delta", "text": first_case}
    assert second_events["content_block_delta"]["delta"] == {"type": "text_delta", "text": second_case}


def test_a_held_stream_does_not_delay_a_unary_call(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """A stream is held open after its first event while another caller makes a plain messages call. The plain
    call must complete with 200 while the stream is still held."""
    held: Final = marker()
    gate: Final = threading.Event()
    provider.script(held, lambda _: anthropic_stream(held, gate_after_first=gate))
    client, response = open_stream(gateway, "/v1/messages", messages_body(held, stream=True))
    with client, response:
        first_block(response)
        unary: Final = gateway.post("/v1/messages", messages_body(marker()))
        assert unary.status_code == 200, unary.text
        assert not gate.is_set()
        gate.set()
        response.read()


def test_a_caller_hanging_up_mid_stream_closes_the_provider_stream(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """A streaming Anthropic call is held after its first event; the caller reads that event and closes its
    connection, then the provider is released and keeps writing paced events. The provider's writes must fail
    (its target appears among the peer's disconnected targets), because without a decoder there is no detached
    drain and the dropped body drops the provider response (`draining_after_detach`, `handler.rs:1153`)."""
    case: Final = marker()
    gate: Final = threading.Event()
    provider.script(case, lambda _: anthropic_stream(case, gate_after_first=gate, pause=0.2))
    client, response = open_stream(gateway, "/v1/messages", messages_body(case, stream=True))
    with client, response:
        assert first_block(response).startswith("event: message_start")
    gate.set()
    gone: Final = eventually(lambda: disconnected_targets(provider.wire), lambda targets: len(targets) > 0, seconds=15)
    assert "/v1/messages" in gone


def test_a_caller_hanging_up_during_a_slow_unary_call_abandons_the_provider_call(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """A non-streaming call whose provider holds its chunked JSON answer before the first byte. The caller times
    out after one second and hangs up; then the provider is released. The provider's write must fail (target in
    the peer's disconnected targets), since dropping the request future drops the provider request.
    UNVERIFIED: hyper notices the closed client connection only when it next polls the connection."""
    case: Final = marker()
    gate: Final = threading.Event()
    body: Final = b'{"id": "msg_dec1_slow", "type": "message", "role": "assistant", "content": []}'
    provider.script(
        case,
        lambda _: Reply(chunks=(body[:20], body[20:]), gate_before_first=gate, pause_between_chunks=0.5),
    )
    with gateway.client(timeout=1) as client, pytest.raises(httpx.ReadTimeout):
        client.post("/v1/messages", json=messages_body(case), headers={"Authorization": f"Bearer {GATEWAY_KEY}"})
    gate.set()
    gone: Final = eventually(lambda: disconnected_targets(provider.wire), lambda targets: len(targets) > 0, seconds=15)
    assert "/v1/messages" in gone


def test_many_streams_at_once_all_end_with_message_stop(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """Eight streaming calls run at once against paced provider streams. Every caller must get 200 and an event
    list that ends with `message_stop` carrying its own delta text; concurrent streams share no buffers."""
    cases: Final = tuple(marker() for _ in range(8))
    for case in cases:
        provider.script(case, lambda _, text=case: anthropic_stream(text, pause=0.05))
    responses: Final = in_threads(
        tuple(lambda case=case: gateway.post("/v1/messages", messages_body(case, stream=True)) for case in cases)
    )
    for case, response in zip(cases, responses, strict=True):
        assert response.status_code == 200, response.text
        events: Final = sse_events(response.text)
        assert events[-1][0] == "message_stop"
        assert dict(events)["content_block_delta"]["delta"] == {"type": "text_delta", "text": case}
