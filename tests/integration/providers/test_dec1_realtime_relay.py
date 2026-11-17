"""Contract: what the proxy does to realtime frames in both directions, and how a session ends.

Client to provider (``RealTimeStreaming.client_ack_messages``, litellm/litellm_core_utils/realtime_streaming.py):
frames are forwarded as sent, except a ``session.update`` to a GA upstream, which is remapped from the flat beta
shape (``modalities``, ``voice``, ``input_audio_format``, ``turn_detection``...) to the GA nested shape and gets
``type: "realtime"`` when absent. A client that sends ``OpenAI-Beta: realtime=v1`` is on the beta protocol: its
``session.update`` is forwarded unchanged and the header is forwarded on the upstream handshake.

Provider to client (``_relay_backend_messages``): JSON events are re-serialized and relayed in order; for a beta
client GA event names are translated back (``conversation.item.added`` -> ``conversation.item.created``,
``response.output_text.delta`` -> ``response.text.delta``); non-JSON text is relayed raw.

Session end (``bidirectional_forward`` / ``_close_client`` and ``realtime_errors.py``): a provider close 1000
is relayed as 1000 with its reason and no error event. Any other provider close sends one ``error`` event of
type ``server_error`` with ``upstream websocket closed with code <c>: <reason>`` and closes the client with the
same code when it is a registered or application code (3000-4999), otherwise 1011. A provider handshake
refusal sends ``Upstream realtime handshake rejected with HTTP <status>`` and closes 1008 for 401/403, 1013 for
429 and 1011 otherwise. Provider frames have no size limit unless ``REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES``
is set (``max_size=None`` on the upstream connection).
"""

from __future__ import annotations

import base64
import json
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.providers.dec1_realtime_support import (
    INTERNAL_ERROR,
    NORMAL_CLOSURE,
    POLICY_VIOLATION,
    TRY_AGAIN_LATER,
    Await,
    AwaitClose,
    Close,
    CloseWith,
    DropTcp,
    Idle,
    PeerScript,
    Reply,
    Send,
    answer_response_create,
    done_event,
    openai_deployment,
    peer_deployment,
    realtime,
    realtime_peer,
    scripted,
    turn_steps,
    upstream_frames,
)

pytestmark: Final = [pytest.mark.timeout(180), pytest.mark.shares_worker_state]

BETA_HEADER: Final = {"OpenAI-Beta": "realtime=v1"}
BETA_SESSION: Final = {
    "modalities": ["text", "audio"],
    "voice": "alloy",
    "input_audio_format": "pcm16",
    "output_audio_format": "g711_ulaw",
    "turn_detection": {"type": "server_vad", "silence_duration_ms": 400},
    "input_audio_transcription": {"model": "whisper-1"},
    "instructions": "be brief",
}
GA_SESSION: Final = {
    "type": "realtime",
    "output_modalities": ["audio"],
    "instructions": "be brief",
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 400},
            "transcription": {"model": "whisper-1"},
        },
        "output": {"voice": "alloy", "format": {"type": "audio/pcmu"}},
    },
}
AUDIO_CHUNK: Final = base64.b64encode(bytes(range(256)) * 4).decode()


def _client_frames(gateway: Gateway, scenario_id: str, frame_type: str) -> list[dict[str, object]]:
    return [frame for frame in upstream_frames(gateway, scenario_id) if frame.get("type") == frame_type]


def test_a_beta_shaped_session_update_reaches_a_ga_upstream_in_the_nested_ga_shape(gateway: Gateway) -> None:
    """A GA client (no ``OpenAI-Beta`` header) sends a flat beta-shaped ``session.update``. The upstream receives
    ``session`` equal to the GA nested shape: ``type`` ``realtime`` injected, ``modalities`` [text, audio] collapsed
    to ``output_modalities`` [audio], formats mapped (``pcm16`` -> ``{"type": "audio/pcm", "rate": 24000}``,
    ``g711_ulaw`` -> ``{"type": "audio/pcmu"}``), voice, turn detection and transcription nested under ``audio``,
    ``instructions`` passed through (``_remap_beta_session_to_ga``)."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(
                Await("session.created"),
                Send({"type": "session.update", "session": BETA_SESSION}),
                Await("session.updated"),
            ),
        )
        updates: Final = _client_frames(gateway, handle.scenario_id, "session.update")
        assert [update["session"] for update in updates] == [GA_SESSION], updates


def test_a_beta_client_session_update_is_forwarded_unchanged(gateway: Gateway) -> None:
    """The same flat ``session.update`` sent by a client that upgraded with ``OpenAI-Beta: realtime=v1`` reaches
    the upstream byte for byte as JSON: ``session`` equals the flat beta object and no ``type`` is injected."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            headers=BETA_HEADER,
            steps=(Await("session.created"), Send({"type": "session.update", "session": BETA_SESSION}), Idle(3)),
        )
        updates: Final = _client_frames(gateway, handle.scenario_id, "session.update")
        assert [update["session"] for update in updates] == [BETA_SESSION], updates


def test_a_ga_shaped_session_update_is_forwarded_with_its_fields_intact(gateway: Gateway) -> None:
    """A GA client sends a session.update already in the GA nested shape (with ``type`` ``realtime``). The remap
    is idempotent: the upstream receives exactly the same ``session`` object."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(
                Await("session.created"),
                Send({"type": "session.update", "session": GA_SESSION}),
                Await("session.updated"),
            ),
        )
        assert [u["session"] for u in _client_frames(gateway, handle.scenario_id, "session.update")] == [GA_SESSION]


def test_the_beta_header_is_forwarded_on_the_upstream_handshake_only_when_the_client_sent_it(gateway: Gateway) -> None:
    """Two sessions on an owned peer, one with ``OpenAI-Beta: realtime=v1`` and one without. The first upstream
    handshake carries ``openai-beta: realtime=v1`` and ``authorization: Bearer dec1-realtime-peer-key``; the
    second carries no ``openai-beta`` header (``OpenAIRealtime._get_additional_headers``)."""
    with realtime_peer() as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        realtime(gateway, f"model={model}", key=gateway.key, headers=BETA_HEADER)
        realtime(gateway, f"model={model}", key=gateway.key)
        handshakes: Final = eventually(peer.drained_handshakes, lambda seen: len(seen) == 2, seconds=15)
        assert handshakes[0].headers.get("openai-beta") == "realtime=v1", handshakes
        assert handshakes[0].headers.get("authorization") == "Bearer dec1-realtime-peer-key", handshakes
        assert "openai-beta" not in handshakes[1].headers, handshakes


def test_audio_append_commit_and_response_create_reach_the_upstream_verbatim(gateway: Gateway) -> None:
    """The client sends ``input_audio_buffer.append`` (base64 PCM), ``input_audio_buffer.commit`` and
    ``response.create`` with a ``response.instructions`` object. The upstream receives the three frames in that
    order with identical JSON bodies; nothing is added or dropped on a session without guardrails."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(), done_event()))
        model: Final = openai_deployment(scenario, handle)
        sent: Final = (
            {"type": "input_audio_buffer.append", "audio": AUDIO_CHUNK, "event_id": "c1"},
            {"type": "input_audio_buffer.commit", "event_id": "c2"},
            {"type": "response.create", "event_id": "c3", "response": {"instructions": "say hi"}},
        )
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), *(Send(frame) for frame in sent), Await("response.done"), Idle(2)),
        )
        frames: Final = upstream_frames(gateway, handle.scenario_id)
        assert list(frames) == list(sent), frames


def test_a_conversation_item_create_reaches_the_upstream_verbatim(gateway: Gateway) -> None:
    """A ``conversation.item.create`` with an ``input_text`` user message is forwarded unchanged to a GA upstream
    (the beta item bridge only runs for Azure beta clients bridged to GA)."""
    item: Final = {
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello there"}]},
    }
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        realtime(gateway, f"model={model}", key=gateway.key, steps=(Await("session.created"), Send(item), Idle(2)))
        assert _client_frames(gateway, handle.scenario_id, "conversation.item.create") == [item]


def test_the_upstream_response_done_reaches_the_client_with_its_id_and_usage(gateway: Gateway) -> None:
    """The scripted upstream answers ``response.create`` with a ``response.done`` carrying a fixed
    ``response.id`` and usage of 10 text in and 5 text out. The client receives one ``response.done`` whose
    ``response.id`` and ``response.usage`` equal what the upstream sent."""
    with gateway.scenario() as scenario:
        event: Final = done_event(response_id="resp_fixed_dec1")
        handle: Final = scripted(scenario, (event,))
        model: Final = openai_deployment(scenario, handle)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key, steps=turn_steps(1))
        done: Final = outcome.of_type("response.done")
        assert [frame["response"]["id"] for frame in done] == ["resp_fixed_dec1"], outcome  # type: ignore[index]
        assert done[0]["response"]["usage"] == event["response"]["usage"], outcome  # type: ignore[index]


def test_the_upstream_session_created_reaches_the_client_first_with_the_upstream_session_id(gateway: Gateway) -> None:
    """The first frame the client receives is the upstream's ``session.created`` with ``session.id``
    ``sess_<scenario id>`` and ``session.model`` ``gpt-4o-realtime-preview`` (the scripted upstream echoes the
    query model): the proxy neither synthesizes nor rewrites the session for an OpenAI upstream."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key)
        first: Final = outcome.frames[0]
        assert first["type"] == "session.created", outcome
        assert first["session"]["id"] == f"sess_{handle.scenario_id}", outcome  # type: ignore[index]
        assert first["session"]["model"] == "gpt-4o-realtime-preview", outcome  # type: ignore[index]


def test_several_upstream_events_for_one_turn_reach_the_client_in_order(gateway: Gateway) -> None:
    """An owned peer answers ``response.create`` with ``response.created``, two ``response.output_text.delta``,
    an unknown ``x.vendor.debug`` event and ``response.done``. A GA client receives exactly those five types in
    that order, unknown types included."""
    events: Final = (
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.output_text.delta", "delta": "a"},
        {"type": "response.output_text.delta", "delta": "b"},
        {"type": "x.vendor.debug", "detail": {"n": 1}},
        done_event(response_id="r1"),
    )

    def on_frame(frame: dict[str, object]) -> tuple[Reply, ...]:
        return tuple(Reply(event) for event in events) if frame.get("type") == "response.create" else ()

    with realtime_peer(PeerScript(on_frame=on_frame)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key, steps=turn_steps(1))
        assert outcome.types == (
            "session.created",
            "response.created",
            "response.output_text.delta",
            "response.output_text.delta",
            "x.vendor.debug",
            "response.done",
        ), outcome


def test_a_beta_client_receives_ga_event_names_translated_back_to_beta(gateway: Gateway) -> None:
    """The owned peer answers with GA names ``conversation.item.added``, ``response.output_text.delta`` and
    ``response.output_audio.delta``. A client that upgraded with ``OpenAI-Beta: realtime=v1`` receives
    ``conversation.item.created``, ``response.text.delta`` and ``response.audio.delta`` instead
    (``_GA_TO_BETA_EVENT_TYPES``); a GA client on the same peer receives the GA names."""
    events: Final = (
        {
            "type": "conversation.item.added",
            "item": {"id": "i1", "type": "message", "role": "assistant", "content": []},
        },
        {"type": "response.output_text.delta", "delta": "x"},
        {"type": "response.output_audio.delta", "delta": "AAAA"},
        done_event(response_id="r2"),
    )

    def on_frame(frame: dict[str, object]) -> tuple[Reply, ...]:
        return tuple(Reply(event) for event in events) if frame.get("type") == "response.create" else ()

    with realtime_peer(PeerScript(on_frame=on_frame)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        beta: Final = realtime(gateway, f"model={model}", key=gateway.key, headers=BETA_HEADER, steps=turn_steps(1))
        ga: Final = realtime(gateway, f"model={model}", key=gateway.key, steps=turn_steps(1))
        assert beta.types[1:4] == ("conversation.item.created", "response.text.delta", "response.audio.delta"), beta
        assert ga.types[1:4] == (
            "conversation.item.added",
            "response.output_text.delta",
            "response.output_audio.delta",
        ), ga


def test_a_non_json_upstream_text_frame_is_relayed_raw(gateway: Gateway) -> None:
    """The owned peer answers ``response.create`` with the text frame ``plain-text-from-provider`` and then a
    ``response.done``. The client receives that exact text (the relay forwards unparseable frames as is) before
    the ``response.done``."""

    def on_frame(frame: dict[str, object]) -> tuple[Reply, ...]:
        if frame.get("type") != "response.create":
            return ()
        return (Reply("plain-text-from-provider"), Reply(done_event()))

    with realtime_peer(PeerScript(on_frame=on_frame)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key, steps=turn_steps(1))
        assert "plain-text-from-provider" in outcome.raw_texts, outcome
        assert outcome.types[-1] == "response.done", outcome


def test_a_non_json_client_text_frame_is_forwarded_to_the_upstream_raw(gateway: Gateway) -> None:
    """The client sends the text frame ``hello-not-json`` after ``session.created``. The JSON decode failure is
    ignored and the text reaches the owned peer unchanged; the session stays open and a following turn
    completes."""
    with (
        realtime_peer(PeerScript(on_frame=answer_response_create(done_event()))) as peer,
        gateway.scenario() as scenario,
    ):
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(
                Await("session.created"),
                Send("hello-not-json"),
                Send({"type": "response.create"}),
                Await("response.done"),
            ),
        )
        assert outcome.types[-1] == "response.done", outcome
        frames: Final = eventually(peer.drained_frames, lambda seen: len(seen) >= 2, seconds=10)
        assert frames[0] == {"type": "__non_json__", "text": "hello-not-json", "binary": False}, frames


def test_a_binary_client_frame_ends_the_session(gateway: Gateway) -> None:
    """The client sends a binary frame. The client loop reads with ``receive_text``, the read fails and is
    treated as a client disconnect (``_receive_client_message`` returns None), so the session ends: the client
    sees the socket close and the peer sees its socket closed. UNVERIFIED: the close code the client observes
    (uvicorn closes an app-finished socket with 1000); the binary payload never reaches the peer."""
    with realtime_peer() as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), Send(b"\x00\x01binary"), AwaitClose()),
        )
        assert outcome.close_code is not None, outcome
        assert all(not frame.get("binary") for frame in peer.drained_frames()), "binary frame reached the provider"


def test_a_two_mebibyte_audio_append_reaches_the_upstream_intact(gateway: Gateway) -> None:
    """The client sends one ``input_audio_buffer.append`` frame of about 2 MiB of base64 audio. It is below
    uvicorn's 16 MiB websocket limit, the provider connection has no size cap, and the owned peer receives the
    same ``audio`` string (equal length and content)."""
    big_audio: Final = base64.b64encode(b"\x11" * (1536 * 1024)).decode()
    with realtime_peer() as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), Send({"type": "input_audio_buffer.append", "audio": big_audio}), Idle(3)),
        )
        frames: Final = eventually(peer.drained_frames, lambda seen: len(seen) == 1, seconds=15)
        assert frames[0]["audio"] == big_audio


def test_a_two_mebibyte_upstream_frame_reaches_the_client(gateway: Gateway) -> None:
    """The owned peer answers ``response.create`` with a ``response.output_audio.delta`` whose ``delta`` is about
    2 MiB, then ``response.done``. With ``REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES`` unset the proxy's upstream
    connection has ``max_size=None`` and the client receives the full delta. A port that adds websockets'
    default 1 MiB cap would close the provider socket with 1009 instead."""
    delta: Final = "A" * (2 * 1024 * 1024)

    def on_frame(frame: dict[str, object]) -> tuple[Reply, ...]:
        if frame.get("type") != "response.create":
            return ()
        return (Reply({"type": "response.output_audio.delta", "delta": delta}), Reply(done_event()))

    with realtime_peer(PeerScript(on_frame=on_frame)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key, steps=turn_steps(1))
        deltas: Final = outcome.of_type("response.output_audio.delta")
        assert [len(str(frame["delta"])) for frame in deltas] == [len(delta)], outcome.types


def test_an_upstream_close_1000_is_relayed_as_1000_with_its_reason_and_no_error_event(gateway: Gateway) -> None:
    """The owned peer answers ``response.create`` with ``response.done`` and then closes with 1000
    ``session finished``. The client gets the ``response.done``, no ``error`` event, and a close 1000 with
    reason ``session finished`` (``BackendClose.ended_cleanly``)."""

    def on_frame(frame: dict[str, object]) -> tuple[Reply | Close, ...]:
        if frame.get("type") != "response.create":
            return ()
        return (Reply(done_event()), Close(1000, "session finished"))

    with realtime_peer(PeerScript(on_frame=on_frame)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key, steps=(*turn_steps(1), AwaitClose()))
        assert outcome.of_type("error") == (), outcome
        assert (outcome.close_code, outcome.close_reason) == (NORMAL_CLOSURE, "session finished"), outcome


@pytest.mark.parametrize(
    ("upstream_code", "client_code"),
    ((4001, 4001), (3000, 3000), (1001, 1001), (1008, 1008), (1011, 1011), (1012, 1012)),
)
def test_an_upstream_close_with_a_relayable_code_reaches_the_client_with_the_same_code_and_an_error_event(
    gateway: Gateway, upstream_code: int, client_code: int
) -> None:
    """After ``session.created`` the owned peer closes with ``<code>`` and reason ``scripted``. The client
    receives one ``error`` event of type ``server_error`` whose message is ``upstream websocket closed with code
    <code>: scripted`` and then a close with the same code and reason ``scripted`` (``client_close_code`` keeps
    registered codes and 3000-4999)."""

    def on_frame(frame: dict[str, object]) -> tuple[Close, ...]:
        return (Close(upstream_code, "scripted"),) if frame.get("type") == "response.create" else ()

    with realtime_peer(PeerScript(on_frame=on_frame)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), Send({"type": "response.create"}), AwaitClose()),
        )
        assert outcome.error_types() == ("server_error",), outcome
        assert outcome.error_messages() == (f"upstream websocket closed with code {upstream_code}: scripted",), outcome
        assert (outcome.close_code, outcome.close_reason) == (client_code, "scripted"), outcome


def test_an_upstream_tcp_drop_is_reported_and_closed_with_1011(gateway: Gateway) -> None:
    """The owned peer aborts its TCP connection without a close frame after ``session.created``. The proxy sees
    close code 1006 from the provider (``backend_close_from``); 1006 may not be sent on the wire, so the client
    receives one ``server_error`` event whose message starts with ``upstream websocket closed with code 1006``
    and a close 1011."""

    def on_frame(frame: dict[str, object]) -> tuple[DropTcp, ...]:
        return (DropTcp(),) if frame.get("type") == "response.create" else ()

    with realtime_peer(PeerScript(on_frame=on_frame)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), Send({"type": "response.create"}), AwaitClose()),
        )
        assert outcome.close_code == INTERNAL_ERROR, outcome
        assert len(outcome.error_messages()) == 1, outcome
        assert outcome.error_messages()[0].startswith("upstream websocket closed with code 1006"), outcome


@pytest.mark.parametrize(
    ("status", "client_code"),
    (
        (401, POLICY_VIOLATION),
        (403, POLICY_VIOLATION),
        (429, TRY_AGAIN_LATER),
        (500, INTERNAL_ERROR),
        (404, INTERNAL_ERROR),
    ),
)
def test_an_upstream_handshake_refusal_maps_its_http_status_to_a_close_code(
    gateway: Gateway, status: int, client_code: int
) -> None:
    """The owned peer answers the provider handshake with HTTP ``<status>``. The client, already accepted by the
    proxy, receives one ``error`` event ``{"type": "server_error", "message": "Upstream realtime handshake rejected
    with HTTP <status>"}`` and a close with ``<client code>`` and the same message as reason
    (``close_after_upstream_handshake_refusal``)."""
    with realtime_peer(PeerScript(refuse_status=status)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key, steps=(AwaitClose(),))
        message: Final = f"Upstream realtime handshake rejected with HTTP {status}"
        assert outcome.handshake_status is None, outcome
        assert outcome.error_types() == ("server_error",), outcome
        assert outcome.error_messages() == (message,), outcome
        assert (outcome.close_code, outcome.close_reason) == (client_code, message), outcome


def test_a_client_close_closes_the_provider_socket(gateway: Gateway) -> None:
    """After one completed turn the client closes with 1000. The proxy leaves the provider ``async with`` block
    and the owned peer observes its socket closed by the proxy with code 1000, so no provider session is left
    running (and billing) after the client is gone."""
    with (
        realtime_peer(PeerScript(on_frame=answer_response_create(done_event()))) as peer,
        gateway.scenario() as scenario,
    ):
        model: Final = peer_deployment(scenario, peer)
        realtime(gateway, f"model={model}", key=gateway.key, steps=(*turn_steps(1), CloseWith(1000)))
        close: Final = peer.closes.get(timeout=15)
        assert close[0] == 1000, close


def test_an_upstream_error_event_mid_session_is_relayed_and_the_session_stays_open(gateway: Gateway) -> None:
    """The owned peer answers the first ``response.create`` with an ``error`` event
    (``invalid_request_error``, code ``bad_thing``) and the second with ``response.done``. The client receives
    the error event unchanged, the socket stays open, and the second turn completes."""
    replies: Final = iter(
        (
            {"type": "error", "error": {"type": "invalid_request_error", "code": "bad_thing", "message": "nope"}},
            done_event(),
        )
    )

    def on_frame(frame: dict[str, object]) -> tuple[Reply, ...]:
        return (Reply(next(replies)),) if frame.get("type") == "response.create" else ()

    with realtime_peer(PeerScript(on_frame=on_frame)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(
                Await("session.created"),
                Send({"type": "response.create"}),
                Await("error"),
                Send({"type": "response.create"}),
                Await("response.done"),
            ),
        )
        assert outcome.types == ("session.created", "error", "response.done"), outcome
        assert outcome.frames[1]["error"] == {  # type: ignore[comparison-overlap]
            "type": "invalid_request_error",
            "code": "bad_thing",
            "message": "nope",
        }, outcome


def test_an_idle_session_survives_past_the_server_ping_interval_and_serves_the_next_turn(gateway: Gateway) -> None:
    """After ``session.created`` the client stays silent for 25 seconds (longer than uvicorn's default 20 second
    ping interval; the client library answers pings). The proxy has no realtime idle timeout of its own, so the
    socket is still open and a ``response.create`` then gets its ``response.done``.
    UNVERIFIED: the proxy's uvicorn ping settings in CI."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), Idle(25), Send({"type": "response.create"}), Await("response.done")),
            timeout=40,
        )
        assert outcome.types == ("session.created", "response.done"), outcome


def test_a_response_cancel_reaches_the_upstream_and_its_error_reaches_the_client(gateway: Gateway) -> None:
    """The client sends ``response.cancel`` with ``event_id`` ``cancel-1`` when no response is active. The
    scripted upstream receives it and answers an ``error`` with ``code`` ``response_cancel_not_active`` and
    ``event_id`` ``cancel-1``; that error reaches the client unchanged because the proxy did not originate the
    cancel (``_is_guardrail_cancel_error`` only swallows its own)."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), Send({"type": "response.cancel", "event_id": "cancel-1"}), Await("error")),
        )
        error: Final = outcome.of_type("error")[0]["error"]
        assert error["code"] == "response_cancel_not_active", outcome  # type: ignore[index]
        assert error["event_id"] == "cancel-1", outcome  # type: ignore[index]
        assert _client_frames(gateway, handle.scenario_id, "response.cancel") == [
            {"type": "response.cancel", "event_id": "cancel-1"}
        ]


def test_an_upstream_session_updated_reply_reaches_the_client_once(gateway: Gateway) -> None:
    """A GA client's ``session.update`` is answered by the scripted upstream with ``session.updated`` merged from
    the GA session. The client receives exactly one ``session.updated`` and its ``session.type`` is ``realtime``
    (the injected type is what the upstream echoed)."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(
                Await("session.created"),
                Send({"type": "session.update", "session": {"instructions": "hi"}}),
                Await("session.updated"),
                Idle(2),
            ),
        )
        updated: Final = outcome.of_type("session.updated")
        assert len(updated) == 1, outcome
        assert updated[0]["session"]["type"] == "realtime", outcome  # type: ignore[index]
        assert json.dumps(updated[0]).count('"instructions": "hi"') == 1, outcome
