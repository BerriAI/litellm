"""Auto-mode safeguards on native /v1/messages providers: the backend's own reply bytes are kept, only the final
``message_delta`` (or the top-level object) gains ``safeguard_results``, and Claude deployments stay untouched."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import pytest
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    CLAUDE_MODEL,
    CLAUDE_TARGET,
    NATIVE_MODEL,
    NATIVE_TARGET,
    OTHER_NATIVE_MODELS,
    SAFEGUARDS,
    Rig,
    auto_mode_body,
    flagged,
    ids_under_review,
    json_body,
    native_backend,
    native_message,
    native_stream_frames,
    nonce,
    one_classifier_call,
    request_json,
    stream_results,
    stream_tool_use_ids,
    text_block,
    tool_id,
    tool_use_block,
    tool_use_ids,
    unavailable,
)
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(graceful_stop_seconds() + 240)

_FRAME_BOUNDARY: Final = re.compile(rb"(?<=\n\n)|(?<=\r\n\r\n)")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def scripted_message(identity: str, **extra: JsonValue) -> Mapping[str, JsonValue]:
    return native_message(identity, content=(text_block("Deleting."), tool_use_block(identity)), **extra)


def frames(raw: bytes) -> tuple[bytes, ...]:
    return tuple(frame for frame in _FRAME_BOUNDARY.split(raw) if frame)


def data_of(frame: bytes) -> dict[str, JsonValue]:
    line: Final = next(line for line in frame.splitlines() if line.startswith(b"data: "))
    return _JSON_OBJECT.validate_json(line.removeprefix(b"data: "))


def is_delta(frame: bytes) -> bool:
    return b'"message_delta"' in frame


def is_message_start(frame: bytes) -> bool:
    return b'"message_start"' in frame


def event_of(frame: bytes) -> bytes:
    return next(line for line in frame.splitlines() if line.startswith(b"event: "))


def restamped(frame: bytes, model: str) -> dict[str, JsonValue]:
    data: Final = data_of(frame)
    return {**data, "message": {**_JSON_OBJECT.validate_python(data["message"]), "model": model}}


def assert_restamped_or_verbatim(before: bytes, after: bytes, model: str) -> None:
    if not is_message_start(before):
        assert after == before, (before, after)
        return
    assert event_of(after) == event_of(before), (before, after)
    assert data_of(after) == restamped(before, model), (before, after)


def assert_only_the_message_start_restamped(sent: bytes, received: bytes, model: str) -> None:
    sent_frames: Final = frames(sent)
    received_frames: Final = frames(received)
    assert len(received_frames) == len(sent_frames), (sent_frames, received_frames)
    for before, after in zip(sent_frames, received_frames, strict=True):
        assert_restamped_or_verbatim(before, after, model)


def assert_only_the_delta_changed(sent: bytes, received: bytes, results: Sequence[JsonValue]) -> None:
    sent_frames: Final = frames(sent)
    received_frames: Final = frames(received)
    assert len(received_frames) == len(sent_frames), (sent_frames, received_frames)
    for before, after in zip(sent_frames, received_frames, strict=True):
        if not is_delta(before):
            assert_restamped_or_verbatim(before, after, NATIVE_MODEL)
            continue
        expected: Final = data_of(before)
        assert data_of(after) == {
            **expected,
            "delta": {**dict(expected["delta"]), "safeguard_results": list(results)},
        }, after  # pyright: ignore[reportArgumentType]  # the scripted delta is an object
        assert after.startswith(b"event: message_delta\ndata: ") and after.endswith(b"\n\n"), after


def test_non_stream_reply_keeps_every_backend_field_and_adds_the_verdict(rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    message: Final = scripted_message(identity, container=None, vendor_extra={"kept": True})
    rig.register(request_nonce, native_backend(message, stream=False))
    body: Final = json_body(rig.post(auto_mode_body(NATIVE_MODEL, request_nonce)))
    assert body == {**message, "model": NATIVE_MODEL, "safeguard_results": flagged((identity,))}, body
    classifier_call, backend_calls = one_classifier_call(rig, request_nonce)
    assert ids_under_review(classifier_call) == (identity,), classifier_call.body
    assert [request.target for request in backend_calls] == [NATIVE_TARGET], backend_calls
    assert request_json(backend_calls[0])["safeguards"] == list(SAFEGUARDS), backend_calls[0].body


def test_stream_keeps_every_frame_verbatim_except_the_stamped_delta(rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    sent: Final = native_stream_frames(scripted_message(identity))
    rig.register(request_nonce, native_backend(scripted_message(identity), stream=True, chunks=sent))
    status, raw = rig.stream(auto_mode_body(NATIVE_MODEL, request_nonce))
    assert status == 200, raw
    assert stream_tool_use_ids(raw) == (identity,), raw
    assert_only_the_delta_changed(b"".join(sent), raw, flagged((identity,)))
    classifier_call, backend_calls = one_classifier_call(rig, request_nonce)
    assert ids_under_review(classifier_call) == (identity,), classifier_call.body
    assert request_json(backend_calls[0])["safeguards"] == list(SAFEGUARDS), backend_calls[0].body


def one_byte_chunks(frame_bytes: Sequence[bytes]) -> tuple[bytes, ...]:
    return tuple(bytes((byte,)) for byte in b"".join(frame_bytes))


def crlf(frame_bytes: Sequence[bytes]) -> tuple[bytes, ...]:
    return tuple(frame.replace(b"\n", b"\r\n") for frame in frame_bytes)


def unterminated_tail(frame_bytes: Sequence[bytes]) -> tuple[bytes, ...]:
    return (*frame_bytes[:-1], frame_bytes[-1].removesuffix(b"\n\n"))


def with_non_json_ping(frame_bytes: Sequence[bytes]) -> tuple[bytes, ...]:
    return (frame_bytes[0], b"event: ping\ndata: not json\n\n", *frame_bytes[1:])


@pytest.mark.parametrize("reshape", (one_byte_chunks, crlf, unterminated_tail, with_non_json_ping))
def test_stream_framing_survives_odd_chunking(
    rig: Rig, reshape: Callable[[Sequence[bytes]], tuple[bytes, ...]]
) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    sent: Final = reshape(native_stream_frames(scripted_message(identity)))
    rig.register(request_nonce, native_backend(scripted_message(identity), stream=True, chunks=sent))
    status, raw = rig.stream(auto_mode_body(NATIVE_MODEL, request_nonce))
    assert status == 200, raw
    assert_only_the_delta_changed(b"".join(sent), raw, flagged((identity,)))
    assert ids_under_review(one_classifier_call(rig, request_nonce)[0]) == (identity,)


def test_truncated_native_tool_use_is_unavailable_in_both_modes(rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    message: Final = scripted_message(identity, stop_reason="max_tokens")
    rig.register(request_nonce, native_backend(message, stream=False))
    body: Final = json_body(rig.post(auto_mode_body(NATIVE_MODEL, request_nonce)))
    assert body["safeguard_results"] == unavailable((identity,), "truncated"), body
    stream_nonce: Final = nonce()
    stream_identity: Final = tool_id(stream_nonce)
    rig.register(stream_nonce, native_backend(scripted_message(stream_identity, stop_reason="max_tokens"), stream=True))
    status, raw = rig.stream(auto_mode_body(NATIVE_MODEL, stream_nonce))
    assert status == 200, raw
    assert stream_results(raw) == unavailable((stream_identity,), "truncated"), raw
    assert rig.classifier_calls(rig.received(request_nonce)) == ()
    assert rig.classifier_calls(rig.received(stream_nonce)) == ()


def test_backend_supplied_results_are_kept_without_a_classifier_call(rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    supplied: Final = unavailable((identity,), "error")
    rig.register(
        request_nonce, native_backend(scripted_message(identity, safeguard_results=list(supplied)), stream=False)
    )
    body: Final = json_body(rig.post(auto_mode_body(NATIVE_MODEL, request_nonce)))
    assert body["safeguard_results"] == supplied, body
    stream_nonce: Final = nonce()
    stream_identity: Final = tool_id(stream_nonce)
    stream_supplied: Final = unavailable((stream_identity,), "error")
    sent: Final = native_stream_frames(
        scripted_message(stream_identity), delta_extra={"safeguard_results": list(stream_supplied)}
    )
    rig.register(stream_nonce, native_backend(scripted_message(stream_identity), stream=True, chunks=sent))
    status, raw = rig.stream(auto_mode_body(NATIVE_MODEL, stream_nonce))
    assert status == 200, raw
    assert_only_the_message_start_restamped(b"".join(sent), raw, NATIVE_MODEL)
    assert rig.classifier_calls(rig.received(request_nonce)) == ()
    assert rig.classifier_calls(rig.received(stream_nonce)) == ()


def test_claude_deployment_forwards_safeguards_and_classifies_nothing(rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    message: Final = native_message(identity, content=(tool_use_block(identity),), model="claude-sonnet-4-6")
    rig.register(request_nonce, native_backend(message, stream=False))
    request: Final = auto_mode_body(CLAUDE_MODEL, request_nonce)
    body: Final = json_body(rig.post(request))
    assert body == {**message, "model": CLAUDE_MODEL}, body
    assert tool_use_ids(body) == (identity,), body
    received: Final = rig.received(request_nonce)
    assert [call.target for call in received] == [CLAUDE_TARGET], received
    assert request_json(received[0])["safeguards"] == request["safeguards"], received[0].body
    assert received[0].headers.get("anthropic-beta") is not None, received[0].headers


@pytest.mark.parametrize(("model", "target"), tuple(OTHER_NATIVE_MODELS.items()))
def test_every_native_messages_provider_is_classified(rig: Rig, model: str, target: str) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    rig.register(request_nonce, native_backend(scripted_message(identity, model=model), stream=True))
    status, raw = rig.stream(auto_mode_body(model, request_nonce))
    assert status == 200, raw
    assert stream_tool_use_ids(raw) == (identity,), raw
    assert stream_results(raw) == flagged((identity,)), raw
    classifier_call, backend_calls = one_classifier_call(rig, request_nonce)
    assert ids_under_review(classifier_call) == (identity,), classifier_call.body
    assert [call.target for call in backend_calls] == [target], backend_calls
    assert request_json(backend_calls[0])["safeguards"] == list(SAFEGUARDS), backend_calls[0].body


NATIVE_TARGETS: Final[Mapping[str, str]] = {NATIVE_MODEL: NATIVE_TARGET, **OTHER_NATIVE_MODELS}


@pytest.mark.parametrize("stream", (False, True), ids=("non_stream", "stream"))
@pytest.mark.parametrize(("model", "target"), tuple(NATIVE_TARGETS.items()))
def test_native_backend_receives_the_request_safeguards_unchanged(
    rig: Rig, model: str, target: str, stream: bool
) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    rig.register(request_nonce, native_backend(scripted_message(identity, model=model), stream=stream))
    body: Final = auto_mode_body(model, request_nonce)
    status: Final = rig.stream(body)[0] if stream else rig.post(body).status_code
    assert status == 200, status
    backend_calls: Final = rig.backend_calls(rig.received(request_nonce))
    assert [call.target for call in backend_calls] == [target], backend_calls
    assert request_json(backend_calls[0])["safeguards"] == body["safeguards"], backend_calls[0].body
