from __future__ import annotations

import asyncio
import os
import sys
import warnings

import fakeredis
import pytest
from websockets.exceptions import ConnectionClosedOK

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_redis_persister import (  # noqa: E402
    VCRCassetteCacheWarning,
    cassette_cache_health,
)
from tests._ws_vcr import (  # noqa: E402
    CASSETTE_TTL_SECONDS,
    RedisLike,
    ReplayConnection,
    WsCassette,
    WsFrame,
    WsSession,
    WsSessionRecorder,
    WsVcrContractDrift,
    WsVcrReplayError,
    WsVcrReplayTimeout,
    build_ws_cassette_client,
    load_ws_cassette,
    save_ws_cassette,
    scrub_secrets,
    text_frames_match,
    ws_redis_key_for,
)


def _server(text: str, client_frames_before: int) -> WsFrame:
    return WsFrame(
        direction="server_to_client",
        opcode="text",
        text=text,
        client_frames_before=client_frames_before,
    )


def _client(text: str) -> WsFrame:
    return WsFrame(direction="client_to_server", opcode="text", text=text)


def _collect_errors():
    errors: list[WsVcrReplayError] = []
    return errors, errors.append


def test_cassette_json_roundtrip_preserves_frames_and_gate():
    cassette = WsCassette(
        sessions=(
            WsSession(
                frames=(
                    _server('{"type":"session.created"}', 0),
                    _client('{"type":"response.create"}'),
                    _server('{"type":"response.done"}', 1),
                    WsFrame(
                        direction="server_to_client", opcode="binary", binary_b64="dGVzdA==", client_frames_before=1
                    ),
                )
            ),
        )
    )

    restored = WsCassette.model_validate_json(cassette.model_dump_json())

    assert restored == cassette
    assert restored.sessions[0].frames[2].client_frames_before == 1
    assert restored.sessions[0].frames[3].opcode == "binary"
    assert restored.sessions[0].frames[3].binary_b64 == "dGVzdA=="


def test_recorder_tracks_client_frame_count_as_causal_gate():
    recorder = WsSessionRecorder()
    recorder.record_server_frame('{"type":"session.created"}')
    recorder.record_client_frame('{"type":"conversation.item.create"}')
    recorder.record_client_frame('{"type":"response.create"}')
    recorder.record_server_frame('{"type":"response.done"}')

    session = recorder.to_session()
    server_frames = [f for f in session.frames if f.direction == "server_to_client"]

    assert server_frames[0].client_frames_before == 0
    assert server_frames[1].client_frames_before == 2


async def test_replay_recv_returns_bytes_when_decode_false_and_str_otherwise():
    session = WsSession(frames=(_server("hello", 0), _server("world", 0)))
    _, on_error = _collect_errors()
    conn = ReplayConnection(session, timeout=1.0, on_error=on_error)

    as_bytes = await conn.recv(decode=False)
    as_str = await conn.recv()

    assert as_bytes == b"hello"
    assert as_str == "world"


async def test_replay_serves_server_frame_only_after_causal_client_count_met():
    session = WsSession(
        frames=(
            _server('{"type":"session.created"}', 0),
            _client('{"type":"response.create"}'),
            _server('{"type":"response.done"}', 1),
        )
    )
    _, on_error = _collect_errors()
    conn = ReplayConnection(session, timeout=2.0, on_error=on_error)

    first = await conn.recv(decode=False)
    assert first == b'{"type":"session.created"}'

    gated = asyncio.ensure_future(conn.recv(decode=False))
    await asyncio.sleep(0.1)
    assert not gated.done(), "gated server frame was released before the recorded client frame was sent"

    await conn.send('{"type":"response.create"}')
    released = await asyncio.wait_for(gated, timeout=1.0)
    assert released == b'{"type":"response.done"}'


async def test_replay_exhausted_server_frames_raise_connection_closed():
    session = WsSession(frames=(_server("only", 0),))
    _, on_error = _collect_errors()
    conn = ReplayConnection(session, timeout=1.0, on_error=on_error)

    await conn.recv(decode=False)
    with pytest.raises(ConnectionClosedOK):
        await conn.recv(decode=False)


async def test_replay_timeout_raises_instead_of_hanging():
    session = WsSession(
        frames=(
            _server('{"type":"session.created"}', 0),
            _server('{"type":"response.done"}', 5),
        )
    )
    errors, on_error = _collect_errors()
    conn = ReplayConnection(session, timeout=0.15, on_error=on_error)

    await conn.recv(decode=False)
    with pytest.raises(WsVcrReplayTimeout):
        await asyncio.wait_for(conn.recv(decode=False), timeout=2.0)
    assert errors and isinstance(errors[0], WsVcrReplayTimeout)


async def test_replay_accepts_client_frame_with_volatile_id_drift():
    recorded_client = _client('{"type":"conversation.item.create","item":{"id":"item_ABC12345","role":"user"}}')
    session = WsSession(frames=(_server("s", 0), recorded_client, _server("done", 1)))
    errors, on_error = _collect_errors()
    conn = ReplayConnection(session, timeout=1.0, on_error=on_error)

    await conn.recv(decode=False)
    await conn.send('{"type":"conversation.item.create","item":{"role":"user","id":"item_ZZ99887766"}}')

    assert errors == []
    assert await conn.recv(decode=False) == b"done"


async def test_replay_rejects_structurally_different_client_frame():
    session = WsSession(frames=(_server("s", 0), _client('{"type":"response.create"}'), _server("done", 1)))
    errors, on_error = _collect_errors()
    conn = ReplayConnection(session, timeout=1.0, on_error=on_error)

    await conn.recv(decode=False)
    with pytest.raises(WsVcrContractDrift):
        await conn.send('{"type":"session.update","session":{"voice":"alloy"}}')
    assert errors and isinstance(errors[0], WsVcrContractDrift)


async def test_replay_rejects_extra_client_frame_beyond_recording():
    session = WsSession(frames=(_server("s", 0), _client('{"type":"response.create"}')))
    errors, on_error = _collect_errors()
    conn = ReplayConnection(session, timeout=1.0, on_error=on_error)

    await conn.send('{"type":"response.create"}')
    with pytest.raises(WsVcrContractDrift):
        await conn.send('{"type":"response.create"}')
    assert errors


def test_text_frames_match_normalizes_ids_and_timestamps_but_not_structure():
    assert text_frames_match(
        '{"type":"x","event_id":"evt_111","ts":"2026-05-25T03:40:37.262045Z"}',
        '{"type":"x","event_id":"evt_999","ts":"2026-06-01T10:00:00Z"}',
    )
    assert not text_frames_match('{"type":"x","text":"hi"}', '{"type":"x","text":"bye"}')
    assert not text_frames_match('{"type":"x"}', '{"type":"x","extra":1}')


def test_scrub_secrets_removes_auth_material():
    scrubbed = scrub_secrets("Authorization: Bearer sk-abcdef123456 and key xai-zzz99988877 raw sk-plainkey123")
    assert "sk-abcdef123456" not in scrubbed
    assert "xai-zzz99988877" not in scrubbed
    assert "sk-plainkey123" not in scrubbed
    assert "Bearer <redacted>" in scrubbed


def test_recorder_scrubs_secrets_in_stored_frames():
    recorder = WsSessionRecorder()
    recorder.record_client_frame('{"authorization":"Bearer sk-supersecretvalue"}')
    stored = recorder.to_session().frames[0].text
    assert stored is not None
    assert "sk-supersecretvalue" not in stored


def _sample_cassette() -> WsCassette:
    return WsCassette(sessions=(WsSession(frames=(_server('{"type":"session.created"}', 0),)),))


def test_save_sets_24h_ttl_and_load_roundtrips():
    fake = fakeredis.FakeStrictRedis()
    key = ws_redis_key_for("tests/llm_translation/realtime/test_x.py::test_y")

    assert save_ws_cassette(fake, key, _sample_cassette(), passed=True) is True

    ttl = fake.ttl(key)
    assert CASSETTE_TTL_SECONDS - 5 <= ttl <= CASSETTE_TTL_SECONDS
    loaded = load_ws_cassette(fake, key)
    assert loaded == _sample_cassette()


def test_save_skipped_when_test_failed_leaves_no_key():
    fake = fakeredis.FakeStrictRedis()
    key = ws_redis_key_for("tests/llm_translation/realtime/test_x.py::test_fail")

    assert save_ws_cassette(fake, key, _sample_cassette(), passed=False) is False
    assert fake.get(key) is None


def test_save_skipped_when_test_failed_preserves_prior_cassette():
    fake = fakeredis.FakeStrictRedis()
    key = ws_redis_key_for("tests/llm_translation/realtime/test_x.py::test_keep")

    save_ws_cassette(fake, key, _sample_cassette(), passed=True)
    newer = WsCassette(sessions=(WsSession(frames=(_server('{"type":"other"}', 0),)),))

    assert save_ws_cassette(fake, key, newer, passed=False) is False
    assert load_ws_cassette(fake, key) == _sample_cassette()


def test_load_missing_key_returns_none():
    fake = fakeredis.FakeStrictRedis()
    assert load_ws_cassette(fake, ws_redis_key_for("never/recorded")) is None


def test_ws_redis_key_uses_distinct_prefix():
    key = ws_redis_key_for("tests/llm_translation/realtime/test_x.py::TestY::test_z")
    assert key.startswith("litellm:vcr:wscassette:")
    assert "::" not in key


def test_build_ws_cassette_client_warns_and_counts_failure_instead_of_silently_disabling():
    def _broken_builder() -> RedisLike:
        raise ValueError("invalid CASSETTE_REDIS_URL")

    failures_before = cassette_cache_health()["load_failures"]
    with pytest.warns(VCRCassetteCacheWarning, match="fall back to live websocket traffic"):
        assert build_ws_cassette_client(builder=_broken_builder) is None
    assert cassette_cache_health()["load_failures"] == failures_before + 1


def test_build_ws_cassette_client_returns_built_client_without_warning():
    fake = fakeredis.FakeStrictRedis()
    with warnings.catch_warnings():
        warnings.simplefilter("error", VCRCassetteCacheWarning)
        assert build_ws_cassette_client(builder=lambda: fake) is fake
