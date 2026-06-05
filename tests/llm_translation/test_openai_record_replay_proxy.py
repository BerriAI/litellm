from __future__ import annotations

import asyncio
import os
import sys

import fakeredis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._openai_record_replay_proxy import (  # noqa: E402
    CASSETTE_TTL_SECONDS,
    RECORD_KEY_PREFIX,
    OpenAIRecordReplay,
)

_OK_BODY = b'{"data":[{"b64_json":"aW1n"}],"usage":{"total_tokens":42}}'


class _Upstream:
    """Stub live upstream; counts calls so replays can be proven offline."""

    def __init__(self, status=200, headers=None, body=_OK_BODY):
        self.calls = 0
        self._status = status
        self._headers = headers if headers is not None else [("content-type", "application/json")]
        self._body = body

    async def __call__(self):
        self.calls += 1
        return self._status, list(self._headers), self._body


def _recorder(client=None):
    return OpenAIRecordReplay(client if client is not None else fakeredis.FakeStrictRedis())


def _run(coro):
    return asyncio.run(coro)


def test_miss_forwards_to_upstream_and_records():
    fake = fakeredis.FakeStrictRedis()
    recorder = _recorder(fake)
    upstream = _Upstream()

    status, headers, body = _run(
        recorder.handle("POST", "/v1/images/generations", b'{"model":"gpt-image-1"}', upstream)
    )

    assert upstream.calls == 1
    assert status == 200
    assert body == _OK_BODY
    key = OpenAIRecordReplay.record_key("POST", "/v1/images/generations", b'{"model":"gpt-image-1"}')
    assert key.startswith(RECORD_KEY_PREFIX)
    assert fake.get(key) is not None


def test_hit_replays_without_calling_upstream():
    recorder = _recorder()
    upstream = _Upstream()
    body_in = b'{"model":"gpt-image-1","prompt":"otter"}'

    first = _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))
    second = _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))

    assert upstream.calls == 1
    assert first == second
    assert second[2] == _OK_BODY


def test_different_body_is_a_separate_recording():
    recorder = _recorder()
    upstream = _Upstream()

    _run(recorder.handle("POST", "/v1/images/generations", b'{"prompt":"otter"}', upstream))
    _run(recorder.handle("POST", "/v1/images/generations", b'{"prompt":"seal"}', upstream))

    assert upstream.calls == 2


def test_record_key_ignores_json_key_order():
    a = OpenAIRecordReplay.record_key("POST", "/v1/images/generations", b'{"model":"x","prompt":"y"}')
    b = OpenAIRecordReplay.record_key("POST", "/v1/images/generations", b'{"prompt":"y","model":"x"}')
    assert a == b


def test_ttl_set_on_write_and_not_refreshed_on_read():
    """A replay must not slide the recording's expiry forward.

    The recording counts down from its last write so it lapses
    ``CASSETTE_TTL_SECONDS`` after capture and the next run re-records live,
    catching provider drift. Refreshing the TTL on a replay would keep an
    actively-replayed recording alive forever and that drift check would never
    run. This mirrors the VCR persister's lapse-after-write contract.
    """
    fake = fakeredis.FakeStrictRedis()
    recorder = _recorder(fake)
    upstream = _Upstream()
    body_in = b'{"model":"gpt-image-1"}'
    key = OpenAIRecordReplay.record_key("POST", "/v1/images/generations", body_in)

    _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))
    assert CASSETTE_TTL_SECONDS - 5 <= fake.ttl(key) <= CASSETTE_TTL_SECONDS

    fake.expire(key, 60)
    _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))

    assert fake.ttl(key) <= 60


def test_replay_drops_framing_headers_so_server_recomputes():
    fake = fakeredis.FakeStrictRedis()
    recorder = _recorder(fake)
    upstream = _Upstream(
        headers=[
            ("content-type", "application/json"),
            ("content-length", "9999"),
            ("transfer-encoding", "chunked"),
            ("content-encoding", "gzip"),
            ("date", "Mon, 01 Jan 2024 00:00:00 GMT"),
            ("server", "cloudflare"),
            ("x-request-id", "req_abc"),
        ]
    )
    body_in = b'{"model":"gpt-image-1"}'

    _, live_headers, _ = _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))
    _, replay_headers, _ = _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))

    for headers in (live_headers, replay_headers):
        names = {k.lower() for k, _ in headers}
        assert names.isdisjoint(
            {
                "content-length",
                "transfer-encoding",
                "content-encoding",
                "date",
                "server",
            }
        )
        assert ("content-type", "application/json") in headers
        assert ("x-request-id", "req_abc") in headers


def test_non_2xx_response_is_not_cached():
    fake = fakeredis.FakeStrictRedis()
    recorder = _recorder(fake)
    upstream = _Upstream(status=500, body=b'{"error":"boom"}')
    body_in = b'{"model":"gpt-image-1"}'
    key = OpenAIRecordReplay.record_key("POST", "/v1/images/generations", body_in)

    status, _, _ = _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))
    assert status == 500
    assert fake.get(key) is None

    _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))
    assert upstream.calls == 2


class _BoomRedis:
    def get(self, *args, **kwargs):
        raise ConnectionError("redis offline")

    def set(self, *args, **kwargs):
        raise ConnectionError("redis offline")


def test_redis_outage_degrades_to_live_passthrough():
    recorder = _recorder(_BoomRedis())
    upstream = _Upstream()
    body_in = b'{"model":"gpt-image-1"}'

    first = _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))
    second = _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))

    assert first[0] == 200 and second[0] == 200
    assert upstream.calls == 2


def test_passthrough_when_no_redis_client_configured():
    recorder = OpenAIRecordReplay(None)
    upstream = _Upstream()
    body_in = b'{"model":"gpt-image-1"}'

    _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))
    _run(recorder.handle("POST", "/v1/images/generations", body_in, upstream))

    assert upstream.calls == 2


class _StubClient:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


def test_app_lifespan_leaves_injected_http_client_open():
    """The app must only close the client it created, never a caller's.

    A caller that injects its own client owns that client's lifecycle; the
    app closing it would break reuse across multiple ``create_app`` calls.
    """
    from starlette.testclient import TestClient

    from tests._openai_record_replay_proxy import create_app

    client = _StubClient()
    app = create_app(recorder=_recorder(), http_client=client)

    with TestClient(app):
        pass

    assert client.closed is False
