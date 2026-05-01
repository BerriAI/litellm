"""Contract tests against real mitmproxy.

The other addon unit tests stub out ``mitmproxy.http`` so they can run
without mitmproxy installed (it's a CI-only dependency installed via
``uv tool install`` rather than declared in ``pyproject.toml``).

That stubbing once hid a real production bug: the addon called
``http.Response.make(status, body, list[(str, str)])`` which the fake
accepted, but real mitmproxy 11 raises ``TypeError: Header fields must
be bytes``. Every cache hit silently fell through to the upstream
because the addon raised inside ``request()``.

This file exercises the *real* mitmproxy ``Response`` contract for the
replay path, so any future regression is caught locally instead of only
showing up as ``Addon error: ...`` in CI logs while tests pass.
"""

from __future__ import annotations

import pytest

mitmproxy_http = pytest.importorskip("mitmproxy.http")

# ``test_addon.py`` (collected alongside this file) installs a stubbed
# ``mitmproxy`` package into ``sys.modules`` so it can run without the
# real dependency. ``importorskip`` will happily return that stub and
# defeat the whole point of *this* file. Detect the stub by checking
# whether the loaded ``Headers`` enforces bytes the way real mitmproxy
# does, and skip if it doesn't.
try:
    mitmproxy_http.Headers([("not", "bytes")])  # type: ignore[arg-type]
except TypeError:
    pass  # real mitmproxy — proceed
except Exception:
    pytest.skip(
        "real mitmproxy not available (fake stub detected); "
        "install ``mitmproxy==11.0.2`` to run these contract tests",
        allow_module_level=True,
    )
else:
    pytest.skip(
        "real mitmproxy not available (fake stub detected); "
        "install ``mitmproxy==11.0.2`` to run these contract tests",
        allow_module_level=True,
    )

from tests.e2e_cassette_proxy.addon import _build_replay_response  # noqa: E402
from tests.e2e_cassette_proxy.redis_store import CachedResponse  # noqa: E402


def test_replay_response_accepts_string_headers_and_emits_bytes():
    """Cached headers are stored as ``str`` (msgpack roundtrip); the
    replay builder must encode them back to ``bytes`` because mitmproxy
    11's ``Headers`` rejects anything else."""
    cached = CachedResponse(
        status_code=200,
        headers=(
            ("Content-Type", "application/json"),
            ("X-Trace-Id", "abc123"),
        ),
        body=b'{"id":"chatcmpl-1"}',
        reason="OK",
    )

    response = _build_replay_response(cached)

    assert isinstance(response, mitmproxy_http.Response)
    assert response.status_code == 200
    # mitmproxy's ``Headers`` decodes back to str on read; the underlying
    # ``fields`` are bytes — verify both directions.
    assert response.headers["Content-Type"] == "application/json"
    assert response.headers["X-Trace-Id"] == "abc123"
    for name, value in response.headers.fields:
        assert isinstance(name, bytes), name
        assert isinstance(value, bytes), value


def test_replay_response_preserves_raw_content_without_double_encoding():
    """When the upstream replied with ``Content-Encoding: gzip`` we recorded
    the *gzipped* bytes (i.e. ``flow.response.raw_content``). On replay the
    body must be set as ``raw_content`` so mitmproxy serves the same wire
    payload — going through ``set_content`` would gzip-encode an already-
    gzipped blob."""
    gzipped_body = (
        b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\xab\xe6"
        b"R\xa8\xe5\x02\x00\x9b\xae\x10\xee\x05\x00\x00\x00"
    )  # arbitrary gzip-magic-prefixed bytes
    cached = CachedResponse(
        status_code=200,
        headers=(
            ("Content-Type", "application/json"),
            ("Content-Encoding", "gzip"),
            ("Content-Length", str(len(gzipped_body))),
        ),
        body=gzipped_body,
        reason="OK",
    )

    response = _build_replay_response(cached)

    assert response.raw_content == gzipped_body, (
        "raw_content must be byte-identical to what we recorded; "
        "going through set_content would re-gzip and corrupt it"
    )


def test_replay_response_handles_repeated_set_cookie_headers():
    """``items(multi=True)`` is used on the recording side specifically so
    multi-value headers like ``Set-Cookie`` aren't silently merged into a
    comma-joined string (which is RFC 7230 illegal for ``Set-Cookie``)."""
    cached = CachedResponse(
        status_code=200,
        headers=(
            ("Set-Cookie", "session=abc; Path=/"),
            ("Set-Cookie", "csrf=xyz; Path=/"),
            ("Content-Type", "text/html"),
        ),
        body=b"<html></html>",
        reason="OK",
    )

    response = _build_replay_response(cached)

    set_cookies = [
        v.decode("latin-1")
        for k, v in response.headers.fields
        if k.lower() == b"set-cookie"
    ]
    assert set_cookies == ["session=abc; Path=/", "csrf=xyz; Path=/"]


def test_replay_response_tolerates_empty_reason_phrase():
    cached = CachedResponse(
        status_code=204,
        headers=(),
        body=b"",
        reason="",
    )

    response = _build_replay_response(cached)

    assert response.status_code == 204
    # mitmproxy expects bytes; an empty reason should default to ``b"OK"``
    # at the wire level so the response is still serializable by
    # mitmproxy's HTTP/1.1 codec. ``reason`` decodes back to str on read.
    assert response.reason == "OK"
