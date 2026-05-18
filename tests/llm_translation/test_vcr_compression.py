"""Unit tests for the zstd compression layer in the cassette persister.

Covers the load-side magic-byte sniff that keeps cassettes recorded
before this change loadable, the round-trip of compressed payloads, and
the size win on representative LLM response bodies.
"""

from __future__ import annotations

import json
import os
import sys

import fakeredis
from vcr.request import Request
from vcr.serialize import serialize as vcr_serialize
from vcr.serializers import yamlserializer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_redis_persister import (  # noqa: E402
    _ZSTD_FRAME_MAGIC,
    _compress,
    _decompress,
    make_redis_persister,
    redis_key_for,
)


def _streaming_cassette_dict() -> dict:
    """Build a cassette body that mimics a 50-event Anthropic SSE stream.

    The repetition pattern (``data: {"type":"content_block_delta",...}\\n\\n``)
    is what makes real streaming cassettes 10-15x compressible, so the
    fixture has to look like one to give meaningful size signals.
    """
    request = Request(
        method="POST",
        uri="https://api.anthropic.com/v1/messages",
        body=b'{"model":"claude","stream":true,"messages":[{"role":"user","content":"hi"}]}',
        headers={"content-type": "application/json"},
    )
    chunks = []
    for i in range(50):
        chunk = (
            "event: content_block_delta\n"
            f"data: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'token_'+str(i)}})}\n\n"
        )
        chunks.append(chunk)
    body = ("".join(chunks)).encode("utf-8")
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["text/event-stream"]},
        "body": {"string": body},
    }
    return {"requests": [request], "responses": [response]}


def test_compress_then_decompress_is_lossless():
    payload = b"hello " * 1000
    out = _compress(payload)
    assert out.startswith(_ZSTD_FRAME_MAGIC)
    assert _decompress(out) == payload


def test_decompress_passes_through_legacy_uncompressed_payload():
    legacy = b"interactions:\n- request:\n    method: POST\n"
    assert _decompress(legacy) == legacy


def test_compress_is_idempotent_when_input_already_starts_with_magic():
    payload = _ZSTD_FRAME_MAGIC + b"already framed"
    assert _compress(payload) == payload


def test_compression_substantially_shrinks_streaming_cassette():
    cassette = _streaming_cassette_dict()
    raw = cassette["responses"][0]["body"]["string"] * 1
    compressed = _compress(raw)
    # SSE bodies have very high token entropy reduction; even a
    # conservative floor of 4x is well below what real cassettes achieve.
    assert len(compressed) <= len(raw) // 4, (
        f"expected ≥4x compression on SSE body; got "
        f"{len(raw)}→{len(compressed)} ({len(raw)/len(compressed):.2f}x)"
    )


def test_persister_compresses_payload_on_save():
    fake = fakeredis.FakeStrictRedis()
    persister = make_redis_persister(client=fake)
    cassette_id = "tests/llm_translation/test_x/test_compress_save"
    persister.save_cassette(cassette_id, _streaming_cassette_dict(), yamlserializer)
    raw_in_redis = fake.get(redis_key_for(cassette_id))
    assert raw_in_redis is not None
    assert raw_in_redis.startswith(_ZSTD_FRAME_MAGIC), (
        "expected the persister to write zstd-framed bytes; first 8 bytes "
        f"were {raw_in_redis[:8]!r}"
    )


def test_persister_loads_legacy_uncompressed_yaml():
    """A cassette stored before compression rolled out must still load.

    Simulates the migration window: old payload is plain YAML bytes;
    the new loader must transparently treat it as already-decompressed.
    """
    fake = fakeredis.FakeStrictRedis()
    persister = make_redis_persister(client=fake)
    cassette_id = "tests/llm_translation/test_x/test_legacy_load"

    # Round-trip a cassette through vcrpy's high-level ``serialize``
    # so the bytes match what the persister wrote before compression
    # rolled out (i.e. the same wrapped ``interactions:`` envelope).
    legacy_dict = _streaming_cassette_dict()
    raw_yaml = vcr_serialize(legacy_dict, yamlserializer)
    raw_bytes = raw_yaml.encode("utf-8") if isinstance(raw_yaml, str) else raw_yaml
    fake.set(redis_key_for(cassette_id), raw_bytes)

    requests, responses = persister.load_cassette(cassette_id, yamlserializer)
    assert len(requests) == 1
    assert responses[0]["status"]["code"] == 200


def test_persister_round_trips_compressed_cassette():
    fake = fakeredis.FakeStrictRedis()
    persister = make_redis_persister(client=fake)
    cassette_id = "tests/llm_translation/test_x/test_round_trip"
    cassette = _streaming_cassette_dict()
    persister.save_cassette(cassette_id, cassette, yamlserializer)
    requests, responses = persister.load_cassette(cassette_id, yamlserializer)
    assert len(requests) == 1
    # vcrpy's YAML deserializer can return either bytes or str depending
    # on how it reconstructs the body; compare the byte content directly
    # so the test is robust to that detail.
    loaded = responses[0]["body"]["string"]
    if isinstance(loaded, str):
        loaded = loaded.encode("utf-8")
    expected = cassette["responses"][0]["body"]["string"]
    if isinstance(expected, str):
        expected = expected.encode("utf-8")
    assert loaded == expected


def test_compression_can_be_disabled_via_env(monkeypatch):
    """``CASSETTE_DISABLE_COMPRESSION=1`` is a debugging escape hatch.

    Useful for diff-friendly cache inspection during incident response;
    we still need to keep load behavior compatible with both formats.
    """
    fake = fakeredis.FakeStrictRedis()
    persister = make_redis_persister(client=fake)
    cassette_id = "tests/llm_translation/test_x/test_disabled"

    monkeypatch.setenv("CASSETTE_DISABLE_COMPRESSION", "1")
    persister.save_cassette(cassette_id, _streaming_cassette_dict(), yamlserializer)
    raw = fake.get(redis_key_for(cassette_id))
    assert raw is not None
    assert not raw.startswith(_ZSTD_FRAME_MAGIC)

    # Loader still works on the uncompressed payload.
    monkeypatch.delenv("CASSETTE_DISABLE_COMPRESSION")
    requests, _responses = persister.load_cassette(cassette_id, yamlserializer)
    assert len(requests) == 1
