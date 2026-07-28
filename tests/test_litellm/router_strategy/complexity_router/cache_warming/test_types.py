import json

import pytest
from pydantic import ValidationError

from litellm.router_strategy.complexity_router.cache_warming.types import (
    CacheWarmingPayload,
    compress_payload,
    decompress_payload,
)


def _payload(**overrides: object) -> CacheWarmingPayload:
    base: dict[str, object] = {
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "messages": [
            {"role": "system", "content": "policy manual é中文 " * 50},
            {"role": "user", "content": [{"type": "text", "text": "summarize rule 7"}]},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup_rule", "parameters": {"type": "object"}}}],
        "call_surface": "chat_completions",
    }
    return CacheWarmingPayload(**{**base, **overrides})


def test_compress_decompress_roundtrip_preserves_payload():
    payload = _payload(system=[{"type": "text", "text": "cached system block"}], call_surface="anthropic_messages")
    blob, _ = compress_payload(payload)
    restored = decompress_payload(blob)
    assert restored.model_dump() == payload.model_dump()
    assert json.dumps(restored.messages) == json.dumps(payload.messages)


def test_roundtrip_preserves_message_key_order():
    payload = _payload(messages=[{"role": "user", "content": "hi", "name": "a"}])
    reordered = _payload(messages=[{"name": "a", "content": "hi", "role": "user"}])
    assert json.dumps(decompress_payload(compress_payload(payload)[0]).messages) != json.dumps(
        decompress_payload(compress_payload(reordered)[0]).messages
    )


def test_decompress_rejects_corrupt_blob():
    with pytest.raises(Exception):
        decompress_payload("not-base64-zlib!!")
    with pytest.raises(ValidationError):
        decompress_payload(compress_and_corrupt())


def compress_and_corrupt() -> str:
    import base64
    import zlib

    return base64.b64encode(zlib.compress(json.dumps({"model": "m"}).encode())).decode("ascii")


def test_payload_sha256_stable_across_key_order():
    ordered = _payload(messages=[{"role": "user", "content": "hi", "name": "a"}])
    reordered = _payload(messages=[{"name": "a", "content": "hi", "role": "user"}])
    different = _payload(messages=[{"role": "user", "content": "bye", "name": "a"}])
    assert compress_payload(ordered)[1] == compress_payload(reordered)[1]
    assert compress_payload(ordered)[1] != compress_payload(different)[1]


def test_compression_shrinks_repetitive_payload():
    payload = _payload()
    blob, _ = compress_payload(payload)
    assert len(blob) < len(payload.model_dump_json())


def test_payload_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        CacheWarmingPayload(
            model="m",
            messages=[],
            call_surface="chat_completions",
            api_key="sk-should-never-be-here",
        )
