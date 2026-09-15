"""Read-only diagnostic: dump VCR cassette episodes from Redis and diff them.

Answers "what is varying between the episodes of a leaking cassette?" by
loading each cassette under
``litellm:vcr:cassette:tests/image_gen_tests/test_image_edits/*`` and, for
every group of same-(method, path) requests, comparing their bodies. For
multipart bodies it splits on the boundary and compares field-by-field so
the output points at the exact part that differs (model / prompt / image[]
bytes / a stray header / the boundary token itself).

Usage:
    CASSETTE_REDIS_URL=redis://... python scripts/diff_vcr_cassette_episodes.py
    CASSETTE_REDIS_URL=redis://... python scripts/diff_vcr_cassette_episodes.py 'tests/image_gen_tests/test_image_edits/*'

Strictly read-only: issues SCAN / GET / TTL / STRLEN only. Never writes or
deletes. Safe to run against the production cassette Redis.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import defaultdict

import redis
import yaml

PREFIX = "litellm:vcr:cassette:"
DEFAULT_MATCH = "tests/image_gen_tests/test_image_edits/*"


def _client() -> redis.Redis:
    url = os.environ.get("CASSETTE_REDIS_URL")
    if not url:
        sys.exit("Set CASSETTE_REDIS_URL to the cassette Redis instance")
    return redis.Redis.from_url(
        url,
        socket_timeout=15,
        socket_connect_timeout=15,
        decode_responses=False,
    )


def _as_bytes(body) -> bytes:
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8", errors="replace")
    return repr(body).encode("utf-8")


def _load_cassette(raw: bytes):
    """Return list of interactions. Tries yaml then json serialization."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        import json

        doc = json.loads(text)
    if not isinstance(doc, dict):
        return []
    return doc.get("interactions", []) or []


def _split_multipart(body: bytes):
    """Split a multipart body into (field_name, header_blob, value_len, value_sha)."""
    # boundary is the bytes up to the first CRLF
    first_line = body.split(b"\r\n", 1)[0]
    if not first_line.startswith(b"--"):
        return None
    boundary = first_line
    parts = body.split(boundary)
    fields = []
    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_blob, _, value = part.partition(b"\r\n\r\n")
        name = b"?"
        for tok in header_blob.split(b";"):
            tok = tok.strip()
            if tok.startswith(b'name="'):
                name = tok[len(b'name="'):].rstrip(b'"')
        fields.append(
            (
                name.decode("latin1"),
                header_blob.decode("latin1"),
                len(value),
                hashlib.sha256(value).hexdigest()[:12],
            )
        )
    return boundary.decode("latin1"), fields


def _diff_two(a: bytes, b: bytes) -> str:
    if a == b:
        return "identical"
    n = min(len(a), len(b))
    off = next((i for i in range(n) if a[i] != b[i]), n)
    start = max(0, off - 60)
    return (
        f"first divergence @ byte {off} (len_a={len(a)} len_b={len(b)})\n"
        f"      a: ...{a[start:off + 60]!r}\n"
        f"      b: ...{b[start:off + 60]!r}"
    )


def _report_cassette(key: str, interactions: list) -> None:
    print(f"\n{'=' * 100}\n{key}\n  episodes: {len(interactions)}")

    by_endpoint = defaultdict(list)
    for idx, inter in enumerate(interactions):
        req = inter.get("request", {}) or {}
        uri = req.get("uri") or req.get("url") or "?"
        method = req.get("method", "?")
        path = uri.split("?", 1)[0]
        body = _as_bytes(req.get("body"))
        by_endpoint[(method, path)].append((idx, body))

    for (method, path), episodes in by_endpoint.items():
        print(f"\n  --- {method} {path}  ({len(episodes)} episode(s)) ---")
        # whole-body hashes
        for idx, body in episodes:
            print(
                f"    ep#{idx:2d} len={len(body):8d} sha={hashlib.sha256(body).hexdigest()[:12]}"
            )

        # multipart field-level analysis on the first episode as a template,
        # then show which fields differ across all episodes.
        parsed = []
        for idx, body in episodes:
            mp = _split_multipart(body)
            parsed.append((idx, body, mp))

        if all(mp for _, _, mp in parsed):
            # boundary tokens
            boundaries = {mp[0] for _, _, mp in parsed}
            print(f"    boundary tokens seen: {boundaries}")
            # field-by-field: collect set of (len, sha) per field name
            field_variants = defaultdict(set)
            field_order = []
            for _, _, (_, fields) in parsed:
                for name, _hdr, vlen, vsha in fields:
                    if name not in field_order:
                        field_order.append(name)
                    field_variants[name].add((vlen, vsha))
            print("    field-level variation across episodes:")
            for name in field_order:
                variants = field_variants[name]
                flag = "  <-- VARIES" if len(variants) > 1 else ""
                print(f"      {name!r}: {len(variants)} distinct value(s){flag}")
                if len(variants) > 1:
                    for vlen, vsha in sorted(variants):
                        print(f"          len={vlen} sha={vsha}")
        else:
            # non-multipart: pairwise diff first two
            if len(episodes) >= 2:
                print("    (non-multipart) diff ep0 vs ep1:")
                print("      " + _diff_two(episodes[0][1], episodes[1][1]))


def main() -> None:
    match = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MATCH
    pattern = f"{PREFIX}{match}".encode()
    client = _client()
    print("PING:", client.ping())

    keys = []
    cursor = 0
    while True:
        cursor, batch = client.scan(cursor=cursor, match=pattern, count=500)
        keys.extend(batch)
        if cursor == 0:
            break
    keys = sorted(k.decode() for k in keys)
    print(f"matched {len(keys)} cassette key(s) for pattern {pattern.decode()}")

    for key in keys:
        raw = client.get(key.encode())
        if raw is None:
            print(f"\n{key}: (gone)")
            continue
        interactions = _load_cassette(raw)
        _report_cassette(key, interactions)


if __name__ == "__main__":
    main()
