"""Cache key derivation for the e2e recording proxy.

The proxy is intentionally promiscuous about *what* it caches: any HTTPS
egress that flows through it is a candidate. The cache key has to be
stable across runs that send equivalent requests, while staying robust
to per-call noise (auth headers, tracing IDs, dates).

We hash a canonical tuple of:

- HTTP method
- scheme + host + port
- URL path
- query string (sorted)
- request body (raw bytes; JSON bodies are re-serialized in canonical
  form so that semantically-equal payloads collide)
- a small allowlist of headers the upstream actually keys on (e.g.
  ``content-type``, ``accept``)

Anything not on that allowlist is dropped before hashing. This is the
same trade-off vcrpy makes when configuring ``filter_headers`` —
strict enough to dedupe equivalent requests, loose enough to ignore
the auth/tracing churn that varies run-to-run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlsplit

CACHE_KEY_PREFIX = "litellm:e2ecass:"

# Headers that materially change what an upstream returns.
DEFAULT_HEADER_ALLOWLIST: Tuple[str, ...] = (
    "accept",
    "accept-encoding",
    "content-type",
    "openai-beta",
    "anthropic-beta",
    "anthropic-version",
    "x-stainless-lang",
)

# Headers we *never* want in the key (auth, tracing, per-request noise).
DEFAULT_HEADER_BLOCKLIST: Tuple[str, ...] = (
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    "openai-api-key",
    "azure-api-key",
    "api-key",
    "cookie",
    "user-agent",
    "x-amz-security-token",
    "x-amz-date",
    "x-amz-content-sha256",
    "amz-sdk-invocation-id",
    "amz-sdk-request",
    "x-goog-api-key",
    "x-goog-user-project",
    "x-request-id",
    "request-id",
    "traceparent",
    "tracestate",
    "x-stainless-arch",
    "x-stainless-os",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
    "x-stainless-package-version",
    "host",
    "content-length",
    "connection",
)


def _canonical_body(body: bytes) -> bytes:
    """JSON bodies often differ only in key order or whitespace; collapse
    those into a single canonical form so the cache hits across sessions.
    Non-JSON bodies are passed through unchanged."""
    if not body:
        return b""
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    return json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_query(raw_query: str) -> str:
    if not raw_query:
        return ""
    pairs = sorted(parse_qsl(raw_query, keep_blank_values=True))
    return "&".join(f"{k}={v}" for k, v in pairs)


def _normalized_headers(
    headers: Mapping[str, str],
    allowlist: Sequence[str],
    blocklist: Sequence[str],
) -> Tuple[Tuple[str, str], ...]:
    allow = {h.lower() for h in allowlist}
    block = {h.lower() for h in blocklist}
    out: list[Tuple[str, str]] = []
    for key, value in headers.items():
        lk = key.lower()
        if lk in block:
            continue
        if allow and lk not in allow:
            continue
        out.append((lk, value))
    out.sort()
    return tuple(out)


def derive_cache_key(
    method: str,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    *,
    allowlist: Optional[Iterable[str]] = None,
    blocklist: Optional[Iterable[str]] = None,
) -> str:
    """Hash the canonicalized (method, url, body, allowlisted-headers) tuple
    into a stable Redis key. Equal inputs (modulo header / JSON noise) always
    produce the same key.
    """
    parts = urlsplit(url)
    canonical_headers = _normalized_headers(
        headers,
        allowlist=(
            tuple(allowlist) if allowlist is not None else DEFAULT_HEADER_ALLOWLIST
        ),
        blocklist=(
            tuple(blocklist) if blocklist is not None else DEFAULT_HEADER_BLOCKLIST
        ),
    )
    canonical_body = _canonical_body(body)
    digest = hashlib.sha256()
    digest.update(method.upper().encode("ascii"))
    digest.update(b"\x1f")
    digest.update(parts.scheme.lower().encode("ascii"))
    digest.update(b"://")
    digest.update(parts.netloc.lower().encode("ascii"))
    digest.update(b"\x1f")
    digest.update(parts.path.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(_canonical_query(parts.query).encode("utf-8"))
    digest.update(b"\x1f")
    for k, v in canonical_headers:
        digest.update(k.encode("ascii"))
        digest.update(b"=")
        digest.update(v.encode("utf-8", errors="replace"))
        digest.update(b"\x1e")
    digest.update(b"\x1f")
    digest.update(canonical_body)
    return f"{CACHE_KEY_PREFIX}{digest.hexdigest()}"
