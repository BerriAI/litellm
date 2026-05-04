"""Redis-backed key-value store for cached HTTP responses.

The on-the-wire format for a cached entry is a single MessagePack blob
(or, if msgpack isn't available, length-prefixed JSON). We pick this
shape over vcrpy's YAML cassettes because:

- Each Redis key holds exactly one (request_summary, response) pair.
  No "growing list of episodes," no `record_mode` semantics, no
  ordering brittleness — see PR #26967 thread for context.
- Binary response bodies (gzip, audio, image) round-trip without
  base64 expansion on the JSON path *because* they're stored as raw
  bytes inside MessagePack.
- We can bound per-key size at the storage layer (oversize responses
  are dropped with a log line and never block the request).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
DEFAULT_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024  # 4 MiB per entry

_log = logging.getLogger("litellm.e2e_cassette_proxy.store")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[E2ECASS] %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


try:
    import msgpack  # type: ignore

    _HAVE_MSGPACK = True
except ImportError:  # pragma: no cover
    _HAVE_MSGPACK = False


@dataclass(frozen=True)
class CachedResponse:
    """Wire-format-agnostic response cached for a single request."""

    status_code: int
    headers: Tuple[Tuple[str, str], ...]
    body: bytes
    reason: str = ""

    def to_blob(self) -> bytes:
        if _HAVE_MSGPACK:
            return msgpack.packb(  # type: ignore[no-any-return]
                {
                    "v": 1,
                    "status": self.status_code,
                    "reason": self.reason,
                    "headers": [[k, v] for k, v in self.headers],
                    "body": self.body,
                },
                use_bin_type=True,
            )
        # JSON fallback: body must be base64 because JSON can't carry raw bytes.
        return json.dumps(
            {
                "v": 1,
                "status": self.status_code,
                "reason": self.reason,
                "headers": [[k, v] for k, v in self.headers],
                "body_b64": base64.b64encode(self.body).decode("ascii"),
            }
        ).encode("utf-8")

    @classmethod
    def from_blob(cls, blob: bytes) -> "CachedResponse":
        if _HAVE_MSGPACK:
            try:
                payload = msgpack.unpackb(blob, raw=False)
            except Exception:  # pragma: no cover - corrupt blob
                payload = json.loads(blob.decode("utf-8"))
        else:
            payload = json.loads(blob.decode("utf-8"))
        body = payload.get("body")
        if body is None and "body_b64" in payload:
            body = base64.b64decode(payload["body_b64"])
        headers = tuple((str(k), str(v)) for k, v in payload.get("headers", []))
        return cls(
            status_code=int(payload["status"]),
            headers=headers,
            body=body or b"",
            reason=str(payload.get("reason", "")),
        )


def _redis_url_from_env() -> Optional[str]:
    for var in ("LITELLM_E2E_CASS_REDIS_URL", "REDIS_URL", "REDIS_SSL_URL"):
        url = os.environ.get(var)
        if url:
            return url
    host = os.environ.get("REDIS_HOST")
    if not host:
        return None
    scheme = "rediss" if os.environ.get("REDIS_SSL", "").lower() == "true" else "redis"
    auth = ""
    if os.environ.get("REDIS_PASSWORD"):
        user = os.environ.get("REDIS_USERNAME", "")
        auth = f"{user}:{os.environ['REDIS_PASSWORD']}@"
    port = os.environ.get("REDIS_PORT", "6379")
    return f"{scheme}://{auth}{host}:{port}"


class RedisCassetteStore:
    """Thin GET/SET wrapper around ``redis.Redis``.

    Operations either succeed or are dropped with a log line — the
    sidecar must never block the request path because the cache is
    misbehaving. ``get`` returning ``None`` is the "miss" signal;
    ``set`` returning ``False`` means "we tried but couldn't persist."
    """

    def __init__(
        self,
        client=None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_payload_bytes
        self._client = client if client is not None else self._build_default_client()

    @staticmethod
    def _build_default_client():
        import redis

        url = _redis_url_from_env()
        if not url:
            raise RuntimeError(
                "Set LITELLM_E2E_CASS_REDIS_URL / REDIS_URL / REDIS_SSL_URL / REDIS_HOST"
                " to enable the e2e cassette proxy"
            )
        return redis.Redis.from_url(
            url,
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=False,
        )

    def get(self, key: str) -> Optional[CachedResponse]:
        try:
            blob = self._client.get(key)
        except Exception as exc:
            _log.info(f"get-failed key={key} err={type(exc).__name__}: {exc}")
            return None
        if blob is None:
            return None
        try:
            return CachedResponse.from_blob(blob)
        except Exception as exc:
            _log.info(f"corrupt-blob key={key} err={type(exc).__name__}: {exc}")
            try:
                self._client.delete(key)
            except Exception:
                pass
            return None

    def set(self, key: str, response: CachedResponse) -> bool:
        blob = response.to_blob()
        if len(blob) > self._max:
            _log.info(
                f"persist-skipped-oversize key={key} bytes={len(blob)} "
                f"max={self._max}"
            )
            return False
        try:
            self._client.set(key, blob, ex=self._ttl)
            return True
        except Exception as exc:
            _log.info(
                f"persist-failed key={key} bytes={len(blob)} "
                f"err={type(exc).__name__}: {exc}"
            )
            return False
