"""Record/replay reverse proxy for the dockerized image-gen spend E2E.

The spend-accuracy test ``tests/test_keys.py::
test_key_info_spend_values_image_generation`` runs the litellm proxy in its own
container and curls it over real HTTP, then asserts the proxy tracked a nonzero
spend for a ``gpt-image-1`` call. That call wildcard-routes to ``openai/*`` on
the real key, so every commit run hit api.openai.com for a paid image and was
exposed to OpenAI outages (the 401 that started this).

This process sits between the proxy and api.openai.com. The proxy points only
its image model's ``api_base`` here; nothing else about the topology changes.
The first request (or the first after a recording lapses) is forwarded live to
OpenAI and recorded; subsequent requests within the TTL replay the recorded
response, so the per-commit run no longer depends on OpenAI being up.

Recordings live in the same Redis cassette store as the VCR persister
(``CASSETTE_REDIS_URL``) and expire ``CASSETTE_TTL_SECONDS`` after their last
write, never refreshed on read. A recording therefore goes stale a day after
capture and the next run past that point re-records live and catches provider
contract drift, exactly matching the lapse-after-write contract in
``tests/_vcr_redis_persister.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Awaitable, Callable, List, Optional, Tuple

CASSETTE_TTL_SECONDS = 24 * 60 * 60
RECORD_KEY_PREFIX = "litellm:openai:record:"
RECORDER_REDIS_URL_ENV = "CASSETTE_REDIS_URL"
UPSTREAM_BASE_URL_ENV = "RECORDER_UPSTREAM_BASE_URL"
DEFAULT_UPSTREAM_BASE_URL = "https://api.openai.com"

Headers = List[Tuple[str, str]]
UpstreamResult = Tuple[int, Headers, bytes]
FetchUpstream = Callable[[], Awaitable[UpstreamResult]]

# Headers the re-serving layer owns and must set itself. Replaying an upstream
# framing header verbatim onto a freshly built response is the same class of bug
# as the Bedrock content-length: 0 regression (#29549): a stale header rides
# along and contradicts the real body. The serving server recomputes
# content-length and sets its own date/server; the stored body is already
# content-decoded so content-encoding must not claim otherwise.
_STRIPPED_RESPONSE_HEADERS = frozenset(
    {
        "content-length",
        "content-encoding",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "upgrade",
        "date",
        "server",
    }
)


def _canonical_body(body: bytes) -> bytes:
    if not body:
        return b""
    try:
        return json.dumps(
            json.loads(body), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (ValueError, TypeError):
        return body


def _sanitize_headers(headers: Headers) -> Headers:
    return [(k, v) for (k, v) in headers if k.lower() not in _STRIPPED_RESPONSE_HEADERS]


class OpenAIRecordReplay:
    """Record-once / replay-from-Redis for upstream OpenAI HTTP calls.

    ``redis_client`` is injected so the process wiring and the tests share one
    code path; pass ``None`` to run as a pure live passthrough (local dev with
    no cassette Redis).
    """

    def __init__(
        self,
        redis_client,
        *,
        upstream_base_url: str = DEFAULT_UPSTREAM_BASE_URL,
        ttl_seconds: int = CASSETTE_TTL_SECONDS,
    ) -> None:
        self._redis = redis_client
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def record_key(method: str, path: str, body: bytes) -> str:
        digest = hashlib.sha256(
            b"\n".join(
                [
                    method.upper().encode("utf-8"),
                    path.encode("utf-8"),
                    _canonical_body(body),
                ]
            )
        ).hexdigest()
        return f"{RECORD_KEY_PREFIX}{digest}"

    async def handle(
        self, method: str, path: str, body: bytes, fetch_upstream: FetchUpstream
    ) -> UpstreamResult:
        key = self.record_key(method, path, body)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        status, headers, resp_body = await fetch_upstream()
        sanitized = _sanitize_headers(headers)
        if 200 <= status < 300:
            self._cache_set(key, status, sanitized, resp_body)
        return status, sanitized, resp_body

    def _cache_get(self, key: str) -> Optional[UpstreamResult]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
        except Exception:
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
            status = int(payload["status"])
            headers = [(str(k), str(v)) for k, v in payload["headers"]]
            resp_body = base64.b64decode(payload["body_b64"])
        except Exception:
            return None
        return status, headers, resp_body

    def _cache_set(self, key: str, status: int, headers: Headers, body: bytes) -> None:
        if self._redis is None:
            return
        payload = json.dumps(
            {
                "status": status,
                "headers": [[k, v] for (k, v) in headers],
                "body_b64": base64.b64encode(body).decode("ascii"),
            }
        )
        try:
            self._redis.set(key, payload, ex=self._ttl_seconds)
        except Exception:
            pass


def _build_default_redis_client():
    url = os.environ.get(RECORDER_REDIS_URL_ENV)
    if not url:
        return None
    import redis

    return redis.Redis.from_url(
        url,
        socket_timeout=5,
        socket_connect_timeout=5,
        decode_responses=False,
    )


def create_app(recorder: Optional[OpenAIRecordReplay] = None, http_client=None):
    import httpx
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse, Response
    from starlette.routing import Route

    if recorder is None:
        recorder = OpenAIRecordReplay(
            redis_client=_build_default_redis_client(),
            upstream_base_url=os.environ.get(
                UPSTREAM_BASE_URL_ENV, DEFAULT_UPSTREAM_BASE_URL
            ),
        )
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def health(_request):
        return PlainTextResponse("ok")

    async def proxy(request):
        body = await request.body()
        path = request.url.path
        full_path = f"{path}?{request.url.query}" if request.url.query else path

        async def fetch_upstream() -> UpstreamResult:
            fwd_headers = {
                k: v for k, v in request.headers.items() if k.lower() != "host"
            }
            upstream = await client.request(
                request.method,
                f"{recorder.upstream_base_url}{full_path}",
                content=body,
                headers=fwd_headers,
            )
            return (
                upstream.status_code,
                list(upstream.headers.items()),
                upstream.content,
            )

        status, headers, resp_body = await recorder.handle(
            request.method, full_path, body, fetch_upstream
        )
        return Response(content=resp_body, status_code=status, headers=dict(headers))

    return Starlette(
        routes=[
            Route("/__recorder_health", health, methods=["GET"]),
            Route(
                "/{path:path}", proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
            ),
        ]
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)
