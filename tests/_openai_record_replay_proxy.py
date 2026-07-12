"""Record/replay reverse proxy for the dockerized real-provider spend E2Es.

Several E2E tests run the litellm proxy in its own container and curl it over
real HTTP, then assert on spend, cost, or rerank output. Those calls reach real
provider APIs (OpenAI image gen and chat, Cohere rerank, Anthropic messages),
so every commit run paid for them and was exposed to provider outages (the 401
that started this).

This process sits between the proxy and the provider. A model points its
``api_base`` here; nothing else about the topology changes. The first request
(or the first after a recording lapses) is forwarded live to the provider and
recorded; subsequent identical requests within the TTL replay the recorded
response, so the per-commit run no longer depends on the provider being up.

One recorder fronts every provider. The default upstream is api.openai.com; a
non-OpenAI model points its ``api_base`` at ``/__recorder_upstream/<host>`` so
the recorder forwards to ``https://<host>`` (folded into the cache key so two
providers sharing a path can't collide). Routing rides ``api_base`` because
some provider handlers drop custom request headers.

Recordings live in the same Redis cassette store as the VCR persister
(``CASSETTE_REDIS_URL``) and expire ``CASSETTE_TTL_SECONDS`` after their last
write, never refreshed on read. A recording therefore goes stale a day after
capture and the next run past that point re-records live and catches provider
contract drift, exactly matching the lapse-after-write contract in
``tests/_vcr_redis_persister.py``.

The process logs its mode at startup (REPLAY when the cassette redis is
reachable, PASSTHROUGH or DEGRADED otherwise) and a HIT/MISS line per request,
so a CI run shows whether it served from the cassette or went live instead of
silently degrading.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from typing import Awaitable, Callable, List, Optional, Tuple

_LOGGER = logging.getLogger("openai_record_replay")
_LOGGER.setLevel(logging.INFO)

CASSETTE_TTL_SECONDS = 24 * 60 * 60
RECORD_KEY_PREFIX = "litellm:openai:record:"
RECORDER_REDIS_URL_ENV = "CASSETTE_REDIS_URL"
UPSTREAM_BASE_URL_ENV = "RECORDER_UPSTREAM_BASE_URL"
DEFAULT_UPSTREAM_BASE_URL = "https://api.openai.com"
# One recorder fronts many providers. A non-default provider is addressed by
# prefixing the request path with ``/__recorder_upstream/<host>/`` via the
# model's ``api_base``. This rides ``api_base`` (which every litellm provider
# honours) rather than a custom header (which some provider handlers, e.g.
# cohere rerank, silently drop).
UPSTREAM_PATH_PREFIX = "/__recorder_upstream/"

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


def _resolve_upstream(path: str, default_upstream: str) -> Tuple[str, str]:
    """Map an incoming request path to ``(upstream_base_url, real_path)``.

    A path under ``/__recorder_upstream/<host>/...`` targets that provider; any
    other path goes to the default upstream unchanged.
    """
    if path.startswith(UPSTREAM_PATH_PREFIX):
        host, _, rest = path[len(UPSTREAM_PATH_PREFIX) :].partition("/")
        return f"https://{host}", f"/{rest}"
    return default_upstream, path


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
    """Record-once / replay-from-Redis for upstream provider HTTP calls.

    ``redis_client`` is injected so the process wiring and the tests share one
    code path; pass ``None`` to run as a pure live passthrough (local dev with
    no cassette Redis). ``upstream_base_url`` is the default provider; per
    request it can be overridden by a ``/__recorder_upstream/<host>/`` path.
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
    def record_key(
        method: str,
        path: str,
        body: bytes,
        upstream_base_url: str = DEFAULT_UPSTREAM_BASE_URL,
    ) -> str:
        digest = hashlib.sha256(
            b"\n".join(
                [
                    upstream_base_url.rstrip("/").encode("utf-8"),
                    method.upper().encode("utf-8"),
                    path.encode("utf-8"),
                    _canonical_body(body),
                ]
            )
        ).hexdigest()
        return f"{RECORD_KEY_PREFIX}{digest}"

    async def handle(
        self,
        method: str,
        path: str,
        body: bytes,
        fetch_upstream: FetchUpstream,
        *,
        upstream_base_url: Optional[str] = None,
    ) -> UpstreamResult:
        key = self.record_key(
            method, path, body, upstream_base_url or self.upstream_base_url
        )
        cached = self._cache_get(key)
        if cached is not None:
            _LOGGER.info("HIT replayed from cassette: %s %s", method, path)
            return cached

        status, headers, resp_body = await fetch_upstream()
        sanitized = _sanitize_headers(headers)
        if not (200 <= status < 300):
            _LOGGER.info(
                "MISS forwarded live, not cached (status=%s): %s %s",
                status,
                method,
                path,
            )
        elif self._cache_set(key, status, sanitized, resp_body):
            _LOGGER.info("MISS forwarded live and recorded: %s %s", method, path)
        else:
            _LOGGER.warning(
                "MISS forwarded live but NOT recorded (redis unset or unreachable): %s %s",
                method,
                path,
            )
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

    def _cache_set(self, key: str, status: int, headers: Headers, body: bytes) -> bool:
        if self._redis is None:
            return False
        payload = json.dumps(
            {
                "status": status,
                "headers": [[k, v] for (k, v) in headers],
                "body_b64": base64.b64encode(body).decode("ascii"),
            }
        )
        try:
            self._redis.set(key, payload, ex=self._ttl_seconds)
            return True
        except Exception:
            return False

    def log_startup_mode(self) -> None:
        if self._redis is None:
            _LOGGER.warning(
                "PASSTHROUGH: %s unset, every request goes live to %s and nothing is cached",
                RECORDER_REDIS_URL_ENV,
                self.upstream_base_url,
            )
            return
        try:
            self._redis.ping()
        except Exception as exc:
            _LOGGER.warning(
                "DEGRADED to live: %s set but cassette redis unreachable (%s); nothing is cached",
                RECORDER_REDIS_URL_ENV,
                type(exc).__name__,
            )
            return
        _LOGGER.info(
            "REPLAY mode: cassette redis reachable, recordings expire %ss after write (no refresh on read)",
            self._ttl_seconds,
        )


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
    import contextlib

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
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        recorder.log_startup_mode()
        try:
            yield
        finally:
            if owns_client:
                await client.aclose()

    async def health(_request):
        return PlainTextResponse("ok")

    async def proxy(request):
        body = await request.body()
        upstream_base_url, real_path = _resolve_upstream(
            request.url.path, recorder.upstream_base_url
        )
        upstream_base_url = upstream_base_url.rstrip("/")
        full_path = (
            f"{real_path}?{request.url.query}" if request.url.query else real_path
        )

        async def fetch_upstream() -> UpstreamResult:
            fwd_headers = {
                k: v for k, v in request.headers.items() if k.lower() != "host"
            }
            upstream = await client.request(
                request.method,
                f"{upstream_base_url}{full_path}",
                content=body,
                headers=fwd_headers,
            )
            return (
                upstream.status_code,
                list(upstream.headers.items()),
                upstream.content,
            )

        status, headers, resp_body = await recorder.handle(
            request.method,
            full_path,
            body,
            fetch_upstream,
            upstream_base_url=upstream_base_url,
        )
        return Response(content=resp_body, status_code=status, headers=dict(headers))

    return Starlette(
        routes=[
            Route("/__recorder_health", health, methods=["GET"]),
            Route(
                "/{path:path}", proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
            ),
        ],
        lifespan=lifespan,
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)
