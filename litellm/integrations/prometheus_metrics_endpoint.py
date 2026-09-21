"""ASGI app for `/metrics` that keeps registry rendering off the event loop.

``prometheus_client.make_asgi_app`` collects and serializes the whole registry
inline in the coroutine, so a large scrape (tens of MB on high cardinality
deployments) blocks every other request on the loop for its whole duration. This
app renders in a worker thread instead, shares one render across concurrent
scrapes that want the same output, and streams the payload back in chunks.
"""

from __future__ import annotations

import asyncio
import gzip
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from prometheus_client import CollectorRegistry
from prometheus_client.exposition import choose_encoder, gzip_accepted
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.types import ASGIApp, Receive, Scope, Send

RESPONSE_CHUNK_SIZE_BYTES: Final = 64 * 1024

_GZIP_HEADERS: Final = MappingProxyType({"Content-Encoding": "gzip"})


@dataclass(frozen=True, slots=True)
class ScrapeRequest:
    """What a scrape asks for, normalized so that header spellings sharing an output share a render."""

    encoder: Callable[[CollectorRegistry], bytes]
    content_type: str
    gzipped: bool
    metric_names: tuple[str, ...]


def parse_scrape_request(accept: str, accept_encoding: str, metric_names: tuple[str, ...]) -> ScrapeRequest:
    encoder, content_type = choose_encoder(accept)
    return ScrapeRequest(
        encoder=encoder,
        content_type=content_type,
        gzipped=gzip_accepted(accept_encoding),
        metric_names=metric_names,
    )


def render_scrape(registry: CollectorRegistry, request: ScrapeRequest) -> bytes:
    rendered: Final = request.encoder(
        registry.restricted_registry(request.metric_names) if request.metric_names else registry  # pyright: ignore[reportArgumentType]  # RestrictedRegistry is registry-shaped but not a subclass
    )
    return gzip.compress(rendered) if request.gzipped else rendered


class CoalescedScrapeRenderer:
    """Renders the registry in a worker thread, sharing one render per distinct output across concurrent scrapes."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self._registry = registry
        self._inflight: Mapping[ScrapeRequest, asyncio.Task[bytes]] = MappingProxyType({})

    def _forget(self, finished: asyncio.Task[bytes]) -> None:
        self._inflight = MappingProxyType({key: task for key, task in self._inflight.items() if task is not finished})

    async def render(self, request: ScrapeRequest) -> bytes:
        inflight: Final = self._inflight.get(request)
        if inflight is not None:
            return await asyncio.shield(inflight)

        task: Final = asyncio.create_task(asyncio.to_thread(render_scrape, self._registry, request))
        self._inflight = MappingProxyType({**self._inflight, request: task})
        task.add_done_callback(self._forget)
        return await asyncio.shield(task)


def _chunks(body: bytes) -> Iterator[bytes]:
    return (body[start : start + RESPONSE_CHUNK_SIZE_BYTES] for start in range(0, len(body), RESPONSE_CHUNK_SIZE_BYTES))


def make_metrics_asgi_app(registry: CollectorRegistry) -> ASGIApp:
    renderer: Final = CoalescedScrapeRenderer(registry)

    async def metrics_app(scope: Scope, receive: Receive, send: Send) -> None:
        request: Final = Request(scope, receive)
        scrape: Final = parse_scrape_request(
            accept=request.headers.get("accept", ""),
            accept_encoding=request.headers.get("accept-encoding", ""),
            metric_names=tuple(request.query_params.getlist("name[]")),
        )
        body: Final = await renderer.render(scrape)
        response: Final = StreamingResponse(
            _chunks(body),
            media_type=scrape.content_type,
            headers=_GZIP_HEADERS if scrape.gzipped else None,
        )
        await response(scope, receive, send)

    return metrics_app
