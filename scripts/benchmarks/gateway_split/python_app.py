import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

import httpx
from fastapi import FastAPI, Request
from headers import clean_headers
from starlette.responses import Response, StreamingResponse

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

BACKEND: Final = os.environ.get("BENCH_BACKEND", "http://127.0.0.1:19400")
RUST: Final = os.environ.get("BENCH_RUST", "http://127.0.0.1:19401")
DISPATCH: Final = os.environ.get("BENCH_DISPATCH") == "1"
STREAM_REQUESTS: Final = os.environ.get("BENCH_STREAM_REQUESTS") == "1"


def create_app() -> FastAPI:
    handler: Final = AsyncHTTPHandler(timeout=httpx.Timeout(30.0, connect=1.0), follow_redirects=False)
    client: Final = handler.client

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        yield
        await client.aclose()

    app: Final = FastAPI(lifespan=lifespan)

    async def forward(request: Request, path: str) -> Response:
        data: Final = request.stream() if STREAM_REQUESTS else await request.body()
        if isinstance(data, bytes) and len(data) > 8 * 1024 * 1024:
            return Response(status_code=413)
        if int(request.headers.get("content-length", "0")) > 8 * 1024 * 1024:
            return Response(status_code=413)
        target: Final = (
            RUST if DISPATCH and path == "v1/messages" and "x-bench-python" not in request.headers else BACKEND
        )
        url: Final = f"{target}/{path}" + (f"?{request.url.query}" if request.url.query else "")
        upstream_request: Final = client.build_request(
            request.method,
            url,
            headers=clean_headers(httpx.Headers(request.headers)),
            content=data,
        )
        try:
            upstream: Final = await client.send(upstream_request, stream=True)
        except httpx.HTTPError:
            return Response(status_code=502)

        async def chunks() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            chunks(),
            status_code=upstream.status_code,
            headers={
                key: value for key, value in clean_headers(upstream.headers).items() if key not in ("date", "server")
            },
        )

    app.add_api_route("/{path:path}", forward, methods=["GET", "POST"])
    return app


app: Final = create_app()
