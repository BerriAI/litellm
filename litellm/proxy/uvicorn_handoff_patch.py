"""
Monkey-patch Uvicorn to record when a request is ready (headers complete)
so LiteLLM middleware can measure "Uvicorn receive → app entry" latency.

Applied when litellm.proxy.proxy_server is imported (so it runs in the process
that serves requests, including gunicorn workers). Sets scope["_uvicorn_received_at"]
(perf_counter) in both httptools and h11 protocol paths when the request
scope is built and before the ASGI app is invoked.
"""
from __future__ import annotations

import time


def _patch_httptools_impl() -> None:
    import uvicorn.protocols.http.httptools_impl as m  # noqa: PLC0415

    _orig_on_headers_complete = m.HttpToolsProtocol.on_headers_complete

    def on_headers_complete(self: m.HttpToolsProtocol) -> None:
        _orig_on_headers_complete(self)
        # Scope is the same dict passed to RequestResponseCycle; stamp it so
        # middleware can measure receive → app entry (timestamp is just after
        # request was ready and task was scheduled).
        self.scope["_uvicorn_received_at"] = time.perf_counter()  # type: ignore[typeddict-unknown-key]

    m.HttpToolsProtocol.on_headers_complete = on_headers_complete  # type: ignore[method-assign]


def _patch_h11_impl() -> None:
    import uvicorn.protocols.http.h11_impl as m  # noqa: PLC0415

    # In h11, scope is built inside data_received; we stamp at start of run_asgi
    # so "received" = when ASGI task began (slightly later than headers-complete).
    _orig_run_asgi = m.RequestResponseCycle.run_asgi

    async def run_asgi(self: m.RequestResponseCycle, app: m.ASGI3Application) -> None:
        if "_uvicorn_received_at" not in self.scope:
            self.scope["_uvicorn_received_at"] = time.perf_counter()  # type: ignore[typeddict-unknown-key]
        await _orig_run_asgi(self, app)

    m.RequestResponseCycle.run_asgi = run_asgi  # type: ignore[method-assign]


def apply_uvicorn_handoff_patch() -> None:
    """Apply patches so scope gets _uvicorn_received_at for handoff timing."""
    _patch_httptools_impl()
    _patch_h11_impl()
