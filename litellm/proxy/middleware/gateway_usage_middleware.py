"""
ASGI middleware that counts total and successful LLM-route requests for
request-based billing via ``LITELLM_USAGE_ENDPOINT``.

"Successful" means the gateway returned a non-5xx status code; 4xx
(auth errors, bad requests) are the caller's fault and still count as the
gateway having handled the request.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.proxy.usage_reporting.gateway_usage_reporter import record_request

_SUCCESS_THRESHOLD = 500


class GatewayUsageMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._tracked_prefixes = _build_tracked_prefixes()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._is_llm_route(scope):
            await self.app(scope, receive, send)
            return

        status_code: int = 0

        async def _send_wrapper(message: dict[str, object]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 0))
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapper)
        finally:
            record_request(succeeded=0 < status_code < _SUCCESS_THRESHOLD)

    def _is_llm_route(self, scope: Scope) -> bool:
        path: str = scope.get("path", "")
        return any(path.startswith(prefix) for prefix in self._tracked_prefixes)


def _build_tracked_prefixes() -> tuple[str, ...]:
    return (
        "/chat/completions",
        "/v1/chat/completions",
        "/completions",
        "/v1/completions",
        "/embeddings",
        "/v1/embeddings",
        "/images/generations",
        "/v1/images/generations",
        "/images/edits",
        "/v1/images/edits",
        "/audio/transcriptions",
        "/v1/audio/transcriptions",
        "/audio/speech",
        "/v1/audio/speech",
        "/moderations",
        "/v1/moderations",
        "/rerank",
        "/v1/rerank",
        "/v2/rerank",
        "/responses",
        "/v1/responses",
        "/realtime",
        "/v1/realtime",
        "/batches",
        "/v1/batches",
        "/fine_tuning",
        "/v1/fine_tuning",
        "/engines/",
        "/openai/deployments/",
        "/openai/v1/realtime",
        "/videos",
        "/v1/videos",
        "/ocr",
        "/v1/ocr",
        "/search",
        "/v1/search",
    )
