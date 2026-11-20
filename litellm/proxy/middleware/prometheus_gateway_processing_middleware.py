from collections.abc import Callable
from time import perf_counter
from typing import Final

from starlette.datastructures import Headers
from starlette.routing import get_route_path
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.gateway_processing import GatewayProcessingTiming, gateway_processing_window

_CHAT_PATHS: Final = frozenset({"/chat/completions", "/v1/chat/completions"})


class PrometheusGatewayProcessingMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        recorder: Callable[[], Callable[[float], None] | None],
    ) -> None:
        self.app: Final = app
        self.recorder: Final = recorder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        record: Final = self.recorder() if get_route_path(scope) in _CHAT_PATHS else None
        if record is None:
            await self.app(scope, receive, send)
            return

        timing: Final = GatewayProcessingTiming(perf_counter())

        async def send_response(message: Message) -> None:
            if message["type"] != "http.response.start" or _streams(message):
                await send(message)
                return
            processing: Final = timing.seconds(perf_counter())
            await send(message)
            if processing is not None:
                try:
                    record(processing)
                except Exception:  # noqa: BLE001  # an exporter failure cannot change the response
                    verbose_proxy_logger.debug("Could not record gateway processing time", exc_info=True)

        with gateway_processing_window(timing):
            await self.app(scope, receive, send_response)


def _streams(message: Message) -> bool:
    return Headers(scope=message).get("content-type", "").lower().startswith("text/event-stream")
