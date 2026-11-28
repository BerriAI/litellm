import sys
from typing import Final

from gunicorn.arbiter import Arbiter
from uvicorn.workers import UvicornWorker

from litellm.constants import DEFAULT_MAX_REQUEST_HEADER_SIZE
from litellm.proxy.shutdown.recycling_http_protocol import RecyclingHTTPProtocol
from litellm.proxy.shutdown.recycling_uvicorn_server import RecyclingUvicornServer


def build_graceful_shutdown_uvicorn_worker(
    timeout_graceful_shutdown: float,
    max_request_header_size: int | None = None,
    *,
    recycle_workers: bool = False,
) -> type[UvicornWorker]:
    class GracefulShutdownUvicornWorker(UvicornWorker):
        def __init__(self, *args: object) -> None:
            self.CONFIG_KWARGS = {
                **UvicornWorker.CONFIG_KWARGS,
                **({"http": RecyclingHTTPProtocol} if recycle_workers else {}),
                "timeout_graceful_shutdown": timeout_graceful_shutdown,
                "h11_max_incomplete_event_size": (
                    max_request_header_size if max_request_header_size is not None else DEFAULT_MAX_REQUEST_HEADER_SIZE
                ),
            }
            super().__init__(*args)

        async def _serve(self) -> None:
            if self.config.limit_max_requests in (None, sys.maxsize):
                await super()._serve()
                return

            self.config.app = self.wsgi
            server: Final = RecyclingUvicornServer(self.config)
            self._install_sigquit_handler()
            await server.serve(sockets=self.sockets)
            if not server.started:
                sys.exit(Arbiter.WORKER_BOOT_ERROR)

    return GracefulShutdownUvicornWorker
