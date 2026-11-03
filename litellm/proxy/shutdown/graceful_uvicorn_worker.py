from uvicorn.workers import UvicornWorker

from litellm.constants import DEFAULT_MAX_REQUEST_HEADER_SIZE


def build_graceful_shutdown_uvicorn_worker(
    timeout_graceful_shutdown: float,
    max_request_header_size: int | None = None,
) -> type[UvicornWorker]:
    class GracefulShutdownUvicornWorker(UvicornWorker):
        def __init__(self, *args: object) -> None:
            self.CONFIG_KWARGS = {
                **UvicornWorker.CONFIG_KWARGS,
                "timeout_graceful_shutdown": timeout_graceful_shutdown,
                "h11_max_incomplete_event_size": (
                    max_request_header_size if max_request_header_size is not None else DEFAULT_MAX_REQUEST_HEADER_SIZE
                ),
            }
            super().__init__(*args)

    return GracefulShutdownUvicornWorker
