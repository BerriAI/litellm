from uvicorn.workers import UvicornWorker


def build_graceful_shutdown_uvicorn_worker(timeout_graceful_shutdown: float) -> type[UvicornWorker]:
    class GracefulShutdownUvicornWorker(UvicornWorker):
        def __init__(self, *args: object) -> None:
            self.CONFIG_KWARGS = {**UvicornWorker.CONFIG_KWARGS, "timeout_graceful_shutdown": timeout_graceful_shutdown}
            super().__init__(*args)

    return GracefulShutdownUvicornWorker
