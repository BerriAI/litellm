import asyncio
import logging
import queue
import socket
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Final

import uvicorn
from starlette.types import ASGIApp


@contextmanager
def asgi_server(app: ASGIApp, *, before_stop: Callable[[], None] | None = None) -> Iterator[str]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port: Final = listener.getsockname()[1]
        server: Final = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                lifespan="on",
                log_level="warning",
                timeout_keep_alive=1,
                timeout_graceful_shutdown=5,
            )
        )
        errors: Final[queue.SimpleQueue[str]] = queue.SimpleQueue()
        loop_ready: Final[Future[asyncio.AbstractEventLoop]] = Future()

        def serve() -> None:
            with asyncio.Runner() as runner:
                loop_ready.set_result(runner.get_loop())
                try:
                    runner.run(server.serve(sockets=[listener]))
                except BaseException as error:
                    errors.put(type(error).__name__ + ": " + str(error))
                if asyncio.all_tasks(runner.get_loop()):
                    errors.put("Owned ASGI loop retained unfinished tasks")

        worker: Final = threading.Thread(target=serve)

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.thread == worker.ident and record.levelno >= logging.ERROR:
                    errors.put(self.format(record))

        handler: Final = Capture()
        logger: Final = logging.getLogger("uvicorn.error")
        logger.addHandler(handler)
        worker.start()
        try:
            deadline: Final = time.monotonic() + 8
            while not server.started:
                assert worker.is_alive() and time.monotonic() < deadline, "Owned ASGI peer failed readiness"
                time.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            if before_stop is not None:
                before_stop()
            server.should_exit = True
            worker.join(timeout=8)
            forced: Final = worker.is_alive()
            if forced:
                server.force_exit = True
                loop: Final = loop_ready.result(timeout=1)

                def cancel_owned() -> None:
                    for task in asyncio.all_tasks(loop):
                        task.cancel()

                loop.call_soon_threadsafe(cancel_owned)
                worker.join(timeout=3)
            logger.removeHandler(handler)
            assert not worker.is_alive(), "Owned ASGI peer survived forced cleanup"
            assert not forced, "Owned ASGI peer required forced cleanup"
            assert not server.server_state.tasks, "Owned ASGI peer retained request tasks"
            assert not server.lifespan.error_occurred and not server.lifespan.shutdown_failed
            assert errors.empty(), tuple(errors.get_nowait() for _ in range(errors.qsize()))
