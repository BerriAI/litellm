import importlib
import logging
import multiprocessing
import os
import signal
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from inspect import signature
from multiprocessing.process import BaseProcess
from socket import socket
from typing import Final, Literal

from pydantic import ConfigDict, ImportString, JsonValue, TypeAdapter, with_config
from typing_extensions import ReadOnly, Required, TypedDict
from uvicorn import Config, Server
from uvicorn import config as uvicorn_config
from uvicorn.supervisors import ChangeReload, Multiprocess


@with_config(ConfigDict(extra="allow"))
class ServerOptions(TypedDict, total=False):
    host: ReadOnly[str]
    port: ReadOnly[int]
    server_header: ReadOnly[bool]
    timeout_graceful_shutdown: ReadOnly[float]
    log_config: ReadOnly[  # mutable-ok: uvicorn.Config requires a concrete logging dict
        str | dict[str, JsonValue] | None
    ]
    log_level: ReadOnly[str | int]
    timeout_keep_alive: ReadOnly[int]
    timeout_worker_healthcheck: ReadOnly[int]
    limit_max_requests: ReadOnly[int]
    limit_max_requests_jitter: ReadOnly[int]
    limit_concurrency: ReadOnly[int]
    ssl_keyfile: ReadOnly[str]
    ssl_certfile: ReadOnly[str]
    loop: ReadOnly[Literal["none", "auto", "asyncio", "uvloop"]]


@with_config(ConfigDict(extra="allow"))
class ReloadOptions(ServerOptions):
    app: Required[ReadOnly[str]]
    reload: ReadOnly[bool]
    reload_dirs: ReadOnly[tuple[str, ...]]
    reload_includes: ReadOnly[tuple[str, ...]]


RELOAD_OPTIONS: Final = TypeAdapter(ReloadOptions)
SERVER_OPTIONS: Final = TypeAdapter(ServerOptions)
ASGI_APP: Final = TypeAdapter(ImportString[Callable[..., object]])
LOGGER: Final = logging.getLogger("uvicorn.error")
RELOAD_WORKER_HEALTHCHECK_SECONDS: Final = 30
RELOADER_ONLY_KEYS: Final = frozenset(ReloadOptions.__annotations__) - frozenset(ServerOptions.__annotations__)


def multi_worker_reload_supported(platform: str = sys.platform) -> bool:
    return platform != "win32" and hasattr(uvicorn_config, "STARTUP_FAILURE")


def server_options(options: ReloadOptions) -> ServerOptions:
    return SERVER_OPTIONS.validate_python(
        {key: value for key, value in options.items() if key not in RELOADER_ONLY_KEYS}
    )


def load_app_or_exit(import_path: str) -> Callable[..., object]:
    module_name: Final = import_path.partition(":")[0]
    try:
        importlib.import_module(module_name)
        return ASGI_APP.validate_python(import_path)
    except Exception:
        LOGGER.exception("Error loading ASGI app %s", import_path)
        sys.exit(uvicorn_config.STARTUP_FAILURE)


def stop_worker_when_pool_exits(pool_parent: BaseProcess) -> None:
    pool_parent.join()
    LOGGER.info("Worker pool process [%s] exited, stopping worker [%s].", pool_parent.pid, os.getpid())
    os.kill(os.getpid(), signal.SIGTERM)


def load_worker_app(import_path: str, worker_app_factory: bool) -> Callable[..., object]:
    pool_parent: Final = multiprocessing.parent_process()
    if pool_parent is not None:
        threading.Thread(target=stop_worker_when_pool_exits, args=(pool_parent,), daemon=True).start()
    worker_app: Final = load_app_or_exit(import_path)
    return ASGI_APP.validate_python(worker_app()) if worker_app_factory else worker_app


def pool_config(options: ReloadOptions, workers: int, worker_overrides: Mapping[str, object] | None = None) -> Config:
    overrides: Final = worker_overrides or {}
    worker_app: Final = TypeAdapter(str).validate_python(overrides.get("app", options["app"]))
    worker_app_factory: Final = TypeAdapter(bool).validate_python(overrides.get("factory", False))
    return Config(
        partial(load_worker_app, worker_app, worker_app_factory),
        factory=True,
        workers=workers,
        **SERVER_OPTIONS.validate_python(
            {
                "timeout_worker_healthcheck": RELOAD_WORKER_HEALTHCHECK_SECONDS,
                **server_options(options),
                **{key: value for key, value in overrides.items() if key not in ("app", "factory")},
            }
        ),  # pyright: ignore[reportArgumentType]  # uvicorn types the float graceful shutdown timeout as int
    )


def stop_pool_when_reloader_exits(reloader: BaseProcess, pool_should_exit: threading.Event) -> None:
    reloader.join()
    LOGGER.info("Reloader process [%s] exited, stopping the worker pool.", reloader.pid)
    pool_should_exit.set()


class ReloadWorkerPool(Multiprocess):
    quit_received: bool = False

    def handle_quit(self) -> None:
        LOGGER.info("Received SIGQUIT, exiting.")
        self.quit_received = True
        self.should_exit.set()

    def terminate_all(self) -> None:
        quit_signal: Final = getattr(signal, "SIGQUIT", None)
        if quit_signal is not None and quit_signal in self.signal_queue:
            self.handle_quit()
        if not self.quit_received:
            super().terminate_all()
            return
        for process in self.processes:
            process.kill()


def _run_worker_pool(config: Config, sockets: Sequence[socket] | None) -> None:
    shared_sockets: Final = list(sockets or ())
    pool: Final = (
        ReloadWorkerPool(
            config=config,
            target=Server(config=config).run,  # pyright: ignore[reportCallIssue]  # Older uvicorn versions require target
            sockets=shared_sockets,
        )
        if "target" in signature(Multiprocess).parameters
        else ReloadWorkerPool(config=config, sockets=shared_sockets)
    )
    reloader: Final = multiprocessing.parent_process()
    if reloader is not None:
        threading.Thread(target=stop_pool_when_reloader_exits, args=(reloader, pool.should_exit), daemon=True).start()
    pool.run()


def run_multi_worker_reload(
    options: ReloadOptions, workers: int, worker_overrides: Mapping[str, object] | None = None
) -> None:
    worker_config: Final = pool_config(options, workers, worker_overrides)
    reload_config: Final = Config(
        options["app"],
        workers=1,
        reload=options["reload"],
        reload_dirs=list(options["reload_dirs"]),
        reload_includes=list(options["reload_includes"]),
        **server_options(options),  # pyright: ignore[reportArgumentType]  # uvicorn types the float graceful shutdown timeout as int
    )
    shared_socket: Final = worker_config.bind_socket()
    try:
        ChangeReload(
            config=reload_config,
            target=partial(_run_worker_pool, worker_config),
            sockets=[shared_socket],
        ).run()
    finally:
        shared_socket.close()
