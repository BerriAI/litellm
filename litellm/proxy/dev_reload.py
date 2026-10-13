import importlib
import logging
import os
import sys
import threading
from collections.abc import Callable, Sequence
from functools import partial
from inspect import signature
from socket import socket
from typing import Final, Literal

from pydantic import ConfigDict, ImportString, JsonValue, TypeAdapter, with_config
from typing_extensions import ReadOnly, Required, TypedDict
from uvicorn import Config, Server
from uvicorn import config as uvicorn_config
from uvicorn.supervisors import ChangeReload, Multiprocess


class ServerOptions(TypedDict, total=False):
    host: ReadOnly[str]
    port: ReadOnly[int]
    server_header: ReadOnly[bool]
    log_config: ReadOnly[str | dict[str, JsonValue]]  # mutable-ok: uvicorn.Config requires a concrete logging dict
    log_level: ReadOnly[str | int]
    timeout_keep_alive: ReadOnly[int]
    timeout_worker_healthcheck: ReadOnly[int]
    limit_max_requests: ReadOnly[int]
    limit_max_requests_jitter: ReadOnly[int]
    limit_concurrency: ReadOnly[int]
    ssl_keyfile: ReadOnly[str]
    ssl_certfile: ReadOnly[str]
    loop: ReadOnly[Literal["none", "auto", "asyncio", "uvloop"]]


@with_config(ConfigDict(extra="forbid"))
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
RELOADER_CHECK_SECONDS: Final = 0.5


def multi_worker_reload_supported() -> bool:
    return hasattr(uvicorn_config, "STARTUP_FAILURE")


def load_app_or_exit(import_path: str) -> Callable[..., object]:
    module_name: Final = import_path.partition(":")[0]
    try:
        importlib.import_module(module_name)
        return ASGI_APP.validate_python(import_path)
    except Exception:
        LOGGER.exception("Error loading ASGI app %s", import_path)
        sys.exit(uvicorn_config.STARTUP_FAILURE)


def pool_config(options: ReloadOptions, workers: int) -> Config:
    return Config(
        partial(load_app_or_exit, options["app"]),
        factory=True,
        workers=workers,
        **SERVER_OPTIONS.validate_python({"timeout_worker_healthcheck": RELOAD_WORKER_HEALTHCHECK_SECONDS, **options}),
    )


def stop_pool_when_reloader_exits(pool: Multiprocess, reloader_pid: int) -> None:
    while not pool.should_exit.wait(RELOADER_CHECK_SECONDS):
        if os.getppid() != reloader_pid:
            LOGGER.info("Reloader process [%d] exited, stopping the worker pool.", reloader_pid)
            pool.should_exit.set()


def _run_worker_pool(config: Config, sockets: Sequence[socket] | None) -> None:
    shared_sockets: Final = list(sockets or ())
    pool: Final = (
        Multiprocess(
            config=config,
            target=Server(config=config).run,  # pyright: ignore[reportCallIssue]  # Older uvicorn versions require target
            sockets=shared_sockets,
        )
        if "target" in signature(Multiprocess).parameters
        else Multiprocess(config=config, sockets=shared_sockets)
    )
    threading.Thread(target=stop_pool_when_reloader_exits, args=(pool, os.getppid()), daemon=True).start()
    pool.run()


def run_multi_worker_reload(options: ReloadOptions, workers: int) -> None:
    worker_config: Final = pool_config(options, workers)
    reload_config: Final = Config(
        options["app"],
        workers=1,
        reload=options["reload"],
        reload_dirs=list(options["reload_dirs"]),
        reload_includes=list(options["reload_includes"]),
        **SERVER_OPTIONS.validate_python(options),
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
