from collections.abc import Sequence
from functools import partial
from inspect import signature
from socket import socket
from typing import Final, Literal

from pydantic import ConfigDict, JsonValue, TypeAdapter, with_config
from typing_extensions import ReadOnly, Required, TypedDict
from uvicorn import Config, Server
from uvicorn.supervisors import ChangeReload, Multiprocess


class WorkerOptions(TypedDict, total=False):
    app: Required[ReadOnly[str]]
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
class ReloadOptions(WorkerOptions):
    reload: ReadOnly[bool]
    reload_dirs: ReadOnly[tuple[str, ...]]
    reload_includes: ReadOnly[tuple[str, ...]]


RELOAD_OPTIONS: Final = TypeAdapter(ReloadOptions)
WORKER_OPTIONS: Final = TypeAdapter(WorkerOptions)


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
    pool.run()


def run_multi_worker_reload(options: ReloadOptions, workers: int) -> None:
    worker_options: Final = WORKER_OPTIONS.validate_python(options)
    worker_config: Final = Config(**worker_options, workers=workers)
    reload_config: Final = Config(
        **worker_options,
        workers=1,
        reload=options["reload"],
        reload_dirs=list(options["reload_dirs"]),
        reload_includes=list(options["reload_includes"]),
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
