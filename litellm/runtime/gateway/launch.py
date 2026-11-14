"""Gateway supervisor: assemble DATABASE_URL, start the in-container PgBouncer, then run uvicorn.

``litellm/runtime/gateway/main.py`` assembles ``DATABASE_URL`` inside every uvicorn worker, which
is fine for a plain Postgres URL but not for the pooler: PgBouncer must be
started exactly once per pod, before the workers fork, and the workers must be
handed the loopback URL it listens on. A pre-existing ``DATABASE_URL`` wins in
``DatabaseURLSettings.apply_to_env`` under password auth, and one marked pooled
wins under token auth too, so exporting it here is enough for every worker to
pick the pooled URL up unchanged.

LiteLLM imports wait for ``main``: every uvicorn worker imports this module again
before it can answer its first 5 s health ping, so a slow host would kill each one.
Workers then load the gateway app through ``worker_app`` without garbage collection
pauses, which hold the GIL long enough to miss a ping.

Run with:
    python -m gateway.launch --workers 4 --host 0.0.0.0 --port 4000
"""

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from typing_extensions import assert_never
from uvicorn.main import main as uvicorn_main

if TYPE_CHECKING:
    from litellm.proxy.db.db_url_settings import DatabaseURLSettings
    from litellm.proxy.db.pgbouncer import PgBouncerError, PgBouncerSettings
    from litellm.proxy.rust_forward.peer import ASGIApp
    from litellm.proxy.rust_forward.server import PythonServer

GATEWAY_APP: Final = "gateway.launch:worker_app"
KEEPALIVE_FLAG: Final = "--timeout-keep-alive"


def uvicorn_argv(argv: Sequence[str], environ: Mapping[str, str]) -> tuple[str, ...]:
    """Honor ``KEEPALIVE_TIMEOUT`` like ``proxy_cli.py`` does, unless the flag was passed explicitly."""
    keepalive: Final = environ.get("KEEPALIVE_TIMEOUT")
    if keepalive is None or any(arg == KEEPALIVE_FLAG or arg.startswith(f"{KEEPALIVE_FLAG}=") for arg in argv):
        return (GATEWAY_APP, "--factory", *argv)
    return (GATEWAY_APP, "--factory", *argv, KEEPALIVE_FLAG, keepalive)


def pool_database_url(
    settings: "DatabaseURLSettings",
    pgbouncer: "PgBouncerSettings",
    environ: Mapping[str, str],
) -> "str | PgBouncerError | None":
    """Start the in-container PgBouncer and return its loopback URL, or None when ``pgbouncer.enabled`` is off.

    The upstream URL is whatever ``apply_to_env`` assembled from the discrete
    ``DATABASE_*`` vars (or an operator-pinned ``DATABASE_URL``). Under token
    auth the pooler mints and renews the upstream token itself.
    """
    from litellm.proxy.db.pgbouncer import PgBouncerError, start_in_container_pgbouncer

    if not pgbouncer.enabled:
        return None
    upstream_url: Final = environ.get("DATABASE_URL")
    if upstream_url is None:
        return PgBouncerError("LITELLM_PGBOUNCER_ENABLED is set but no DATABASE_URL could be assembled")
    return start_in_container_pgbouncer(pgbouncer, upstream_url, token_auth=settings.token_auth())


def worker_app() -> "ASGIApp":
    from litellm_cli.main import without_gc_pauses

    with without_gc_pauses():
        from gateway.main import app
        from litellm.proxy.rust_forward.mode import rust_forward_mode
        from litellm.proxy.rust_forward.peer import wrap
    return wrap(app, rust_forward_mode(os.environ))


def _serve(argv: Sequence[str]) -> None:
    uvicorn_main(tuple(argv), prog_name="uvicorn")


def python_server(argv: Sequence[str]) -> "PythonServer":
    """What uvicorn would serve the public port with, read from the same flags and environment it reads."""
    from litellm.proxy.rust_forward.server import PythonServer, Server, UvicornCliParams
    from litellm.proxy.shutdown.graceful_shutdown_manager import GracefulShutdownManager

    params: Final = UvicornCliParams.model_validate(
        uvicorn_main.make_context("uvicorn", list(uvicorn_argv(argv, os.environ))).params
    )
    return PythonServer(
        server=Server.UVICORN,
        listener=params.listener,
        host=params.host,
        port=params.port,
        keepalive_seconds=params.timeout_keep_alive,
        graceful_timeout_seconds=GracefulShutdownManager.get_timeout()
        if params.timeout_graceful_shutdown is None
        else params.timeout_graceful_shutdown,
        trusted_proxies=params.trusted_proxies,
    )


def configure_database() -> None:
    from litellm.proxy.db.db_url_settings import DatabaseURLSettings
    from litellm.proxy.db.pgbouncer import PgBouncerError, PgBouncerSettings, export_pooled_database_url

    settings: Final = DatabaseURLSettings.from_env()
    settings.apply_to_env()
    pooled_url: Final = pool_database_url(settings, PgBouncerSettings(), os.environ)
    if isinstance(pooled_url, PgBouncerError):
        sys.exit(f"LiteLLM gateway: in-container pgbouncer could not start: {pooled_url.reason}")
    if pooled_url is not None:
        export_pooled_database_url(pooled_url)


def _serve_python_alone(argv: Sequence[str], serve: Callable[[Sequence[str]], None]) -> None:
    configure_database()
    serve(uvicorn_argv(argv, os.environ))


def _python_server_or_exit(argv: Sequence[str]) -> "PythonServer":
    """Report ``--help``, ``--version`` and a bad flag the way uvicorn's own CLI does, since it never gets to."""
    import click

    try:
        return python_server(argv)
    except click.exceptions.Exit as stopped:
        sys.exit(stopped.exit_code)
    except click.ClickException as error:
        error.show()
        sys.exit(error.exit_code)


def main(argv: Sequence[str], serve: Callable[[Sequence[str]], None] = _serve) -> None:
    from pydantic import ValidationError

    from litellm.proxy.rust_forward.mode import (
        Mode,
        Supervised,
        environment_error,
        rust_forward_mode,
        stop_with_supervisor,
        uvicorn_flags,
    )
    from litellm.proxy.rust_forward.supervisor import supervise
    from litellm.proxy.rust_forward.supervisor.preflight import StartupError

    try:
        mode: Final = rust_forward_mode(os.environ)
    except ValidationError as error:
        sys.exit(environment_error(error))
    match mode:
        case Mode.DISABLED:
            _serve_python_alone(argv, serve)
        case Mode.SUPERVISOR:
            supervised: Final = supervise(_python_server_or_exit(argv), before_start=configure_database)
            if isinstance(supervised, StartupError):
                _serve_python_alone(argv, serve)
            else:
                sys.exit(supervised)
        case Supervised():
            stop_with_supervisor(mode)
            serve((*uvicorn_argv(argv, os.environ), *uvicorn_flags(mode)))
        case _:
            assert_never(mode)


if __name__ == "__main__":
    main(sys.argv[1:])
