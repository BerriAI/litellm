"""Gateway supervisor: assemble DATABASE_URL, start the in-container PgBouncer, then run uvicorn.

``gateway/main.py`` assembles ``DATABASE_URL`` inside every uvicorn worker, which
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
import sysconfig
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from typing_extensions import assert_never
from uvicorn.main import main as uvicorn_main

from litellm_cli.main import without_gc_pauses

if TYPE_CHECKING:
    from fastapi import FastAPI

    from litellm.proxy.db.db_url_settings import DatabaseURLSettings
    from litellm.proxy.db.pgbouncer import PgBouncerError, PgBouncerSettings
    from litellm.proxy.rust_gateway import Supervised

GATEWAY_APP: Final = "gateway.launch:worker_app"
KEEPALIVE_FLAG: Final = "--timeout-keep-alive"


def uvicorn_argv(argv: Sequence[str], environ: Mapping[str, str], supervised: "Supervised | None") -> tuple[str, ...]:
    """Honor ``KEEPALIVE_TIMEOUT`` like ``proxy_cli.py`` does, unless the flag was passed explicitly."""
    keepalive: Final = environ.get("KEEPALIVE_TIMEOUT")
    inherited_socket: Final = (
        (
            "--fd",
            str(supervised.listener_fd),
            "--no-proxy-headers",
        )
        if supervised is not None
        else ()
    )
    if keepalive is None or any(arg == KEEPALIVE_FLAG or arg.startswith(f"{KEEPALIVE_FLAG}=") for arg in argv):
        return (GATEWAY_APP, "--factory", *argv, *inherited_socket)
    return (GATEWAY_APP, "--factory", *argv, KEEPALIVE_FLAG, keepalive, *inherited_socket)


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


def worker_app() -> "FastAPI":
    with without_gc_pauses():
        from gateway.main import app
    return app


def _serve(argv: Sequence[str]) -> None:
    uvicorn_main(tuple(argv), prog_name="uvicorn")


def public_listener(argv: Sequence[str], binary: str | None) -> tuple[str, str, int, Mapping[str, str]]:
    """Validate the uvicorn flags for Rust forwarding and return the binary, public address and child environment."""
    from pydantic import TypeAdapter

    from litellm.proxy.rust_gateway import PROXY_HEADERS, TRUSTED_PROXIES, startup_error

    params: Final[Mapping[str, object]] = uvicorn_main.make_context(
        "uvicorn", list(uvicorn_argv(argv, os.environ, None))
    ).params
    error: Final = startup_error(
        binary,
        {
            "--ssl-keyfile": params["ssl_keyfile"],
            "--ssl-certfile": params["ssl_certfile"],
            "--ssl-ciphers": any(arg.split("=")[0] == "--ssl-ciphers" for arg in argv),
            "--uds": params["uds"] is not None,
            "--fd": params["fd"] is not None,
        },
    )
    if error is not None:
        sys.exit(error)
    assert binary is not None
    host, port = TypeAdapter(tuple[str, int]).validate_python((params["host"], params["port"]))
    operator_trust: Final = TypeAdapter(str | None).validate_python(params["forwarded_allow_ips"])
    proxy_headers: Final = TypeAdapter(bool).validate_python(params["proxy_headers"])
    return (
        binary,
        host,
        port,
        {PROXY_HEADERS: str(proxy_headers).lower()}
        | ({} if operator_trust is None else {TRUSTED_PROXIES: operator_trust}),
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


def main(argv: Sequence[str], serve: Callable[[Sequence[str]], None] = _serve) -> None:
    from litellm.proxy.rust_gateway import (
        Disabled,
        Supervised,
        Supervisor,
        gateway_binary,
        rust_gateway_mode,
        supervise,
    )

    mode: Final = rust_gateway_mode(os.environ)
    match mode:
        case Disabled():
            configure_database()
            serve(uvicorn_argv(argv, os.environ, None))
        case Supervisor():
            binary, host, port, trust = public_listener(argv, gateway_binary(sysconfig.get_path("scripts")))
            configure_database()
            command: Final = (sys.executable, "-m", "gateway.launch", *argv)
            sys.exit(supervise(binary, command, host, port, {**os.environ, **trust}))
        case Supervised():
            serve(uvicorn_argv(argv, os.environ, mode))
        case _:
            assert_never(mode)


if __name__ == "__main__":
    main(sys.argv[1:])
