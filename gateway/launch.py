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
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

import click
from uvicorn.main import main as uvicorn_main

from litellm_cli.main import without_gc_pauses

if TYPE_CHECKING:
    from fastapi import FastAPI

    from litellm.proxy.db.db_url_settings import DatabaseURLSettings
    from litellm.proxy.db.pgbouncer import PgBouncerError, PgBouncerSettings

GATEWAY_APP: Final = "gateway.launch:worker_app"
KEEPALIVE_FLAG: Final = "--timeout-keep-alive"


def uvicorn_argv(argv: Sequence[str], environ: Mapping[str, str]) -> tuple[str, ...]:
    """Honor ``KEEPALIVE_TIMEOUT`` like ``proxy_cli.py`` does, unless the flag was passed explicitly."""
    keepalive: Final = environ.get("KEEPALIVE_TIMEOUT")
    fd: Final = environ.get("_LITELLM_GATEWAY_FD")
    inherited_socket: Final = (
        ("--fd", fd, "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1") if fd is not None else ()
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


def forward_listener(argv: Sequence[str]) -> tuple[str, str, int] | None:
    if not (
        "--rust_gateway" in argv or click.BOOL.convert(os.environ.get("LITELLM_RUST_GATEWAY", "false"), None, None)
    ):
        return None
    from pydantic import TypeAdapter

    from litellm.proxy.rust_gateway import gateway_binary, startup_error, unsupported_proxy_trust

    forwarded: Final = tuple(arg for arg in argv if arg != "--rust_gateway")
    params: Final[Mapping[str, object]] = uvicorn_main.make_context(
        "uvicorn", list(uvicorn_argv(forwarded, os.environ))
    ).params
    binary: Final = gateway_binary()
    error: Final = startup_error(
        binary,
        {
            "--ssl-keyfile": params["ssl_keyfile"],
            "--ssl-certfile": params["ssl_certfile"],
            "--ssl-ciphers": any(arg.split("=")[0] == "--ssl-ciphers" for arg in forwarded),
            "--uds": params["uds"] is not None,
            "--fd": params["fd"] is not None,
            "--forwarded-allow-ips": unsupported_proxy_trust(
                TypeAdapter(str | None).validate_python(params["forwarded_allow_ips"])
            ),
        },
    )
    if error is not None:
        sys.exit(error)
    host, port = TypeAdapter(tuple[str, int]).validate_python((params["host"], params["port"]))
    assert binary is not None
    return binary, host, port


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
    listener: Final = forward_listener(argv)
    if "_LITELLM_GATEWAY_FD" not in os.environ:
        configure_database()
    if listener is not None:
        from litellm.proxy.rust_gateway import supervise

        binary, host, port = listener
        forwarded: Final = tuple(arg for arg in argv if arg != "--rust_gateway")
        sys.exit(supervise(binary, (sys.executable, "-m", "gateway.launch", *forwarded), host, port))
    serve(uvicorn_argv(argv, os.environ))


if __name__ == "__main__":
    main(sys.argv[1:])
