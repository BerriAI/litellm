"""Gateway supervisor: assemble DATABASE_URL, start the in-container PgBouncer, then run uvicorn.

``gateway/main.py`` assembles ``DATABASE_URL`` inside every uvicorn worker, which
is fine for a plain Postgres URL but not for the pooler: PgBouncer must be
started exactly once per pod, before the workers fork, and the workers must be
handed the loopback URL it listens on. A pre-existing ``DATABASE_URL`` wins in
``DatabaseURLSettings.apply_to_env`` under password auth, and one marked pooled
wins under token auth too, so exporting it here is enough for every worker to
pick the pooled URL up unchanged.

Run with:
    python -m gateway.launch --workers 4 --host 0.0.0.0 --port 4000
"""

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from uvicorn.main import main as uvicorn_main

from litellm.proxy.db.db_url_settings import DatabaseURLSettings
from litellm.proxy.db.pgbouncer import (
    PgBouncerError,
    PgBouncerSettings,
    export_pooled_database_url,
    start_in_container_pgbouncer,
)

GATEWAY_APP: Final = "gateway.main:app"
KEEPALIVE_FLAG: Final = "--timeout-keep-alive"


def uvicorn_argv(argv: Sequence[str], environ: Mapping[str, str]) -> tuple[str, ...]:
    """Honor ``KEEPALIVE_TIMEOUT`` like ``proxy_cli.py`` does, unless the flag was passed explicitly."""
    keepalive: Final = environ.get("KEEPALIVE_TIMEOUT")
    if keepalive is None or any(arg == KEEPALIVE_FLAG or arg.startswith(f"{KEEPALIVE_FLAG}=") for arg in argv):
        return (GATEWAY_APP, *argv)
    return (GATEWAY_APP, *argv, KEEPALIVE_FLAG, keepalive)


def pool_database_url(
    settings: DatabaseURLSettings,
    pgbouncer: PgBouncerSettings,
    environ: Mapping[str, str],
) -> str | PgBouncerError | None:
    """Start the in-container PgBouncer and return its loopback URL, or None when ``pgbouncer.enabled`` is off.

    The upstream URL is whatever ``apply_to_env`` assembled from the discrete
    ``DATABASE_*`` vars (or an operator-pinned ``DATABASE_URL``). Under token
    auth the pooler mints and renews the upstream token itself.
    """
    if not pgbouncer.enabled:
        return None
    upstream_url: Final = environ.get("DATABASE_URL")
    if upstream_url is None:
        return PgBouncerError("LITELLM_PGBOUNCER_ENABLED is set but no DATABASE_URL could be assembled")
    return start_in_container_pgbouncer(pgbouncer, upstream_url, token_auth=settings.token_auth())


def _serve(argv: Sequence[str]) -> None:
    uvicorn_main(tuple(argv), prog_name="uvicorn")


def main(argv: Sequence[str], serve: Callable[[Sequence[str]], None] = _serve) -> None:
    settings: Final = DatabaseURLSettings.from_env()
    settings.apply_to_env()
    pooled_url: Final = pool_database_url(settings, PgBouncerSettings(), os.environ)
    if isinstance(pooled_url, PgBouncerError):
        sys.exit(f"LiteLLM gateway: in-container pgbouncer could not start: {pooled_url.reason}")
    if pooled_url is not None:
        export_pooled_database_url(pooled_url)
    serve(uvicorn_argv(argv, os.environ))


if __name__ == "__main__":
    main(sys.argv[1:])
