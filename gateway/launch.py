"""Gateway supervisor: assemble DATABASE_URL, start the in-container PgBouncer, then run uvicorn.

``gateway/main.py`` assembles ``DATABASE_URL`` inside every uvicorn worker, which
is fine for a plain Postgres URL but not for the pooler: PgBouncer must be
started exactly once per pod, before the workers fork, and the workers must be
handed the loopback URL it listens on. A pre-existing ``DATABASE_URL`` wins in
``DatabaseURLSettings.apply_to_env`` under password auth, so setting it here is
enough for every worker to pick the pooled URL up unchanged.

Run with:
    python -m gateway.launch --workers 4 --host 0.0.0.0 --port 4000
"""

import os
import sys
from collections.abc import MutableMapping, Sequence
from typing import Final

from uvicorn.main import main as uvicorn_main

from litellm.proxy.db.db_url_settings import DatabaseURLSettings
from litellm.proxy.db.pgbouncer import PgBouncerError, PgBouncerSettings, start_in_container_pgbouncer

GATEWAY_APP: Final = "gateway.main:app"


def pool_database_url(
    settings: DatabaseURLSettings,
    pgbouncer: PgBouncerSettings,
    environ: MutableMapping[str, str],
) -> PgBouncerError | None:
    """Point ``environ["DATABASE_URL"]`` at an in-container PgBouncer when ``pgbouncer.enabled``.

    The upstream URL is whatever ``apply_to_env`` assembled from the discrete
    ``DATABASE_*`` vars (or an operator-pinned ``DATABASE_URL``). Token auth is
    rejected by the pooler itself, since it holds one password for its lifetime.
    """
    if not pgbouncer.enabled:
        return None
    upstream_url: Final = environ.get("DATABASE_URL")
    if upstream_url is None:
        return PgBouncerError("LITELLM_PGBOUNCER_ENABLED is set but no DATABASE_URL could be assembled")
    pooled_url: Final = start_in_container_pgbouncer(
        pgbouncer, upstream_url, token_auth_enabled=settings.token_auth() is not None
    )
    if isinstance(pooled_url, PgBouncerError):
        return pooled_url
    environ["DATABASE_URL"] = pooled_url
    return None


def main(argv: Sequence[str]) -> None:
    settings: Final = DatabaseURLSettings.from_env()
    settings.apply_to_env()
    failed: Final = pool_database_url(settings, PgBouncerSettings(), os.environ)
    if failed is not None:
        sys.exit(f"LiteLLM gateway: in-container pgbouncer could not start: {failed.reason}")
    uvicorn_main((GATEWAY_APP, *argv), prog_name="uvicorn")


if __name__ == "__main__":
    main(sys.argv[1:])
