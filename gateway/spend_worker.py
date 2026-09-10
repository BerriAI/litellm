"""Spend sidecar entrypoint for the gateway image.

Assembles ``DATABASE_URL`` the way ``gateway.launch`` does, but instead of starting a PgBouncer it
points the URL at the one the gateway container already runs on the pod's loopback (the sidecar
must see the same ``DATABASE_URL`` inputs and ``LITELLM_PGBOUNCER_*`` values as the gateway
container), then hands off to ``litellm.proxy.spend_worker``.

Run with:
    python -m gateway.spend_worker [--address unix:///var/run/litellm/spend-worker.sock]
"""

import os
import sys
from collections.abc import Mapping, Sequence
from typing import Final

from litellm.proxy.db.db_url_settings import DatabaseURLSettings
from litellm.proxy.db.pgbouncer import PgBouncerError, PgBouncerSettings, pooled_database_url
from litellm.proxy.spend_worker import main as spend_worker_main


def pod_pgbouncer_database_url(pgbouncer: PgBouncerSettings, environ: Mapping[str, str]) -> str | PgBouncerError | None:
    """The gateway container's PgBouncer URL for ``environ["DATABASE_URL"]``, or None when PgBouncer is off."""
    if not pgbouncer.enabled:
        return None
    upstream_url: Final = environ.get("DATABASE_URL")
    if upstream_url is None:
        return PgBouncerError("LITELLM_PGBOUNCER_ENABLED is set but no DATABASE_URL could be assembled")
    return pooled_database_url(upstream_url, pgbouncer)


def main(argv: Sequence[str]) -> None:
    DatabaseURLSettings.from_env().apply_to_env()
    pooled: Final = pod_pgbouncer_database_url(PgBouncerSettings(), os.environ)
    if isinstance(pooled, PgBouncerError):
        sys.exit(f"LiteLLM spend worker: cannot use the pod's pgbouncer: {pooled.reason}")
    if pooled is not None:
        os.environ["DATABASE_URL"] = pooled
    spend_worker_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
