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
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from types import MappingProxyType
from typing import Final

from uvicorn.main import main as uvicorn_main

from litellm.proxy.db.db_url_settings import DatabaseURLSettings
from litellm.proxy.db.pgbouncer import (
    PGBOUNCER_MANAGED_SHUTDOWN_SECONDS,
    ManagedPgBouncerRuntime,
    PgBouncerError,
    PgBouncerSettings,
    export_pooled_database_url,
    pgbouncer_startup_signals,
    start_in_container_pgbouncer,
    start_managed_pgbouncer,
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
    return start_in_container_pgbouncer(
        pgbouncer,
        upstream_url,
        token_auth=settings.token_auth(),
        shutdown_timeout_seconds=PGBOUNCER_MANAGED_SHUTDOWN_SECONDS,
    )


def managed_pool_database_url(
    settings: DatabaseURLSettings,
    pgbouncer: PgBouncerSettings,
    environ: Mapping[str, str],
) -> ManagedPgBouncerRuntime | PgBouncerError | None:
    upstream_url: Final = environ.get("DATABASE_URL")
    if not upstream_url:
        return None
    effective_settings: Final = pgbouncer if pgbouncer.enabled else pgbouncer.model_copy(update={"enabled": True})
    return start_managed_pgbouncer(effective_settings, upstream_url, token_auth=settings.token_auth())


def _pooled_child_environments(
    python_environment: Mapping[str, str],
    rust_environment: Mapping[str, str],
    pooler: ManagedPgBouncerRuntime | None,
) -> tuple[Mapping[str, str], Mapping[str, str]]:
    if pooler is None:
        return (
            MappingProxyType({**python_environment, "LITELLM_PGBOUNCER_ENABLED": "false"}),
            rust_environment,
        )
    pooled_values: Final = {
        "DATABASE_URL": pooler.pooled_url,
        "LITELLM_PGBOUNCER_POOLED_DATABASE_URL": "true",
    }
    return (
        MappingProxyType(
            {
                **python_environment,
                **pooled_values,
                "LITELLM_PGBOUNCER_ENABLED": "false",
            }
        ),
        MappingProxyType({**rust_environment, **pooled_values}),
    )


def _serve(argv: Sequence[str]) -> None:
    uvicorn_main(tuple(argv), prog_name="uvicorn")


def _stop_starting_pooler(pooler: ManagedPgBouncerRuntime, timeout: int) -> None:
    pooler.stop(time.monotonic() + timeout)


def main(argv: Sequence[str], serve: Callable[[Sequence[str]], None] = _serve) -> None:
    gateway_mode: Final = os.environ.get("LITELLM_RUST_GATEWAY")
    if gateway_mode not in (None, "true", "false"):
        sys.exit("LiteLLM gateway: LITELLM_RUST_GATEWAY must be exactly 'true' or 'false'")
    if gateway_mode == "true":
        from dataclasses import replace

        from gateway.supervisor import ConfigurationFailure, build_configuration, startup_scope, supervise

        with ExitStack() as startup_cleanup, startup_scope():
            rust_settings: Final = DatabaseURLSettings.from_env()
            rust_settings.apply_to_env()
            preflight_configuration: Final = build_configuration(argv, os.environ, sys.executable)
            if isinstance(preflight_configuration, ConfigurationFailure):
                sys.exit(f"LiteLLM gateway supervisor: {preflight_configuration.reason}")
            pooler: Final = managed_pool_database_url(rust_settings, PgBouncerSettings(), os.environ)
            if isinstance(pooler, PgBouncerError):
                sys.exit(f"LiteLLM gateway: in-container pgbouncer could not start: {pooler.reason}")
            if pooler is not None:
                startup_cleanup.callback(
                    _stop_starting_pooler, pooler, preflight_configuration.termination_grace_timeout
                )
                export_pooled_database_url(pooler.pooled_url)
            python_environment, rust_environment = _pooled_child_environments(
                preflight_configuration.python_environment,
                preflight_configuration.rust_environment,
                pooler,
            )
            final_configuration: Final = replace(
                preflight_configuration,
                python_environment=python_environment,
                rust_environment=rust_environment,
                managed_pooler=pooler,
            )
            failure: Final = supervise(final_configuration)
            startup_cleanup.pop_all()
            if failure is not None:
                sys.exit(f"LiteLLM gateway supervisor: {failure.reason}")
            return

    pgbouncer: Final = PgBouncerSettings()
    with pgbouncer_startup_signals(pgbouncer.enabled):
        settings: Final = DatabaseURLSettings.from_env()
        settings.apply_to_env()
        pooled_url: Final = pool_database_url(settings, pgbouncer, os.environ)
        if isinstance(pooled_url, PgBouncerError):
            sys.exit(f"LiteLLM gateway: in-container pgbouncer could not start: {pooled_url.reason}")
        if pooled_url is not None:
            export_pooled_database_url(pooled_url)
        serve(uvicorn_argv(argv, os.environ))


if __name__ == "__main__":
    main(sys.argv[1:])
