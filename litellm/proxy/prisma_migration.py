"""Standalone entrypoint for applying database migrations and generating the Prisma client.

Migration failures fail the entrypoint by default; set ENFORCE_PRISMA_MIGRATION_CHECK=false
for log-only behavior. A failed 'prisma generate' is always log-only: every shipped image
bakes the client at build time, and refreshing it writes into site-packages, which an
arbitrary non-root uid or a read-only root filesystem cannot do.

That same refresh is also skipped outright when the installed client already came from the
schema being generated from, because it can then only rewrite the client with what it
already holds. It is not free: it is the slowest step of the migrations Job (~10s on a
whole CPU, ~40s on a quarter of one), it runs unbounded after the database is already
migrated, and a Job that is still inside it looks identical to a Job still migrating.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("./"))

from collections.abc import Callable
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.proxy_cli import run_server
from litellm.secret_managers.main import str_to_bool

SCHEMA_FILE: Final = Path("schema.prisma")


def generated_client_dir() -> Path | None:
    """Directory 'prisma generate' writes the client into, or None if prisma is missing."""
    try:
        import prisma
    except ImportError:
        return None
    return Path(prisma.__file__).parent if prisma.__file__ else None


def client_already_generated_from(schema: Path, client_dir: Path | None) -> bool:
    """True when the installed client was generated from this exact schema.

    'prisma generate' copies its source schema into the generated package, so identical
    bytes mean a regeneration would reproduce what is already on disk.
    """
    if client_dir is None:
        return False
    try:
        return (client_dir / SCHEMA_FILE.name).read_bytes() == schema.read_bytes()
    except OSError:
        return False


def installed_client_is_current() -> bool:
    return client_already_generated_from(SCHEMA_FILE, generated_client_dir())


def generate_prisma_client() -> subprocess.CompletedProcess[str]:
    return subprocess.run(("prisma", "generate"), capture_output=True, text=True)


def main(
    start_server: Callable[..., object] = run_server,
    client_is_current: Callable[[], bool] = installed_client_is_current,
    generate_client: Callable[[], subprocess.CompletedProcess[str]] = generate_prisma_client,
) -> int:
    enforce_prisma_migration_check: Final = str_to_bool(os.getenv("ENFORCE_PRISMA_MIGRATION_CHECK")) is not False
    run_server_args: Final = (
        ("--skip_server_startup", "--enforce_prisma_migration_check")
        if enforce_prisma_migration_check
        else ("--skip_server_startup",)
    )
    start_server(run_server_args, standalone_mode=False)

    if client_is_current():
        verbose_proxy_logger.info(
            "Skipping 'prisma generate': the installed client already comes from %s",
            SCHEMA_FILE,
        )
        return 0

    verbose_proxy_logger.info("Running 'prisma generate'...")
    result: Final = generate_client()
    verbose_proxy_logger.info("'prisma generate' stdout: %s", result.stdout)

    if result.returncode != 0:
        verbose_proxy_logger.warning(
            "'prisma generate' exited %s; continuing with the client baked at image build time. stderr: %s",
            result.returncode,
            result.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
