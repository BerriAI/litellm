"""Standalone entrypoint for applying database migrations and generating the Prisma client.

The entrypoint enforces migration failures by default. Set
ENFORCE_PRISMA_MIGRATION_CHECK=false to preserve log-only behavior for migration and
Prisma generate failures.
"""

import os
import subprocess
import sys
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.proxy_cli import run_server
from litellm.secret_managers.main import str_to_bool


def main() -> int:
    enforce_prisma_migration_check: Final = (
        str_to_bool(os.getenv("ENFORCE_PRISMA_MIGRATION_CHECK")) is not False
    )
    run_server_args: Final = (
        ("--skip_server_startup", "--enforce_prisma_migration_check")
        if enforce_prisma_migration_check
        else ("--skip_server_startup",)
    )
    run_server(run_server_args, standalone_mode=False)

    verbose_proxy_logger.info("Running 'prisma generate'...")
    result: Final = subprocess.run(("prisma", "generate"), capture_output=True, text=True)
    verbose_proxy_logger.info("'prisma generate' stdout: %s", result.stdout)
    exit_code: Final = result.returncode

    if exit_code != 0:
        verbose_proxy_logger.info("'prisma generate' failed with exit code %s.", exit_code)
        verbose_proxy_logger.error("'prisma generate' stderr: %s", result.stderr)
        if enforce_prisma_migration_check:
            return exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
