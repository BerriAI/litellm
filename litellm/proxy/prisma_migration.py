"""Standalone entrypoint for applying database migrations and generating the Prisma client.

Migration failures fail the entrypoint by default; set ENFORCE_PRISMA_MIGRATION_CHECK=false
for log-only behavior. A failed 'prisma generate' is always log-only: every shipped image
bakes the client at build time, and refreshing it writes into site-packages, which an
arbitrary non-root uid or a read-only root filesystem cannot do.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("./"))

from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.proxy_cli import run_server
from litellm.secret_managers.main import str_to_bool


def main() -> int:
    enforce_prisma_migration_check: Final = str_to_bool(os.getenv("ENFORCE_PRISMA_MIGRATION_CHECK")) is not False
    run_server_args: Final = (
        ("--skip_server_startup", "--enforce_prisma_migration_check")
        if enforce_prisma_migration_check
        else ("--skip_server_startup",)
    )
    run_server(run_server_args, standalone_mode=False)

    verbose_proxy_logger.info("Running 'prisma generate'...")
    result: Final = subprocess.run(("prisma", "generate"), capture_output=True, text=True)
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
