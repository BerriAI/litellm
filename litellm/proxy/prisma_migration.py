"""Standalone entrypoint for applying database migrations and generating the Prisma client.

A failed migration fails the entrypoint, the same way it fails proxy startup. A failed
'prisma generate' is log-only: every shipped image bakes the client at build time, and
refreshing it writes into site-packages, which an arbitrary non-root uid or a read-only
root filesystem cannot do.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("./"))

from typing import Final

from litellm_proxy_extras.prisma_toolchain import resolve_prisma_argv

from litellm._logging import verbose_proxy_logger
from litellm.proxy.proxy_cli import run_server


def main() -> int:
    run_server(("--skip_server_startup",), standalone_mode=False)

    verbose_proxy_logger.info("Running 'prisma generate'...")
    result: Final = subprocess.run(resolve_prisma_argv(("prisma", "generate")), capture_output=True, text=True)
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
