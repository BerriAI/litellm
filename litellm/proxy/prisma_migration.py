"""Standalone entrypoint for applying database migrations and generating the Prisma client.

A failed migration fails the entrypoint, the same way it fails proxy startup. A failed
'prisma generate' is log-only: every shipped image bakes the client at build time, and
refreshing it writes into site-packages, which an arbitrary non-root uid or a read-only
root filesystem cannot do. When the prisma package is not writable it is skipped instead,
because its subprocesses roughly double the job's peak memory for a step that cannot succeed.
"""

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable
from importlib.machinery import ModuleSpec

sys.path.insert(0, os.path.abspath("./"))

from typing import Final

from litellm_proxy_extras.prisma_toolchain import resolve_prisma_argv

from litellm._logging import verbose_proxy_logger
from litellm.proxy.proxy_cli import run_server


def _is_writable_dir(path: str) -> bool:
    return os.access(path, os.W_OK)


def prisma_package_is_writable(
    find_spec: Callable[[str], ModuleSpec | None] = importlib.util.find_spec,
    is_writable: Callable[[str], bool] = _is_writable_dir,
) -> bool:
    """'prisma generate' writes the client into the installed prisma package, so it can only succeed here."""
    spec: Final = find_spec("prisma")
    locations: Final = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    return bool(locations) and all(is_writable(location) for location in locations)


def main(can_generate: Callable[[], bool] = prisma_package_is_writable) -> int:
    run_server(("--skip_server_startup",), standalone_mode=False)

    if not can_generate():
        verbose_proxy_logger.info(
            "Skipping 'prisma generate' because the prisma package is not writable, using the client baked at image build time"
        )
        return 0

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
