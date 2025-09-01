# What is this?
## Script to apply initial prisma migration on Docker setup

import os
import subprocess
import sys

sys.path.insert(
    0, os.path.abspath("./")
)  # Adds the parent directory to the system path

from litellm._logging import verbose_proxy_logger
from litellm.proxy.proxy_cli import run_server

# Call the Click command with standalone_mode=False
run_server(["--skip_server_startup"], standalone_mode=False)

# run prisma generate
verbose_proxy_logger.info("Running 'prisma generate'...")
result = subprocess.run(["prisma", "generate"], capture_output=True, text=True)
verbose_proxy_logger.info(f"'prisma generate' stdout: {result.stdout}")  # Log stdout
exit_code = result.returncode

if exit_code != 0:
    verbose_proxy_logger.info(f"'prisma generate' failed with exit code {exit_code}.")
    verbose_proxy_logger.error(
        f"'prisma generate' stderr: {result.stderr}"
    )  # Log stderr
