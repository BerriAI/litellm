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

# Run prisma generate
verbose_proxy_logger.info("Running 'prisma generate'...")
try:
    result = subprocess.run(["prisma", "generate"], capture_output=True, text=True)
    if result.returncode != 0:
        if "Permission denied" in result.stderr:
            verbose_proxy_logger.warning(
                f"Permission denied during 'prisma generate'. Skipping generation, assuming client is pre-generated. Error: {result.stderr}"
            )
        else:
            verbose_proxy_logger.info(
                f"'prisma generate' failed with exit code {result.returncode}."
            )
            verbose_proxy_logger.error(
                f"'prisma generate' stderr: {result.stderr}"
            )  # Log stderr
except Exception as e:
    verbose_proxy_logger.error(f"Error running prisma generate: {e}")
