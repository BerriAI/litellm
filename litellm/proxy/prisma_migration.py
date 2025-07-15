# What is this?
## Script to apply initial prisma migration on Docker setup

import os
import sys

sys.path.insert(
    0, os.path.abspath("./")
)  # Adds the parent directory to the system path

from litellm.proxy.proxy_cli import run_server

run_server(
    use_prisma_migrate=True,
    skip_server_startup=True,
)
