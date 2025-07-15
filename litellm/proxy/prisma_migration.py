# What is this?
## Script to apply initial prisma migration on Docker setup

import os
import subprocess
import sys

sys.path.insert(
    0, os.path.abspath("./")
)  # Adds the parent directory to the system path

result = subprocess.run(
    ["litellm", "--use_prisma_migrate", "--skip_server_startup"],
    capture_output=True,
    text=True,
)
