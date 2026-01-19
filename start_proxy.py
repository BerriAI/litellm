#!/usr/bin/env python3
"""Start litellm proxy with config file."""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from litellm.proxy.proxy_cli import run_server

    # Run the server
    # Disable loop type to use default asyncio (avoid uvloop dependency)
    os.environ["LITELLM_PROXY_LOOP_TYPE"] = "none"

    sys.argv = [
        "litellm",
        "--config", "reproduce_json_logs_config.yaml",
        "--port", "4000"
    ]

    run_server()
