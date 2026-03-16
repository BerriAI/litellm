#!/usr/bin/env python3
"""Start LiteLLM proxy with fast-litellm acceleration enabled."""
import fast_litellm  # noqa: F401 - Must be imported before litellm to apply Rust patches
from litellm.proxy.proxy_cli import run_server

if __name__ == "__main__":
    run_server()
