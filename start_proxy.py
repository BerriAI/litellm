#!/usr/bin/env python3
"""Start LiteLLM proxy for testing."""
import os
import sys

# Set config path before importing
os.environ["LITELLM_CONFIG_PATH"] = "/workspace/test_proxy_config.yaml"
os.environ["LITELLM_MASTER_KEY"] = "sk-1234"

# Don't modify sys.path - let it use the workspace naturally
import uvicorn
from litellm.proxy.proxy_server import app

if __name__ == "__main__":
    print("Starting LiteLLM proxy on port 4000...")
    print(f"Config: {os.environ.get('LITELLM_CONFIG_PATH')}")
    print(f"Master key: {os.environ.get('LITELLM_MASTER_KEY')}")
    uvicorn.run(app, host="0.0.0.0", port=4000, log_level="info")
