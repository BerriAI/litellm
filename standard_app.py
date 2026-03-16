"""
Gunicorn wrapper for standard LiteLLM proxy (no fast-litellm).

Usage:
    CONFIG_FILE_PATH=benchmark_config.yaml \
    gunicorn standard_app:app --preload -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:4000
"""

import os

# Set config file path before anything else loads
os.environ.setdefault("CONFIG_FILE_PATH", "benchmark_config.yaml")

from litellm.proxy.proxy_server import app  # noqa: F401, E402
