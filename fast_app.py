"""
Gunicorn wrapper for LiteLLM proxy with fast-litellm Rust acceleration.

Usage:
    gunicorn fast_app:app --preload -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:4000
"""

import fast_litellm  # noqa: F401  — Apply Rust acceleration before litellm loads

from litellm.proxy.proxy_server import app  # noqa: F401, E402
