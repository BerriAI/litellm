"""
Purpose
- Common helpers for ndsmokes: reachability checks, env parsing, HTTP post.

Scope
- DOES: simple TCP reachability, env-driven base URL, tiny httpx POST wrapper
- DOES NOT: assert semantics, retry logic, long timeouts
"""
from __future__ import annotations
import os, socket
from urllib.parse import urlsplit
import httpx

def can_connect(host: str, port: int, timeout: float = 0.75) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def parse_base(env_key: str, default: str) -> tuple[str, int, str]:
    base = os.getenv(env_key, default).rstrip("/")
    p = urlsplit(base)
    host = p.hostname or "127.0.0.1"
    port = p.port or (443 if p.scheme == "https" else 80)
    return host, port, base

def post_json(url: str, payload: dict, timeout: float = 10.0) -> httpx.Response:
    return httpx.post(url, json=payload, timeout=timeout)

def require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"Missing required env: {var}")
    return val