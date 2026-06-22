"""Shared fixtures + env contract for realtime e2e tests.

These tests assume an externally running litellm proxy (configured with the
realtime model aliases in realtime_e2e_config.yaml). They are gated behind the
``realtime_e2e`` marker and skip cleanly when the proxy or provider creds are
absent, so they never break the default unit run.

Env contract:
  LITELLM_PROXY_WS_URL   ws base, default ws://0.0.0.0:4000
  LITELLM_PROXY_URL      http base for health check, default http://0.0.0.0:4000
  LITELLM_PROXY_API_KEY  virtual key / master key, default sk-1234
"""

from __future__ import annotations

import os

import httpx
import pytest

from .providers import RealtimeProvider


@pytest.fixture(scope="session")
def proxy_ws_url() -> str:
    return os.environ.get("LITELLM_PROXY_WS_URL", "ws://0.0.0.0:4000")


@pytest.fixture(scope="session")
def proxy_http_url() -> str:
    return os.environ.get("LITELLM_PROXY_URL", "http://0.0.0.0:4000")


@pytest.fixture(scope="session")
def proxy_api_key() -> str:
    return os.environ.get("LITELLM_PROXY_API_KEY", "sk-1234")


@pytest.fixture(scope="session", autouse=True)
def require_proxy(proxy_http_url: str) -> None:
    try:
        response = httpx.get(f"{proxy_http_url}/health/liveliness", timeout=5)
        response.raise_for_status()
    except Exception as exc:
        pytest.skip(
            f"litellm proxy not reachable at {proxy_http_url}: {exc}. "
            "Start it with the realtime config (see README.md)."
        )


def skip_if_creds_missing(provider: RealtimeProvider) -> None:
    missing = tuple(var for var in provider.required_env if not os.environ.get(var))
    if missing:
        pytest.skip(f"{provider.id}: missing env {', '.join(missing)}")
