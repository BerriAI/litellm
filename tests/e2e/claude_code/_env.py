"""Proxy-env resolution for the claude_code compat cells.

Historically each cell hardcoded ``LITELLM_PROXY_BASE_URL`` and
``LITELLM_PROXY_API_KEY``, which do not match the names the rest of
``tests/e2e/`` reads (``LITELLM_PROXY_URL`` / ``LITELLM_MASTER_KEY`` from
``e2e_config.py``). That drift means anyone standing up a live proxy for
one suite has to set two extra env vars for another, and CI wiring has to
export both spellings. Everything under ``claude_code/`` now goes through
``resolve_proxy`` / ``require_proxy`` here, so the naming lives in one
place and either spelling works (primary wins on tie).
"""

from __future__ import annotations

import os
from typing import Mapping, NamedTuple

import pytest


class ProxyConfig(NamedTuple):
    base_url: str
    api_key: str


PRIMARY_BASE_URL_ENV = "LITELLM_PROXY_URL"
PRIMARY_API_KEY_ENV = "LITELLM_MASTER_KEY"

LEGACY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
LEGACY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"


def _pick(mapping: Mapping[str, str], primary: str, legacy: str) -> str | None:
    return mapping.get(primary) or mapping.get(legacy) or None


def resolve_proxy_from(mapping: Mapping[str, str]) -> ProxyConfig | None:
    """Pure resolver: takes an env mapping, returns a ProxyConfig if
    both a base URL and an API key are present under either the primary
    or legacy names, else None. Extracted so tests can exercise it
    without mutating ``os.environ``."""
    base_url = _pick(mapping, PRIMARY_BASE_URL_ENV, LEGACY_BASE_URL_ENV)
    api_key = _pick(mapping, PRIMARY_API_KEY_ENV, LEGACY_API_KEY_ENV)
    if not base_url or not api_key:
        return None
    return ProxyConfig(base_url=base_url, api_key=api_key)


def resolve_proxy(env: Mapping[str, str] | None = None) -> ProxyConfig | None:
    """Convenience wrapper that defaults to ``os.environ``. Prefer
    calling ``resolve_proxy_from(env)`` from tests so nothing has to
    reach into the process environment."""
    return resolve_proxy_from(os.environ if env is None else env)


def require_proxy(
    compat_result,
    *,
    env: Mapping[str, str] | None = None,
) -> ProxyConfig:
    """Return proxy config or hard-fail the test. Matches the fail-not-
    skip contract in ``tests/e2e/conftest.py``: a live test with the
    proxy env unset is a real failure, not a skip.

    ``env`` is injected for tests; production callers pass nothing and
    the process env is used. This keeps tests off ``monkeypatch.setenv``
    for a check that is a pure function of its inputs."""
    cfg = resolve_proxy(env)
    if cfg is None:
        compat_result.set(
            {
                "status": "fail",
                "error": (
                    f"missing required env: set {PRIMARY_BASE_URL_ENV} and "
                    f"{PRIMARY_API_KEY_ENV} (or the legacy "
                    f"{LEGACY_BASE_URL_ENV} / {LEGACY_API_KEY_ENV}) to "
                    "point at a running LiteLLM proxy"
                ),
            }
        )
        pytest.fail(
            f"{PRIMARY_BASE_URL_ENV} / {PRIMARY_API_KEY_ENV} not configured",
            pytrace=False,
        )
    return cfg
