"""Proxy-env resolution for the claude_code compat cells.

Uses the same ``LITELLM_PROXY_URL`` / ``LITELLM_MASTER_KEY`` names as
``e2e_config.py`` and the rest of ``tests/e2e/``. Everything under
``claude_code/`` goes through ``resolve_proxy`` / ``require_proxy`` here
so the naming lives in one place.
"""

from __future__ import annotations

import os
from typing import Mapping, NamedTuple

import pytest

from proxy_client import ProxyClient, build_proxy_client


class ProxyConfig(NamedTuple):
    base_url: str
    api_key: str


PRIMARY_BASE_URL_ENV = "LITELLM_PROXY_URL"
PRIMARY_API_KEY_ENV = "LITELLM_MASTER_KEY"


def resolve_proxy_from(mapping: Mapping[str, str]) -> ProxyConfig | None:
    """Pure resolver: takes an env mapping, returns a ProxyConfig if
    both a base URL and an API key are present, else None. Extracted so
    tests can exercise it without mutating ``os.environ``."""
    base_url = mapping.get(PRIMARY_BASE_URL_ENV) or None
    api_key = mapping.get(PRIMARY_API_KEY_ENV) or None
    if not base_url or not api_key:
        return None
    return ProxyConfig(base_url=base_url, api_key=api_key)


def resolve_proxy(env: Mapping[str, str] | None = None) -> ProxyConfig | None:
    """Convenience wrapper that defaults to ``os.environ``. Prefer
    calling ``resolve_proxy_from(env)`` from tests so nothing has to
    reach into the process environment."""
    return resolve_proxy_from(os.environ if env is None else env)


def _fail_missing_proxy_env(compat_result) -> None:
    compat_result.set(
        {
            "status": "fail",
            "error": (
                f"missing required env: set {PRIMARY_BASE_URL_ENV} and "
                f"{PRIMARY_API_KEY_ENV} to point at a running LiteLLM proxy"
            ),
        }
    )
    pytest.fail(
        f"{PRIMARY_BASE_URL_ENV} / {PRIMARY_API_KEY_ENV} not configured",
        pytrace=False,
    )


def require_proxy(
    compat_result,
    *,
    env: Mapping[str, str] | None = None,
) -> ProxyConfig:
    """Return the proxy config (base URL + master key), or hard-fail
    the test.

    ``env`` is injected for tests; production callers pass nothing and
    the process env is used. This keeps tests off ``monkeypatch.setenv``
    for a check that is a pure function of its inputs."""
    cfg = resolve_proxy(env)
    if cfg is None:
        _fail_missing_proxy_env(compat_result)
    return cfg


class ProxyClientConfig(NamedTuple):
    client: ProxyClient
    api_key: str


def require_proxy_client(
    compat_result,
    *,
    env: Mapping[str, str] | None = None,
) -> ProxyClientConfig:
    """Return the shared ``ProxyClient`` plus the master key the HTTP probes
    authenticate with, or hard-fail the test.

    Both planes of the built ``ProxyClient`` point at the one resolved base URL,
    so the probes reuse the shared transport (split control/data-plane routing,
    timeout, typed ``Result``) rather than hand-rolling ``httpx``. The api_key is
    returned alongside because the probes call ``/v1/messages`` with the master
    key (the compat matrix's credential), the same way the CLI rows do."""
    cfg = require_proxy(compat_result, env=env)
    client = build_proxy_client(
        base_url=cfg.base_url,
        master_key=cfg.api_key,
        control_plane_base_url=cfg.base_url,
    )
    return ProxyClientConfig(client=client, api_key=cfg.api_key)
