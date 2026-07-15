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
from typing import Callable, Mapping, NamedTuple

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


def _fail_missing_proxy_env(compat_result) -> None:
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


def require_proxy(
    compat_result,
    *,
    env: Mapping[str, str] | None = None,
) -> ProxyConfig:
    """Return the proxy control-plane config (base URL + master key), or
    hard-fail the test.

    ``env`` is injected for tests; production callers pass nothing and
    the process env is used. This keeps tests off ``monkeypatch.setenv``
    for a check that is a pure function of its inputs.

    SECURITY: the ``api_key`` returned here is the proxy master key.
    Callers that shell out to untrusted subprocesses (the ``claude`` CLI)
    must NOT pass it as ``ANTHROPIC_AUTH_TOKEN``; a compromised CLI
    release would then have admin capabilities (create/delete keys,
    register deployments, read spend logs, drain budgets). Cell code
    paths use ``require_compat_cli_credentials`` instead, which returns
    an inference-only session key minted by the compat fixture. The
    only legitimate use of ``require_proxy`` is control-plane work
    inside the fixture itself."""
    cfg = resolve_proxy(env)
    if cfg is None:
        _fail_missing_proxy_env(compat_result)
    return cfg


CliKeyProvider = Callable[[], str | None]


def require_compat_cli_credentials(
    compat_result,
    *,
    cli_key_provider: CliKeyProvider,
    env: Mapping[str, str] | None = None,
) -> ProxyConfig:
    """Return the proxy base URL paired with the inference-only compat
    CLI key, or hard-fail the test.

    Cells never skip. Two hard-fail paths:

    - Proxy env unset → fail. A live cell with no proxy configured is
      a config error the dev should see loudly.

    - CLI key provider returned falsy (fixture didn't mint one, or
      minted an empty string) → fail. This means the environment is
      missing provider credentials the fixture needs to register the
      compat deployments, or the fixture itself hit a bug. Either way
      it's a real setup error, not a "you don't have the creds" skip.
      The error names the concrete env var to set (``ANTHROPIC_API_KEY``
      is the minimum to get the three Anthropic-tier deployments).

    Purpose: cells hand this credential to the ``claude`` CLI subprocess.
    Using the master key here would leak admin capabilities to the CLI
    (see ``require_proxy``). ``cli_key_provider`` is a zero-arg callable
    that returns the fixture-minted session key. Injected so tests can
    bind an in-memory key without running the fixture or reaching into
    any global state."""
    cfg = resolve_proxy(env)
    if cfg is None:
        _fail_missing_proxy_env(compat_result)
    cli_key = cli_key_provider()
    if not cli_key:
        compat_result.set(
            {
                "status": "fail",
                "error": (
                    "compat CLI key not available. The claude_code "
                    "compat fixture either registered zero deployments "
                    "(no provider credentials in env - export "
                    "ANTHROPIC_API_KEY at minimum, plus AWS_ACCESS_KEY_ID"
                    "/AZURE_FOUNDRY_API_KEY/VERTEXAI_PROJECT+"
                    "GOOGLE_APPLICATION_CREDENTIALS for the other "
                    "providers), or /key/generate did not round-trip "
                    "cleanly against the proxy."
                ),
            }
        )
        pytest.fail(
            "compat CLI key not available for this session",
            pytrace=False,
        )
    return ProxyConfig(base_url=cfg.base_url, api_key=cli_key)
