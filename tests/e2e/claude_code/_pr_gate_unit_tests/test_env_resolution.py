"""Regression tests for ``claude_code/_env.py``.

Pin the resolution rules so a future edit cannot silently reintroduce
the ``LITELLM_PROXY_BASE_URL`` / ``LITELLM_PROXY_API_KEY`` naming drift
that made every ``claude_code`` cell fail with "not configured" even
when the surrounding e2e suite had a live proxy configured under the
suite-wide ``LITELLM_PROXY_URL`` / ``LITELLM_MASTER_KEY`` names.
"""

from __future__ import annotations

import pytest

from claude_code._env import (
    LEGACY_API_KEY_ENV,
    LEGACY_BASE_URL_ENV,
    PRIMARY_API_KEY_ENV,
    PRIMARY_BASE_URL_ENV,
    ProxyConfig,
    require_proxy,
    resolve_proxy_from,
)


class _CompatResultStub:
    """Minimal stand-in for the compat_result fixture used by cells."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def set(self, payload: dict[str, str]) -> None:
        self.calls.append(payload)


def test_primary_env_names_match_suite_wide_config() -> None:
    """The primary names claude_code reads must exactly match the ones
    ``e2e_config.py`` reads for the rest of the suite. Anything else
    silently reintroduces the drift this refactor cleaned up."""
    assert PRIMARY_BASE_URL_ENV == "LITELLM_PROXY_URL"
    assert PRIMARY_API_KEY_ENV == "LITELLM_MASTER_KEY"


def test_returns_none_when_no_env_is_set() -> None:
    assert resolve_proxy_from({}) is None


def test_returns_none_when_only_url_is_set() -> None:
    assert (
        resolve_proxy_from({PRIMARY_BASE_URL_ENV: "http://localhost:4000"}) is None
    )


def test_returns_none_when_only_key_is_set() -> None:
    assert resolve_proxy_from({PRIMARY_API_KEY_ENV: "sk-1234"}) is None


def test_primary_pair_resolves() -> None:
    cfg = resolve_proxy_from(
        {
            PRIMARY_BASE_URL_ENV: "http://localhost:4000",
            PRIMARY_API_KEY_ENV: "sk-1234",
        }
    )
    assert cfg == ProxyConfig("http://localhost:4000", "sk-1234")


def test_legacy_pair_resolves_when_primary_missing() -> None:
    """Existing CI wiring that only exports the legacy names must keep
    working — otherwise this refactor breaks stage on the way in."""
    cfg = resolve_proxy_from(
        {
            LEGACY_BASE_URL_ENV: "http://legacy:4000",
            LEGACY_API_KEY_ENV: "sk-legacy",
        }
    )
    assert cfg == ProxyConfig("http://legacy:4000", "sk-legacy")


def test_primary_wins_over_legacy_when_both_set() -> None:
    """When both spellings are present, the suite-wide names take
    precedence. Otherwise a stale legacy export in the environment
    would silently override a caller who set the primary names."""
    cfg = resolve_proxy_from(
        {
            PRIMARY_BASE_URL_ENV: "http://primary:4000",
            LEGACY_BASE_URL_ENV: "http://legacy:4000",
            PRIMARY_API_KEY_ENV: "sk-primary",
            LEGACY_API_KEY_ENV: "sk-legacy",
        }
    )
    assert cfg == ProxyConfig("http://primary:4000", "sk-primary")


def test_mixed_url_primary_key_legacy_resolves() -> None:
    """One spelling per var is fine — mixing across pairs must still
    resolve, so a partial migration doesn't strand a caller."""
    cfg = resolve_proxy_from(
        {
            PRIMARY_BASE_URL_ENV: "http://primary:4000",
            LEGACY_API_KEY_ENV: "sk-legacy",
        }
    )
    assert cfg == ProxyConfig("http://primary:4000", "sk-legacy")


def test_empty_string_env_is_treated_as_unset() -> None:
    """``os.environ.get`` on an exported-but-empty var returns "" which
    is falsy. The resolver must treat that as unset so a shell that
    accidentally exports ``LITELLM_PROXY_URL=`` doesn't turn into a
    "" base_url that hits the wrong endpoint."""
    assert (
        resolve_proxy_from(
            {PRIMARY_BASE_URL_ENV: "", PRIMARY_API_KEY_ENV: "sk-1234"}
        )
        is None
    )


def test_require_proxy_fails_with_helpful_message_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error the user sees must name BOTH the primary and legacy
    env vars — otherwise they can't tell why the test is failing when
    they only set the legacy pair, or vice versa."""
    for name in (
        PRIMARY_BASE_URL_ENV,
        PRIMARY_API_KEY_ENV,
        LEGACY_BASE_URL_ENV,
        LEGACY_API_KEY_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    compat = _CompatResultStub()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        require_proxy(compat)
    assert PRIMARY_BASE_URL_ENV in str(excinfo.value)
    assert PRIMARY_API_KEY_ENV in str(excinfo.value)
    assert compat.calls and compat.calls[0]["status"] == "fail"
    assert LEGACY_BASE_URL_ENV in compat.calls[0]["error"]


def test_require_proxy_returns_config_when_primary_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LEGACY_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(LEGACY_API_KEY_ENV, raising=False)
    monkeypatch.setenv(PRIMARY_BASE_URL_ENV, "http://localhost:4000")
    monkeypatch.setenv(PRIMARY_API_KEY_ENV, "sk-1234")
    cfg = require_proxy(_CompatResultStub())
    assert cfg == ProxyConfig("http://localhost:4000", "sk-1234")
