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
    require_compat_cli_credentials,
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


def test_require_proxy_fails_with_helpful_message_when_env_empty() -> None:
    """The error the user sees must name BOTH the primary and legacy
    env vars — otherwise they can't tell why the test is failing when
    they only set the legacy pair, or vice versa."""
    compat = _CompatResultStub()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        require_proxy(compat, env={})
    assert PRIMARY_BASE_URL_ENV in str(excinfo.value)
    assert PRIMARY_API_KEY_ENV in str(excinfo.value)
    assert compat.calls and compat.calls[0]["status"] == "fail"
    assert LEGACY_BASE_URL_ENV in compat.calls[0]["error"]


def test_require_proxy_returns_config_when_primary_env_supplied() -> None:
    cfg = require_proxy(
        _CompatResultStub(),
        env={
            PRIMARY_BASE_URL_ENV: "http://localhost:4000",
            PRIMARY_API_KEY_ENV: "sk-1234",
        },
    )
    assert cfg == ProxyConfig("http://localhost:4000", "sk-1234")


def test_require_proxy_returns_config_when_only_legacy_env_supplied() -> None:
    cfg = require_proxy(
        _CompatResultStub(),
        env={
            LEGACY_BASE_URL_ENV: "http://legacy:4000",
            LEGACY_API_KEY_ENV: "sk-legacy",
        },
    )
    assert cfg == ProxyConfig("http://legacy:4000", "sk-legacy")


def test_require_proxy_leaves_compat_result_untouched_on_success() -> None:
    """A successful resolution must NOT append a spurious fail entry.
    Would have silently poisoned every compat cell's result rows."""
    compat = _CompatResultStub()
    require_proxy(
        compat,
        env={
            PRIMARY_BASE_URL_ENV: "http://localhost:4000",
            PRIMARY_API_KEY_ENV: "sk-1234",
        },
    )
    assert compat.calls == []


# ---------------------------------------------------------------------------
# require_compat_cli_credentials — the inference-only key path cells use.
# ---------------------------------------------------------------------------


def test_require_compat_cli_credentials_returns_cli_key_not_master() -> None:
    """SECURITY-critical: the returned ``api_key`` must be the CLI key
    the provider hands us, not the master key from env. Cells feed this
    into the ``claude`` CLI subprocess as ANTHROPIC_AUTH_TOKEN."""
    cfg = require_compat_cli_credentials(
        _CompatResultStub(),
        cli_key_provider=lambda: "sk-cli-scoped",
        env={
            PRIMARY_BASE_URL_ENV: "http://localhost:4000",
            PRIMARY_API_KEY_ENV: "sk-master-must-not-leak",
        },
    )
    assert cfg.api_key == "sk-cli-scoped"
    assert cfg.api_key != "sk-master-must-not-leak"


def test_require_compat_cli_credentials_hard_fails_without_master_fallback() -> None:
    """If the fixture never bound a CLI key (returns ``None``), the
    helper must hard-fail. A silent fallback to the master key would
    defeat the whole point of minting a scoped key in the first place."""
    compat = _CompatResultStub()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        require_compat_cli_credentials(
            compat,
            cli_key_provider=lambda: None,
            env={
                PRIMARY_BASE_URL_ENV: "http://localhost:4000",
                PRIMARY_API_KEY_ENV: "sk-master",
            },
        )
    assert "compat CLI key" in str(excinfo.value)
    assert compat.calls and compat.calls[0]["status"] == "fail"


def test_require_compat_cli_credentials_treats_empty_key_as_missing() -> None:
    """The provider returning an empty string is the same failure mode as
    ``None`` - a truthy check would happily hand the CLI a blank
    Authorization header and the request would 401, but with a
    surprising error trail. Fail loudly with the same message instead."""
    with pytest.raises(pytest.fail.Exception):
        require_compat_cli_credentials(
            _CompatResultStub(),
            cli_key_provider=lambda: "",
            env={
                PRIMARY_BASE_URL_ENV: "http://localhost:4000",
                PRIMARY_API_KEY_ENV: "sk-master",
            },
        )


def test_require_compat_cli_credentials_fails_on_missing_env_before_calling_provider() -> None:
    """The env-missing guard fires first. If the proxy env is unset,
    the provider must never be called - otherwise a hypothetical provider
    that reaches out to the network to fetch a key would waste that call."""
    provider_calls = 0

    def counting_provider() -> str:
        nonlocal provider_calls
        provider_calls += 1
        return "sk-cli"

    with pytest.raises(pytest.fail.Exception):
        require_compat_cli_credentials(
            _CompatResultStub(),
            cli_key_provider=counting_provider,
            env={},
        )
    assert provider_calls == 0, (
        "cli_key_provider must not be called before env resolution succeeds"
    )
