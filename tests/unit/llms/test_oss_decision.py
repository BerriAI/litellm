from typing import Final

import pytest

from litellm.llms.oss_decision import OssDecisionProvider, oss_connection, validate_oss_request

pytestmark: Final = pytest.mark.parametrize("provider", ["laya", "bespoke"])


@pytest.mark.parametrize(
    ("base", "key", "expected_base", "expected_key"),
    [
        (None, None, "http://decision.test/root", "oss-env-key"),
        ("http://custom.test/", None, "http://custom.test", None),
        ("http://custom.test/", "explicit-key", "http://custom.test", "explicit-key"),
    ],
)
def test_oss_credentials_stay_with_their_configured_destination(
    monkeypatch: pytest.MonkeyPatch,
    provider: OssDecisionProvider,
    base: str | None,
    key: str | None,
    expected_base: str,
    expected_key: str | None,
) -> None:
    monkeypatch.setenv(f"{provider.upper()}_API_BASE", "http://decision.test/root/")
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "oss-env-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "never-send-this")
    monkeypatch.setenv("NIMBLE_API_KEY", "never-send-nimble-search-key")
    connection: Final = oss_connection(provider, base, key)
    assert (connection.api_base, connection.api_key) == (expected_base, expected_key)
    assert "key" not in repr(connection)


@pytest.mark.parametrize(
    "base",
    ["", "ftp://laya.test", "http://user:password@laya.test", "https://laya.test?key=x", "http://laya.test/#x"],
)
def test_oss_rejects_ambiguous_server_urls(provider: OssDecisionProvider, base: str) -> None:
    with pytest.raises(ValueError, match=provider):
        oss_connection(provider, base)


def test_oss_missing_server_does_not_fall_back_to_typesafe(
    monkeypatch: pytest.MonkeyPatch, provider: OssDecisionProvider
) -> None:
    monkeypatch.delenv(f"{provider.upper()}_API_BASE", raising=False)
    monkeypatch.setenv("TYPESAFE_API_BASE", "https://typesafe.test")
    monkeypatch.setenv("NIMBLE_API_BASE", "https://nimble-search.test")
    with pytest.raises(ValueError, match=f"{provider.upper()}_API_BASE"):
        oss_connection(provider)


def test_oss_request_accepts_the_name_ollama_serves_nimble_under_only_for_bespoke(provider: OssDecisionProvider) -> None:
    body: Final = {"model": "nimble"}
    if provider == "bespoke":
        assert validate_oss_request(provider, body) == "nimble"
        return
    with pytest.raises(ValueError, match=f"{provider} model must be one of"):
        validate_oss_request(provider, body)
