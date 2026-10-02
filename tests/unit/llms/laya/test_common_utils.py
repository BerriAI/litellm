from collections.abc import Mapping
from typing import Final

import pytest

from litellm.llms.laya.common_utils import laya_connection, laya_response_model


@pytest.mark.parametrize(
    ("base", "key", "expected_base", "expected_key"),
    [
        (None, None, "http://laya.test/root", "laya-env-key"),
        ("http://custom.test/", None, "http://custom.test", None),
        ("http://custom.test/", "explicit-key", "http://custom.test", "explicit-key"),
    ],
)
def test_laya_credentials_stay_with_their_configured_destination(
    monkeypatch: pytest.MonkeyPatch,
    base: str | None,
    key: str | None,
    expected_base: str,
    expected_key: str | None,
) -> None:
    monkeypatch.setenv("LAYA_API_BASE", "http://laya.test/root/")
    monkeypatch.setenv("LAYA_API_KEY", "laya-env-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "never-send-this")
    connection: Final = laya_connection(base, key)
    assert (connection.api_base, connection.api_key) == (expected_base, expected_key)
    assert "key" not in repr(connection)


@pytest.mark.parametrize(
    "base",
    ["", "ftp://laya.test", "http://user:password@laya.test", "https://laya.test?key=x", "http://laya.test/#x"],
)
def test_laya_rejects_ambiguous_server_urls(base: str) -> None:
    with pytest.raises(ValueError, match="Laya"):
        laya_connection(base)


def test_laya_missing_server_does_not_fall_back_to_typesafe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAYA_API_BASE", raising=False)
    monkeypatch.setenv("TYPESAFE_API_BASE", "https://typesafe.test")
    with pytest.raises(ValueError, match="LAYA_API_BASE"):
        laya_connection()


@pytest.mark.parametrize(
    ("routing", "requested", "expected"),
    [
        ({"model": "multilingual"}, "english", "multilingual"),
        (None, "english", "english"),
        ({"model": 42}, "english", "english"),
        (None, None, "unknown"),
    ],
)
def test_laya_identity_tracks_the_checkpoint_not_the_shared_agent_name(
    routing: Mapping[str, object] | None, requested: str | None, expected: str
) -> None:
    assert laya_response_model({"model": "laya-rl-agent", "routing": routing}, requested) == expected
