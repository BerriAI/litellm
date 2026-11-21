import json
from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.realtime_endpoints import live_endpoints
from litellm.proxy.realtime_endpoints.live_endpoints import _attachable_session, _session_token

_NOW: Final = 1_800_000_000
_OWNER: Final = UserAPIKeyAuth(token="owner-key-hash", team_id="team-a")


@pytest.fixture(autouse=True)
def _fixed_clock_and_salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-live-session-test-salt")
    monkeypatch.setattr(live_endpoints.time, "time", lambda: _NOW)


def test_session_token_hides_the_upstream_id_and_opens_for_its_owner() -> None:
    token: Final = _session_token("live_upstream", "gpt-live-1", _OWNER)

    assert "live_upstream" not in token
    assert _attachable_session(token, _OWNER) == ("live_upstream", "gpt-live-1")


@pytest.mark.parametrize(
    ("caller", "expected"),
    [
        (UserAPIKeyAuth(token="other-key-hash", team_id="team-a"), ("live_upstream", "gpt-live-1")),
        (UserAPIKeyAuth(token="other-key-hash", team_id="team-b"), None),
        (UserAPIKeyAuth(token="other-key-hash"), None),
        (UserAPIKeyAuth(), None),
    ],
    ids=["same-team", "other-team", "other-key-no-team", "no-identity"],
)
def test_session_token_opens_only_for_the_owning_key_or_team(
    caller: UserAPIKeyAuth, expected: tuple[str, str] | None
) -> None:
    assert _attachable_session(_session_token("live_upstream", "gpt-live-1", _OWNER), caller) == expected


def test_session_token_from_a_key_without_a_team_does_not_open_for_another_teamless_key() -> None:
    token: Final = _session_token("live_upstream", "gpt-live-1", UserAPIKeyAuth(token="owner-key-hash"))

    assert _attachable_session(token, UserAPIKeyAuth(token="other-key-hash")) is None


def test_expired_session_token_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    token: Final = _session_token("live_upstream", "gpt-live-1", _OWNER)
    monkeypatch.setattr(live_endpoints.time, "time", lambda: _NOW + live_endpoints._LIVE_SESSION_TOKEN_TTL_SECONDS + 1)

    assert _attachable_session(token, _OWNER) is None


def _token_with(**overrides: object) -> str:
    decrypted: Final = decrypt_value_helper(
        value=_session_token("live_upstream", "gpt-live-1", _OWNER), key="live_session_id"
    )
    assert decrypted is not None
    return encrypt_value_helper(json.dumps({**json.loads(decrypted), **overrides}))


@pytest.mark.parametrize("token", ["live_upstream", ""], ids=["raw-upstream-id", "empty"])
def test_a_session_id_the_proxy_did_not_issue_is_rejected(token: str) -> None:
    assert _attachable_session(token, _OWNER) is None


@pytest.mark.parametrize("plaintext", ["not json", "[]"], ids=["not-json", "not-an-object"])
def test_an_encrypted_value_that_is_not_a_session_token_is_rejected(plaintext: str) -> None:
    assert _attachable_session(encrypt_value_helper(plaintext), _OWNER) is None


@pytest.mark.parametrize(
    "overrides",
    [{"v": "v1"}, {"session_id": 7}, {"model": None}, {"expires_at": "soon"}],
    ids=["realtime-token-version", "bad-session-id", "no-model", "bad-expiry"],
)
def test_a_malformed_session_token_is_rejected(overrides: dict[str, object]) -> None:
    assert _attachable_session(_token_with(**overrides), _OWNER) is None
