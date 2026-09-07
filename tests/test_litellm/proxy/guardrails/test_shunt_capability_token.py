"""Unit tests for litellm.proxy.guardrails.shunt_capability_token."""

import pytest

from litellm.proxy.guardrails.shunt_capability_token import (
    TOKEN_TTL_SECONDS,
    mint_shunt_capability_token,
    open_shunt_capability_token,
)


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234-test-salt-key")


class TestMintRequiresExactlyOneIdentity:
    def test_neither_field_raises(self):
        with pytest.raises(ValueError, match="exactly one of key_hash or master_key"):
            mint_shunt_capability_token(key_hash=None, master_key=None)

    def test_both_fields_raises(self):
        with pytest.raises(ValueError, match="exactly one of key_hash or master_key"):
            mint_shunt_capability_token(key_hash="abc", master_key="sk-1234")


class TestRoundTrip:
    def test_key_hash_round_trips(self):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None)
        grant = open_shunt_capability_token(token)
        assert grant is not None
        assert grant.key_hash == "deadbeef"
        assert grant.master_key is None

    def test_master_key_round_trips(self):
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-real-master-key")
        grant = open_shunt_capability_token(token)
        assert grant is not None
        assert grant.master_key == "sk-real-master-key"
        assert grant.key_hash is None

    def test_token_carries_the_shunt_prefix(self):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None)
        assert token.startswith("shunt_cap_v1:")

    def test_token_never_contains_the_raw_master_key_in_plaintext(self):
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-real-master-key")
        assert "sk-real-master-key" not in token


class TestExpiry:
    def test_fresh_token_is_valid(self):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None, now=1_000_000)
        assert open_shunt_capability_token(token, now=1_000_000) is not None

    def test_token_valid_just_before_expiry(self):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None, now=1_000_000)
        assert open_shunt_capability_token(token, now=1_000_000 + TOKEN_TTL_SECONDS - 1) is not None

    def test_token_expired_after_ttl(self):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None, now=1_000_000)
        assert open_shunt_capability_token(token, now=1_000_000 + TOKEN_TTL_SECONDS + 1) is None

    def test_replay_within_ttl_still_opens(self):
        """Deliberately not single-use: a retried Bash command must still authenticate."""
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None, now=1_000_000)
        first = open_shunt_capability_token(token, now=1_000_005)
        second = open_shunt_capability_token(token, now=1_000_010)
        assert first is not None
        assert second is not None
        assert first.key_hash == second.key_hash


class TestMalformedInput:
    def test_wrong_prefix_returns_none(self):
        assert open_shunt_capability_token("not-a-shunt-token") is None

    def test_empty_string_returns_none(self):
        assert open_shunt_capability_token("") is None

    def test_prefix_with_garbage_payload_returns_none(self):
        assert open_shunt_capability_token("shunt_cap_v1:not-valid-ciphertext") is None

    def test_a_sealed_but_differently_shaped_payload_returns_none(self):
        """Cross-type confusion: another sealed value's ciphertext must not parse as a grant."""
        from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

        foreign = "shunt_cap_v1:" + encrypt_value_helper('{"totally": "unrelated"}')
        assert open_shunt_capability_token(foreign) is None
