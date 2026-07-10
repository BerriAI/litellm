"""Spec tests for the DCR-bridge envelope consumer helpers.

These pin the two consumer contracts the admission edge depends on: the master-key key
derivation is deterministic, domain-separated, and always yields a >= 32-byte signing key
(so mint and open agree without persisting keys), and the ``Authorization`` classifier is
total over the three cases admission branches on (non-envelope, valid envelope, and
envelope-shaped-but-unopenable), never leaking the recovered upstream token in a repr.
"""

from datetime import datetime, timedelta, timezone

from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (
    BridgeEnvelopeAdmitted,
    BridgeEnvelopeInvalid,
    NotBridgeEnvelope,
    envelope_keys_from_master_key,
    resolve_bridge_envelope,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
    ENVELOPE_PREFIX,
    EnvelopeIdentity,
    EnvelopeKeys,
    SealedEnvelope,
    UpstreamTokenGrant,
    mint_envelope,
)

_NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)
_MASTER_KEY = "sk-master-key-for-derivation-tests-0123456789"
_ACCESS_TOKEN = "upstream-access-token-do-not-leak-8f14e45fceea"
_IDENTITY = EnvelopeIdentity(user_id="user-123", server_id="srv-456")


def _grant() -> UpstreamTokenGrant:
    return UpstreamTokenGrant(access_token=SecretStr(_ACCESS_TOKEN), token_type="Bearer", expires_in=600)


def _sealed_token(keys: EnvelopeKeys, now: datetime = _NOW) -> str:
    sealed = mint_envelope(_IDENTITY, _grant(), keys, now)
    assert isinstance(sealed, SealedEnvelope)
    return sealed.token.get_secret_value()


def test_key_derivation_is_deterministic():
    assert envelope_keys_from_master_key(_MASTER_KEY) == envelope_keys_from_master_key(_MASTER_KEY)


def test_key_derivation_signing_and_encryption_differ():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    assert keys.signing_key.get_secret_value() != keys.encryption_key.get_secret_value()


def test_key_derivation_differs_by_master_key():
    a = envelope_keys_from_master_key(_MASTER_KEY)
    b = envelope_keys_from_master_key(_MASTER_KEY + "x")
    assert a.signing_key.get_secret_value() != b.signing_key.get_secret_value()
    assert a.encryption_key.get_secret_value() != b.encryption_key.get_secret_value()


def test_key_derivation_signing_key_meets_hs256_floor_for_short_master_key():
    keys = envelope_keys_from_master_key("x")
    assert len(keys.signing_key.get_secret_value()) >= 32


def test_derived_keys_round_trip_mint_and_open():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    result = resolve_bridge_envelope(_sealed_token(keys), keys, _NOW)
    assert isinstance(result, BridgeEnvelopeAdmitted)
    assert result.identity == _IDENTITY


def test_resolve_non_envelope_is_not_bridge_envelope():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    assert isinstance(resolve_bridge_envelope("Bearer sk-some-litellm-key", keys, _NOW), NotBridgeEnvelope)
    assert isinstance(resolve_bridge_envelope("plain-token", keys, _NOW), NotBridgeEnvelope)


def test_resolve_valid_envelope_returns_identity_and_upstream_authorization():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    result = resolve_bridge_envelope(_sealed_token(keys), keys, _NOW)
    assert isinstance(result, BridgeEnvelopeAdmitted)
    assert result.identity == _IDENTITY
    assert result.upstream_authorization.get_secret_value() == f"Bearer {_ACCESS_TOKEN}"


def test_resolve_strips_optional_bearer_scheme_before_detection():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    token = _sealed_token(keys)
    assert token.startswith(ENVELOPE_PREFIX)
    bare = resolve_bridge_envelope(token, keys, _NOW)
    prefixed = resolve_bridge_envelope(f"Bearer {token}", keys, _NOW)
    lower = resolve_bridge_envelope(f"bearer {token}", keys, _NOW)
    assert isinstance(bare, BridgeEnvelopeAdmitted)
    assert isinstance(prefixed, BridgeEnvelopeAdmitted)
    assert isinstance(lower, BridgeEnvelopeAdmitted)
    assert prefixed.upstream_authorization.get_secret_value() == bare.upstream_authorization.get_secret_value()


def test_resolve_expired_envelope_is_invalid_not_admitted():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    token = _sealed_token(keys, now=_NOW)
    later = _NOW + timedelta(seconds=601)
    assert isinstance(resolve_bridge_envelope(token, keys, later), BridgeEnvelopeInvalid)


def test_resolve_envelope_minted_under_a_different_master_key_is_invalid():
    minted = envelope_keys_from_master_key(_MASTER_KEY)
    other = envelope_keys_from_master_key("a-completely-different-master-key")
    assert isinstance(resolve_bridge_envelope(_sealed_token(minted), other, _NOW), BridgeEnvelopeInvalid)


def test_resolve_tampered_envelope_is_invalid():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    token = _sealed_token(keys)
    tampered = token[:-4] + ("aaaa" if token[-4:] != "aaaa" else "bbbb")
    result = resolve_bridge_envelope(tampered, keys, _NOW)
    assert isinstance(result, BridgeEnvelopeInvalid)


def test_admitted_result_repr_never_leaks_upstream_token():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    result = resolve_bridge_envelope(_sealed_token(keys), keys, _NOW)
    assert isinstance(result, BridgeEnvelopeAdmitted)
    assert _ACCESS_TOKEN not in repr(result)
    assert _ACCESS_TOKEN not in str(result)
