"""Spec tests for the DCR-bridge envelope producer and consumer helpers.

These pin the contracts the token endpoint and admission edge depend on: the master-key
key derivation is deterministic, domain-separated (keyed HMAC), and always yields a
>= 32-byte signing key; the ``Authorization`` classifier is total over the cases admission
branches on (non-envelope, valid envelope bound to this server, envelope-shaped-but-
unopenable, and envelope minted for a different server); the producer helper round-trips
through the consumer; and no path leaks the upstream token in a repr.
"""

from datetime import datetime, timedelta, timezone

from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (
    BridgeEnvelopeAdmitted,
    BridgeEnvelopeInvalid,
    NotBridgeEnvelope,
    build_bridge_token_response,
    envelope_keys_from_master_key,
    is_bridge_envelope_shaped,
    resolve_bridge_envelope,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
    ENVELOPE_PREFIX,
    EnvelopeIdentity,
    EnvelopeKeys,
    EnvelopeTooLarge,
    SealedEnvelope,
    UpstreamTokenGrant,
    mint_envelope,
)

_NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)
_MASTER_KEY = "sk-master-key-for-derivation-tests-0123456789"
_ACCESS_TOKEN = "upstream-access-token-do-not-leak-8f14e45fceea"
_IDENTITY = EnvelopeIdentity(user_id="user-123", server_id="srv-456")
_SERVER_ID = _IDENTITY.server_id


def _grant() -> UpstreamTokenGrant:
    return UpstreamTokenGrant(access_token=SecretStr(_ACCESS_TOKEN), token_type="Bearer", expires_in=600)


def _sealed_token(keys: EnvelopeKeys, now: datetime = _NOW, identity: EnvelopeIdentity = _IDENTITY) -> str:
    sealed = mint_envelope(identity, _grant(), keys, now)
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


def test_key_derivation_is_cached_so_the_memory_hard_kdf_runs_once_per_key():
    """The scrypt KDF is intentionally expensive to resist offline guessing, so it must be cached:
    repeated calls for the same master key return the identical object rather than re-deriving,
    keeping the per-request admission path free. Returning a distinct object each call would mean
    the cache was dropped and every open would pay the memory-hard cost."""
    first = envelope_keys_from_master_key("sk-cache-probe-key-9988776655")
    assert envelope_keys_from_master_key("sk-cache-probe-key-9988776655") is first


def test_derived_keys_round_trip_mint_and_open():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    result = resolve_bridge_envelope(_sealed_token(keys), keys, _NOW, _SERVER_ID)
    assert isinstance(result, BridgeEnvelopeAdmitted)
    assert result.identity == _IDENTITY


def test_resolve_non_envelope_is_not_bridge_envelope():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    assert isinstance(resolve_bridge_envelope("Bearer sk-some-litellm-key", keys, _NOW, _SERVER_ID), NotBridgeEnvelope)
    assert isinstance(resolve_bridge_envelope("plain-token", keys, _NOW, _SERVER_ID), NotBridgeEnvelope)


def test_resolve_valid_envelope_returns_identity_and_upstream_authorization():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    result = resolve_bridge_envelope(_sealed_token(keys), keys, _NOW, _SERVER_ID)
    assert isinstance(result, BridgeEnvelopeAdmitted)
    assert result.identity == _IDENTITY
    assert result.upstream_authorization.get_secret_value() == f"Bearer {_ACCESS_TOKEN}"


def test_resolve_strips_optional_bearer_scheme_before_detection():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    token = _sealed_token(keys)
    assert token.startswith(ENVELOPE_PREFIX)
    bare = resolve_bridge_envelope(token, keys, _NOW, _SERVER_ID)
    prefixed = resolve_bridge_envelope(f"Bearer {token}", keys, _NOW, _SERVER_ID)
    lower = resolve_bridge_envelope(f"bearer {token}", keys, _NOW, _SERVER_ID)
    assert isinstance(bare, BridgeEnvelopeAdmitted)
    assert isinstance(prefixed, BridgeEnvelopeAdmitted)
    assert isinstance(lower, BridgeEnvelopeAdmitted)
    assert prefixed.upstream_authorization.get_secret_value() == bare.upstream_authorization.get_secret_value()


def test_resolve_expired_envelope_is_invalid_not_admitted():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    token = _sealed_token(keys, now=_NOW)
    later = _NOW + timedelta(seconds=601)
    assert isinstance(resolve_bridge_envelope(token, keys, later, _SERVER_ID), BridgeEnvelopeInvalid)


def test_resolve_envelope_minted_under_a_different_master_key_is_invalid():
    minted = envelope_keys_from_master_key(_MASTER_KEY)
    other = envelope_keys_from_master_key("a-completely-different-master-key")
    assert isinstance(resolve_bridge_envelope(_sealed_token(minted), other, _NOW, _SERVER_ID), BridgeEnvelopeInvalid)


def test_resolve_tampered_envelope_is_invalid():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    token = _sealed_token(keys)
    tampered = token[:-4] + ("aaaa" if token[-4:] != "aaaa" else "bbbb")
    result = resolve_bridge_envelope(tampered, keys, _NOW, _SERVER_ID)
    assert isinstance(result, BridgeEnvelopeInvalid)


def test_resolve_envelope_minted_for_another_server_is_invalid():
    """An envelope sealed for server A must be rejected when presented to server B, so a
    captured or misrouted envelope cannot forward one server's upstream credential to
    another. The valid access token stays sealed; the mismatch alone fails the resolve."""
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    other_server_identity = EnvelopeIdentity(user_id=_IDENTITY.user_id, server_id="srv-OTHER")
    token = _sealed_token(keys, identity=other_server_identity)
    result = resolve_bridge_envelope(token, keys, _NOW, _SERVER_ID)
    assert isinstance(result, BridgeEnvelopeInvalid)


def test_resolve_matching_server_binding_is_admitted():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    result = resolve_bridge_envelope(_sealed_token(keys), keys, _NOW, "srv-456")
    assert isinstance(result, BridgeEnvelopeAdmitted)


def test_resolve_non_ascii_server_id_stays_total_and_does_not_raise():
    """The server-binding check must not raise on a non-ASCII server_id (an admin can register a
    unicode server_id); it stays total and returns a typed result. A matching non-ASCII id admits,
    a mismatching one is BridgeEnvelopeInvalid, and neither raises."""
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    unicode_identity = EnvelopeIdentity(user_id=_IDENTITY.user_id, server_id="srv-café")
    token = _sealed_token(keys, identity=unicode_identity)
    assert isinstance(resolve_bridge_envelope(token, keys, _NOW, "srv-café"), BridgeEnvelopeAdmitted)
    assert isinstance(resolve_bridge_envelope(token, keys, _NOW, "srv-cafe"), BridgeEnvelopeInvalid)


def test_resolve_strips_bearer_with_extra_whitespace():
    """A Bearer scheme separated by extra spaces or a tab still yields the envelope, so a client
    using non-minimal but legal whitespace is not misclassified as a non-envelope."""
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    token = _sealed_token(keys)
    for header in (f"Bearer  {token}", f"Bearer\t{token}", f"  Bearer {token}"):
        assert isinstance(resolve_bridge_envelope(header, keys, _NOW, _SERVER_ID), BridgeEnvelopeAdmitted)


def test_admitted_result_repr_never_leaks_upstream_token():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    result = resolve_bridge_envelope(_sealed_token(keys), keys, _NOW, _SERVER_ID)
    assert isinstance(result, BridgeEnvelopeAdmitted)
    assert _ACCESS_TOKEN not in repr(result)
    assert _ACCESS_TOKEN not in str(result)


def test_build_bridge_token_response_round_trips_through_the_consumer():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    sealed = build_bridge_token_response(_IDENTITY, _grant(), keys, _NOW)
    assert isinstance(sealed, SealedEnvelope)
    assert sealed.token.get_secret_value().startswith(ENVELOPE_PREFIX)
    result = resolve_bridge_envelope(sealed.token.get_secret_value(), keys, _NOW, _SERVER_ID)
    assert isinstance(result, BridgeEnvelopeAdmitted)
    assert result.identity == _IDENTITY
    assert result.upstream_authorization.get_secret_value() == f"Bearer {_ACCESS_TOKEN}"


def test_build_bridge_token_response_oversized_grant_returns_error_value():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    huge = UpstreamTokenGrant(access_token=SecretStr("x" * 20000), token_type="Bearer")
    result = build_bridge_token_response(_IDENTITY, huge, keys, _NOW)
    assert isinstance(result, EnvelopeTooLarge)


def test_build_bridge_token_response_repr_never_leaks_upstream_token():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    sealed = build_bridge_token_response(_IDENTITY, _grant(), keys, _NOW)
    assert isinstance(sealed, SealedEnvelope)
    assert _ACCESS_TOKEN not in repr(sealed)
    assert _ACCESS_TOKEN not in str(sealed)


def test_is_bridge_envelope_shaped_detects_envelope_with_and_without_bearer():
    keys = envelope_keys_from_master_key(_MASTER_KEY)
    token = _sealed_token(keys)
    assert is_bridge_envelope_shaped(token) is True
    assert is_bridge_envelope_shaped(f"Bearer {token}") is True
    assert is_bridge_envelope_shaped(f"bearer {token}") is True


def test_is_bridge_envelope_shaped_rejects_non_envelope_bearer():
    assert is_bridge_envelope_shaped("Bearer sk-some-litellm-key") is False
    assert is_bridge_envelope_shaped("plain-upstream-token") is False
    assert is_bridge_envelope_shaped("") is False
