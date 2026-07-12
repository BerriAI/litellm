"""Spec tests for the sealed-envelope module (oauth_delegate DCR bridge).

The envelope is the single client-held bearer carrying both a litellm identity and the
encrypted upstream grant, with zero server-side storage. These tests pin the security
contract: an envelope opens only under the exact keys that minted it, tampering with any
signed byte is detected, expiry is enforced against the injected clock (capped by the
module TTL ceiling), oversized envelopes are rejected rather than truncated, and no
error value, model repr, or raised exception ever contains the inner access token.
"""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr, ValidationError

from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
    ENVELOPE_ISSUER,
    ENVELOPE_PREFIX,
    MAX_ENVELOPE_BYTES,
    MAX_ENVELOPE_TTL_SECONDS,
    BadSignature,
    DecryptFailed,
    EnvelopeIdentity,
    EnvelopeKeys,
    EnvelopeTooLarge,
    Expired,
    MalformedPayload,
    NotAnEnvelope,
    OpenedEnvelope,
    SealedEnvelope,
    UpstreamTokenGrant,
    is_envelope,
    mint_envelope,
    open_envelope,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value, encrypt_value

_NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)
_SIGNING_KEY = "unit-test-signing-key-0123456789abcdef0123456789abcdef"
_ENCRYPTION_KEY = "unit-test-encryption-key-fedcba9876543210fedcba9876543210"
_OTHER_SIGNING_KEY = "other-signing-key-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_OTHER_ENCRYPTION_KEY = "other-encryption-key-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_KEYS = EnvelopeKeys(signing_key=SecretStr(_SIGNING_KEY), encryption_key=SecretStr(_ENCRYPTION_KEY))
_WRONG_SIGNING = EnvelopeKeys(signing_key=SecretStr(_OTHER_SIGNING_KEY), encryption_key=SecretStr(_ENCRYPTION_KEY))
_WRONG_ENCRYPTION = EnvelopeKeys(signing_key=SecretStr(_SIGNING_KEY), encryption_key=SecretStr(_OTHER_ENCRYPTION_KEY))
_ACCESS_TOKEN = "upstream-access-token-do-not-leak-8f14e45fceea"
_REFRESH_TOKEN = "upstream-refresh-token-do-not-leak-1d0aa4b7"
_IDENTITY = EnvelopeIdentity(server_id="srv-456", key_hash="hashed-key-123")


def _full_grant() -> UpstreamTokenGrant:
    return UpstreamTokenGrant(
        access_token=SecretStr(_ACCESS_TOKEN),
        token_type="Bearer",
        refresh_token=SecretStr(_REFRESH_TOKEN),
        scope="read:tools write:tools",
        expires_in=600,
    )


def _minimal_grant() -> UpstreamTokenGrant:
    return UpstreamTokenGrant(access_token=SecretStr(_ACCESS_TOKEN), token_type="Bearer")


def _sealed_token(grant: UpstreamTokenGrant, keys: EnvelopeKeys = _KEYS) -> str:
    sealed = mint_envelope(_IDENTITY, grant, keys, _NOW)
    assert isinstance(sealed, SealedEnvelope)
    return sealed.token.get_secret_value()


def _unverified_claims(sealed_token: str) -> dict[str, object]:
    return jwt.decode(sealed_token.removeprefix(ENVELOPE_PREFIX), options={"verify_signature": False})


def _forge(claims: dict[str, object], signing_key: str = _SIGNING_KEY) -> str:
    return ENVELOPE_PREFIX + jwt.encode(claims, signing_key, algorithm="HS256")


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _hand_crafted_hs256(payload: dict[str, object], signing_key: str = _SIGNING_KEY) -> str:
    """Assemble an HS256 envelope from raw bytes, bypassing PyJWT's encode-side claim
    guards (it refuses to build a token with a non-string ``iss``). This is the real
    attacker path: a client crafts the compact JWT directly, so any registered claim can
    carry a hostile JSON type."""
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    body = _b64url(json.dumps(payload).encode("utf-8"))
    signing_input = f"{header}.{body}".encode("ascii")
    signature = _b64url(hmac.new(signing_key.encode("utf-8"), signing_input, hashlib.sha256).digest())
    return ENVELOPE_PREFIX + f"{header}.{body}.{signature}"


def _tampered(sealed_token: str, segment: int, index: int) -> str:
    parts = sealed_token.removeprefix(ENVELOPE_PREFIX).split(".")
    original = parts[segment][index]
    replacement = "A" if original in "QRST" else "Q"
    mutated = parts[segment][:index] + replacement + parts[segment][index + 1 :]
    rebuilt = ".".join(parts[:segment] + [mutated] + parts[segment + 1 :])
    return ENVELOPE_PREFIX + rebuilt


def test_round_trip_recovers_identity_and_grant_exactly():
    grant = _full_grant()
    token = _sealed_token(grant)
    assert is_envelope(token)
    opened = open_envelope(token, _KEYS, _NOW)
    assert isinstance(opened, OpenedEnvelope)
    assert opened.identity == _IDENTITY
    assert opened.grant == grant
    assert opened.grant.access_token.get_secret_value() == _ACCESS_TOKEN
    assert opened.grant.refresh_token is not None
    assert opened.grant.refresh_token.get_secret_value() == _REFRESH_TOKEN


def test_minimal_grant_round_trips_without_none_leakage_into_claims():
    token = _sealed_token(_minimal_grant())
    claims = _unverified_claims(token)
    blob = claims["grant"]
    assert isinstance(blob, str)
    plaintext = decrypt_value(value=base64.urlsafe_b64decode(blob), signing_key=_ENCRYPTION_KEY)
    assert set(json.loads(plaintext)) == {"access_token", "token_type"}
    opened = open_envelope(token, _KEYS, _NOW)
    assert isinstance(opened, OpenedEnvelope)
    assert opened.grant.refresh_token is None
    assert opened.grant.scope is None
    assert opened.grant.expires_in is None


def test_claim_layout_and_no_plaintext_token_in_envelope():
    token = _sealed_token(_full_grant())
    claims = _unverified_claims(token)
    assert set(claims) == {"iss", "iat", "exp", "server_id", "key_hash", "grant"}
    assert claims["iss"] == ENVELOPE_ISSUER
    assert claims["iat"] == int(_NOW.timestamp())
    assert claims["exp"] == int(_NOW.timestamp()) + 600
    assert claims["server_id"] == "srv-456"
    assert claims["key_hash"] == "hashed-key-123"
    assert _ACCESS_TOKEN not in token
    assert _ACCESS_TOKEN not in json.dumps(claims)
    assert _REFRESH_TOKEN not in json.dumps(claims)


@pytest.mark.parametrize(
    "expires_in, expected_ttl",
    [
        (600, 600),
        (MAX_ENVELOPE_TTL_SECONDS + 82800, MAX_ENVELOPE_TTL_SECONDS),
        (None, MAX_ENVELOPE_TTL_SECONDS),
    ],
)
def test_exp_is_min_of_upstream_expires_in_and_cap(expires_in, expected_ttl):
    grant = UpstreamTokenGrant(access_token=SecretStr(_ACCESS_TOKEN), token_type="Bearer", expires_in=expires_in)
    sealed = mint_envelope(_IDENTITY, grant, _KEYS, _NOW)
    assert isinstance(sealed, SealedEnvelope)
    assert sealed.expires_at == _NOW + timedelta(seconds=expected_ttl)


def test_expiry_honored_against_injected_clock():
    token = _sealed_token(_full_grant())
    assert isinstance(open_envelope(token, _KEYS, _NOW + timedelta(seconds=599)), OpenedEnvelope)
    assert isinstance(open_envelope(token, _KEYS, _NOW + timedelta(seconds=600)), Expired)
    assert isinstance(open_envelope(token, _KEYS, _NOW + timedelta(seconds=601)), Expired)


def test_ttl_cap_enforced_on_open_even_when_upstream_token_lives_longer():
    grant = UpstreamTokenGrant(access_token=SecretStr(_ACCESS_TOKEN), token_type="Bearer", expires_in=86400)
    token = _sealed_token(grant)
    just_before_cap = _NOW + timedelta(seconds=MAX_ENVELOPE_TTL_SECONDS - 1)
    at_cap = _NOW + timedelta(seconds=MAX_ENVELOPE_TTL_SECONDS)
    assert isinstance(open_envelope(token, _KEYS, just_before_cap), OpenedEnvelope)
    assert isinstance(open_envelope(token, _KEYS, at_cap), Expired)


def test_tampering_any_payload_or_signature_byte_is_bad_signature():
    token = _sealed_token(_full_grant())
    parts = token.removeprefix(ENVELOPE_PREFIX).split(".")
    for segment in (1, 2):
        for index in range(len(parts[segment])):
            result = open_envelope(_tampered(token, segment, index), _KEYS, _NOW)
            assert isinstance(result, BadSignature), f"segment {segment} index {index}: {result!r}"


def test_tampering_header_bytes_never_opens():
    token = _sealed_token(_full_grant())
    parts = token.removeprefix(ENVELOPE_PREFIX).split(".")
    for index in range(len(parts[0])):
        result = open_envelope(_tampered(token, 0, index), _KEYS, _NOW)
        assert isinstance(result, (BadSignature, MalformedPayload)), f"header index {index}: {result!r}"


def test_alg_none_is_rejected():
    claims = _unverified_claims(_sealed_token(_full_grant()))
    unsigned = ENVELOPE_PREFIX + jwt.encode(claims, None, algorithm="none")
    assert isinstance(open_envelope(unsigned, _KEYS, _NOW), MalformedPayload)


def test_wrong_signing_key_is_bad_signature():
    token = _sealed_token(_full_grant())
    assert isinstance(open_envelope(token, _WRONG_SIGNING, _NOW), BadSignature)


def test_wrong_encryption_key_is_decrypt_failed():
    token = _sealed_token(_full_grant())
    assert isinstance(open_envelope(token, _WRONG_ENCRYPTION, _NOW), DecryptFailed)


def test_ciphertext_swapped_from_another_envelope_is_decrypt_failed():
    claims_a = _unverified_claims(_sealed_token(_full_grant(), keys=_KEYS))
    claims_b = _unverified_claims(_sealed_token(_minimal_grant(), keys=_WRONG_ENCRYPTION))
    swapped = _forge({**claims_a, "grant": claims_b["grant"]})
    assert isinstance(open_envelope(swapped, _KEYS, _NOW), DecryptFailed)


def test_wrong_issuer_is_malformed_payload():
    claims = _unverified_claims(_sealed_token(_full_grant()))
    assert isinstance(open_envelope(_forge({**claims, "iss": "evil-issuer"}), _KEYS, _NOW), MalformedPayload)


def test_missing_identity_claim_is_malformed_payload():
    claims = _unverified_claims(_sealed_token(_full_grant()))
    forged = _forge({key: value for key, value in claims.items() if key != "key_hash"})
    assert isinstance(open_envelope(forged, _KEYS, _NOW), MalformedPayload)


@pytest.mark.parametrize("identity_claim", ["server_id", "key_hash"])
def test_signed_empty_identity_claim_is_malformed_payload_not_a_raise(identity_claim):
    claims = _unverified_claims(_sealed_token(_full_grant()))
    forged = _forge({**claims, identity_claim: ""})
    assert isinstance(open_envelope(forged, _KEYS, _NOW), MalformedPayload)
    intact = open_envelope(_forge(claims), _KEYS, _NOW)
    assert isinstance(intact, OpenedEnvelope)
    assert intact.identity == _IDENTITY


def test_lone_surrogate_candidate_is_malformed_payload_not_a_raise():
    surrogate_candidate = ENVELOPE_PREFIX + "\ud800abc.def.ghi"
    result = open_envelope(surrogate_candidate, _KEYS, _NOW)
    assert isinstance(result, MalformedPayload)


@pytest.mark.parametrize(
    "override",
    [
        {"iat": [1]},
        {"iat": {}},
        {"iat": float("inf")},
        {"nbf": None},
        {"nbf": [1]},
    ],
)
def test_hostile_iat_nbf_types_are_malformed_payload_not_a_raise(override):
    claims = _unverified_claims(_sealed_token(_full_grant()))
    forged = _forge({**claims, **override})
    result = open_envelope(forged, _KEYS, _NOW)
    assert isinstance(result, MalformedPayload)


@pytest.mark.parametrize("hostile_iss", [["litellm-mcp-bridge"], 5, {"iss": "x"}])
def test_non_string_issuer_claim_is_malformed_payload_not_a_raise(hostile_iss):
    claims = _unverified_claims(_sealed_token(_full_grant()))
    forged = _hand_crafted_hs256({**claims, "iss": hostile_iss})
    result = open_envelope(forged, _KEYS, _NOW)
    assert isinstance(result, MalformedPayload)


@pytest.mark.parametrize("hostile_exp", ["600", 600.5, [600]])
def test_non_int_exp_claim_is_malformed_payload_not_a_raise(hostile_exp):
    claims = _unverified_claims(_sealed_token(_full_grant()))
    forged = _hand_crafted_hs256({**claims, "exp": hostile_exp})
    result = open_envelope(forged, _KEYS, _NOW)
    assert isinstance(result, MalformedPayload)


def test_unexpected_extra_claim_is_malformed_payload():
    claims = _unverified_claims(_sealed_token(_full_grant()))
    forged = _forge({**claims, "role": "admin"})
    assert isinstance(open_envelope(forged, _KEYS, _NOW), MalformedPayload)


def test_future_iat_opens_against_injected_now_not_wall_clock():
    future = _NOW + timedelta(seconds=100_000)
    sealed = mint_envelope(_IDENTITY, _full_grant(), _KEYS, future)
    assert isinstance(sealed, SealedEnvelope)
    opened = open_envelope(sealed.token.get_secret_value(), _KEYS, future)
    assert isinstance(opened, OpenedEnvelope)
    assert opened.identity == _IDENTITY
    assert opened.grant == _full_grant()


def test_rs256_signed_token_is_rejected_against_the_hs256_pin():
    claims = _unverified_claims(_sealed_token(_full_grant()))
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rs256_token = ENVELOPE_PREFIX + jwt.encode(claims, private_key, algorithm="RS256")
    result = open_envelope(rs256_token, _KEYS, _NOW)
    assert isinstance(result, MalformedPayload)


@pytest.mark.parametrize("short_key", ["", "too-short", "x" * 31])
def test_signing_key_below_hs256_minimum_is_rejected_at_construction(short_key):
    with pytest.raises(ValidationError):
        EnvelopeKeys(signing_key=SecretStr(short_key), encryption_key=SecretStr(_ENCRYPTION_KEY))


def test_signing_key_at_hs256_minimum_is_accepted():
    keys = EnvelopeKeys(signing_key=SecretStr("y" * 32), encryption_key=SecretStr(_ENCRYPTION_KEY))
    assert keys.signing_key.get_secret_value() == "y" * 32


def test_correctly_signed_garbage_grant_blob_is_decrypt_failed():
    claims = _unverified_claims(_sealed_token(_full_grant()))
    forged = _forge({**claims, "grant": "not-a-ciphertext"})
    assert isinstance(open_envelope(forged, _KEYS, _NOW), DecryptFailed)


def test_decryptable_blob_that_is_not_a_grant_is_malformed_payload():
    claims = _unverified_claims(_sealed_token(_full_grant()))
    wrong_shape = base64.urlsafe_b64encode(
        bytes(encrypt_value(value=json.dumps({"nope": 1}), signing_key=_ENCRYPTION_KEY))
    ).decode("ascii")
    forged = _forge({**claims, "grant": wrong_shape})
    assert isinstance(open_envelope(forged, _KEYS, _NOW), MalformedPayload)


def _mint_with_token_len(n: int) -> SealedEnvelope | EnvelopeTooLarge:
    grant = UpstreamTokenGrant(access_token=SecretStr("a" * n), token_type="Bearer")
    return mint_envelope(_IDENTITY, grant, _KEYS, _NOW)


def _largest_token_len_that_mints(lo: int, hi: int) -> int:
    if hi - lo <= 1:
        return lo
    mid = (lo + hi) // 2
    if isinstance(_mint_with_token_len(mid), SealedEnvelope):
        return _largest_token_len_that_mints(mid, hi)
    return _largest_token_len_that_mints(lo, mid)


def test_oversized_grant_is_a_typed_mint_error_never_truncated():
    result = _mint_with_token_len(30000)
    assert isinstance(result, EnvelopeTooLarge)
    assert result.tag == "envelope_too_large"
    assert result.size_bytes > MAX_ENVELOPE_BYTES
    assert result.max_bytes == MAX_ENVELOPE_BYTES


def test_size_cap_boundary_just_under_succeeds_and_just_over_fails():
    assert isinstance(_mint_with_token_len(1), SealedEnvelope)
    assert isinstance(_mint_with_token_len(30000), EnvelopeTooLarge)
    largest = _largest_token_len_that_mints(1, 30000)
    assert largest > 6000
    sealed = _mint_with_token_len(largest)
    assert isinstance(sealed, SealedEnvelope)
    assert len(sealed.token.get_secret_value().encode("utf-8")) <= MAX_ENVELOPE_BYTES
    overflowing = _mint_with_token_len(largest + 1)
    assert isinstance(overflowing, EnvelopeTooLarge)
    assert overflowing.size_bytes > MAX_ENVELOPE_BYTES
    opened = open_envelope(sealed.token.get_secret_value(), _KEYS, _NOW)
    assert isinstance(opened, OpenedEnvelope)


def test_open_size_guard_measures_bytes_not_characters():
    """The open-side size guard must reject on UTF-8 byte length, matching mint's cap, so a
    hostile multi-byte candidate whose character count is under the cap but whose byte count is
    over it is rejected up front rather than reaching the expensive HMAC/decrypt path. Patching
    _decode_claims to fail loudly proves the guard short-circuits before decode."""
    from unittest.mock import patch

    from litellm.proxy._experimental.mcp_server.outbound_credentials import envelope

    multibyte_body = "é" * 7000  # 7000 chars, 14000 UTF-8 bytes
    candidate = ENVELOPE_PREFIX + multibyte_body
    assert len(candidate) <= MAX_ENVELOPE_BYTES
    assert len(candidate.encode("utf-8")) > MAX_ENVELOPE_BYTES

    with patch.object(envelope, "_decode_claims", side_effect=AssertionError("decode reached")) as decode:
        result = open_envelope(candidate, _KEYS, _NOW)

    assert isinstance(result, MalformedPayload)
    decode.assert_not_called()


def test_open_size_guard_rejects_oversize_character_count_before_decode():
    """A candidate whose character count already exceeds the cap is rejected up front, before the
    decode path, so an arbitrarily long hostile string is not run through HMAC/decrypt. The cheap
    character precheck makes this O(1) since UTF-8 byte length is never below character length."""
    from unittest.mock import patch

    from litellm.proxy._experimental.mcp_server.outbound_credentials import envelope

    candidate = ENVELOPE_PREFIX + ("a" * (MAX_ENVELOPE_BYTES + 1))
    assert len(candidate) > MAX_ENVELOPE_BYTES

    with patch.object(envelope, "_decode_claims", side_effect=AssertionError("decode reached")) as decode:
        result = open_envelope(candidate, _KEYS, _NOW)

    assert isinstance(result, MalformedPayload)
    decode.assert_not_called()


def test_is_envelope_detects_only_prefixed_values():
    assert is_envelope(_sealed_token(_full_grant()))
    raw_jwt = jwt.encode({"sub": "user-123"}, _SIGNING_KEY, algorithm="HS256")
    assert not is_envelope(raw_jwt)
    assert not is_envelope("some-random-opaque-token")
    assert not is_envelope("")


def test_open_on_non_envelope_input_is_not_an_envelope():
    raw_jwt = jwt.encode({"sub": "user-123"}, _SIGNING_KEY, algorithm="HS256")
    assert isinstance(open_envelope(raw_jwt, _KEYS, _NOW), NotAnEnvelope)
    assert isinstance(open_envelope("", _KEYS, _NOW), NotAnEnvelope)
    assert isinstance(open_envelope(_ACCESS_TOKEN, _KEYS, _NOW), NotAnEnvelope)


def test_open_on_prefixed_garbage_is_malformed_payload():
    assert isinstance(open_envelope(ENVELOPE_PREFIX + "garbage", _KEYS, _NOW), MalformedPayload)
    assert isinstance(open_envelope(ENVELOPE_PREFIX + _ACCESS_TOKEN, _KEYS, _NOW), MalformedPayload)


def test_no_result_value_ever_reveals_the_access_token():
    grant = _full_grant()
    sealed = mint_envelope(_IDENTITY, grant, _KEYS, _NOW)
    assert isinstance(sealed, SealedEnvelope)
    token = sealed.token.get_secret_value()
    oversized_grant = UpstreamTokenGrant(access_token=SecretStr(_ACCESS_TOKEN + "x" * 30000), token_type="Bearer")
    values = (
        sealed,
        open_envelope(token, _KEYS, _NOW),
        mint_envelope(_IDENTITY, oversized_grant, _KEYS, _NOW),
        open_envelope(_ACCESS_TOKEN, _KEYS, _NOW),
        open_envelope(ENVELOPE_PREFIX + _ACCESS_TOKEN, _KEYS, _NOW),
        open_envelope(token, _WRONG_SIGNING, _NOW),
        open_envelope(token, _WRONG_ENCRYPTION, _NOW),
        open_envelope(token, _KEYS, _NOW + timedelta(seconds=601)),
        grant,
    )
    for value in values:
        assert _ACCESS_TOKEN not in repr(value)
        assert _ACCESS_TOKEN not in str(value)
        assert _REFRESH_TOKEN not in repr(value)
        assert _REFRESH_TOKEN not in str(value)


def test_non_positive_expires_in_is_rejected_at_construction_without_leaking():
    for bad_expires_in in (0, -5):
        with pytest.raises(ValidationError) as excinfo:
            UpstreamTokenGrant(
                access_token=SecretStr(_ACCESS_TOKEN),
                token_type="Bearer",
                expires_in=bad_expires_in,
            )
        assert _ACCESS_TOKEN not in str(excinfo.value)
        assert _ACCESS_TOKEN not in repr(excinfo.value)


def test_empty_identity_and_key_fields_are_rejected_at_construction():
    with pytest.raises(ValidationError):
        EnvelopeIdentity(server_id="", key_hash="hashed-key-123")
    with pytest.raises(ValidationError):
        EnvelopeIdentity(server_id="srv-456", key_hash="")
    with pytest.raises(ValidationError):
        EnvelopeKeys(signing_key=SecretStr(""), encryption_key=SecretStr(_ENCRYPTION_KEY))
    with pytest.raises(ValidationError):
        EnvelopeKeys(signing_key=SecretStr(_SIGNING_KEY), encryption_key=SecretStr(""))
    with pytest.raises(ValidationError):
        UpstreamTokenGrant(access_token=SecretStr(""), token_type="Bearer")


def test_public_models_are_frozen():
    sealed = mint_envelope(_IDENTITY, _full_grant(), _KEYS, _NOW)
    assert isinstance(sealed, SealedEnvelope)
    opened = open_envelope(sealed.token.get_secret_value(), _KEYS, _NOW)
    assert isinstance(opened, OpenedEnvelope)
    with pytest.raises(ValidationError):
        sealed.token = SecretStr("overwritten")
    with pytest.raises(ValidationError):
        opened.grant = _minimal_grant()
    with pytest.raises(ValidationError):
        _IDENTITY.key_hash = "someone-elses-hash"
