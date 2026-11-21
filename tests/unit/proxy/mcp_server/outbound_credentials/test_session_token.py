"""Tests for the identity-only gateway session token (mint/open, hostile-input totality)."""

from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.utils import base64url_decode, base64url_encode
from pydantic import SecretStr, TypeAdapter, ValidationError

from litellm.proxy.mcp_server.outbound_credentials.session_token import (
    MAX_SESSION_TOKEN_BYTES,
    SESSION_ISSUER,
    SESSION_REFRESH_PREFIX,
    SESSION_REFRESH_TTL_SECONDS,
    SESSION_TOKEN_PREFIX,
    SESSION_TTL_SECONDS,
    AsymmetricSessionKeys,
    MintedSessionToken,
    NotASessionToken,
    OpenedSessionToken,
    SessionBadSignature,
    SessionExpired,
    SessionKeys,
    SessionMalformed,
    SessionPrincipal,
    SessionRotatedPublicKey,
    SessionTokenTooLarge,
    is_session_refresh_token,
    is_session_token,
    mint_session_refresh_token,
    mint_session_token,
    open_session_refresh_token,
    open_session_token,
    session_public_key_pem,
)


def _rsa_private_pem(bits: int = 2048) -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


_RSA_PEM_A = _rsa_private_pem()
_RSA_PEM_B = _rsa_private_pem()

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
KEYS = SessionKeys(signing_key=SecretStr("k" * 32))
OTHER_KEYS = SessionKeys(signing_key=SecretStr("x" * 32))
PRINCIPAL = SessionPrincipal(user_id="user-123", client_id="llm_client_abc")


def _mint_access() -> str:
    minted = mint_session_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    return minted.token.get_secret_value()


def _mint_refresh() -> str:
    minted = mint_session_refresh_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    return minted.token.get_secret_value()


def _corrupt_signature(token: str) -> str:
    unsigned, signature = token.rsplit(".", 1)
    raw = base64url_decode(signature)
    return f"{unsigned}.{base64url_encode(bytes((raw[0] ^ 0x01,)) + raw[1:]).decode()}"


def _sign_claims(payload: dict, prefix: str = SESSION_TOKEN_PREFIX, keys: SessionKeys = KEYS) -> str:
    return prefix + jwt.encode(payload, keys.signing_key.get_secret_value(), algorithm="HS256")


def _valid_claims(**overrides) -> dict:
    base = {
        "iss": SESSION_ISSUER,
        "iat": int(NOW.timestamp()),
        "exp": int((NOW + timedelta(seconds=600)).timestamp()),
        "jti": "jti-fixed",
        "kind": "session",
        "user_id": "user-123",
        "client_id": "llm_client_abc",
    }
    return {**base, **overrides}


def test_access_round_trip_recovers_principal_and_caps_ttl():
    minted = mint_session_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    assert minted.expires_at == NOW + timedelta(seconds=SESSION_TTL_SECONDS)
    token = minted.token.get_secret_value()
    assert is_session_token(token)
    assert not is_session_refresh_token(token)
    opened = open_session_token(token, KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == PRINCIPAL


def test_refresh_round_trip_recovers_principal_and_caps_ttl():
    minted = mint_session_refresh_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    assert minted.expires_at == NOW + timedelta(seconds=SESSION_REFRESH_TTL_SECONDS)
    token = minted.token.get_secret_value()
    assert is_session_refresh_token(token)
    opened = open_session_refresh_token(token, KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == PRINCIPAL


def test_access_token_reprefixed_as_refresh_is_rejected_by_signed_kind():
    body = _mint_access().removeprefix(SESSION_TOKEN_PREFIX)
    swapped = SESSION_REFRESH_PREFIX + body
    assert isinstance(open_session_refresh_token(swapped, KEYS, NOW), SessionMalformed)


def test_refresh_token_reprefixed_as_access_is_rejected_by_signed_kind():
    body = _mint_refresh().removeprefix(SESSION_REFRESH_PREFIX)
    swapped = SESSION_TOKEN_PREFIX + body
    assert isinstance(open_session_token(swapped, KEYS, NOW), SessionMalformed)


def test_refresh_token_is_not_an_access_token_at_the_edge():
    assert isinstance(open_session_token(_mint_refresh(), KEYS, NOW), NotASessionToken)


def test_expired_access_token_is_expired_not_malformed():
    token = _mint_access()
    at_expiry = NOW + timedelta(seconds=SESSION_TTL_SECONDS)
    assert isinstance(open_session_token(token, KEYS, at_expiry), SessionExpired)
    after = NOW + timedelta(seconds=SESSION_TTL_SECONDS + 1)
    assert isinstance(open_session_token(token, KEYS, after), SessionExpired)


def test_still_valid_one_second_before_expiry():
    token = _mint_access()
    just_before = NOW + timedelta(seconds=SESSION_TTL_SECONDS - 1)
    assert isinstance(open_session_token(token, KEYS, just_before), OpenedSessionToken)


def test_tampered_signature_is_bad_signature():
    token = _mint_access()
    assert isinstance(open_session_token(_corrupt_signature(token), KEYS, NOW), SessionBadSignature)


def test_key_rotation_invalidates_outstanding_tokens():
    token = _mint_access()
    assert isinstance(open_session_token(token, OTHER_KEYS, NOW), SessionBadSignature)


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("sk-9876", NotASessionToken),
        ("llm_env_something", NotASessionToken),
        ("", NotASessionToken),
        (SESSION_TOKEN_PREFIX, SessionMalformed),
        (SESSION_TOKEN_PREFIX + "not-a-jwt", SessionMalformed),
        (SESSION_TOKEN_PREFIX + "\ud800garbage", SessionMalformed),
        (SESSION_TOKEN_PREFIX + "a" * (MAX_SESSION_TOKEN_BYTES + 1), SessionMalformed),
    ],
)
def test_hostile_candidates_never_raise(candidate, expected):
    assert isinstance(open_session_token(candidate, KEYS, NOW), expected)


def test_multibyte_candidate_over_byte_cap_but_under_char_cap_is_rejected():
    filler = "€" * (MAX_SESSION_TOKEN_BYTES // 3)
    candidate = SESSION_TOKEN_PREFIX + filler
    assert len(candidate) <= MAX_SESSION_TOKEN_BYTES
    assert isinstance(open_session_token(candidate, KEYS, NOW), SessionMalformed)


def test_alg_none_token_is_rejected():
    unsigned = jwt.api_jws.encode(b'{"iss":"litellm-mcp-gateway"}', key=None, algorithm="none")
    assert isinstance(open_session_token(SESSION_TOKEN_PREFIX + unsigned, KEYS, NOW), SessionMalformed)


@pytest.mark.parametrize(
    "claims",
    [
        _valid_claims(iss="wrong-issuer"),
        _valid_claims(exp=str(int((NOW + timedelta(seconds=600)).timestamp()))),
        _valid_claims(iat="evil"),
        _valid_claims(kind="access"),
        _valid_claims(user_id=""),
        _valid_claims(family=""),
        _valid_claims(nbf=0),
        {k: v for k, v in _valid_claims().items() if k != "client_id"},
        {k: v for k, v in _valid_claims().items() if k != "exp"},
    ],
)
def test_signed_but_malformed_claims_are_rejected_without_raising(claims):
    token = _sign_claims(claims)
    assert isinstance(open_session_token(token, KEYS, NOW), SessionMalformed)


def test_signed_claims_with_exact_shape_open():
    token = _sign_claims(_valid_claims())
    opened = open_session_token(token, KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal.user_id == "user-123"


def test_oversized_client_id_fails_mint_with_typed_error_not_truncation():
    principal = SessionPrincipal(user_id="user-123", client_id="c" * (MAX_SESSION_TOKEN_BYTES + 100))
    minted = mint_session_token(principal, KEYS, NOW)
    assert isinstance(minted, SessionTokenTooLarge)
    assert minted.max_bytes == MAX_SESSION_TOKEN_BYTES


def test_empty_principal_fields_rejected_at_construction():
    with pytest.raises(ValidationError):
        SessionPrincipal(user_id="", client_id="c")
    with pytest.raises(ValidationError):
        SessionPrincipal(user_id="u", client_id="")
    with pytest.raises(ValidationError):
        SessionPrincipal(user_id="u", client_id="c", resource_server_ids=())


def test_short_signing_key_rejected_at_construction():
    with pytest.raises(ValidationError):
        SessionKeys(signing_key=SecretStr("short"))


def test_two_mints_of_the_same_principal_are_distinct_tokens():
    first = mint_session_token(PRINCIPAL, KEYS, NOW)
    second = mint_session_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(first, MintedSessionToken) and isinstance(second, MintedSessionToken)
    assert first.token.get_secret_value() != second.token.get_secret_value()


def test_minted_token_repr_never_leaks_value():
    minted = mint_session_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    assert minted.token.get_secret_value() not in repr(minted)


def _decoded_claims(token: str, prefix: str) -> dict:
    return jwt.decode(
        token.removeprefix(prefix),
        KEYS.signing_key.get_secret_value(),
        algorithms=["HS256"],
        options={"verify_exp": False},
    )


def test_single_server_scope_keeps_the_scalar_wire_claim():
    principal = SessionPrincipal(
        user_id="user-123",
        client_id="llm_client_abc",
        resource_server_ids=("alpha-id",),
    )
    minted = mint_session_token(principal, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)

    claims = _decoded_claims(minted.token.get_secret_value(), SESSION_TOKEN_PREFIX)
    opened = open_session_token(minted.token.get_secret_value(), KEYS, NOW)
    assert claims["resource_server_id"] == "alpha-id"
    assert "resource_server_ids" not in claims
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal.resource_server_ids == ("alpha-id",)


def test_multi_server_scope_round_trips_through_the_list_claim():
    principal = SessionPrincipal(
        user_id="user-123",
        client_id="llm_client_abc",
        resource_server_ids=("alpha-id", "beta-id"),
    )
    minted = mint_session_token(principal, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)

    claims = _decoded_claims(minted.token.get_secret_value(), SESSION_TOKEN_PREFIX)
    opened = open_session_token(minted.token.get_secret_value(), KEYS, NOW)
    assert claims["resource_server_ids"] == ["alpha-id", "beta-id"]
    assert "resource_server_id" not in claims
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == principal


def test_token_with_both_server_scope_claims_is_malformed():
    token = _sign_claims(_valid_claims(resource_server_id="alpha-id", resource_server_ids=["alpha-id", "beta-id"]))

    assert isinstance(open_session_token(token, KEYS, NOW), SessionMalformed)


def test_mcp_principal_wire_claims_carry_no_audience_or_team_keys():
    access_claims = _decoded_claims(_mint_access(), SESSION_TOKEN_PREFIX)
    refresh_claims = _decoded_claims(_mint_refresh(), SESSION_REFRESH_PREFIX)
    for claims in (access_claims, refresh_claims):
        assert "audience" not in claims
        assert "team_id" not in claims


def test_legacy_signed_claims_open_with_no_audience_and_no_team():
    opened = open_session_token(_sign_claims(_valid_claims()), KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal.audience is None
    assert opened.principal.team_id is None


def test_proxy_api_audience_and_team_round_trip_through_the_refresh_token():
    principal = SessionPrincipal(user_id="user-123", client_id="llm_client_abc", audience="proxy_api", team_id="team-b")
    minted = mint_session_refresh_token(principal, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    token = minted.token.get_secret_value()
    claims = _decoded_claims(token, SESSION_REFRESH_PREFIX)
    assert claims["audience"] == "proxy_api"
    assert claims["team_id"] == "team-b"
    opened = open_session_refresh_token(token, KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == principal


def test_refresh_token_family_is_rooted_at_the_first_jti_and_carried_through_rotation():
    root = mint_session_refresh_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(root, MintedSessionToken)
    root_token = root.token.get_secret_value()
    assert "family" not in _decoded_claims(root_token, SESSION_REFRESH_PREFIX)
    opened_root = open_session_refresh_token(root_token, KEYS, NOW)
    assert isinstance(opened_root, OpenedSessionToken)
    assert opened_root.family == opened_root.jti

    rotated = mint_session_refresh_token(PRINCIPAL, KEYS, NOW, family=opened_root.family)
    assert isinstance(rotated, MintedSessionToken)
    rotated_token = rotated.token.get_secret_value()
    assert _decoded_claims(rotated_token, SESSION_REFRESH_PREFIX)["family"] == opened_root.jti
    opened_rotated = open_session_refresh_token(rotated_token, KEYS, NOW)
    assert isinstance(opened_rotated, OpenedSessionToken)
    assert opened_rotated.jti != opened_root.jti
    assert opened_rotated.family == opened_root.jti

    twice_rotated = mint_session_refresh_token(PRINCIPAL, KEYS, NOW, family=opened_rotated.family)
    assert isinstance(twice_rotated, MintedSessionToken)
    opened_twice = open_session_refresh_token(twice_rotated.token.get_secret_value(), KEYS, NOW)
    assert isinstance(opened_twice, OpenedSessionToken)
    assert opened_twice.family == opened_root.jti


def test_signed_claims_with_an_unknown_audience_are_rejected():
    token = _sign_claims(_valid_claims(audience="bogus"))
    assert isinstance(open_session_token(token, KEYS, NOW), SessionMalformed)


def test_signed_claims_with_a_non_string_team_are_rejected():
    token = _sign_claims(_valid_claims(team_id=42))
    assert isinstance(open_session_token(token, KEYS, NOW), SessionMalformed)


def test_principal_rejects_an_unknown_audience_at_construction():
    with pytest.raises(ValidationError):
        SessionPrincipal(user_id="user-123", client_id="llm_client_abc", audience="mcp")


RSA_KEYS = AsymmetricSessionKeys(private_key_pem=SecretStr(_RSA_PEM_A), kid="2026-01")
OTHER_RSA_KEYS = AsymmetricSessionKeys(private_key_pem=SecretStr(_RSA_PEM_B), kid="2025-06")


@pytest.mark.parametrize("bits", (2048, 4096))
def test_rs256_reuses_one_private_key_load_for_validation_public_key_and_token_pairs(bits: int) -> None:
    private_pem: Final = _rsa_private_pem(bits)
    public_key: Final = serialization.load_pem_private_key(private_pem.encode(), password=None).public_key()
    expected_public_pem: Final = public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    with (
        patch(
            "cryptography.hazmat.primitives.serialization.load_pem_private_key",
            wraps=serialization.load_pem_private_key,
        ) as load_private_key,
        patch("jwt.algorithms.load_pem_private_key", load_private_key),
    ):
        for kid in ("first", "second", "first"):
            keys: Final = AsymmetricSessionKeys(private_key_pem=SecretStr(private_pem), kid=kid)
            assert session_public_key_pem(keys) == expected_public_pem
            for mint, prefix, kind, ttl in (
                (mint_session_token, SESSION_TOKEN_PREFIX, "session", SESSION_TTL_SECONDS),
                (mint_session_refresh_token, SESSION_REFRESH_PREFIX, "session_refresh", SESSION_REFRESH_TTL_SECONDS),
            ):
                minted: Final = mint(PRINCIPAL, keys, NOW)
                assert isinstance(minted, MintedSessionToken)
                encoded: Final = minted.token.get_secret_value().removeprefix(prefix)
                claims: Final = TypeAdapter(dict[str, object]).validate_python(
                    jwt.decode(
                        encoded, public_key, algorithms=["RS256"], issuer=SESSION_ISSUER, options={"verify_exp": False}
                    )
                )
                header: Final = jwt.get_unverified_header(encoded)
                assert (header["alg"], header["kid"]) == ("RS256", kid)
                assert (claims["user_id"], claims["client_id"], claims["kind"]) == (
                    PRINCIPAL.user_id,
                    PRINCIPAL.client_id,
                    kind,
                )
                assert claims["exp"] == int((NOW + timedelta(seconds=ttl)).timestamp())
        assert load_private_key.call_count == 1


@pytest.mark.parametrize("refresh", (False, True))
def test_rs256_changed_pem_with_the_same_kid_signs_and_verifies_with_the_new_key(refresh: bool) -> None:
    first: Final = AsymmetricSessionKeys(private_key_pem=SecretStr(_RSA_PEM_A), kid="same-kid")
    rotated: Final = AsymmetricSessionKeys(private_key_pem=SecretStr(_RSA_PEM_B), kid=first.kid)
    mint: Final = mint_session_refresh_token if refresh else mint_session_token
    open_token: Final = open_session_refresh_token if refresh else open_session_token
    original: Final = mint(PRINCIPAL, first, NOW)
    current: Final = mint(PRINCIPAL, rotated, NOW)
    assert isinstance(original, MintedSessionToken)
    assert isinstance(current, MintedSessionToken)
    assert session_public_key_pem(first) != session_public_key_pem(rotated)
    assert isinstance(open_token(original.token.get_secret_value(), rotated, NOW), SessionBadSignature)
    assert isinstance(open_token(current.token.get_secret_value(), first, NOW), SessionBadSignature)
    opened: Final = open_token(current.token.get_secret_value(), rotated, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == PRINCIPAL


@pytest.mark.parametrize(
    "private_pem, error",
    (
        ("not a pem", "private_key_pem is not a loadable unencrypted PEM private key"),
        (_rsa_private_pem(bits=1024), "private_key_pem must be an RSA key of at least 2048 bits"),
        (
            ec.generate_private_key(ec.SECP256R1())
            .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
            .decode(),
            "private_key_pem must be an unencrypted RSA private key in PEM format",
        ),
        (
            rsa.generate_private_key(public_exponent=65537, key_size=2048)
            .private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.BestAvailableEncryption(b"test-password"),
            )
            .decode(),
            "private_key_pem is not a loadable unencrypted PEM private key",
        ),
    ),
    ids=("invalid", "weak", "non-rsa", "encrypted"),
)
def test_rs256_invalid_private_keys_are_rejected_on_every_attempt(private_pem: str, error: str) -> None:
    with patch(
        "cryptography.hazmat.primitives.serialization.load_pem_private_key", wraps=serialization.load_pem_private_key
    ) as load_private_key:
        for _ in range(2):
            with pytest.raises(ValidationError, match=error):
                AsymmetricSessionKeys(private_key_pem=SecretStr(private_pem), kid="rejected")
        assert load_private_key.call_count == 2


def test_rs256_access_round_trip_with_kid_and_alg_pinned_in_header():
    minted = mint_session_token(PRINCIPAL, RSA_KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    token = minted.token.get_secret_value()
    header = jwt.get_unverified_header(token.removeprefix(SESSION_TOKEN_PREFIX))
    assert header["alg"] == "RS256"
    assert header["kid"] == "2026-01"
    opened = open_session_token(token, RSA_KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == PRINCIPAL


def test_rs256_refresh_round_trip():
    minted = mint_session_refresh_token(PRINCIPAL, RSA_KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    token = minted.token.get_secret_value()
    opened = open_session_refresh_token(token, RSA_KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == PRINCIPAL


def test_rs256_token_verifies_with_public_key_only():
    minted = mint_session_token(PRINCIPAL, RSA_KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    public_pem = session_public_key_pem(RSA_KEYS)
    assert "PUBLIC KEY" in public_pem
    assert "PRIVATE" not in public_pem
    claims = jwt.decode(
        minted.token.get_secret_value().removeprefix(SESSION_TOKEN_PREFIX),
        public_pem,
        algorithms=["RS256"],
        issuer=SESSION_ISSUER,
        options={"verify_exp": False},
    )
    assert claims["user_id"] == "user-123"


def test_rs256_tampered_signature_is_bad_signature():
    minted = mint_session_token(PRINCIPAL, RSA_KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    token = minted.token.get_secret_value()
    assert isinstance(open_session_token(_corrupt_signature(token), RSA_KEYS, NOW), SessionBadSignature)


def test_rs256_expired_token_is_expired():
    minted = mint_session_token(PRINCIPAL, RSA_KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    after = NOW + timedelta(seconds=SESSION_TTL_SECONDS + 1)
    assert isinstance(open_session_token(minted.token.get_secret_value(), RSA_KEYS, after), SessionExpired)


def test_hs256_token_is_rejected_in_rs256_mode():
    assert isinstance(open_session_token(_mint_access(), RSA_KEYS, NOW), SessionBadSignature)


def test_hs256_token_claiming_the_current_kid_is_rejected_by_alg_pinning():
    token = SESSION_TOKEN_PREFIX + jwt.encode(
        _valid_claims(),
        KEYS.signing_key.get_secret_value(),
        algorithm="HS256",
        headers={"kid": RSA_KEYS.kid},
    )
    assert isinstance(open_session_token(token, RSA_KEYS, NOW), SessionMalformed)


def test_rs256_token_is_rejected_in_hs256_mode():
    minted = mint_session_token(PRINCIPAL, RSA_KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    assert isinstance(open_session_token(minted.token.get_secret_value(), KEYS, NOW), SessionMalformed)


def test_rs256_token_from_an_unknown_kid_is_bad_signature():
    minted = mint_session_token(PRINCIPAL, OTHER_RSA_KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    assert isinstance(open_session_token(minted.token.get_secret_value(), RSA_KEYS, NOW), SessionBadSignature)


def test_rs256_token_signed_by_a_foreign_key_claiming_the_current_kid_is_bad_signature():
    token = SESSION_TOKEN_PREFIX + jwt.encode(
        _valid_claims(),
        _RSA_PEM_B,
        algorithm="RS256",
        headers={"kid": RSA_KEYS.kid},
    )
    assert isinstance(open_session_token(token, RSA_KEYS, NOW), SessionBadSignature)


def test_alg_none_token_with_the_current_kid_is_rejected_in_rs256_mode():
    unsigned = jwt.api_jws.encode(
        b'{"iss":"litellm-mcp-gateway"}', key=None, algorithm="none", headers={"kid": RSA_KEYS.kid}
    )
    assert isinstance(open_session_token(SESSION_TOKEN_PREFIX + unsigned, RSA_KEYS, NOW), SessionMalformed)


def test_rotation_previous_public_key_still_verifies_until_removed():
    minted = mint_session_token(PRINCIPAL, OTHER_RSA_KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    token = minted.token.get_secret_value()
    rotated = AsymmetricSessionKeys(
        private_key_pem=SecretStr(_RSA_PEM_A),
        kid="2026-01",
        previous_public_keys=(
            SessionRotatedPublicKey(kid="2025-06", public_key_pem=session_public_key_pem(OTHER_RSA_KEYS)),
        ),
    )
    opened = open_session_token(token, rotated, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == PRINCIPAL
    assert isinstance(open_session_token(token, RSA_KEYS, NOW), SessionBadSignature)


def test_rotation_window_still_enforces_expiry_and_tamper_on_the_previous_key():
    minted = mint_session_token(PRINCIPAL, OTHER_RSA_KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    token = minted.token.get_secret_value()
    rotated = AsymmetricSessionKeys(
        private_key_pem=SecretStr(_RSA_PEM_A),
        kid="2026-01",
        previous_public_keys=(
            SessionRotatedPublicKey(kid="2025-06", public_key_pem=session_public_key_pem(OTHER_RSA_KEYS)),
        ),
    )
    after = NOW + timedelta(seconds=SESSION_TTL_SECONDS + 1)
    assert isinstance(open_session_token(token, rotated, after), SessionExpired)
    assert isinstance(open_session_token(_corrupt_signature(token), rotated, NOW), SessionBadSignature)


def test_weak_or_garbage_private_key_pem_rejected_at_construction():
    with pytest.raises(ValidationError):
        AsymmetricSessionKeys(private_key_pem=SecretStr(_rsa_private_pem(bits=1024)), kid="weak")
    with pytest.raises(ValidationError):
        AsymmetricSessionKeys(private_key_pem=SecretStr("not a pem"), kid="junk")
    with pytest.raises(ValidationError):
        SessionRotatedPublicKey(kid="junk", public_key_pem="not a pem")
    with pytest.raises(ValidationError):
        SessionRotatedPublicKey(kid="private-half", public_key_pem=_RSA_PEM_A)


def test_weak_rotated_public_key_rejected_at_construction():
    weak_public = (
        serialization.load_pem_private_key(_rsa_private_pem(bits=1024).encode(), password=None)
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    with pytest.raises(ValidationError):
        SessionRotatedPublicKey(kid="2024-01", public_key_pem=weak_public)


def test_duplicate_kids_rejected_at_construction():
    previous = SessionRotatedPublicKey(kid="2025-06", public_key_pem=session_public_key_pem(OTHER_RSA_KEYS))
    with pytest.raises(ValidationError):
        AsymmetricSessionKeys(private_key_pem=SecretStr(_RSA_PEM_A), kid="2025-06", previous_public_keys=(previous,))
    with pytest.raises(ValidationError):
        AsymmetricSessionKeys(
            private_key_pem=SecretStr(_RSA_PEM_A), kid="2026-01", previous_public_keys=(previous, previous)
        )


def test_asymmetric_keys_repr_never_leaks_the_private_key():
    assert _RSA_PEM_A not in repr(RSA_KEYS)
