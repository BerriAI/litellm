"""The gateway-credential recognizer must be complete by construction: every credential this
gateway mints classifies as `gateway_credential` (so it is never exchange subject material), and
a real external token classifies as `external`. These tests mint the real gateway formats so a
newly added mint that this recognizer fails to catch fails a test here, not in production."""

from datetime import datetime, timezone

import jwt as pyjwt
import pytest

import litellm.proxy.proxy_server as proxy_server
from litellm.proxy._experimental.mcp_server.outbound_credentials.credential_provenance import (
    classify_inbound_provenance,
    is_gateway_issued_credential,
)

_MASTER = "sk-master-key-for-provenance-tests"


@pytest.fixture(autouse=True)
def _master_key(monkeypatch):
    monkeypatch.setattr(proxy_server, "master_key", _MASTER)


def _hs256(claims: dict, key: str) -> str:
    return pyjwt.encode(claims, key, algorithm="HS256")


def _real_session_token() -> str:
    from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
        session_keys_from_master_key,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
        SessionPrincipal,
        mint_session_token,
    )

    minted = mint_session_token(
        SessionPrincipal(user_id="u", client_id="c"),
        session_keys_from_master_key(_MASTER),
        datetime.now(timezone.utc),
    )
    return minted.token.get_secret_value()


def _gateway_credentials() -> dict[str, str]:
    now = 2_000_000_000
    return {
        "master_key": _MASTER,
        "virtual_key": "sk-" + "a" * 40,
        # The three the prefix denylist missed: master-key HS256 JWTs with no prefix and no iss.
        "ui_login_session": _hs256({"user_id": "u", "key": "sk-embedded", "user_role": "proxy_admin"}, _MASTER),
        "onboarding": _hs256({"token_type": "litellm_onboarding", "user_id": "u", "exp": now}, _MASTER),
        "byok_session": _hs256({"user_id": "u", "server_id": "s", "type": "byok_session", "exp": now}, _MASTER),
        # Prefixed gateway JWTs.
        "mcp_session_real": _real_session_token(),
        "envelope_prefixed": "llm_env_" + _hs256({"iss": "litellm-mcp-bridge"}, "derived-key"),
        # Reserved-issuer inner tokens presented without their prefix (derived-key signed).
        "session_inner_no_prefix": _hs256({"iss": "litellm-mcp-gateway", "exp": now}, "some-derived-key"),
        "envelope_inner_no_prefix": _hs256({"iss": "litellm-mcp-bridge", "exp": now}, "some-derived-key"),
    }


def _external_jwt_tokens() -> dict[str, str]:
    """Non-gateway JWTs: id_token candidates id_jag can exchange (`external_jwt`)."""
    now = 2_000_000_000
    return {
        "external_idp_jwt": _hs256({"iss": "https://okta.example.com", "sub": "u", "aud": "gw", "exp": now}, "idp-key"),
        "external_jwt_no_exp": _hs256({"iss": "https://idp.example.com", "sub": "u"}, "another-idp-key"),
    }


def _external_opaque_tokens() -> dict[str, str]:
    """Non-gateway, non-JWT bearers: not id_tokens, so id_jag rejects them (`external_opaque`)."""
    return {
        "opaque_access_token": "opaque-access-token-from-an-idp-1234567890",
        # Three dot-segments but not a decodable JWT; the finding's example.
        "jwt_shaped_but_undecodable": "abc.def.ghi",
    }


# Evaluated once, and the case NAME is the test id. A minted session token embeds the current time
# and a random jti, so deriving the id from the token value instead would make it differ on every
# collection; under xdist the workers then disagree about which tests exist and the run errors out.
_GATEWAY_CREDENTIALS = _gateway_credentials()
_EXTERNAL_JWT_TOKENS = _external_jwt_tokens()
_EXTERNAL_OPAQUE_TOKENS = _external_opaque_tokens()


@pytest.mark.parametrize("name,token", list(_GATEWAY_CREDENTIALS.items()), ids=list(_GATEWAY_CREDENTIALS))
def test_every_gateway_credential_is_recognized(name, token):
    assert is_gateway_issued_credential(token) is True
    assert classify_inbound_provenance(token) == "gateway_credential"


@pytest.mark.parametrize("name,token", list(_EXTERNAL_JWT_TOKENS.items()), ids=list(_EXTERNAL_JWT_TOKENS))
def test_external_jwt_is_an_id_token_candidate(name, token):
    assert is_gateway_issued_credential(token) is False
    assert classify_inbound_provenance(token) == "external_jwt"


@pytest.mark.parametrize("name,token", list(_EXTERNAL_OPAQUE_TOKENS.items()), ids=list(_EXTERNAL_OPAQUE_TOKENS))
def test_external_opaque_token_is_not_an_id_token(name, token):
    """A non-gateway non-JWT bearer is not an id_token; id_jag must not forward it (the finding)."""
    assert is_gateway_issued_credential(token) is False
    assert classify_inbound_provenance(token) == "external_opaque"


@pytest.mark.parametrize("empty", [None, ""])
def test_absent_inbound_token_classifies_absent(empty):
    assert classify_inbound_provenance(empty) == "absent"


@pytest.mark.parametrize(
    "temporal_claims",
    [
        {"exp": 1},  # long expired
        {"nbf": 9_999_999_999},  # not yet valid
        {"iat": 9_999_999_999},  # issued in the future
    ],
)
def test_temporally_invalid_gateway_jwt_is_still_recognized(temporal_claims):
    """Recognition determines provenance, not usability: an expired or not-yet-valid gateway JWT
    is still this gateway's and must classify as gateway_credential, or it drops to the
    exchangeable-external path and is disclosed upstream."""
    token = _hs256({"user_id": "u", "key": "sk-embedded", **temporal_claims}, _MASTER)
    assert is_gateway_issued_credential(token) is True
    assert classify_inbound_provenance(token) == "gateway_credential"


def test_expired_external_jwt_stays_an_id_token_candidate():
    """An expired token that is NOT ours stays external_jwt; the org authorization server, not
    this recognizer, is the authority on whether the caller's own token is still valid."""
    token = _hs256({"iss": "https://idp.example.com", "sub": "u", "exp": 1}, "external-idp-key")
    assert is_gateway_issued_credential(token) is False
    assert classify_inbound_provenance(token) == "external_jwt"


def test_encrypted_login_blob_is_recognized_by_decryption():
    """CLI / experimental UI login tokens are encrypted, not JWT-signed; decryption succeeding is
    proof they are this gateway's, so they are recognized without a prefix."""
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    blob = encrypt_value_helper('{"token": "cli-session-abc", "is_session_token": true}')
    assert is_gateway_issued_credential(blob) is True


def test_recognizer_does_not_crash_without_a_master_key(monkeypatch):
    """With no master key configured the crypto checks are skipped; an external token is still
    classified external rather than raising into egress."""
    monkeypatch.setattr(proxy_server, "master_key", None)
    assert is_gateway_issued_credential("some.external.jwt") is False
    # A prefix-bearing gateway credential is still caught with no master key.
    assert is_gateway_issued_credential("sk-still-caught-by-prefix") is True


@pytest.mark.parametrize(
    "adversarial",
    [
        "café-tökèn-with-non-ascii",  # the reported crash: non-ASCII bytes hit the master-key compare
        "bearer-\U0001f600-emoji",  # supplementary-plane characters
        "\x00\x01control-bytes",  # control characters
        "aaa.é.ccc",  # non-ASCII inside a jwt-shaped value
    ],
)
def test_non_ascii_inbound_bearer_never_raises_into_egress(adversarial):
    """The inbound bearer is fully attacker-controlled, so recognition must be TOTAL: a non-ASCII
    bearer used to reach ``secrets.compare_digest(token, master_key)`` and raise ``TypeError``,
    500-ing every egress. It must now classify as a non-gateway external token without raising.
    Reverting the digest-based master-key compare makes this test raise."""
    assert is_gateway_issued_credential(adversarial) is False
    assert classify_inbound_provenance(adversarial) in ("external_jwt", "external_opaque")


def test_master_key_compare_matches_only_the_exact_key(monkeypatch):
    """The digest-based compare must still be exact equality, not a length match or near miss. A
    master key with no gateway prefix isolates the compare from the other checks: the exact key is
    recognized, a value that only shares its length or a prefix of it is not."""
    plain_master = "plain-master-key-no-gateway-prefix"
    monkeypatch.setattr(proxy_server, "master_key", plain_master)
    assert is_gateway_issued_credential(plain_master) is True
    assert is_gateway_issued_credential(plain_master + "x") is False
    assert is_gateway_issued_credential(plain_master[:-1] + "Z") is False
