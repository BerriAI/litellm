import base64
import json
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.identity.extractors.jwt import extract_jwt_principal


def _b64url(payload: dict) -> str:
    raw = json.dumps(payload).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _build_unverified_jwt(claims: dict) -> str:
    header = _b64url({"alg": "HS256", "typ": "JWT"})
    body = _b64url(claims)
    return f"{header}.{body}.sig"


def test_returns_none_for_non_jwt():
    assert extract_jwt_principal("sk-abc") is None
    assert extract_jwt_principal(None) is None
    assert extract_jwt_principal("") is None


def test_extracts_sub_iss_aud():
    token = _build_unverified_jwt(
        {"sub": "user-1", "iss": "https://idp.example", "aud": "litellm"}
    )
    p = extract_jwt_principal(token)
    assert p is not None
    assert p.sub == "user-1"
    assert p.iss == "https://idp.example"
    assert p.aud == "litellm"


def test_string_scope_is_split():
    token = _build_unverified_jwt({"sub": "u", "scope": "read write admin"})
    p = extract_jwt_principal(token)
    assert p is not None
    assert p.scopes == ("read", "write", "admin")


def test_list_scope_is_preserved():
    token = _build_unverified_jwt({"sub": "u", "scope": ["a", "b"]})
    p = extract_jwt_principal(token)
    assert p is not None
    assert p.scopes == ("a", "b")


def test_scp_claim_supported():
    token = _build_unverified_jwt({"sub": "u", "scp": "read"})
    p = extract_jwt_principal(token)
    assert p is not None
    assert p.scopes == ("read",)


def test_raw_claims_preserved():
    claims = {"sub": "u", "custom": {"groups": ["g1"]}}
    token = _build_unverified_jwt(claims)
    p = extract_jwt_principal(token)
    assert p is not None
    assert p.claims["custom"] == {"groups": ["g1"]}
