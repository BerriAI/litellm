from typing import Final

import jwt
import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.vertex import assert_vertex_authorization, service_account_json

_PROJECT: Final = "fixture-project"
_CLAIMS: Final[dict[str, JsonValue]] = {
    "iss": f"scripted@{_PROJECT}.iam.gserviceaccount.com",
    "sub": f"scripted@{_PROJECT}.iam.gserviceaccount.com",
    "scope": "https://www.googleapis.com/auth/cloud-platform",
    "iat": 1000,
    "exp": 4600,
}


def _token(claims: dict[str, JsonValue], *, project: str = _PROJECT, kid: str = "scripted") -> str:
    credentials: Final = TypeAdapter(dict[str, str]).validate_json(service_account_json(project, "http://unused"))
    return jwt.encode(claims, credentials["private_key"], algorithm="RS256", headers={"kid": kid})


@pytest.mark.parametrize("with_null_audience", [False, True])
def test_vertex_peer_verifies_a_service_account_signature_and_claims(with_null_audience: bool) -> None:
    bearer: Final = f"Bearer {_token({**_CLAIMS, **({'aud': None} if with_null_audience else {})})}"
    assert assert_vertex_authorization(bearer, _PROJECT, oauth_token=None, now=2000) is None


@pytest.mark.parametrize(
    "changed",
    [
        {"iss": "other@fixture-project.iam.gserviceaccount.com"},
        {"sub": "other@fixture-project.iam.gserviceaccount.com"},
        {"scope": "https://www.googleapis.com/auth/devstorage.read_only"},
        {"iat": 2001},
        {"exp": 2000},
        {"exp": "4600"},
        {"iat": True},
        {"aud": "https://wrong.googleapis.com"},
        {"extra": "unexpected"},
    ],
    ids=[
        "issuer",
        "subject",
        "scope",
        "future-issued",
        "expired",
        "mistyped-expiry",
        "boolean-issued",
        "audience",
        "extra-claim",
    ],
)
def test_vertex_peer_rejects_a_signed_token_with_the_wrong_claims(changed: dict[str, JsonValue]) -> None:
    bearer: Final = f"Bearer {_token({**_CLAIMS, **changed})}"
    with pytest.raises((AssertionError, jwt.InvalidTokenError)):
        assert_vertex_authorization(bearer, _PROJECT, oauth_token=None, now=2000)


@pytest.mark.parametrize("field", ["iss", "sub", "scope", "iat", "exp"])
def test_vertex_peer_rejects_a_signed_token_missing_a_required_claim(field: str) -> None:
    bearer: Final = f"Bearer {_token({key: value for key, value in _CLAIMS.items() if key != field})}"
    with pytest.raises((AssertionError, jwt.InvalidTokenError)):
        assert_vertex_authorization(bearer, _PROJECT, oauth_token=None, now=2000)


def test_vertex_peer_rejects_a_token_signed_by_another_projects_key() -> None:
    bearer: Final = f"Bearer {_token(dict(_CLAIMS), project='unregistered-for-this-project')}"
    with pytest.raises(AssertionError, match="not signed by this project's fixture service account"):
        assert_vertex_authorization(bearer, _PROJECT, oauth_token=None, now=2000)


def test_vertex_peer_rejects_an_unknown_signing_key_id() -> None:
    bearer: Final = f"Bearer {_token(dict(_CLAIMS), kid='not-scripted')}"
    with pytest.raises(AssertionError):
        assert_vertex_authorization(bearer, _PROJECT, oauth_token=None, now=2000)


@pytest.mark.parametrize("bearer", [None, "", "scripted-token", "Basic scripted-token", "Bearer malformed"])
def test_vertex_peer_rejects_a_missing_or_malformed_bearer(bearer: str | None) -> None:
    with pytest.raises((AssertionError, jwt.InvalidTokenError)):
        assert_vertex_authorization(bearer, _PROJECT, oauth_token=None, now=2000)


@pytest.mark.parametrize("token", ["scripted-token", "scripted-vertex-access-token"])
def test_vertex_peer_keeps_the_exact_oauth_token_contract_on_oauth_routes(token: str) -> None:
    assert assert_vertex_authorization(f"Bearer {token}", _PROJECT, oauth_token=token, now=2000) is None
    with pytest.raises(jwt.InvalidTokenError):
        assert_vertex_authorization("Bearer incorrect-token", _PROJECT, oauth_token=token, now=2000)


def test_vertex_peer_does_not_accept_an_oauth_token_on_a_jwt_only_route() -> None:
    with pytest.raises(jwt.InvalidTokenError):
        assert_vertex_authorization("Bearer scripted-token", _PROJECT, oauth_token=None, now=2000)
