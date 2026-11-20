#### What this tests ####
#    Unit tests for JWT-Auth

import base64
import os
import time

from dotenv import load_dotenv

from litellm._uuid import uuid

load_dotenv()

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Final

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.caching.caching import DualCache
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.proxy._types import (
    JWTLiteLLMRoleMap,
    LiteLLM_JWTAuth,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
)
from litellm.proxy.auth import core
from litellm.proxy.auth import jwt as proxy_jwt
from litellm.proxy.auth.host.facts import EndUserRow
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.rust_bridge._native import RustAuthSettings, RustIdentity
from litellm.types.proxy.agent_identity import ManagedAgentContext
from litellm.types.proxy.auth.user_roles import LitellmUserRoles

public_key = {
    "kty": "RSA",
    "e": "AQAB",
    "n": "qIgOQfEVrrErJC0E7gsHXi6rs_V0nyFY5qPFui2-tv0o4CwpwDzgfBtLO7o_wLiguq0lnu54sMT2eLNoRiiPuLvv6bg7Iy1H9yc5_4Jf5oYEOrqN5o9ZBOoYp1q68Pv0oNJYyZdGu5ZJfd7V4y953vB2XfEKgXCsAkhVhlvIUMiDNKWoMDWsyb2xela5tRURZ2mJAXcHfSC_sYdZxIA2YYrIHfoevq_vTlaz0qVSe_uOKjEpgOAS08UUrgda4CQL11nzICiIQzc6qmjIQt2cjzB2D_9zb4BYndzEtfl0kwAT0z_I85S3mkwTqHU-1BvKe_4MG4VG3dAAeffLPXJyXQ",
    "alg": "RS256",
}


def test_load_config_with_custom_role_names():
    config = {"general_settings": {"litellm_proxy_roles": {"admin_jwt_scope": "litellm-proxy-admin"}}}
    proxy_roles = LiteLLM_JWTAuth(**config.get("general_settings", {}).get("litellm_proxy_roles", {}))

    print(f"proxy_roles: {proxy_roles}")

    assert proxy_roles.admin_jwt_scope == "litellm-proxy-admin"


# test_load_config_with_custom_role_names()


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params
    from litellm.proxy.utils import PrismaClient, ProxyLogging

    proxy_logging_obj = ProxyLogging(user_api_key_cache=UserApiKeyCache())

    ### add connection pool + pool timeout args
    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    # Assuming PrismaClient is a class that needs to be instantiated
    prisma_client = PrismaClient(database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj)

    return prisma_client


@pytest.fixture
def team_token_tuple():
    import json

    import jwt
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Generate a private / public key pair using RSA algorithm
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    # Get private key in PEM format
    private_key = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Get public key in PEM format
    public_key = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    public_key_obj = serialization.load_pem_public_key(public_key, backend=default_backend())

    # Convert RSA public key object to JWK (JSON Web Key)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key_obj))

    # VALID TOKEN
    ## GENERATE A TOKEN
    # Assuming the current time is in UTC
    expiration_time = int((datetime.now() + timedelta(minutes=10)).timestamp())

    team_id = f"team123_{uuid.uuid4()}"
    payload = {
        "sub": "user123",
        "exp": expiration_time,  # set the token to expire in 10 minutes
        "scope": "litellm_team",
        "client_id": team_id,
        "aud": None,
    }

    # Generate the JWT token
    # But before, you should convert bytes to string
    private_key_str = private_key.decode("utf-8")

    ## team token
    token = jwt.encode(payload, private_key_str, algorithm="RS256")

    return team_id, token, public_jwk


import pytest


@pytest.fixture
def public_jwt_key():
    import json

    import jwt
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Generate a private / public key pair using RSA algorithm
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    # Get private key in PEM format
    private_key = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Get public key in PEM format
    public_key = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    public_key_obj = serialization.load_pem_public_key(public_key, backend=default_backend())

    # Convert RSA public key object to JWK (JSON Web Key)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key_obj))

    return {"private_key": private_key, "public_jwk": public_jwk}


async def mock_user_object(*args, **kwargs):
    print(f"Args: {args}")
    print(f"kwargs: {kwargs}")
    assert kwargs["user_id_upsert"] is True
    # Return a mock user object
    user_id = kwargs.get("user_id")
    user_email = kwargs.get("user_email")
    return LiteLLM_UserTable(spend=0, user_id=user_id, max_budget=None, user_email=user_email)


def _base64url_encode_bytes(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _base64url_encode_int(value: int) -> str:
    value_bytes = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return _base64url_encode_bytes(value=value_bytes)


def _get_rsa_key_and_jwk(kid: str):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "n": _base64url_encode_int(value=public_numbers.n),
        "e": _base64url_encode_int(value=public_numbers.e),
        "kid": kid,
        "alg": "RS256",
        "use": "sig",
    }
    return private_key, jwk


def _encode_rsa_jwt(
    private_key,
    issuer: str,
    audience: str,
    kid: str,
    extra_claims: dict | None = None,
) -> str:
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    current_time = int(time.time())
    claims = {
        "sub": "test-subject",
        "iss": issuer,
        "aud": audience,
        "iat": current_time,
        "exp": current_time + 300,
    }
    if extra_claims:
        claims.update(extra_claims)

    return jwt.encode(
        claims,
        private_key_pem,
        algorithm="RS256",
        headers={"kid": kid},
    )


def test_multi_issuer_jwt_requires_audience_unless_explicitly_disabled(
    monkeypatch,
):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    issuer = "https://issuer.example.com"
    jwks_url = f"{issuer}/keys"

    with pytest.raises(Exception, match="must configure audience or set") as exc:
        LiteLLM_JWTAuth(
            issuers=[
                {
                    "issuer": issuer,
                    "jwks_url": jwks_url,
                }
            ]
        )

    assert "must configure audience" in str(exc.value)


def test_jwt_user_takes_its_role_and_limits_from_its_user_row(
    jwt_identity: Callable[..., RustIdentity.Jwt],
) -> None:
    user: Final = LiteLLM_UserTable(user_id="u", user_role="internal_user", tpm_limit=5)

    token: Final = proxy_jwt.auth_from_result(proxy_jwt.jwt_result(jwt_identity(user=user), "jwt"))

    assert (token.user_role, token.user_tpm_limit, token.user_email) == (
        LitellmUserRoles.INTERNAL_USER,
        5,
        "u@example.com",
    )


def test_jwt_result_carries_every_row_the_identity_names(
    jwt_identity: Callable[..., RustIdentity.Jwt],
) -> None:
    team: Final = LiteLLM_TeamTable(team_id="t2", models=["gpt-b"], tpm_limit=9)
    membership: Final = LiteLLM_TeamMembership(user_id="u", team_id="t2")
    org: Final = LiteLLM_OrganizationTable(
        organization_id="o1", budget_id="b", models=[], created_by="x", updated_by="x"
    )
    end_user: Final = LiteLLM_EndUserTable(user_id="eu", blocked=False)

    result: Final = proxy_jwt.jwt_result(
        jwt_identity(
            team_id="t2",
            org_id="o1",
            end_user_id="eu",
            team=team,
            membership=membership,
            org=org,
            end_user=EndUserRow(end_user_id="eu", end_user=end_user, params={}),
        ),
        "jwt",
    )

    assert (
        result["token"],
        result["team_object"],
        result.get("team_membership"),
        result["org_object"],
        result["end_user_object"],
    ) == ("jwt", team, membership, org, end_user)
    assert proxy_jwt.auth_from_result(result).team_tpm_limit == 9


def test_delegated_managed_agent_carries_its_one_shot_proof(
    jwt_identity: Callable[..., RustIdentity.Jwt],
) -> None:
    managed: Final = ManagedAgentContext(agent_id="a", mode="delegated", user_id="u")

    token: Final = proxy_jwt.auth_from_result(
        proxy_jwt.jwt_result(jwt_identity(agent_id="a", managed_agent=managed), "jwt")
    )

    assert token.managed_agent_context == managed
    assert token._managed_delegation_verified is True  # pyright: ignore[reportPrivateUsage]  # the proof is private
    assert token.user_role == LitellmUserRoles.INTERNAL_USER


def test_claim_reading_maps_each_configured_claim() -> None:
    reading: Final = proxy_jwt.ClaimReading(
        LiteLLM_JWTAuth(
            user_id_jwt_field="sub",
            user_email_jwt_field="email",
            team_ids_jwt_field="groups",
            roles_jwt_field="roles",
            jwt_litellm_role_map=[JWTLiteLLMRoleMap(jwt_role="admin-*", litellm_role=LitellmUserRoles.PROXY_ADMIN)],
            user_allowed_email_domain="example.com",
        )
    )
    token: Final[dict[str, object]] = {
        "sub": "user-1",
        "email": "dev@example.com",
        "groups": ["g1", "g2"],
        "roles": ["reader", "admin-eu"],
    }

    assert reading.get_user_id(token, default_value=None) == "user-1"
    assert reading.get_user_email(token, default_value=None) == "dev@example.com"
    assert reading.get_team_ids_from_jwt(token) == ["g1", "g2"]
    assert reading.get_all_jwt_team_ids(token) == ["g1", "g2"]
    assert reading.map_jwt_role_to_litellm_role(token) == LitellmUserRoles.PROXY_ADMIN
    assert reading.map_jwt_role_to_litellm_role({**token, "roles": ["reader"]}) is None
    assert reading.get_user_id({}, default_value="fallback") == "fallback"
    assert reading.get_user_email({}, default_value="fallback@example.com") == "fallback@example.com"
    assert reading.is_allowed_domain("dev@example.com") is True
    assert reading.is_allowed_domain("dev@elsewhere.com") is False


def test_a_token_presents_the_claims_it_carries_and_no_others() -> None:
    signing_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    claims: Final[dict[str, object]] = {"sub": "user-1", "exp": 4102444800}
    token: Final = jwt.encode(claims, signing_key, algorithm="RS256")

    assert proxy_jwt.is_jwt(token) is True
    assert proxy_jwt.unverified_claims(token) == claims
    assert proxy_jwt.presents_claims(token, claims) is True
    assert proxy_jwt.presents_claims(token, {**claims, "sub": "user-2"}) is False


_JWKS_URL: Final = "https://idp.example/jwks"
_ISSUER: Final = "https://idp.example"
_SIGNING_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _Jwks:
    async def get(self, url: str, params: object = None, headers: object = None) -> httpx.Response:
        key: Final = jwt.algorithms.RSAAlgorithm.to_jwk(_SIGNING_KEY.public_key(), as_dict=True)
        return httpx.Response(200, json={"keys": [{**key, "kid": "k1"}]}, request=httpx.Request("GET", url))


def _signed(claims: dict[str, object], key: rsa.RSAPrivateKey = _SIGNING_KEY) -> str:
    return jwt.encode(
        {"exp": int(time.time()) + 600, "iss": _ISSUER, **claims}, key, algorithm="RS256", headers={"kid": "k1"}
    )


@pytest.fixture
def jwt_settings(monkeypatch: pytest.MonkeyPatch) -> RustAuthSettings:
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", _JWKS_URL)
    monkeypatch.setenv("JWT_ISSUER", _ISSUER)
    monkeypatch.setattr(proxy_server, "jwt_http_handler", _Jwks())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    return RustAuthSettings(
        {"enable_jwt_auth": True}, LiteLLM_JWTAuth(), core.HostFacts(master_key=None, premium=True, has_db=False)
    )


@pytest.mark.asyncio
async def test_verify_returns_the_claims_of_a_token_the_jwks_signed(jwt_settings: RustAuthSettings) -> None:
    verified: Final = await proxy_jwt.verify(jwt_settings, _signed({"sub": "user-1"}))

    assert verified["sub"] == "user-1"


@pytest.mark.asyncio
async def test_verify_refuses_a_token_another_key_signed(jwt_settings: RustAuthSettings) -> None:
    forged: Final = _signed({"sub": "user-1"}, rsa.generate_private_key(public_exponent=65537, key_size=2048))

    with pytest.raises(Exception, match="Validation fails"):
        _ = await proxy_jwt.verify(jwt_settings, forged)


@pytest.mark.asyncio
async def test_an_assertion_for_another_audience_is_rejected_not_raised(jwt_settings: RustAuthSettings) -> None:
    rejected: Final = await proxy_jwt.verify_assertion(
        jwt_settings, _signed({"sub": "user-1", "aud": "other-app"}), ["litellm"], ["sub"]
    )
    accepted: Final = await proxy_jwt.verify_assertion(
        jwt_settings, _signed({"sub": "user-1", "aud": "litellm"}), ["litellm"], ["sub"]
    )

    assert isinstance(rejected, proxy_jwt.AssertionRejection)
    assert not isinstance(accepted, proxy_jwt.AssertionRejection)
    assert accepted["sub"] == "user-1"
