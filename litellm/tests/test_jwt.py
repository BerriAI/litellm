#### What this tests ####
#    Unit tests for JWT-Auth

import sys, os, asyncio, time, random
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm.proxy._types import LiteLLM_JWTAuth
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.caching import DualCache
from datetime import datetime, timedelta

public_key = {
    "kty": "RSA",
    "e": "AQAB",
    "n": "qIgOQfEVrrErJC0E7gsHXi6rs_V0nyFY5qPFui2-tv0o4CwpwDzgfBtLO7o_wLiguq0lnu54sMT2eLNoRiiPuLvv6bg7Iy1H9yc5_4Jf5oYEOrqN5o9ZBOoYp1q68Pv0oNJYyZdGu5ZJfd7V4y953vB2XfEKgXCsAkhVhlvIUMiDNKWoMDWsyb2xela5tRURZ2mJAXcHfSC_sYdZxIA2YYrIHfoevq_vTlaz0qVSe_uOKjEpgOAS08UUrgda4CQL11nzICiIQzc6qmjIQt2cjzB2D_9zb4BYndzEtfl0kwAT0z_I85S3mkwTqHU-1BvKe_4MG4VG3dAAeffLPXJyXQ",
    "alg": "RS256",
}


def test_load_config_with_custom_role_names():
    config = {
        "general_settings": {
            "litellm_proxy_roles": {"admin_jwt_scope": "litellm-proxy-admin"}
        }
    }

    proxy_roles = LiteLLM_JWTAuth(
        **config.get("general_settings", {}).get("litellm_proxy_roles", {})
    )

    print(f"proxy_roles: {proxy_roles}")

    assert proxy_roles.admin_jwt_scope == "litellm-proxy-admin"


# test_load_config_with_custom_role_names()


@pytest.mark.asyncio
async def test_token_single_public_key():
    import jwt

    jwt_handler = JWTHandler()

    backend_keys = {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "e": "AQAB",
                "n": "qIgOQfEVrrErJC0E7gsHXi6rs_V0nyFY5qPFui2-tv0o4CwpwDzgfBtLO7o_wLiguq0lnu54sMT2eLNoRiiPuLvv6bg7Iy1H9yc5_4Jf5oYEOrqN5o9ZBOoYp1q68Pv0oNJYyZdGu5ZJfd7V4y953vB2XfEKgXCsAkhVhlvIUMiDNKWoMDWsyb2xela5tRURZ2mJAXcHfSC_sYdZxIA2YYrIHfoevq_vTlaz0qVSe_uOKjEpgOAS08UUrgda4CQL11nzICiIQzc6qmjIQt2cjzB2D_9zb4BYndzEtfl0kwAT0z_I85S3mkwTqHU-1BvKe_4MG4VG3dAAeffLPXJyXQ",
                "alg": "RS256",
            }
        ]
    }

    # set cache
    cache = DualCache()

    await cache.async_set_cache(key="litellm_jwt_auth_keys", value=backend_keys["keys"])

    jwt_handler.user_api_key_cache = cache

    public_key = await jwt_handler.get_public_key(kid=None)

    assert public_key is not None
    assert isinstance(public_key, dict)
    assert (
        public_key["n"]
        == "qIgOQfEVrrErJC0E7gsHXi6rs_V0nyFY5qPFui2-tv0o4CwpwDzgfBtLO7o_wLiguq0lnu54sMT2eLNoRiiPuLvv6bg7Iy1H9yc5_4Jf5oYEOrqN5o9ZBOoYp1q68Pv0oNJYyZdGu5ZJfd7V4y953vB2XfEKgXCsAkhVhlvIUMiDNKWoMDWsyb2xela5tRURZ2mJAXcHfSC_sYdZxIA2YYrIHfoevq_vTlaz0qVSe_uOKjEpgOAS08UUrgda4CQL11nzICiIQzc6qmjIQt2cjzB2D_9zb4BYndzEtfl0kwAT0z_I85S3mkwTqHU-1BvKe_4MG4VG3dAAeffLPXJyXQ"
    )


@pytest.mark.asyncio
async def test_valid_invalid_token():
    """
    Tests
    - valid token
    - invalid token
    """
    import jwt, json
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend

    # Generate a private / public key pair using RSA algorithm
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
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

    public_key_obj = serialization.load_pem_public_key(
        public_key, backend=default_backend()
    )

    # Convert RSA public key object to JWK (JSON Web Key)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key_obj))

    assert isinstance(public_jwk, dict)

    # set cache
    cache = DualCache()

    await cache.async_set_cache(key="litellm_jwt_auth_keys", value=[public_jwk])

    jwt_handler = JWTHandler()

    jwt_handler.user_api_key_cache = cache

    # VALID TOKEN
    ## GENERATE A TOKEN
    # Assuming the current time is in UTC
    expiration_time = int((datetime.utcnow() + timedelta(minutes=10)).timestamp())

    payload = {
        "sub": "user123",
        "exp": expiration_time,  # set the token to expire in 10 minutes
        "scope": "litellm-proxy-admin",
    }

    # Generate the JWT token
    # But before, you should convert bytes to string
    private_key_str = private_key.decode("utf-8")
    token = jwt.encode(payload, private_key_str, algorithm="RS256")

    ## VERIFY IT WORKS

    # verify token

    response = await jwt_handler.auth_jwt(token=token)

    assert response is not None
    assert isinstance(response, dict)

    print(f"response: {response}")

    # INVALID TOKEN
    ## GENERATE A TOKEN
    # Assuming the current time is in UTC
    expiration_time = int((datetime.utcnow() + timedelta(minutes=10)).timestamp())

    payload = {
        "sub": "user123",
        "exp": expiration_time,  # set the token to expire in 10 minutes
        "scope": "litellm-NO-SCOPE",
    }

    # Generate the JWT token
    # But before, you should convert bytes to string
    private_key_str = private_key.decode("utf-8")
    token = jwt.encode(payload, private_key_str, algorithm="RS256")

    ## VERIFY IT WORKS

    # verify token

    try:
        response = await jwt_handler.auth_jwt(token=token)
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


@pytest.fixture
def prisma_client():
    import litellm
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.proxy.proxy_cli import append_query_params

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    ### add connection pool + pool timeout args
    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    # Assuming DBClient is a class that needs to be instantiated
    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    return prisma_client


@pytest.mark.asyncio
async def test_team_token_output(prisma_client):
    import jwt, json
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
    from fastapi import Request
    from starlette.datastructures import URL
    from litellm.proxy.proxy_server import user_api_key_auth, new_team
    from litellm.proxy._types import NewTeamRequest, UserAPIKeyAuth
    import litellm
    import uuid

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Generate a private / public key pair using RSA algorithm
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
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

    public_key_obj = serialization.load_pem_public_key(
        public_key, backend=default_backend()
    )

    # Convert RSA public key object to JWK (JSON Web Key)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key_obj))

    assert isinstance(public_jwk, dict)

    # set cache
    cache = DualCache()

    await cache.async_set_cache(key="litellm_jwt_auth_keys", value=[public_jwk])

    jwt_handler = JWTHandler()

    jwt_handler.user_api_key_cache = cache

    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    # VALID TOKEN
    ## GENERATE A TOKEN
    # Assuming the current time is in UTC
    expiration_time = int((datetime.utcnow() + timedelta(minutes=10)).timestamp())

    team_id = f"team123_{uuid.uuid4()}"
    payload = {
        "sub": "user123",
        "exp": expiration_time,  # set the token to expire in 10 minutes
        "scope": "litellm_team",
        "client_id": team_id,
    }

    # Generate the JWT token
    # But before, you should convert bytes to string
    private_key_str = private_key.decode("utf-8")

    ## team token
    token = jwt.encode(payload, private_key_str, algorithm="RS256")

    ## admin token
    payload = {
        "sub": "user123",
        "exp": expiration_time,  # set the token to expire in 10 minutes
        "scope": "litellm_proxy_admin",
    }

    admin_token = jwt.encode(payload, private_key_str, algorithm="RS256")

    ## VERIFY IT WORKS

    # verify token

    response = await jwt_handler.auth_jwt(token=token)

    ## RUN IT THROUGH USER API KEY AUTH

    """
    - 1. Initial call should fail -> team doesn't exist
    - 2. Create team via admin token 
    - 3. 2nd call w/ same team -> call should succeed -> assert UserAPIKeyAuth object correctly formatted
    """

    bearer_token = "Bearer " + token

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    ## 1. INITIAL TEAM CALL - should fail
    # use generated key to auth in
    setattr(litellm.proxy.proxy_server, "general_settings", {"enable_jwt_auth": True})
    setattr(litellm.proxy.proxy_server, "jwt_handler", jwt_handler)
    try:
        result = await user_api_key_auth(request=request, api_key=bearer_token)
        pytest.fail("Team doesn't exist. This should fail")
    except Exception as e:
        pass

    ## 2. CREATE TEAM W/ ADMIN TOKEN - should succeed
    try:
        bearer_token = "Bearer " + admin_token

        request._url = URL(url="/team/new")
        result = await user_api_key_auth(request=request, api_key=bearer_token)
        await new_team(
            data=NewTeamRequest(
                team_id=team_id,
                tpm_limit=100,
                rpm_limit=99,
                models=["gpt-3.5-turbo", "gpt-4"],
            ),
            user_api_key_dict=result,
        )
    except Exception as e:
        pytest.fail(f"This should not fail - {str(e)}")

    ## 3. 2nd CALL W/ TEAM TOKEN - should succeed
    bearer_token = "Bearer " + token
    request._url = URL(url="/chat/completions")
    try:
        team_result: UserAPIKeyAuth = await user_api_key_auth(
            request=request, api_key=bearer_token
        )
    except Exception as e:
        pytest.fail(f"Team exists. This should not fail - {e}")

    ## 4. ASSERT USER_API_KEY_AUTH format (used for tpm/rpm limiting in parallel_request_limiter.py)

    assert team_result.team_tpm_limit == 100
    assert team_result.team_rpm_limit == 99
    assert team_result.team_models == ["gpt-3.5-turbo", "gpt-4"]
