#### What this tests ####
#    Unit tests for JWT-Auth

import asyncio
import os
import random
import sys
import time
import traceback
import uuid

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from litellm.caching import DualCache
from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLMRoutes
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.management_endpoints.team_endpoints import new_team
from litellm.proxy.proxy_server import chat_completion

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


@pytest.mark.parametrize("audience", [None, "litellm-proxy"])
@pytest.mark.asyncio
async def test_valid_invalid_token(audience):
    """
    Tests
    - valid token
    - invalid token
    """
    import json

    import jwt
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    os.environ.pop("JWT_AUDIENCE", None)
    if audience:
        os.environ["JWT_AUDIENCE"] = audience

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
        "aud": audience,
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
        "aud": audience,
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
    from litellm.proxy.proxy_cli import append_query_params
    from litellm.proxy.utils import PrismaClient, ProxyLogging

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


@pytest.fixture
def team_token_tuple():
    import json
    import uuid

    import jwt
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm
    from litellm.proxy._types import NewTeamRequest, UserAPIKeyAuth
    from litellm.proxy.proxy_server import user_api_key_auth

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
        "aud": None,
    }

    # Generate the JWT token
    # But before, you should convert bytes to string
    private_key_str = private_key.decode("utf-8")

    ## team token
    token = jwt.encode(payload, private_key_str, algorithm="RS256")

    return team_id, token, public_jwk


@pytest.mark.parametrize("audience", [None, "litellm-proxy"])
@pytest.mark.asyncio
async def test_team_token_output(prisma_client, audience):
    import json
    import uuid

    import jwt
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm
    from litellm.proxy._types import NewTeamRequest, UserAPIKeyAuth
    from litellm.proxy.proxy_server import user_api_key_auth

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    await litellm.proxy.proxy_server.prisma_client.connect()

    os.environ.pop("JWT_AUDIENCE", None)
    if audience:
        os.environ["JWT_AUDIENCE"] = audience

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

    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="client_id")

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
        "aud": audience,
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
        "aud": audience,
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
    setattr(
        litellm.proxy.proxy_server,
        "general_settings",
        {
            "enable_jwt_auth": True,
        },
    )
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
            http_request=Request(scope={"type": "http"}),
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


@pytest.mark.parametrize("audience", [None, "litellm-proxy"])
@pytest.mark.parametrize(
    "team_id_set, default_team_id",
    [(True, False), (False, True)],
)
@pytest.mark.parametrize("user_id_upsert", [True, False])
@pytest.mark.asyncio
async def test_user_token_output(
    prisma_client, audience, team_id_set, default_team_id, user_id_upsert
):
    import uuid

    args = locals()
    print(f"received args - {args}")
    if default_team_id:
        default_team_id = "team_id_12344_{}".format(uuid.uuid4())
    """
    - If user required, check if it exists
    - fail initial request (when user doesn't exist)
    - create user
    - retry -> it should pass now
    """
    import json
    import uuid

    import jwt
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm
    from litellm.proxy._types import NewTeamRequest, NewUserRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        new_user,
        user_info,
    )
    from litellm.proxy.proxy_server import user_api_key_auth

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    await litellm.proxy.proxy_server.prisma_client.connect()

    os.environ.pop("JWT_AUDIENCE", None)
    if audience:
        os.environ["JWT_AUDIENCE"] = audience

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

    jwt_handler.litellm_jwtauth.user_id_jwt_field = "sub"
    jwt_handler.litellm_jwtauth.team_id_default = default_team_id
    jwt_handler.litellm_jwtauth.user_id_upsert = user_id_upsert

    if team_id_set:
        jwt_handler.litellm_jwtauth.team_id_jwt_field = "client_id"

    # VALID TOKEN
    ## GENERATE A TOKEN
    # Assuming the current time is in UTC
    expiration_time = int((datetime.utcnow() + timedelta(minutes=10)).timestamp())

    team_id = f"team123_{uuid.uuid4()}"
    user_id = f"user123_{uuid.uuid4()}"
    payload = {
        "sub": user_id,
        "exp": expiration_time,  # set the token to expire in 10 minutes
        "scope": "litellm_team",
        "client_id": team_id,
        "aud": audience,
    }

    # Generate the JWT token
    # But before, you should convert bytes to string
    private_key_str = private_key.decode("utf-8")

    ## team token
    token = jwt.encode(payload, private_key_str, algorithm="RS256")

    ## admin token
    payload = {
        "sub": user_id,
        "exp": expiration_time,  # set the token to expire in 10 minutes
        "scope": "litellm_proxy_admin",
        "aud": audience,
    }

    admin_token = jwt.encode(payload, private_key_str, algorithm="RS256")

    ## VERIFY IT WORKS

    # verify token

    response = await jwt_handler.auth_jwt(token=token)

    ## RUN IT THROUGH USER API KEY AUTH

    """
    - 1. Initial call should fail -> team doesn't exist
    - 2. Create team via admin token 
    - 3. 2nd call w/ same team -> call should fail -> user doesn't exist
    - 4. Create user via admin token
    - 5. 3rd call w/ same team, same user -> call should succeed
    - 6. assert user api key auth format
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
            http_request=Request(scope={"type": "http"}),
        )
        if default_team_id:
            await new_team(
                data=NewTeamRequest(
                    team_id=default_team_id,
                    tpm_limit=100,
                    rpm_limit=99,
                    models=["gpt-3.5-turbo", "gpt-4"],
                ),
                user_api_key_dict=result,
                http_request=Request(scope={"type": "http"}),
            )
    except Exception as e:
        pytest.fail(f"This should not fail - {str(e)}")

    ## 3. 2nd CALL W/ TEAM TOKEN - should fail
    bearer_token = "Bearer " + token
    request._url = URL(url="/chat/completions")
    try:
        team_result: UserAPIKeyAuth = await user_api_key_auth(
            request=request, api_key=bearer_token
        )
        if user_id_upsert == False:
            pytest.fail(f"User doesn't exist. this should fail")
    except Exception as e:
        pass

    ## 4. Create user
    if user_id_upsert:
        ## check if user already exists
        try:
            bearer_token = "Bearer " + admin_token

            request._url = URL(url="/team/new")
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            await user_info(user_id=user_id)
        except Exception as e:
            pytest.fail(f"This should not fail - {str(e)}")
    else:
        try:
            bearer_token = "Bearer " + admin_token

            request._url = URL(url="/team/new")
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            await new_user(
                data=NewUserRequest(
                    user_id=user_id,
                ),
            )
        except Exception as e:
            pytest.fail(f"This should not fail - {str(e)}")

    ## 5. 3rd call w/ same team, same user -> call should succeed
    bearer_token = "Bearer " + token
    request._url = URL(url="/chat/completions")
    try:
        team_result: UserAPIKeyAuth = await user_api_key_auth(
            request=request, api_key=bearer_token
        )
    except Exception as e:
        pytest.fail(f"Team exists. This should not fail - {e}")

    ## 6. ASSERT USER_API_KEY_AUTH format (used for tpm/rpm limiting in parallel_request_limiter.py AND cost tracking)

    if team_id_set or default_team_id is not None:
        assert team_result.team_tpm_limit == 100
        assert team_result.team_rpm_limit == 99
        assert team_result.team_models == ["gpt-3.5-turbo", "gpt-4"]
    assert team_result.user_id == user_id


@pytest.mark.parametrize("audience", [None, "litellm-proxy"])
@pytest.mark.asyncio
async def test_allowed_routes_admin(prisma_client, audience):
    """
    Add a check to make sure jwt proxy admin scope can access all allowed admin routes

    - iterate through allowed endpoints
    - check if admin passes user_api_key_auth for them
    """
    import json
    import uuid

    import jwt
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm
    from litellm.proxy._types import NewTeamRequest, UserAPIKeyAuth
    from litellm.proxy.proxy_server import user_api_key_auth

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    await litellm.proxy.proxy_server.prisma_client.connect()

    os.environ.pop("JWT_AUDIENCE", None)
    if audience:
        os.environ["JWT_AUDIENCE"] = audience

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

    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="client_id")

    # VALID TOKEN
    ## GENERATE A TOKEN
    # Assuming the current time is in UTC
    expiration_time = int((datetime.utcnow() + timedelta(minutes=10)).timestamp())

    # Generate the JWT token
    # But before, you should convert bytes to string
    private_key_str = private_key.decode("utf-8")

    ## admin token
    payload = {
        "sub": "user123",
        "exp": expiration_time,  # set the token to expire in 10 minutes
        "scope": "litellm_proxy_admin",
        "aud": audience,
    }

    admin_token = jwt.encode(payload, private_key_str, algorithm="RS256")

    # verify token

    response = await jwt_handler.auth_jwt(token=admin_token)

    ## RUN IT THROUGH USER API KEY AUTH

    """
    - 1. Initial call should fail -> team doesn't exist
    - 2. Create team via admin token 
    - 3. 2nd call w/ same team -> call should succeed -> assert UserAPIKeyAuth object correctly formatted
    """

    bearer_token = "Bearer " + admin_token

    pseudo_routes = jwt_handler.litellm_jwtauth.admin_allowed_routes

    actual_routes = []
    for route in pseudo_routes:
        if route in LiteLLMRoutes.__members__:
            actual_routes.extend(LiteLLMRoutes[route].value)

    for route in actual_routes:
        request = Request(scope={"type": "http"})

        request._url = URL(url=route)

        ## 1. INITIAL TEAM CALL - should fail
        # use generated key to auth in
        setattr(
            litellm.proxy.proxy_server,
            "general_settings",
            {
                "enable_jwt_auth": True,
            },
        )
        setattr(litellm.proxy.proxy_server, "jwt_handler", jwt_handler)
        try:
            result = await user_api_key_auth(request=request, api_key=bearer_token)
        except Exception as e:
            raise e


from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_team_cache_update_called():
    import litellm
    from litellm.proxy.proxy_server import user_api_key_cache

    # Use setattr to replace the method on the user_api_key_cache object
    cache = DualCache()

    setattr(
        litellm.proxy.proxy_server,
        "user_api_key_cache",
        cache,
    )

    with patch.object(cache, "async_get_cache", new=AsyncMock()) as mock_call_cache:
        cache.async_get_cache = mock_call_cache
        # Call the function under test
        await litellm.proxy.proxy_server.update_cache(
            token=None, user_id=None, end_user_id=None, team_id="1234", response_cost=20
        )  # type: ignore

        await asyncio.sleep(3)
        mock_call_cache.assert_awaited_once()
