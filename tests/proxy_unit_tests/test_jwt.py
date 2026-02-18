#### What this tests ####
#    Unit tests for JWT-Auth

import asyncio
import os
import random
import sys
import time
import traceback
from litellm._uuid import uuid

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, HTTPException
from fastapi.routing import APIRoute
from fastapi.responses import Response
import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LiteLLM_UserTable,
    LiteLLMRoutes,
    JWTAuthBuilderResult,
)
from litellm.proxy.auth.handle_jwt import JWTHandler, JWTAuthManager
from litellm.proxy.management_endpoints.team_endpoints import new_team
from litellm.proxy.proxy_server import chat_completion
from typing import Literal

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
async def test_token_single_public_key(monkeypatch):
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

    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://example.com/public-key")

    # set cache
    cache = DualCache()

    await cache.async_set_cache(
        key="litellm_jwt_auth_keys_https://example.com/public-key",
        value=backend_keys["keys"],
    )

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
async def test_valid_invalid_token(audience, monkeypatch):
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

    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://example.com/public-key")

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

    await cache.async_set_cache(
        key="litellm_jwt_auth_keys_https://example.com/public-key", value=[public_jwk]
    )

    jwt_handler = JWTHandler()

    jwt_handler.user_api_key_cache = cache

    # VALID TOKEN
    ## GENERATE A TOKEN
    # Assuming the current time is in UTC
    expiration_time = int((datetime.now() + timedelta(minutes=10)).timestamp())

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
    expiration_time = int((datetime.now() + timedelta(minutes=10)).timestamp())

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

    # Assuming PrismaClient is a class that needs to be instantiated
    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    return prisma_client


@pytest.fixture
def team_token_tuple():
    import json
    from litellm._uuid import uuid

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


@pytest.mark.parametrize("audience", [None, "litellm-proxy"])
@pytest.mark.asyncio
async def test_team_token_output(prisma_client, audience, monkeypatch):
    import json
    from litellm._uuid import uuid

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

    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://example.com/public-key")

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

    await cache.async_set_cache(
        key="litellm_jwt_auth_keys_https://example.com/public-key", value=[public_jwk]
    )

    jwt_handler = JWTHandler()

    jwt_handler.user_api_key_cache = cache

    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="client_id")

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

    request = Request(scope={"type": "http", "headers": []})
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
            http_request=Request(scope={"type": "http", "headers": []}),
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
async def aaaatest_user_token_output(
    prisma_client, audience, team_id_set, default_team_id, user_id_upsert, monkeypatch
):
    from litellm._uuid import uuid

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
    from litellm._uuid import uuid

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

    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://example.com/public-key")

    # set cache
    cache = DualCache()

    await cache.async_set_cache(
        key="litellm_jwt_auth_keys_https://example.com/public-key", value=[public_jwk]
    )

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
    expiration_time = int((datetime.now() + timedelta(minutes=10)).timestamp())

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

    request = Request(scope={"type": "http", "headers": []})
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
            http_request=Request(scope={"type": "http", "headers": []}),
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
                http_request=Request(scope={"type": "http", "headers": []}),
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
            await user_info(request=request, user_id=user_id)
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


@pytest.mark.parametrize("admin_allowed_routes", [None, ["ui_routes"]])
@pytest.mark.parametrize("audience", [None, "litellm-proxy"])
@pytest.mark.asyncio
async def test_allowed_routes_admin(
    prisma_client, audience, admin_allowed_routes, monkeypatch
):
    """
    Add a check to make sure jwt proxy admin scope can access all allowed admin routes

    - iterate through allowed endpoints
    - check if admin passes user_api_key_auth for them
    """
    import json
    from litellm._uuid import uuid

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

    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://example.com/public-key")

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

    await cache.async_set_cache(
        key="litellm_jwt_auth_keys_https://example.com/public-key", value=[public_jwk]
    )

    jwt_handler = JWTHandler()

    jwt_handler.user_api_key_cache = cache

    if admin_allowed_routes:
        jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
            team_id_jwt_field="client_id", admin_allowed_routes=admin_allowed_routes
        )
    else:
        jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="client_id")

    # VALID TOKEN
    ## GENERATE A TOKEN
    # Assuming the current time is in UTC
    expiration_time = int((datetime.now() + timedelta(minutes=10)).timestamp())

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

    print(f"admin_token: {admin_token}")
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
        request = Request(scope={"type": "http", "headers": []})

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
            token=None,
            user_id=None,
            end_user_id=None,
            team_id="1234",
            response_cost=20,
            parent_otel_span=None,
        )  # type: ignore

        await asyncio.sleep(3)
        mock_call_cache.assert_awaited_once()


@pytest.fixture
def public_jwt_key():
    import json

    import jwt
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

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

    return {"private_key": private_key, "public_jwk": public_jwk}


async def mock_user_object(*args, **kwargs):
    print("Args: {}".format(args))
    print("kwargs: {}".format(kwargs))
    assert kwargs["user_id_upsert"] is True
    # Return a mock user object
    user_id = kwargs.get("user_id")
    user_email = kwargs.get("user_email")
    return LiteLLM_UserTable(
        spend=0, 
        user_id=user_id, 
        max_budget=None, 
        user_email=user_email
    )


@pytest.mark.parametrize(
    "user_email, should_work", [("ishaan@berri.ai", True), ("krrish@tassle.xyz", False)]
)
@pytest.mark.asyncio
async def test_allow_access_by_email(
    public_jwt_key, user_email, should_work, monkeypatch
):
    """
    Allow anyone with an `@xyz.com` email make a request to the proxy.

    Relevant issue: https://github.com/BerriAI/litellm/issues/5605
    """
    import jwt
    from starlette.datastructures import URL

    from litellm.proxy._types import NewTeamRequest, UserAPIKeyAuth
    from litellm.proxy.proxy_server import user_api_key_auth

    public_jwk = public_jwt_key["public_jwk"]
    private_key = public_jwt_key["private_key"]

    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://example.com/public-key")

    # set cache
    cache = DualCache()

    await cache.async_set_cache(
        key="litellm_jwt_auth_keys_https://example.com/public-key", value=[public_jwk]
    )

    jwt_handler = JWTHandler()

    jwt_handler.user_api_key_cache = cache

    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        user_email_jwt_field="email",
        user_allowed_email_domain="berri.ai",
        user_id_upsert=True,
    )

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
        "aud": "litellm-proxy",
        "email": user_email,
    }

    # Generate the JWT token
    # But before, you should convert bytes to string
    private_key_str = private_key.decode("utf-8")

    ## team token
    token = jwt.encode(payload, private_key_str, algorithm="RS256")

    ## VERIFY IT WORKS
    # Expect the call to succeed
    response = await jwt_handler.auth_jwt(token=token)
    assert response is not None  # Adjust this based on your actual response check

    ## RUN IT THROUGH USER API KEY AUTH
    bearer_token = "Bearer " + token

    request = Request(scope={"type": "http", "headers": []})

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
    setattr(litellm.proxy.proxy_server, "prisma_client", {})

    # AsyncMock(
    #     return_value=LiteLLM_UserTable(
    #         spend=0, user_id=user_email, max_budget=None, user_email=user_email
    #     )
    # ),
    with patch.object(
        litellm.proxy.auth.handle_jwt,
        "get_user_object",
        side_effect=mock_user_object,
    ) as mock_client:
        if should_work:
            # Expect the call to succeed
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            assert result is not None  # Adjust this based on your actual response check
        else:
            # Expect the call to fail
            with pytest.raises(
                Exception
            ):  # Replace with the actual exception raised on failure
                resp = await user_api_key_auth(request=request, api_key=bearer_token)
                print(resp)


def test_get_public_key_from_jwk_url():
    import litellm
    from litellm.proxy.auth.handle_jwt import JWTHandler

    jwt_handler = JWTHandler()

    jwk_response = [
        {
            "kty": "RSA",
            "alg": "RS256",
            "kid": "RaPJB8QVptWHjHcoHkVlUWO4f0D3BtcY6iSDXgGVBgk",
            "use": "sig",
            "e": "AQAB",
            "n": "zgLDu57gLpkzzIkKrTKQVyjK8X40hvu6X_JOeFjmYmI0r3bh7FTOmre5rTEkDOL-1xvQguZAx4hjKmCzBU5Kz84FbsGiqM0ug19df4kwdTS6XOM6YEKUZrbaw4P7xTPsbZj7W2G_kxWNm3Xaxq6UKFdUF7n9snnBKKD6iUA-cE6HfsYmt9OhYZJfy44dbAbuanFmAsWw97SHrPFL3ueh3Ixt19KgpF4iSsXNg3YvoesdFM8psmivgePyyHA8k7pK1Yq7rNQX1Q9nzhvP-F7ocFbP52KYPlaSTu30YwPTVTFKYpDNmHT1fZ7LXZZNLrP_7-NSY76HS2ozSpzjsGVelQ",
        }
    ]

    public_key = jwt_handler.parse_keys(
        keys=jwk_response,
        kid="RaPJB8QVptWHjHcoHkVlUWO4f0D3BtcY6iSDXgGVBgk",
    )

    assert public_key is not None
    assert public_key == jwk_response[0]


@pytest.mark.asyncio
async def test_end_user_jwt_auth(monkeypatch):
    import litellm
    from litellm.proxy.auth.handle_jwt import JWTHandler
    from litellm.caching import DualCache
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.proxy_server import user_api_key_auth
    import json

    monkeypatch.delenv("JWT_AUDIENCE", None)
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://example.com/public-key")
    jwt_handler = JWTHandler()

    litellm_jwtauth = LiteLLM_JWTAuth(
        end_user_id_jwt_field="sub",
    )

    cache = DualCache()

    keys = [
        {
            "kid": "d-1733370597545",
            "alg": "RS256",
            "kty": "RSA",
            "use": "sig",
            "n": "j5Ik60FJSUIPMVdMSU8vhYvyPPH7rUsUllNI0BfBlIkgEYFk2mg4KX1XDQc6mcKFjbq9k_7TSkHWKnfPhNkkb0MdmZLKbwTrmte2k8xWDxp-rSmZpIJwC1zuPDx5joLHBgIb09-K2cPL2rvwzP75WtOr_QLXBerHAbXx8cOdI7mrSRWJ9iXbKv_pLDnZHnGNld75tztb8nCtgrywkF010jGi1xxaT8UKsTvK-QkIBkYI6m6WR9LMnG2OZm-ExuvNPUenfYUsqnynPF4SMNZwyQqJfavSLKI8uMzB2s9pcbx5HfQwIOYhMlgBHjhdDn2IUSnXSJqSsN6RQO18M2rgPQ",
            "e": "AQAB",
        },
        {
            "kid": "s-f836dd32-ef71-426a-8804-946a7f230bc9",
            "alg": "RS256",
            "kty": "RSA",
            "use": "sig",
            "n": "2A5-ZA18YKn7M4OtxsfXBc3Z7n2WyHTxbK4GEBlmD9T9TDr4sbJaI4oHfTvzsAC3H2r2YkASzrCISXMXQJjLHoeLgDVcKs8qTdLj7K5FNT9fA0kU9ayUjSGrqkz57SG7oNf9Wp__Qa-H-bs6Z8_CEfBy0JA9QSHUfrdOXp4vCB_qLn6DE0DJH9ELAq_0nktVQk_oxlvXlGtVZSZe31mNNgiD__RJMogf-SIFcYOkMLVGTTEBYiCk1mHxXS6oJZaVSWiBgHzu5wkra5AfQLUVelQaupT5H81hFPmiceEApf_2DacnqqRV4-Nl8sjhJtuTXiprVS2Z5r2pOMz_kVGNgw",
            "e": "AQAB",
        },
    ]

    cache.set_cache(
        key="litellm_jwt_auth_keys_https://example.com/public-key",
        value=keys,
    )

    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=cache,
        litellm_jwtauth=litellm_jwtauth,
        leeway=100000000000000,
    )

    token = "eyJraWQiOiJkLTE3MzMzNzA1OTc1NDUiLCJ0eXAiOiJKV1QiLCJ2ZXJzaW9uIjoiNCIsImFsZyI6IlJTMjU2In0.eyJpYXQiOjE3MzM1MDcyNzcsImV4cCI6MTczMzUwNzg3Nywic3ViIjoiODFiM2U1MmEtNjdhNi00ZWZiLTk2NDUtNzA1MjdlMTAxNDc5IiwidElkIjoicHVibGljIiwic2Vzc2lvbkhhbmRsZSI6Ijg4M2Y4YWFmLWUwOTEtNGE1Ny04YTJhLTRiMjcwMmZhZjMzYyIsInJlZnJlc2hUb2tlbkhhc2gxIjoiNDVhNDRhYjlmOTMwMGQyOTY4ZjkxODZkYWQ0YTcwY2QwNjk2YzBiNTBmZmUxZmQ4ZTM2YzU1NGU0MWE4ODU0YiIsInBhcmVudFJlZnJlc2hUb2tlbkhhc2gxIjpudWxsLCJhbnRpQ3NyZlRva2VuIjpudWxsLCJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjMwMDEvYXV0aC9zdCIsImxlZ2FjeV9jb21wYW55X2lkIjoxNTI0OTM5LCJsZWdhY3lfaWQiOjM5NzAyNzk1LCJzY29wZSI6WyJza2lsbF91cCJdLCJzdC1ldiI6eyJ0IjoxNzMzNTA3Mjc4NzAwLCJ2IjpmYWxzZX19.XlYrT6dRIjaZKkJtdr7C_UuxajFRbNpA9BnIsny3rxiPVyS8rhIBwxW12tZwgttRywmXrXK-msowFhWU4XdL5Qfe4lwZb2HTbDeGiQPvQTlOjWWYMhgCoKdPtjCQsAcW45rg7aQ0p42JFQPoAQa8AnGfxXpgx2vSR7njiZ3ZZyHerDdKQHyIGSFVOxoK0TgR-hxBVY__Wjg8UTKgKSz9KU_uwnPgpe2DeYmP-LTK2oeoygsVRmbldY_GrrcRe3nqYcUfFkxSs0FSsoSv35jIxiptXfCjhEB1Y5eaJhHEjlYlP2rw98JysYxjO2rZbAdUpL3itPeo3T2uh1NZr_lArw"

    response = await jwt_handler.auth_jwt(token=token)

    assert response is not None

    end_user_id = jwt_handler.get_end_user_id(
        token=response,
        default_value=None,
    )

    assert end_user_id is not None

    ## CHECK USER API KEY AUTH ##
    from starlette.datastructures import URL

    bearer_token = "Bearer " + token

    api_route = APIRoute(path="/chat/completions", endpoint=chat_completion)
    request = Request(
        {
            "type": "http",
            "route": api_route,
            "path": "/chat/completions",
            "headers": [(b"authorization", f"Bearer {bearer_token}".encode("latin-1"))],
            "method": "POST",
        }
    )

    async def return_body():
        body_dict = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        }
        # Serialize the dictionary to JSON and encode it to bytes
        return json.dumps(body_dict).encode("utf-8")

    request.body = return_body

    ## 1. INITIAL TEAM CALL - should fail
    # use generated key to auth in
    from litellm import Router
    from litellm.types.router import RouterGeneralSettings
    
    # Create a router with pass_through_all_models enabled
    router = Router(
        model_list=[],
        router_general_settings=RouterGeneralSettings(
            pass_through_all_models=True
        ),
    )
    
    setattr(
        litellm.proxy.proxy_server,
        "general_settings",
        {"enable_jwt_auth": True},
    )
    setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        router,
    )
    setattr(litellm.proxy.proxy_server, "prisma_client", {})
    setattr(litellm.proxy.proxy_server, "jwt_handler", jwt_handler)
    from litellm.proxy.proxy_server import cost_tracking

    cost_tracking()
    result = await user_api_key_auth(request=request, api_key=bearer_token)
    
    # Assert that end_user_id is correctly extracted from JWT token's 'sub' field
    assert result.end_user_id == "81b3e52a-67a6-4efb-9645-70527e101479"

    temp_response = Response()
    from litellm.proxy.hooks.proxy_track_cost_callback import (
        _should_track_cost_callback,
    )

    # Mock the actual LLM completion call
    mock_response = litellm.ModelResponse(
        id="chatcmpl-mock",
        choices=[
            litellm.Choices(
                finish_reason="stop",
                index=0,
                message=litellm.Message(
                    content="Hello! I'm doing well, thank you for asking.",
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        usage=litellm.Usage(
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25,
        ),
    )

    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)) as mock_completion:
        resp = await chat_completion(
            request=request,
            fastapi_response=temp_response,
            model="gpt-4o",
            user_api_key_dict=result,
        )

        assert resp is not None

        await asyncio.sleep(1)

        # Verify the completion was called with correct end_user_id
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args.kwargs
        
        # end_user_id is passed in metadata as 'user_api_key_end_user_id'
        metadata = call_kwargs.get("metadata", {})
        assert metadata.get("user_api_key_end_user_id") == "81b3e52a-67a6-4efb-9645-70527e101479"


def test_can_rbac_role_call_route():
    from litellm.proxy.auth.handle_jwt import JWTAuthManager
    from litellm.proxy._types import RoleBasedPermissions
    from litellm.proxy._types import LitellmUserRoles

    with pytest.raises(HTTPException):
        JWTAuthManager.can_rbac_role_call_route(
            rbac_role=LitellmUserRoles.TEAM,
            general_settings={
                "role_permissions": [
                    RoleBasedPermissions(
                        role=LitellmUserRoles.TEAM, routes=["/v1/chat/completions"]
                    )
                ]
            },
            route="/v1/embeddings",
        )


def test_user_api_key_auth_jwt_hashing():
    """
    Test that JWT tokens are properly hashed in UserAPIKeyAuth
    This test ensures that when a JWT token is passed as an API key,
    it gets hashed with the "hashed-jwt-" prefix.

    Critical: This was a security fix for users
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler
    
    # Test with a JWT token (3 parts separated by dots)
    jwt_token = "test-jwt-token-header.payload.signature"
    
    # Create UserAPIKeyAuth instance with JWT
    user_auth = UserAPIKeyAuth(api_key=jwt_token)
    
    # Verify that the API key is hashed with "hashed-jwt-" prefix
    # critical - the raw JWT token should not be in the api_key or token
    assert user_auth.api_key.startswith("hashed-jwt-")
    assert user_auth.token.startswith("hashed-jwt-")
    assert jwt_token not in user_auth.api_key
    assert jwt_token not in user_auth.token

    
    # Test with a regular API key (should not be hashed)
    regular_api_key = "sk-1234567890abcdef"
    user_auth_regular = UserAPIKeyAuth(api_key=regular_api_key)
    
    # Verify that regular API key is hashed normally (without "hashed-jwt-" prefix)
    assert not user_auth_regular.api_key.startswith("hashed-jwt-")
    assert not user_auth_regular.token.startswith("hashed-jwt-")
    
    # Test with a non-JWT, non-sk string (should not be hashed)
    non_jwt_key = "some-random-key"
    user_auth_non_jwt = UserAPIKeyAuth(api_key=non_jwt_key)
    
    # Verify that non-JWT key is not hashed
    assert user_auth_non_jwt.api_key == non_jwt_key
    assert user_auth_non_jwt.token == non_jwt_key


def test_jwt_handler_is_jwt_static_method():
    """
    Test that JWTHandler.is_jwt is a static method and works correctly
    """
    from litellm.proxy.auth.handle_jwt import JWTHandler
    
    # Test with valid JWT format
    valid_jwt = "test-jwt-token-header.payload.signature"
    assert JWTHandler.is_jwt(valid_jwt) == True
    
    # Test with invalid JWT format (only 2 parts)
    invalid_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"
    assert JWTHandler.is_jwt(invalid_jwt) == False
    
    # Test with regular API key
    regular_key = "sk-1234567890abcdef"
    assert JWTHandler.is_jwt(regular_key) == False
    
    # Test with empty string
    assert JWTHandler.is_jwt("") == False


@pytest.mark.parametrize(
    "requested_model, should_work",
    [
        ("gpt-3.5-turbo-testing", True),
        ("gpt-4o", False),
    ],
)
def test_check_scope_based_access(requested_model, should_work):
    from litellm.proxy.auth.handle_jwt import JWTAuthManager
    from litellm.proxy._types import ScopeMapping

    args = {
        "scope_mappings": [
            ScopeMapping(
                models=["anthropic-claude"],
                routes=["/v1/chat/completions"],
                scope="litellm.api.consumer",
            ),
            ScopeMapping(
                models=["gpt-3.5-turbo-testing"],
                routes=None,
                scope="litellm.api.gpt_3_5_turbo",
            ),
        ],
        "scopes": [
            "profile",
            "groups-scope",
            "email",
            "litellm.api.gpt_3_5_turbo",
            "litellm.api.consumer",
        ],
        "request_data": {
            "model": requested_model,
            "messages": [{"role": "user", "content": "Hey, how's it going 1234?"}],
        },
        "general_settings": {
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "team_id_jwt_field": "client_id",
                "team_id_upsert": True,
                "scope_mappings": [
                    {
                        "scope": "litellm.api.consumer",
                        "models": ["anthropic-claude"],
                        "routes": ["/v1/chat/completions"],
                    },
                    {
                        "scope": "litellm.api.gpt_3_5_turbo",
                        "models": ["gpt-3.5-turbo-testing"],
                    },
                ],
                "enforce_scope_based_access": True,
                "enforce_rbac": True,
            },
        },
    }

    if should_work:
        JWTAuthManager.check_scope_based_access(**args)
    else:
        with pytest.raises(HTTPException):
            JWTAuthManager.check_scope_based_access(**args)


@pytest.mark.asyncio
async def test_custom_validate_called():
    # Setup
    mock_custom_validate = MagicMock(return_value=True)

    jwt_handler = MagicMock()
    jwt_handler.litellm_jwtauth = MagicMock(
        custom_validate=mock_custom_validate,
        allowed_routes=["/chat/completions"],
        oidc_userinfo_enabled=False,
    )
    jwt_handler.auth_jwt = AsyncMock(return_value={"sub": "test_user"})

    try:
        await JWTAuthManager.auth_builder(
            api_key="test",
            jwt_handler=jwt_handler,
            request_data={},
            general_settings={},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )
    except Exception:
        pass
    # Assert custom_validate was called with the jwt token
    mock_custom_validate.assert_called_once_with({"sub": "test_user"})


@pytest.mark.asyncio
async def test_auth_jwt_es256_jwk_path(monkeypatch):
    import time, base64, jwt
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization

    monkeypatch.delenv("JWT_AUDIENCE", raising=False)

    def b64url_uint(n: int, size: int) -> str:
        return base64.urlsafe_b64encode(n.to_bytes(size, "big")).rstrip(b"=").decode()

    ec_key = ec.generate_private_key(ec.SECP256R1())
    ec_priv_pem = ec_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    pub = ec_key.public_key().public_numbers()
    ec_jwk = {
        "kty": "EC",
        "crv": "P-256",
        "x": b64url_uint(pub.x, 32),
        "y": b64url_uint(pub.y, 32),
        "kid": "ec1",
        "alg": "ES256",
        "use": "sig",
    }

    now = int(time.time())
    token = jwt.encode(
        {"sub": "alice", "aud": "litellm-proxy", "iss": "http://example", "iat": now, "exp": now + 300},
        ec_priv_pem,
        algorithm="ES256",
        headers={"kid": "ec1"},
    )

    h = JWTHandler()
    with patch.object(h, "get_public_key", new=AsyncMock(return_value=ec_jwk)):
        claims = await h.auth_jwt(token)
        assert claims["sub"] == "alice"


@pytest.mark.asyncio
async def test_auth_jwt_rs256_regression(monkeypatch):
    """
    Regression: RSA path must still work (kty RSA, n/e) after EC support.
    """
    import time, base64, jwt
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    monkeypatch.delenv("JWT_AUDIENCE", raising=False)

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsa_priv_pem = rsa_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = rsa_key.public_key().public_numbers()

    def b64url(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    n = pub.n.to_bytes((pub.n.bit_length() + 7) // 8, "big")
    e = pub.e.to_bytes((pub.e.bit_length() + 7) // 8, "big")
    rsa_jwk = {
        "kty": "RSA",
        "n": b64url(n),
        "e": b64url(e),
        "kid": "rsa1",
        "alg": "RS256",
        "use": "sig",
    }

    now = int(time.time())
    token = jwt.encode(
        {"sub": "bob", "aud": "litellm-proxy", "iss": "http://example", "iat": now, "exp": now + 300},
        rsa_priv_pem,
        algorithm="RS256",
        headers={"kid": "rsa1"},
    )

    h = JWTHandler()
    with patch.object(h, "get_public_key", new=AsyncMock(return_value=rsa_jwk)):
        claims = await h.auth_jwt(token)
        assert claims["sub"] == "bob"


@pytest.mark.asyncio
async def test_auth_jwt_mismatched_key_fails(monkeypatch):
    """
    Negative: ES256 token must fail if JWKS returns an RSA key (mismatch).
    """
    import time, base64, jwt
    from cryptography.hazmat.primitives.asymmetric import ec, rsa
    from cryptography.hazmat.primitives import serialization

    monkeypatch.delenv("JWT_AUDIENCE", raising=False)

    # ES256 token
    ec_key = ec.generate_private_key(ec.SECP256R1())
    ec_priv_pem = ec_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    now = int(time.time())
    token = jwt.encode(
        {"sub": "mallory", "aud": "litellm-proxy", "iss": "http://example", "iat": now, "exp": now + 300},
        ec_priv_pem,
        algorithm="ES256",
        headers={"kid": "ec1"},
    )

    # RSA JWK (wrong key)
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = rsa_key.public_key().public_numbers()

    def b64url(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    rsa_jwk = {
        "kty": "RSA",
        "n": b64url(pub.n.to_bytes((pub.n.bit_length() + 7) // 8, "big")),
        "e": b64url(pub.e.to_bytes((pub.e.bit_length() + 7) // 8, "big")),
        "kid": "rsa1",
        "alg": "RS256",
        "use": "sig",
    }

    h = JWTHandler()
    with patch.object(h, "get_public_key", new=AsyncMock(return_value=rsa_jwk)):
        with pytest.raises(Exception) as exc:
            await h.auth_jwt(token)
        assert "Validation fails" in str(exc.value)