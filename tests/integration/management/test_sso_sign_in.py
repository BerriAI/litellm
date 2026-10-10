import pytest
from fastapi.testclient import TestClient
from fastapi import Request, Header
from unittest.mock import patch, MagicMock, AsyncMock

import os

import litellm
from litellm.proxy.proxy_server import app
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.proxy.management_endpoints.ui_sso import auth_callback
from litellm.proxy._types import LitellmUserRoles
import jwt
import time
from litellm.caching.caching import DualCache

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "mock_google_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "mock_google_client_secret")
    monkeypatch.setenv("PROXY_BASE_URL", "http://testserver")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "mock_master_key")


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params

    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    litellm.proxy.proxy_server.litellm_proxy_budget_name = (
        f"litellm-proxy-budget-{time.time()}"
    )
    litellm.proxy.proxy_server.user_custom_key_generate = None

    return prisma_client


@patch("fastapi_sso.sso.google.GoogleSSO")
@pytest.mark.asyncio
async def test_auth_callback_new_user(mock_google_sso, mock_env_vars, prisma_client):
    """
    Tests that a new SSO Sign In user is by default given an 'INTERNAL_USER_VIEW_ONLY' role
    """
    from litellm._uuid import uuid
    import litellm

    litellm.turn_on_debug()

    unique_user_id = str(uuid.uuid4())
    unique_user_email = f"newuser{unique_user_id}@example.com"

    try:
        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        await litellm.proxy.proxy_server.prisma_client.connect()

        litellm.proxy.proxy_server.master_key = "mock_master_key"

        mock_sso_result = MagicMock()
        mock_sso_result.email = unique_user_email
        mock_sso_result.id = unique_user_id
        mock_sso_result.provider = "google"
        mock_sso_result.user_role = (
            None                                                           
        )
        mock_google_sso.return_value.verify_and_process = AsyncMock(
            return_value=mock_sso_result
        )

        mock_request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "scheme": "http",
                "server": ("testserver", 80),
                "path": "/sso/callback",
                "query_string": b"",
                "headers": {},
            }
        )

        response = await auth_callback(request=mock_request)

        assert response.status_code == 303
        assert response.headers["location"].startswith(
            f"http://testserver/ui/?login=success"
        )

        user = await prisma_client.db.litellm_usertable.find_first(
            where={"user_id": unique_user_id}
        )
        print("inserted user from SSO", user)
        assert user is not None
        assert user.user_email == unique_user_email
        assert user.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        assert user.metadata == {"auth_provider": "google"}

    finally:
        await prisma_client.db.litellm_usertable.delete(
            where={"user_id": unique_user_id}
        )


@patch("fastapi_sso.sso.google.GoogleSSO")
@pytest.mark.asyncio
async def test_auth_callback_new_user_with_sso_default(
    mock_google_sso, mock_env_vars, prisma_client
):
    """
    When litellm_settings.default_internal_user_params.user_role = 'INTERNAL_USER'

    Tests that a new SSO Sign In user is by default given an 'INTERNAL_USER' role
    """
    from litellm._uuid import uuid

    unique_user_id = str(uuid.uuid4())
    unique_user_email = f"newuser{unique_user_id}@example.com"

    try:
        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        litellm.default_internal_user_params = {
            "user_role": LitellmUserRoles.INTERNAL_USER.value
        }
        await litellm.proxy.proxy_server.prisma_client.connect()

        litellm.proxy.proxy_server.master_key = "mock_master_key"

        mock_sso_result = MagicMock()
        mock_sso_result.email = unique_user_email
        mock_sso_result.id = unique_user_id
        mock_sso_result.provider = "google"
        mock_google_sso.return_value.verify_and_process = AsyncMock(
            return_value=mock_sso_result
        )

        mock_request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "scheme": "http",
                "server": ("testserver", 80),
                "path": "/sso/callback",
                "query_string": b"",
                "headers": {},
            }
        )

        response = await auth_callback(request=mock_request)

        assert response.status_code == 303
        assert response.headers["location"].startswith(
            f"http://testserver/ui/?login=success"
        )

        user = await prisma_client.db.litellm_usertable.find_first(
            where={"user_id": unique_user_id}
        )
        print("inserted user from SSO", user)
        assert user is not None
        assert user.user_email == unique_user_email
        assert user.user_role == LitellmUserRoles.INTERNAL_USER

    finally:
        await prisma_client.db.litellm_usertable.delete(
            where={"user_id": unique_user_id}
        )
        litellm.default_internal_user_params = None
