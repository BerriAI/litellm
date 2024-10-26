import os
import sys
import traceback
import uuid
import datetime as dt
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

load_dotenv()
import io
import os
import time


# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException, Request
import pytest
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth

# Replace the actual hash_token function with our mock
import litellm.proxy.auth.route_checks


# Mock objects and functions
class MockRequest:
    def __init__(self, query_params=None):
        self.query_params = query_params or {}


def mock_hash_token(token):
    return token


litellm.proxy.auth.route_checks.hash_token = mock_hash_token


# Test is_llm_api_route
def test_is_llm_api_route():
    assert RouteChecks.is_llm_api_route("/v1/chat/completions") is True
    assert RouteChecks.is_llm_api_route("/v1/completions") is True
    assert RouteChecks.is_llm_api_route("/v1/embeddings") is True
    assert RouteChecks.is_llm_api_route("/v1/images/generations") is True
    assert RouteChecks.is_llm_api_route("/v1/threads/thread_12345") is True
    assert RouteChecks.is_llm_api_route("/bedrock/model/invoke") is True
    assert RouteChecks.is_llm_api_route("/vertex-ai/text") is True
    assert RouteChecks.is_llm_api_route("/gemini/generate") is True
    assert RouteChecks.is_llm_api_route("/cohere/generate") is True

    # check non-matching routes
    assert RouteChecks.is_llm_api_route("/some/random/route") is False
    assert RouteChecks.is_llm_api_route("/key/regenerate/82akk800000000jjsk") is False
    assert RouteChecks.is_llm_api_route("/key/82akk800000000jjsk/delete") is False


# Test _route_matches_pattern
def test_route_matches_pattern():
    # check matching routes
    assert (
        RouteChecks._route_matches_pattern(
            "/threads/thread_12345", "/threads/{thread_id}"
        )
        is True
    )
    assert (
        RouteChecks._route_matches_pattern(
            "/key/regenerate/82akk800000000jjsk", "/key/{token_id}/regenerate"
        )
        is False
    )
    assert (
        RouteChecks._route_matches_pattern(
            "/v1/chat/completions", "/v1/chat/completions"
        )
        is True
    )
    assert (
        RouteChecks._route_matches_pattern(
            "/v1/models/gpt-4", "/v1/models/{model_name}"
        )
        is True
    )

    # check non-matching routes
    assert (
        RouteChecks._route_matches_pattern(
            "/v1/chat/completionz/thread_12345", "/v1/chat/completions/{thread_id}"
        )
        is False
    )
    assert (
        RouteChecks._route_matches_pattern(
            "/v1/{thread_id}/messages", "/v1/messages/thread_2345"
        )
        is False
    )


@pytest.fixture
def route_checks():
    return RouteChecks()


@pytest.mark.asyncio
async def test_llm_api_route(route_checks):
    """
    Internal User is allowed to access all LLM API routes
    """
    assert (
        await route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/v1/chat/completions",
            request=MockRequest(),
            valid_token=UserAPIKeyAuth(api_key="test_key"),
            api_key="test_key",
            request_data={},
        )
        is None
    )


@pytest.mark.asyncio
async def test_key_info_route_allowed(route_checks):
    """
    Internal User is allowed to access /key/info route
    """
    assert (
        await route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/key/info",
            request=MockRequest(query_params={"key": "test_key"}),
            valid_token=UserAPIKeyAuth(api_key="test_key"),
            api_key="test_key",
            request_data={},
        )
        is None
    )


@pytest.mark.asyncio
async def test_key_info_route_forbidden(route_checks):
    """
    Internal User is not allowed to access /key/info route for a key they're not using in Authenticated API Key
    """
    with pytest.raises(HTTPException) as exc_info:
        await route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/key/info",
            request=MockRequest(query_params={"key": "wrong_key"}),
            valid_token=UserAPIKeyAuth(api_key="test_key"),
            api_key="test_key",
            request_data={},
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_user_info_route_allowed(route_checks):
    """
    Internal User is allowed to access /user/info route for their own user_id
    """
    assert (
        await route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/user/info",
            request=MockRequest(query_params={"user_id": "test_user"}),
            valid_token=UserAPIKeyAuth(api_key="test_key", user_id="test_user"),
            api_key="test_key",
            request_data={},
        )
        is None
    )


@pytest.mark.asyncio
async def test_user_info_route_forbidden(route_checks):
    """
    Internal User is not allowed to access /user/info route for a different user_id
    """
    with pytest.raises(HTTPException) as exc_info:
        await route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/user/info",
            request=MockRequest(query_params={"user_id": "wrong_user"}),
            valid_token=UserAPIKeyAuth(api_key="test_key", user_id="test_user"),
            api_key="test_key",
            request_data={},
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_can_user_access_user_info_own_id():
    valid_token = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    result = await RouteChecks._can_user_access_user_info(
        valid_token, user_id="test_user"
    )
    assert result is True


@pytest.mark.asyncio
async def test_can_user_access_user_info_different_id():
    valid_token = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    with pytest.raises(HTTPException) as exc_info:
        await RouteChecks._can_user_access_user_info(valid_token, user_id="other_user")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_can_user_access_user_info_org_admin():
    valid_token = UserAPIKeyAuth(api_key="test_key", user_id="admin_user")
    user_obj = LiteLLM_UserTable(user_id="admin_user", max_budget=None, user_email=None)

    with patch.object(
        RouteChecks, "_is_org_admin_for_user_id", new_callable=AsyncMock
    ) as mock_is_admin:
        mock_is_admin.return_value = True
        result = await RouteChecks._can_user_access_user_info(
            valid_token, user_id="other_user", user_obj=user_obj
        )
        assert result is True


@pytest.mark.asyncio
async def test_is_org_admin_for_user_id():
    user_id = "test_user"
    admin_user_obj = LiteLLM_UserTable(
        user_id="admin_user", max_budget=None, user_email=None
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        mock_db = AsyncMock()
        mock_prisma.db = mock_db
        mock_db.litellm_usertable.find_unique.return_value = AsyncMock(
            organization_memberships=[AsyncMock(organization_id="org1")]
        )

        with patch(
            "litellm.proxy.auth.route_checks.OrganizationRoleBasedAccessChecks._user_is_admin_in_org",
            return_value=True,
        ):
            result = await RouteChecks._is_org_admin_for_user_id(
                user_id, admin_user_obj
            )
            assert result is True


@pytest.mark.asyncio
async def test_is_org_admin_for_user_id_not_admin():
    user_id = "test_user"
    non_admin_user_obj = LiteLLM_UserTable(
        user_id="non_admin_user", max_budget=None, user_email=None
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        mock_db = AsyncMock()
        mock_prisma.db = mock_db
        mock_db.litellm_usertable.find_unique.return_value = AsyncMock(
            organization_memberships=[AsyncMock(organization_id="org1")]
        )

        with patch(
            "litellm.proxy.auth.route_checks.OrganizationRoleBasedAccessChecks._user_is_admin_in_org",
            return_value=False,
        ):
            result = await RouteChecks._is_org_admin_for_user_id(
                user_id, non_admin_user_obj
            )
            assert result is False
