import traceback
from litellm._uuid import uuid
import datetime as dt
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

load_dotenv()
import io
import time


# this file is to test litellm/proxy

import asyncio
import logging

from fastapi import HTTPException
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


@pytest.fixture
def route_checks():
    return RouteChecks()


def test_llm_api_route(route_checks):
    """
    Internal User is allowed to access all LLM API routes
    """
    assert (
        route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/v1/chat/completions",
            request=MockRequest(),
            valid_token=UserAPIKeyAuth(api_key="test_key"),
            request_data={},
        )
        is None
    )


def test_key_info_route_allowed(route_checks):
    """
    Internal User is allowed to access /key/info route
    """
    assert (
        route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/key/info",
            request=MockRequest(query_params={"key": "test_key"}),
            valid_token=UserAPIKeyAuth(api_key="test_key"),
            request_data={},
        )
        is None
    )


def test_user_info_route_allowed(route_checks):
    """
    Internal User is allowed to access /user/info route for their own user_id
    """
    assert (
        route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/user/info",
            request=MockRequest(query_params={"user_id": "test_user"}),
            valid_token=UserAPIKeyAuth(api_key="test_key", user_id="test_user"),
            request_data={},
        )
        is None
    )


def test_user_info_route_forbidden(route_checks):
    """
    Internal User is not allowed to access /user/info route for a different user_id
    """
    with pytest.raises(HTTPException) as exc_info:
        route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/user/info",
            request=MockRequest(query_params={"user_id": "wrong_user"}),
            valid_token=UserAPIKeyAuth(api_key="test_key", user_id="test_user"),
            request_data={},
        )
    assert exc_info.value.status_code == 403
