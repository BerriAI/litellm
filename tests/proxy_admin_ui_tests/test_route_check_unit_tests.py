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
            api_key="test_key",
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
            api_key="test_key",
            request_data={},
        )
        is None
    )


def test_key_info_route_forbidden(route_checks):
    """
    Internal User is not allowed to access /key/info route for a key they're not using in Authenticated API Key
    """
    with pytest.raises(HTTPException) as exc_info:
        route_checks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/key/info",
            request=MockRequest(query_params={"key": "wrong_key"}),
            valid_token=UserAPIKeyAuth(api_key="test_key"),
            api_key="test_key",
            request_data={},
        )
    assert exc_info.value.status_code == 403


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
            api_key="test_key",
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
            api_key="test_key",
            request_data={},
        )
    assert exc_info.value.status_code == 403
