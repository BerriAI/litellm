# What is this?
## Unit tests for user_api_key_auth helper functions

import os
import sys

import litellm.proxy
import litellm.proxy.proxy_server

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from starlette.datastructures import URL
from litellm._logging import verbose_proxy_logger
import logging
import litellm
from litellm.proxy.auth.user_api_key_auth import (
    user_api_key_auth,
    UserAPIKeyAuth,
    get_api_key_from_custom_header,
)
from fastapi import WebSocket, HTTPException, status

from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles


class Request:
    def __init__(self, client_ip: Optional[str] = None, headers: Optional[dict] = None):
        self.client = MagicMock()
        self.client.host = client_ip
        self.headers: Dict[str, str] = {}


@pytest.mark.parametrize(
    "allowed_ips, client_ip, expected_result",
    [
        (None, "127.0.0.1", True),  # No IP restrictions, should be allowed
        (["127.0.0.1"], "127.0.0.1", True),  # IP in allowed list
        (["192.168.1.1"], "127.0.0.1", False),  # IP not in allowed list
        ([], "127.0.0.1", False),  # Empty allowed list, no IP should be allowed
        (["192.168.1.1", "10.0.0.1"], "10.0.0.1", True),  # IP in allowed list
        (
            ["192.168.1.1"],
            None,
            False,
        ),  # Request with no client IP should not be allowed
    ],
)
def test_check_valid_ip(
    allowed_ips: Optional[List[str]], client_ip: Optional[str], expected_result: bool
):
    from litellm.proxy.auth.auth_utils import _check_valid_ip

    request = Request(client_ip)

    assert _check_valid_ip(allowed_ips, request)[0] == expected_result  # type: ignore


# test x-forwarder for is used when user has opted in


@pytest.mark.parametrize(
    "allowed_ips, client_ip, expected_result",
    [
        (None, "127.0.0.1", True),  # No IP restrictions, should be allowed
        (["127.0.0.1"], "127.0.0.1", True),  # IP in allowed list
        (["192.168.1.1"], "127.0.0.1", False),  # IP not in allowed list
        ([], "127.0.0.1", False),  # Empty allowed list, no IP should be allowed
        (["192.168.1.1", "10.0.0.1"], "10.0.0.1", True),  # IP in allowed list
        (
            ["192.168.1.1"],
            None,
            False,
        ),  # Request with no client IP should not be allowed
    ],
)
def test_check_valid_ip_sent_with_x_forwarded_for(
    allowed_ips: Optional[List[str]], client_ip: Optional[str], expected_result: bool
):
    from litellm.proxy.auth.auth_utils import _check_valid_ip

    request = Request(client_ip, headers={"X-Forwarded-For": client_ip})

    assert _check_valid_ip(allowed_ips, request, use_x_forwarded_for=True)[0] == expected_result  # type: ignore


@pytest.mark.asyncio
async def test_check_blocked_team():
    """
    cached valid_token obj has team_blocked = true

    cached team obj has team_blocked = false

    assert team is not blocked
    """
    import asyncio
    import time

    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy._types import (
        LiteLLM_TeamTable,
        LiteLLM_TeamTableCachedObj,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    _team_id = "1234"
    user_key = "sk-12345678"

    valid_token = UserAPIKeyAuth(
        team_id=_team_id,
        team_blocked=True,
        token=hash_token(user_key),
        last_refreshed_at=time.time(),
    )
    await asyncio.sleep(1)
    team_obj = LiteLLM_TeamTableCachedObj(
        team_id=_team_id, blocked=False, last_refreshed_at=time.time()
    )
    hashed_token = hash_token(user_key)
    print(f"STORING TOKEN UNDER KEY={hashed_token}")
    user_api_key_cache.set_cache(key=hashed_token, value=valid_token)
    user_api_key_cache.set_cache(key="team_id:{}".format(_team_id), value=team_obj)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", "hello-world")

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    await user_api_key_auth(request=request, api_key="Bearer " + user_key)


@pytest.mark.asyncio
async def test_team_object_has_object_permission_id():
    """
    Ensure the team object passed into common_checks contains the team's object_permission_id.
    """
    import asyncio
    import time

    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy.proxy_server import hash_token, user_api_key_cache
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    team_id = "team-vector"
    permission_id = "perm-vector-123"
    user_key = "sk-12345678"
    hashed_key = hash_token(user_key)

    valid_token = UserAPIKeyAuth(
        team_id=team_id,
        token=hashed_key,
        last_refreshed_at=time.time(),
        team_object_permission_id=permission_id,
    )
    user_api_key_cache.set_cache(key=hashed_key, value=valid_token)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", "test-client")

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
    ) as mock_common_checks:
        mock_common_checks.return_value = True
        await user_api_key_auth(request=request, api_key="Bearer " + user_key)

        assert mock_common_checks.await_args is not None
        team_object = mock_common_checks.await_args.kwargs.get("team_object")
        assert team_object is not None
        assert team_object.object_permission_id == permission_id


@pytest.mark.parametrize(
    "user_role, expected_role",
    [
        ("app_user", "internal_user"),
        ("internal_user", "internal_user"),
        ("proxy_admin_viewer", "proxy_admin_viewer"),
    ],
)
@pytest.mark.asyncio
async def test_returned_user_api_key_auth(user_role, expected_role):
    from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import _return_user_api_key_auth_obj
    from datetime import datetime

    new_obj = await _return_user_api_key_auth_obj(
        user_obj=LiteLLM_UserTable(
            user_role=user_role, user_id="", max_budget=None, user_email=""
        ),
        api_key="hello-world",
        parent_otel_span=None,
        valid_token_dict={},
        route="/chat/completion",
        start_time=datetime.now(),
    )

    assert new_obj.user_role == expected_role


@pytest.mark.parametrize("key_ownership", ["user_key", "team_key"])
@pytest.mark.asyncio
async def test_aaauser_personal_budgets(key_ownership):
    """
    Set a personal budget on a user

    - have it only apply when key belongs to user -> raises BudgetExceededError
    - if key belongs to team, have key respect team budget -> allows call to go through
    """
    import asyncio
    import time

    from fastapi import Request
    from starlette.datastructures import URL
    import litellm

    from litellm.proxy._types import LiteLLM_UserTable, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    _user_id = "1234"
    user_key = "sk-12345678"

    if key_ownership == "user_key":
        valid_token = UserAPIKeyAuth(
            token=hash_token(user_key),
            last_refreshed_at=time.time(),
            user_id=_user_id,
            spend=20,
        )
    elif key_ownership == "team_key":
        valid_token = UserAPIKeyAuth(
            token=hash_token(user_key),
            last_refreshed_at=time.time(),
            user_id=_user_id,
            team_id="my-special-team",
            team_max_budget=100,
            spend=20,
        )

    user_obj = LiteLLM_UserTable(
        user_id=_user_id, spend=11, max_budget=10, user_email=""
    )
    user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)
    user_api_key_cache.set_cache(key="{}".format(_user_id), value=user_obj)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", "hello-world")

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    test_user_cache = getattr(litellm.proxy.proxy_server, "user_api_key_cache")

    assert test_user_cache.get_cache(key=hash_token(user_key)) == valid_token

    try:
        await user_api_key_auth(request=request, api_key="Bearer " + user_key)

        if key_ownership == "user_key":
            pytest.fail("Expected this call to fail. User is over limit.")
    except Exception:
        if key_ownership == "team_key":
            pytest.fail("Expected this call to work. Key is below team budget.")


@pytest.mark.asyncio
@pytest.mark.parametrize("prohibited_param", ["api_base", "base_url"])
async def test_user_api_key_auth_fails_with_prohibited_params(prohibited_param):
    """
    Relevant issue: https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997
    """
    import json

    from fastapi import Request

    # Setup
    user_key = "sk-1234"

    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    # Create request with prohibited parameter in body
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    async def return_body():
        body = {prohibited_param: "https://custom-api.com"}
        return bytes(json.dumps(body), "utf-8")

    request.body = return_body
    try:
        response = await user_api_key_auth(
            request=request, api_key="Bearer " + user_key
        )
    except Exception as e:
        print("error str=", str(e))
        error_message = str(e.message)
        print("error message=", error_message)
        assert "is not allowed in request body" in error_message


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "route, should_raise_error",
    [
        ("/embeddings", False),
        ("/chat/completions", True),
        ("/completions", True),
        ("/models", True),
        ("/v1/embeddings", True),
    ],
)
async def test_auth_with_allowed_routes(route, should_raise_error):
    # Setup
    user_key = "sk-1234"

    general_settings = {"allowed_routes": ["/embeddings"]}
    from fastapi import Request

    from litellm.proxy import proxy_server

    initial_general_settings = getattr(proxy_server, "general_settings")

    setattr(proxy_server, "master_key", "sk-1234")
    setattr(proxy_server, "general_settings", general_settings)

    request = Request(scope={"type": "http"})
    request._url = URL(url=route)

    if should_raise_error:
        try:
            await user_api_key_auth(request=request, api_key="Bearer " + user_key)
            pytest.fail("Expected this call to fail. User is over limit.")
        except Exception as e:
            print("error str=", str(e.message))
            error_str = str(e.message)
            assert "Route" in error_str and "not allowed" in error_str
            pass
    else:
        await user_api_key_auth(request=request, api_key="Bearer " + user_key)

    setattr(proxy_server, "general_settings", initial_general_settings)


@pytest.mark.parametrize(
    "route, user_role, expected_result",
    [
        # Proxy Admin checks
        ("/global/spend/logs", "proxy_admin", True),
        ("/key/delete", "proxy_admin", False),
        ("/key/generate", "proxy_admin", False),
        ("/key/regenerate", "proxy_admin", False),
        # Internal User checks - allowed routes
        ("/global/spend/logs", "internal_user", True),
        ("/key/delete", "internal_user", False),
        ("/key/generate", "internal_user", False),
        ("/key/82akk800000000jjsk/regenerate", "internal_user", False),
        # Internal User Viewer
        ("/key/generate", "internal_user_viewer", False),
        # Internal User checks - disallowed routes
        ("/organization/member_add", "internal_user", False),
    ],
)
def test_is_ui_route_allowed(route, user_role, expected_result):
    from litellm.proxy.auth.auth_checks import _is_ui_route
    from litellm.proxy._types import LiteLLM_UserTable

    user_obj = LiteLLM_UserTable(
        user_id="3b803c0e-666e-4e99-bd5c-6e534c07e297",
        max_budget=None,
        spend=0.0,
        model_max_budget={},
        model_spend={},
        user_email="my-test-email@1234.com",
        models=[],
        tpm_limit=None,
        rpm_limit=None,
        user_role=user_role,
        organization_memberships=[],
    )

    received_args: dict = {
        "route": route,
        "user_obj": user_obj,
    }
    try:
        assert _is_ui_route(**received_args) == expected_result
    except Exception as e:
        # If expected result is False, we expect an error
        if expected_result is False:
            pass
        else:
            raise e


@pytest.mark.parametrize(
    "route, user_role, expected_result",
    [
        ("/key/generate", "internal_user_viewer", False),
    ],
)
def test_is_api_route_allowed(route, user_role, expected_result):
    from litellm.proxy.auth.auth_checks import _is_api_route_allowed
    from litellm.proxy._types import LiteLLM_UserTable

    user_obj = LiteLLM_UserTable(
        user_id="3b803c0e-666e-4e99-bd5c-6e534c07e297",
        max_budget=None,
        spend=0.0,
        model_max_budget={},
        model_spend={},
        user_email="my-test-email@1234.com",
        models=[],
        tpm_limit=None,
        rpm_limit=None,
        user_role=user_role,
        organization_memberships=[],
    )

    received_args: dict = {
        "route": route,
        "user_obj": user_obj,
    }
    try:
        assert _is_api_route_allowed(**received_args) == expected_result
    except Exception as e:
        # If expected result is False, we expect an error
        if expected_result is False:
            pass
        else:
            raise e


@pytest.mark.asyncio
async def test_auth_not_connected_to_db():
    """
    ensure requests don't fail when `prisma_client` = None
    """
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    user_key = "sk-12345678"

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", None)
    setattr(
        litellm.proxy.proxy_server,
        "general_settings",
        {"allow_requests_on_db_unavailable": True},
    )

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    valid_token = await user_api_key_auth(request=request, api_key="Bearer " + user_key)
    print("got valid token", valid_token)
    assert valid_token.key_name == "failed-to-connect-to-db"
    assert valid_token.token == "failed-to-connect-to-db"


@pytest.mark.parametrize(
    "headers, custom_header_name, expected_api_key",
    [
        # Test with valid Bearer token
        ({"x-custom-api-key": "Bearer sk-12345678"}, "x-custom-api-key", "sk-12345678"),
        # Test with raw token (no Bearer prefix)
        ({"x-custom-api-key": "Bearer sk-12345678"}, "x-custom-api-key", "sk-12345678"),
        # Test with empty header value
        ({"x-custom-api-key": ""}, "x-custom-api-key", ""),
        # Test with missing header
        ({}, "X-Custom-API-Key", ""),
        # Test with different header casing
        ({"X-CUSTOM-API-KEY": "Bearer sk-12345678"}, "X-Custom-API-Key", "sk-12345678"),
    ],
)
def test_get_api_key_from_custom_header(headers, custom_header_name, expected_api_key):
    verbose_proxy_logger.setLevel(logging.DEBUG)

    # Mock the Request object
    request = MagicMock(spec=Request)
    request.headers = headers

    # Call the function and verify it doesn't raise an exception

    api_key = get_api_key_from_custom_header(
        request=request, custom_litellm_key_header_name=custom_header_name
    )
    assert api_key == expected_api_key


from litellm.proxy._types import LitellmUserRoles


@pytest.mark.parametrize(
    "user_role, auth_user_id, requested_user_id, expected_result",
    [
        (LitellmUserRoles.PROXY_ADMIN, "1234", None, True),
        (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, None, "1234", True),
        (LitellmUserRoles.TEAM, "1234", None, False),
        (LitellmUserRoles.TEAM, None, None, False),
        (LitellmUserRoles.TEAM, "1234", "1234", True),
    ],
)
def test_allowed_route_inside_route(
    user_role, auth_user_id, requested_user_id, expected_result
):
    from litellm.proxy.auth.auth_checks import allowed_route_check_inside_route
    from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles

    assert (
        allowed_route_check_inside_route(
            user_api_key_dict=UserAPIKeyAuth(user_role=user_role, user_id=auth_user_id),
            requested_user_id=requested_user_id,
        )
        == expected_result
    )


def test_read_request_body():
    from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
    from fastapi import Request

    payload = "()" * 1000000
    request = Request(scope={"type": "http"})

    async def return_body():
        return payload

    request.body = return_body
    result = _read_request_body(request)
    assert result is not None


@pytest.mark.asyncio
async def test_auth_with_form_data_and_model():
    """
    Test user_api_key_auth when:
    1. Request has form data instead of JSON body
    2. Virtual key has a model set
    """
    from fastapi import Request
    from starlette.datastructures import URL, FormData
    from litellm.proxy.proxy_server import (
        hash_token,
        user_api_key_cache,
        user_api_key_auth,
    )

    # Setup
    user_key = "sk-12345678"

    # Create a virtual key with a specific model
    valid_token = UserAPIKeyAuth(
        token=hash_token(user_key),
        models=["gpt-4"],
    )

    # Store the virtual key in cache
    user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", "hello-world")

    # Create request with form data
    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "headers": [(b"content-type", b"application/x-www-form-urlencoded")],
        }
    )
    request._url = URL(url="/chat/completions")

    # Mock form data
    form_data = FormData([("key1", "value1"), ("key2", "value2")])

    async def return_form_data():
        return form_data

    request.form = return_form_data

    # Test user_api_key_auth with form data request
    response = await user_api_key_auth(request=request, api_key="Bearer " + user_key)
    assert response.models == ["gpt-4"], "Model from virtual key should be preserved"


@pytest.mark.asyncio
async def test_soft_budget_alert():
    """
    Test that when a token's spend exceeds soft_budget, it triggers a budget alert but allows the request
    """
    import asyncio
    import time

    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    # Setup
    user_key = "sk-12345"
    soft_budget = 10
    current_spend = 15  # Spend exceeds soft budget

    # Create a valid token with soft budget
    valid_token = UserAPIKeyAuth(
        token=hash_token(user_key),
        soft_budget=soft_budget,
        spend=current_spend,
        last_refreshed_at=time.time(),
    )

    # Store in cache
    user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)

    # Mock proxy server settings
    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", AsyncMock())

    # Create request
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    # Track if budget_alerts was called
    alert_called = False
    original_budget_alerts = litellm.proxy.proxy_server.proxy_logging_obj.budget_alerts

    async def mock_budget_alerts(*args, **kwargs):
        nonlocal alert_called
        if kwargs.get("type") == "soft_budget":
            alert_called = True
        return await original_budget_alerts(*args, **kwargs)

    # Patch the budget_alerts method
    setattr(
        litellm.proxy.proxy_server.proxy_logging_obj,
        "budget_alerts",
        mock_budget_alerts,
    )

    try:
        # Call user_api_key_auth
        response = await user_api_key_auth(
            request=request, api_key="Bearer " + user_key
        )

        # Assert the request was allowed (no exception raised)
        assert response is not None
        # Assert the alert was triggered
        await asyncio.sleep(3)
        assert alert_called == True, "Soft budget alert should have been triggered"

    finally:
        # Restore original budget_alerts
        setattr(
            litellm.proxy.proxy_server.proxy_logging_obj,
            "budget_alerts",
            original_budget_alerts,
        )


def test_is_allowed_route():
    from litellm.proxy.auth.auth_checks import _is_allowed_route
    from litellm.proxy._types import UserAPIKeyAuth
    import datetime

    request = MagicMock()

    args = {
        "route": "/embeddings",
        "token_type": "api",
        "request": request,
        "request_data": {"input": ["hello world"], "model": "embedding-small"},
        "valid_token": UserAPIKeyAuth(
            token="9644159bc181998825c44c788b1526341ed2e825d1b6f562e23173759e14bb86",
            key_name="sk-...CJjQ",
            key_alias=None,
            spend=0.0,
            max_budget=None,
            expires=None,
            models=[],
            aliases={},
            config={},
            user_id=None,
            team_id=None,
            max_parallel_requests=None,
            metadata={},
            tpm_limit=None,
            rpm_limit=None,
            budget_duration=None,
            budget_reset_at=None,
            allowed_cache_controls=[],
            permissions={},
            model_spend={},
            model_max_budget={},
            soft_budget_cooldown=False,
            blocked=None,
            litellm_budget_table=None,
            org_id=None,
            created_at=MagicMock(),
            updated_at=MagicMock(),
            team_spend=None,
            team_alias=None,
            team_tpm_limit=None,
            team_rpm_limit=None,
            team_max_budget=None,
            team_models=[],
            team_blocked=False,
            soft_budget=None,
            team_model_aliases=None,
            team_member_spend=None,
            team_member=None,
            team_metadata=None,
            end_user_id=None,
            end_user_tpm_limit=None,
            end_user_rpm_limit=None,
            end_user_max_budget=None,
            last_refreshed_at=1736990277.432638,
            api_key=None,
            user_role=None,
            allowed_model_region=None,
            parent_otel_span=None,
            rpm_limit_per_model=None,
            tpm_limit_per_model=None,
            user_tpm_limit=None,
            user_rpm_limit=None,
        ),
        "user_obj": None,
    }

    assert _is_allowed_route(**args)


@pytest.mark.parametrize(
    "user_obj, expected_result",
    [
        (None, False),  # Case 1: user_obj is None
        (
            LiteLLM_UserTable(
                user_role=LitellmUserRoles.PROXY_ADMIN.value,
                user_id="1234",
                user_email="test@test.com",
                max_budget=None,
                spend=0.0,
            ),
            True,
        ),  # Case 2: user_role is PROXY_ADMIN
        (
            LiteLLM_UserTable(
                user_role="OTHER_ROLE",
                user_id="1234",
                user_email="test@test.com",
                max_budget=None,
                spend=0.0,
            ),
            False,
        ),  # Case 3: user_role is not PROXY_ADMIN
    ],
)
def test_is_user_proxy_admin(user_obj, expected_result):
    from litellm.proxy.auth.auth_checks import _is_user_proxy_admin

    assert _is_user_proxy_admin(user_obj) == expected_result


@pytest.mark.parametrize(
    "user_obj, expected_role",
    [
        (None, None),  # Case 1: user_obj is None (should return None)
        (
            LiteLLM_UserTable(
                user_role=LitellmUserRoles.PROXY_ADMIN.value,
                user_id="1234",
                user_email="test@test.com",
                max_budget=None,
                spend=0.0,
            ),
            LitellmUserRoles.PROXY_ADMIN,
        ),  # Case 2: user_role is PROXY_ADMIN (should return LitellmUserRoles.PROXY_ADMIN)
        (
            LiteLLM_UserTable(
                user_role="OTHER_ROLE",
                user_id="1234",
                user_email="test@test.com",
                max_budget=None,
                spend=0.0,
            ),
            LitellmUserRoles.INTERNAL_USER,
        ),  # Case 3: invalid user_role (should return LitellmUserRoles.INTERNAL_USER)
    ],
)
def test_get_user_role(user_obj, expected_role):
    from litellm.proxy.auth.user_api_key_auth import _get_user_role

    assert _get_user_role(user_obj) == expected_role


@pytest.mark.asyncio
async def test_user_api_key_auth_websocket():
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_websocket

    # Prepare a mock WebSocket object
    mock_websocket = MagicMock(spec=WebSocket)
    mock_websocket.query_params = {"model": "some_model"}
    mock_websocket.headers = {"authorization": "Bearer some_api_key"}
    # Mock the scope attribute that user_api_key_auth_websocket accesses
    mock_websocket.scope = {"headers": [(b"authorization", b"Bearer some_api_key")]}
    # Mock the url attribute
    mock_websocket.url = URL(url="/ws")

    # Mock the return value of `user_api_key_auth` when it's called within the `user_api_key_auth_websocket` function
    with patch(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth", autospec=True
    ) as mock_user_api_key_auth:

        # Make the call to the WebSocket function
        await user_api_key_auth_websocket(mock_websocket)

        # Assert that `user_api_key_auth` was called with the correct parameters
        mock_user_api_key_auth.assert_called_once()

        # Get the request object that was passed to user_api_key_auth
        request_arg = mock_user_api_key_auth.call_args.kwargs["request"]
        
        # Verify that the request has headers set
        assert hasattr(request_arg, "headers"), "Request object should have headers attribute"
        assert "authorization" in request_arg.headers, "Request headers should contain authorization"
        assert request_arg.headers["authorization"] == "Bearer some_api_key"

        assert (
            mock_user_api_key_auth.call_args.kwargs["api_key"] == "Bearer some_api_key"
        )


@pytest.mark.parametrize("enforce_rbac", [True, False])
@pytest.mark.asyncio
async def test_jwt_user_api_key_auth_builder_enforce_rbac(enforce_rbac, monkeypatch):
    from litellm.proxy.auth.handle_jwt import JWTHandler, JWTAuthManager
    from unittest.mock import patch, Mock
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.caching import DualCache

    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "my-fake-url")
    monkeypatch.setenv("JWT_AUDIENCE", "api://LiteLLM_Proxy-dev")

    local_cache = DualCache()

    keys = [
        {
            "kty": "RSA",
            "use": "sig",
            "kid": "z1rsYHHJ9-8mggt4HsZu8BKkBPw",
            "x5t": "z1rsYHHJ9-8mggt4HsZu8BKkBPw",
            "n": "pOe4GbleFDT1u5ioOQjNMmhvkDVoVD9cBKvX7AlErtWA_D6wc1w1iwkd6arYVCPObZbAB4vLSXrlpBSOuP6VYnXw_cTgniv_c82ra-mfqCpM-SbqzZ3sVqlcE_bwxvci_4PrxAW4R85ok12NXyZ2371H3yGevabi35AlVm-bQ24azo1hLK_0DzB6TxsAIOTOcKfIugOfqP-B2R4vR4u6pYftS8MWcxegr9iJ5JNtubI1X2JHpxJhkRoMVwKFna2GXmtzdxLi3yS_GffVCKfTbFMhalbJS1lSmLqhmLZZL-lrQZ6fansTl1vcGcoxnzPTwBkZMks0iVV4yfym_gKBXQ",
            "e": "AQAB",
            "x5c": [
                "MIIC/TCCAeWgAwIBAgIIQk8Qok6pfXkwDQYJKoZIhvcNAQELBQAwLTErMCkGA1UEAxMiYWNjb3VudHMuYWNjZXNzY29udHJvbC53aW5kb3dzLm5ldDAeFw0yNDExMjcwOTA0MzlaFw0yOTExMjcwOTA0MzlaMC0xKzApBgNVBAMTImFjY291bnRzLmFjY2Vzc2NvbnRyb2wud2luZG93cy5uZXQwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCk57gZuV4UNPW7mKg5CM0yaG+QNWhUP1wEq9fsCUSu1YD8PrBzXDWLCR3pqthUI85tlsAHi8tJeuWkFI64/pVidfD9xOCeK/9zzatr6Z+oKkz5JurNnexWqVwT9vDG9yL/g+vEBbhHzmiTXY1fJnbfvUffIZ69puLfkCVWb5tDbhrOjWEsr/QPMHpPGwAg5M5wp8i6A5+o/4HZHi9Hi7qlh+1LwxZzF6Cv2Inkk225sjVfYkenEmGRGgxXAoWdrYZea3N3EuLfJL8Z99UIp9NsUyFqVslLWVKYuqGYtlkv6WtBnp9qexOXW9wZyjGfM9PAGRkySzSJVXjJ/Kb+AoFdAgMBAAGjITAfMB0GA1UdDgQWBBSTO5FmUwwGS+1CNqg2uNgjxUjFijANBgkqhkiG9w0BAQsFAAOCAQEAok04z0ICMEHGqDTzx6eD7vvJP8itJTCSz8JcZcGVJofJpViGF3bNnyeSPa7vNDYP1Ps9XBvw3/n2s+yynZ8EwFxMyxCZRCSbLv0N+cAbH3rmZqGcgMJszZVwcFUtXQPTe1ZRyHtEyOB+PVFH7K7obysRVO/cC6EGqIF3pYWzez/dtMaXRAkdTNlz0ko62WoA4eMPwUFCITjW/Jxfxl0BNUbo82PXXKhaeVJb+EgFG5b/pWWPswWmBoQhmD5G1UODvEACHRl/cHsPPqe4YE+6D1/wMno/xqqyGltnk8v0d4TpNcQMn9oM19V+OGgrzWOvvXhvnhqUIVGMsRlyBGNHAw=="
            ],
            "cloud_instance_name": "microsoftonline.com",
            "issuer": "https://login.microsoftonline.com/bdfd79b3-8401-47fb-a764-6e595c455b05/v2.0",
        }
    ]

    local_cache.set_cache(
        key="litellm_jwt_auth_keys_my-fake-url",
        value=keys,
    )

    litellm_jwtauth = LiteLLM_JWTAuth(
        **{
            "admin_jwt_scope": "litellm_proxy_endpoints_access",
            "admin_allowed_routes": ["openai_routes", "info_routes"],
            "public_key_ttl": 600,
            "enforce_rbac": enforce_rbac,
        }
    )

    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=local_cache,
        litellm_jwtauth=litellm_jwtauth,
        leeway=10000000000000,
    )
    args = {
        "api_key": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6InoxcnNZSEhKOS04bWdndDRIc1p1OEJLa0JQdyIsImtpZCI6InoxcnNZSEhKOS04bWdndDRIc1p1OEJLa0JQdyJ9.eyJhdWQiOiJhcGk6Ly9MaXRlTExNX1Byb3h5LWRldiIsImlzcyI6Imh0dHBzOi8vc3RzLndpbmRvd3MubmV0L2JkZmQ3OWIzLTg0MDEtNDdmYi1hNzY0LTZlNTk1YzQ1NWIwNS8iLCJpYXQiOjE3MzcyNDE3ODEsIm5iZiI6MTczNzI0MTc4MSwiZXhwIjoxNzM3MjQ1NjgxLCJhaW8iOiJrMlJnWUpBNE5hZGg4MGJQdXlyRmxlV1o3dHZiQUE9PSIsImFwcGlkIjoiOGNjZjNkMDItMmNkNi00N2I5LTgxODUtMGVkYjI0YWJjZjY5IiwiYXBwaWRhY3IiOiIxIiwiaWRwIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvYmRmZDc5YjMtODQwMS00N2ZiLWE3NjQtNmU1OTVjNDU1YjA1LyIsIm9pZCI6IjQ0YTg3YTYzLWFiNTUtNDc4NS1iMmFmLTMzNjllZWM4ZTEzOSIsInJoIjoiMS5BYjBBczNuOXZRR0UtMGVuWkc1WlhFVmJCY0VDbkl6NHJxaE9wZ2E0UGZSZjBsbTlBQUM5QUEuIiwic3ViIjoiNDRhODdhNjMtYWI1NS00Nzg1LWIyYWYtMzM2OWVlYzhlMTM5IiwidGlkIjoiYmRmZDc5YjMtODQwMS00N2ZiLWE3NjQtNmU1OTVjNDU1YjA1IiwidXRpIjoiY3ltNVhlcmhIMHVMSlNZU1JyQmhBQSIsInZlciI6IjEuMCJ9.UooJjM9pS-wgYsExqgHdrYyQhp7NbwAsr7au9dWJaLpsufXeyHJSg-Xd5VJ4RsDVJiDes3jkC7WeoAiaCfzEHpAum-p_aqqLYXf1QIYbi1hLC0m7y_klFcqMp11WbDa9TSTvg-o8q3x2Y5su8X23ymlFih4OP17b7JA6a4_2MybU5QkCEW1tQK6VspuuXzeDHvbfGeGYcIptHFyfttHMHHXRtX1o9bX7gOR_dwFITAXD18T4ZdAN_0y6f1OtVF9TMWQhMXhKU8ahn8TSg_CXmPl9T_1gV3ZWLvVtcdVrWs82fDz3-2lEw28z4bQEr1Z5xoAz7srhx1WEBu_ioAcQiA",
        "jwt_handler": jwt_handler,
        "route": "/v1/chat/completions",
        "prisma_client": None,
        "user_api_key_cache": Mock(),
        "parent_otel_span": None,
        "proxy_logging_obj": Mock(),
        "request_data": {},
        "general_settings": {},
    }

    if enforce_rbac:
        with pytest.raises(HTTPException):
            await JWTAuthManager.auth_builder(**args)
    else:
        await JWTAuthManager.auth_builder(**args)


def test_user_api_key_auth_end_user_str():
    from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth

    user_api_key_args = {
        "api_key": "sk-1234",
        "parent_otel_span": None,
        "user_role": LitellmUserRoles.PROXY_ADMIN,
        "end_user_id": "1",
        "user_id": "default_user_id",
    }

    user_api_key_auth = UserAPIKeyAuth(**user_api_key_args)
    assert user_api_key_auth.end_user_id == "1"


def test_can_rbac_role_call_model():
    from litellm.proxy.auth.handle_jwt import JWTAuthManager
    from litellm.proxy._types import RoleBasedPermissions

    roles_based_permissions = [
        RoleBasedPermissions(
            role=LitellmUserRoles.INTERNAL_USER,
            models=["gpt-4"],
        ),
        RoleBasedPermissions(
            role=LitellmUserRoles.PROXY_ADMIN,
            models=["anthropic-claude"],
        ),
    ]

    assert JWTAuthManager.can_rbac_role_call_model(
        rbac_role=LitellmUserRoles.INTERNAL_USER,
        general_settings={"role_permissions": roles_based_permissions},
        model="gpt-4",
    )

    with pytest.raises(HTTPException):
        JWTAuthManager.can_rbac_role_call_model(
            rbac_role=LitellmUserRoles.INTERNAL_USER,
            general_settings={"role_permissions": roles_based_permissions},
            model="gpt-4o",
        )

    with pytest.raises(HTTPException):
        JWTAuthManager.can_rbac_role_call_model(
            rbac_role=LitellmUserRoles.PROXY_ADMIN,
            general_settings={"role_permissions": roles_based_permissions},
            model="gpt-4o",
        )


def test_can_rbac_role_call_model_no_role_permissions():
    from litellm.proxy.auth.handle_jwt import JWTAuthManager

    assert JWTAuthManager.can_rbac_role_call_model(
        rbac_role=LitellmUserRoles.INTERNAL_USER,
        general_settings={},
        model="gpt-4",
    )

    assert JWTAuthManager.can_rbac_role_call_model(
        rbac_role=LitellmUserRoles.PROXY_ADMIN,
        general_settings={"role_permissions": []},
        model="anthropic-claude",
    )


@pytest.mark.parametrize(
    "route, request_data, expected_model",
    [
        ("/v1/chat/completions", {"model": "gpt-4"}, "gpt-4"),
        ("/v1/completions", {"model": "gpt-4"}, "gpt-4"),
        ("/v1/chat/completions", {}, None),
        ("/v1/completions", {}, None),
        ("/openai/deployments/gpt-4", {}, "gpt-4"),
        ("/openai/deployments/gpt-4", {"model": "gpt-4o"}, "gpt-4o"),
    ],
)
def test_get_model_from_request(route, request_data, expected_model):
    from litellm.proxy.auth.user_api_key_auth import get_model_from_request

    assert get_model_from_request(request_data, route) == expected_model


@pytest.mark.asyncio
async def test_jwt_non_admin_team_route_access(monkeypatch):
    """
    Test that a non-admin JWT user cannot access team management routes
    """
    from fastapi import Request, HTTPException
    from starlette.datastructures import URL
    from unittest.mock import patch
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    import json
    from litellm.proxy._types import ProxyException

    mock_jwt_response = {
        "is_proxy_admin": False,
        "team_id": None,
        "team_object": None,
        "user_id": None,
        "user_object": None,
        "org_id": None,
        "org_object": None,
        "end_user_id": None,
        "end_user_object": None,
        "token": "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJmR09YQTNhbHFObjByRzJ6OHJQT1FLZVVMSWxCNDFnVWl4VDJ5WE1QVG1ZIn0.eyJleHAiOjE3NDI2MDAzODIsImlhdCI6MTc0MjYwMDA4MiwianRpIjoiODRhNjZmZjAtMTE5OC00YmRkLTk1NzAtNWZhMjNhZjYxMmQyIiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo4MDgwL3JlYWxtcy9saXRlbGxtLXJlYWxtIiwiYXVkIjoiYWNjb3VudCIsInN1YiI6ImZmMGZjOGNiLWUyMjktNDkyYy05NzYwLWNlYzVhMDYxNmI2MyIsInR5cCI6IkJlYXJlciIsImF6cCI6ImxpdGVsbG0tdGVzdC1jbGllbnQtaWQiLCJzaWQiOiI4MTYwNjIxOC0yNmZmLTQwMjAtOWQxNy05Zjc0YmFlNTBkODUiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbImh0dHA6Ly9sb2NhbGhvc3Q6NDAwMC8qIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsImRlZmF1bHQtcm9sZXMtbGl0ZWxsbS1yZWFsbSIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGdyb3Vwcy1zY29wZSBlbWFpbCBsaXRlbGxtLmFwaS5jb25zdW1lciIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiS3Jpc2ggRGhvbGFraWEiLCJncm91cHMiOlsiL28zX21pbmlfYWNjZXNzIl0sInByZWZlcnJlZF91c2VybmFtZSI6ImtycmlzaGRoMiIsImdpdmVuX25hbWUiOiJLcmlzaCIsImZhbWlseV9uYW1lIjoiRGhvbGFraWEiLCJlbWFpbCI6ImtycmlzaGRob2xha2lhMkBnbWFpbC5jb20ifQ.Fu2ErZhnfez-bhn_XmjkywcFdZHcFUSvzIzfdNiEowdA0soLmCyqf9731amP6m68shd9qk11e0mQhxFIAIxZPojViC1Csc9TBXLRRQ8ESMd6gPIj-DBkKVkQSZLJ1uibsh4Oo2RViGtqWVcEt32T8U_xhGdtdzNkJ8qy_e0fdNDsUnhmSaTQvmZJYarW0roIrkC-zYZrX3fftzbQfavSu9eqdfPf6wUttIrkaWThWUuORy-xaeZfSmvsGbEg027hh6QwlChiZTSF8R6bRxoqfPN3ZaGFFgbBXNRYZA_eYi2IevhIwJHi_r4o1UvtKAJyfPefm-M6hCfkN_6da4zsog",
    }

    # Create request
    request = Request(
        scope={"type": "http", "headers": [(b"authorization", b"Bearer fake.jwt.token")]}
    )
    request._url = URL(url="/team/new")

    monkeypatch.setattr(
        litellm.proxy.proxy_server, "general_settings", {"enable_jwt_auth": True}
    )

    # Mock JWTAuthManager.auth_builder
    with patch(
        "litellm.proxy.auth.handle_jwt.JWTAuthManager.auth_builder",
        return_value=mock_jwt_response,
    ):
        try:
            await user_api_key_auth(request=request, api_key="Bearer fake.jwt.token")
            pytest.fail(
                "Expected this call to fail. Non-admin user should not access team routes."
            )
        except ProxyException as e:
            print("e", e)
            assert "Only proxy admin can be used to generate" in str(e.message)


@pytest.mark.asyncio
async def test_x_litellm_api_key():
    """
    Check if auth can pick up x-litellm-api-key header, even if Bearer token is provided
    """
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy._types import (
        LiteLLM_TeamTable,
        LiteLLM_TeamTableCachedObj,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    master_key = "sk-1234"

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    setattr(litellm.proxy.proxy_server, "prisma_client", "hello-world")

    ignored_key = "aj12445"

    # Create request with headers as bytes
    request = Request(
        scope={
            "type": "http"
        }
    )
    request._url = URL(url="/chat/completions")

    valid_token = await user_api_key_auth(request=request, api_key="Bearer " + ignored_key, custom_litellm_key_header=master_key)
    assert valid_token.token == hash_token(master_key)


@pytest.mark.asyncio
async def test_user_api_key_from_query_param():
    """Ensure user_api_key_auth reads API key from `key` query parameter."""
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    user_key = "sk-query-1234"
    user_api_key_cache.set_cache(key=hash_token(user_key), value=UserAPIKeyAuth(token=hash_token(user_key)))

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", "hello-world")

    request = Request(
        scope={
            "type": "http",
            "path": "/v1beta/models/gemini:streamGenerateContent",
            "query_string": f"alt=sse&key={user_key}".encode(),
        }
    )
    request._url = URL(url=f"/v1beta/models/gemini:streamGenerateContent?alt=sse&key={user_key}")

    async def return_body():
        return b"{}"

    request.body = return_body

    valid_token = await user_api_key_auth(request=request, api_key="")
    assert valid_token.token == hash_token(user_key)

