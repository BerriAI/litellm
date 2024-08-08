# What is this?
## Unit tests for user_api_key_auth helper functions

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

import litellm


class Request:
    def __init__(self, client_ip: Optional[str] = None):
        self.client = MagicMock()
        self.client.host = client_ip


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
    from litellm.proxy.auth.user_api_key_auth import _check_valid_ip

    request = Request(client_ip)

    assert _check_valid_ip(allowed_ips, request) == expected_result  # type: ignore


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
    user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)
    user_api_key_cache.set_cache(key="team_id:{}".format(_team_id), value=team_obj)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", "hello-world")

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    await user_api_key_auth(request=request, api_key="Bearer " + user_key)


@pytest.mark.parametrize(
    "user_role, expected_role",
    [
        ("app_user", "internal_user"),
        ("internal_user", "internal_user"),
        ("proxy_admin_viewer", "proxy_admin_viewer"),
    ],
)
def test_returned_user_api_key_auth(user_role, expected_role):
    from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import _return_user_api_key_auth_obj

    new_obj = _return_user_api_key_auth_obj(
        user_obj=LiteLLM_UserTable(
            user_role=user_role, user_id="", max_budget=None, user_email=""
        ),
        api_key="hello-world",
        parent_otel_span=None,
        valid_token_dict={},
        route="/chat/completion",
    )

    assert new_obj.user_role == expected_role


@pytest.mark.parametrize("key_ownership", ["user_key", "team_key"])
@pytest.mark.asyncio
async def test_user_personal_budgets(key_ownership):
    """
    Set a personal budget on a user

    - have it only apply when key belongs to user -> raises BudgetExceededError
    - if key belongs to team, have key respect team budget -> allows call to go through
    """
    import asyncio
    import time

    from fastapi import Request
    from starlette.datastructures import URL

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
    await asyncio.sleep(1)
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

    try:
        await user_api_key_auth(request=request, api_key="Bearer " + user_key)

        if key_ownership == "user_key":
            pytest.fail("Expected this call to fail. User is over limit.")
    except Exception:
        if key_ownership == "team_key":
            pytest.fail("Expected this call to work. Key is below team budget.")
