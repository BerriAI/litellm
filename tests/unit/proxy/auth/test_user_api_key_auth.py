# What is this?
## Unit tests for user_api_key_auth helper functions


import json
import logging
from typing import Dict, Final, List, Mapping, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, WebSocket, status
from starlette.datastructures import URL
from starlette.requests import Request as StarletteRequest
from starlette.types import Message

import litellm
import litellm.proxy
import litellm.proxy.proxy_server
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_AccessGroupTable, LiteLLM_UserTable, LitellmUserRoles, ProxyException
from litellm.proxy.auth.customer_hierarchy_auth import (
    CustomerBudgetExceededError,
    customer_member_cache_key,
    customer_org_alias_cache_key,
    resolve_customer_hierarchy,
)
from litellm.proxy.auth.user_api_key_auth import (
    UserAPIKeyAuth,
    _authorize_authenticated_request,
    get_api_key_from_custom_header,
    user_api_key_auth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerRequestIdentity, CustomerScopeAuth
from tests._master_key import MASTER_KEY
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "denial,code",
    (
        ("scope", "403"),
        ("mismatch", "403"),
        ("unknown", "403"),
        ("model", "403"),
        ("model_budget", "403"),
        ("budget", "422"),
    ),
)
async def test_customer_identity_is_retained_only_after_ownership_authorization(
    monkeypatch: pytest.MonkeyPatch, denial: str, code: str
) -> None:
    server: Final = litellm.proxy.proxy_server
    cache: Final = UserApiKeyCache()
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="customer-org-b",
        customer_org_alias="customer-b",
        customer_team_id="customer-team-b",
        customer_member_id="customer-member-b",
        scopes=(
            CustomerScopeAuth(
                key="customer_org",
                entity_id="customer-org-b",
                spend=1,
                max_budget=10 if denial == "model" else 0,
                allowed_models=("other-model",) if denial in ("model", "model_budget") else (),
            ),
        ),
    )
    await cache.async_set_cache(key=customer_member_cache_key("customer-member-b"), value=hierarchy)
    await cache.async_set_cache(
        key=customer_org_alias_cache_key("customer-a"),
        value=CustomerHierarchyAuth(customer_org_id="customer-org-a", customer_org_alias="customer-a", scopes=()),
    )
    database: Final = MagicMock()
    database.db.litellm_customerorganizationtable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr(server, "prisma_client", database)
    monkeypatch.setattr(server, "user_api_key_cache", cache)
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "general_settings", {})
    identity: Final = UserAPIKeyAuth(
        token="verified-customer-key",
        metadata={
            "customer_org_id": "customer-org-b" if denial in ("budget", "model", "model_budget") else "customer-org-a"
        },
        customer_hierarchy=hierarchy,
    )
    headers: Final = ((b"x-litellm-customer-member-id", b"customer-member-b"),) + (
        ((b"x-litellm-customer-org-alias", b"customer-a" if denial == "mismatch" else b"unknown-customer"),)
        if denial in ("mismatch", "unknown")
        else ()
    )
    request: Final = StarletteRequest({"type": "http", "method": "POST", "path": "/v1/responses", "headers": headers})
    with pytest.raises(ProxyException) as denied:
        await _authorize_authenticated_request(
            user_api_key_auth_obj=identity,
            request=request,
            request_data={"model": "qa-model", "input": "Hello"},
            route="/v1/responses",
            api_key="verified-customer-key",
        )
    assert denied.value.code == code
    assert identity.customer_hierarchy == (hierarchy if denial == "budget" else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("/v1/responses", "/v1/chat/completions", "/v1/messages"))
async def test_budget_denial_persists_the_hierarchy_used_by_the_common_resolver(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    server: Final = litellm.proxy.proxy_server
    cache: Final = UserApiKeyCache()
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="customer-org-b",
        customer_member_id="customer-member-b",
        scopes=(CustomerScopeAuth(key="customer_org", entity_id="customer-org-b", spend=1, max_budget=0.5),),
    )
    await cache.async_set_cache(key=customer_member_cache_key("customer-member-b"), value=hierarchy)
    store: Final = FakeSpendCounters().store()
    await store.increment("spend:customer_org:customer-org-b:lifetime", 2)
    database: Final = MagicMock()
    monkeypatch.setattr(server, "prisma_client", database)
    monkeypatch.setattr(server, "user_api_key_cache", cache)
    monkeypatch.setattr(server, "spend_counter_store", store)
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "general_settings", {})
    with pytest.raises(CustomerBudgetExceededError) as resolved:
        await resolve_customer_hierarchy(
            CustomerRequestIdentity(customer_member_id="customer-member-b"), database, cache, None, "qa-model", route
        )
    identity: Final = UserAPIKeyAuth(token="verified-customer-key")
    request: Final = StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": route,
            "headers": ((b"x-litellm-customer-member-id", b"customer-member-b"),),
        }
    )
    with pytest.raises(ProxyException) as denied:
        await _authorize_authenticated_request(
            identity, request, {"model": "qa-model", "input": "Hello"}, route, "verified-customer-key"
        )
    assert denied.value.code == "422"
    assert identity.customer_hierarchy == resolved.value.customer_hierarchy


class Request:
    def __init__(self, client_ip: Optional[str] = None, headers: Optional[dict] = None):
        self.client = MagicMock()
        self.client.host = client_ip
        self.headers: Dict[str, str] = {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,fallbacks,allowed", (("model-a", [], True), ("model-b", [], False), ("model-a", ["model-b"], False))
)
async def test_virtual_key_model_policy_on_first_lookup_and_cache_hit(
    monkeypatch: pytest.MonkeyPatch, model: str, fallbacks: list[str], allowed: bool
) -> None:
    server: Final = litellm.proxy.proxy_server
    cache: Final = UserApiKeyCache()
    key: Final = "sk-central-model-check"
    stored: Final = UserAPIKeyAuth(token=server.hash_token(key), models=["model-a"])
    database: Final = MagicMock(get_data=AsyncMock(side_effect=(stored, AssertionError("unexpected cache miss"))))
    monkeypatch.setattr(server, "master_key", MASTER_KEY)
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(server, "user_api_key_cache", cache)
    monkeypatch.setattr(server, "prisma_client", database)
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "user_custom_auth", None)
    for _ in range(2):
        request: Final = StarletteRequest(
            {"type": "http", "method": "POST", "path": "/chat/completions", "headers": []},
            receive=AsyncMock(
                return_value={
                    "type": "http.request",
                    "body": json.dumps({"model": model, "fallbacks": fallbacks}).encode(),
                    "more_body": False,
                }
            ),
        )
        if allowed:
            identity: Final = await user_api_key_auth(request=request, api_key=f"Bearer {key}")
            assert identity.models == ["model-a"]
        else:
            with pytest.raises(ProxyException) as denied:
                await user_api_key_auth(request=request, api_key=f"Bearer {key}")
            assert denied.value.code == "403"
            assert "model-b" in denied.value.message


@pytest.mark.asyncio
async def test_scim_deactivation_precedes_central_key_model_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    server: Final = litellm.proxy.proxy_server
    key: Final = "sk-deactivated-owner"
    cache: Final = UserApiKeyCache()
    stored: Final = UserAPIKeyAuth(token=server.hash_token(key), models=["model-a"], user_id="inactive-owner")
    cache.set_cache(
        "inactive-owner",
        LiteLLM_UserTable(user_id="inactive-owner", metadata={"scim_active": False}),
        model_type=LiteLLM_UserTable,
    )
    monkeypatch.setattr(server, "master_key", MASTER_KEY)
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(server, "user_api_key_cache", cache)
    monkeypatch.setattr(server, "prisma_client", MagicMock(get_data=AsyncMock(return_value=stored)))
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "user_custom_auth", None)
    request: Final = StarletteRequest(
        {"type": "http", "method": "POST", "path": "/chat/completions", "headers": []},
        receive=AsyncMock(return_value={"type": "http.request", "body": b'{"model":"model-b"}'}),
    )
    with pytest.raises(ProxyException) as denied:
        await user_api_key_auth(request=request, api_key=f"Bearer {key}")
    assert denied.value.code == "401"
    assert "deactivated via SCIM" in denied.value.message


@pytest.mark.asyncio
async def test_restricted_key_model_list_denial_names_the_denied_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server: Final = litellm.proxy.proxy_server
    cache: Final = UserApiKeyCache()
    group: Final = LiteLLM_AccessGroupTable(
        access_group_id="science-id", access_group_name="science", access_model_names=["model-a"]
    )
    cache.set_cache("access_group_rule_name:science", group, model_type=LiteLLM_AccessGroupTable)
    cache.set_cache("access_group_id:science-id", group, model_type=LiteLLM_AccessGroupTable)
    monkeypatch.setattr(server, "user_api_key_cache", cache)
    monkeypatch.setattr(server, "prisma_client", MagicMock())
    monkeypatch.setattr(
        server,
        "general_settings",
        {"access_group_rules": [{"attribute": "user_id", "equals": "science", "access_group": "science"}]},
    )
    monkeypatch.setattr(server, "llm_router", None)
    identity: Final = UserAPIKeyAuth(user_id="science", models=["model-a"])
    request: Final = StarletteRequest({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    with pytest.raises(ProxyException) as denied:
        await _authorize_authenticated_request(
            identity, request, {"model": ["model-a", "model-b"]}, "/chat/completions", "synthetic",
            force_virtual_key_checks=True,
        )
    assert denied.value.code == "403"
    assert "model-b" in denied.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,enabled,allowed",
    (("group-model", True, True), ("outside", True, False), ("group-model", False, False), ("explicit", False, True)),
)
async def test_user_access_group_rules_are_enforced_at_central_authorization(
    monkeypatch: pytest.MonkeyPatch, model: str, enabled: bool, allowed: bool
) -> None:
    server: Final = litellm.proxy.proxy_server
    cache: Final = UserApiKeyCache()
    group: Final = LiteLLM_AccessGroupTable(
        access_group_id="science-id", access_group_name="science", access_model_names=["group-model"]
    )
    cache.set_cache("access_group_rule_name:science", group, model_type=LiteLLM_AccessGroupTable)
    cache.set_cache("access_group_id:science-id", group, model_type=LiteLLM_AccessGroupTable)
    cache.set_cache("science", LiteLLM_UserTable(user_id="science"), model_type=LiteLLM_UserTable)
    monkeypatch.setattr(
        server,
        "general_settings",
        {"access_group_rules": [{"attribute": "user_id", "equals": "science", "access_group": "science"}]}
        if enabled
        else {},
    )
    monkeypatch.setattr(server, "user_api_key_cache", cache)
    monkeypatch.setattr(server, "prisma_client", MagicMock())
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "user_custom_auth", None)
    principal: Final = UserAPIKeyAuth(models=["explicit"], user_id="science")
    request: Final = StarletteRequest({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    if allowed:
        assert (
            await _authorize_authenticated_request(
                principal, request, {"model": model}, "/chat/completions", "synthetic", force_virtual_key_checks=True
            )
            is None
        )
        assert principal.access_group_ids == (["science-id"] if enabled else None)
    else:
        with pytest.raises(ProxyException, match="model"):
            await _authorize_authenticated_request(
                principal, request, {"model": model}, "/chat/completions", "synthetic", force_virtual_key_checks=True
            )


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
def test_check_valid_ip(allowed_ips: Optional[List[str]], client_ip: Optional[str], expected_result: bool):
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
    user_key = "sk-98765678"

    valid_token = UserAPIKeyAuth(
        team_id=_team_id,
        team_blocked=True,
        token=hash_token(user_key),
        last_refreshed_at=time.time(),
    )
    await asyncio.sleep(1)
    team_obj = LiteLLM_TeamTableCachedObj(team_id=_team_id, blocked=False, last_refreshed_at=time.time())
    hashed_token = hash_token(user_key)
    print(f"STORING TOKEN UNDER KEY={hashed_token}")
    user_api_key_cache.set_cache(key=hashed_token, value=valid_token)
    user_api_key_cache.set_cache(key="team_id:{}".format(_team_id), value=team_obj)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    setattr(litellm.proxy.proxy_server, "prisma_client", "hello-world")

    request = Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
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
    user_key = "sk-98765678"
    hashed_key = hash_token(user_key)

    valid_token = UserAPIKeyAuth(
        team_id=team_id,
        token=hashed_key,
        last_refreshed_at=time.time(),
        team_object_permission_id=permission_id,
        team_models=["gpt-4o"],
    )
    user_api_key_cache.set_cache(key=hashed_key, value=valid_token)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    setattr(litellm.proxy.proxy_server, "prisma_client", "test-client")

    request = Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    request._url = URL(url="/chat/completions")

    with patch("litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock) as mock_common_checks:
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
        user_obj=LiteLLM_UserTable(user_role=user_role, user_id="", max_budget=None, user_email=""),
        api_key="hello-world",
        parent_otel_span=None,
        valid_token_dict={},
        route="/chat/completion",
        start_time=datetime.now(),
    )

    assert new_obj.user_role == expected_role


class _NoMembershipRowPrisma:
    class db:
        class litellm_teammembership:
            @staticmethod
            async def find_unique(where: dict[str, dict[str, str]], include: dict[str, bool]) -> None:
                return None


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

    from litellm.proxy._types import (
        LiteLLM_UserTable,
        ProxyErrorTypes,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    _user_id = "1234"
    user_key = "sk-98765678"

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
            team_models=["gpt-4o"],
            spend=20,
        )

    user_obj = LiteLLM_UserTable(user_id=_user_id, spend=11, max_budget=10, user_email="")
    user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)
    user_api_key_cache.set_cache(key="{}".format(_user_id), value=user_obj)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    setattr(litellm.proxy.proxy_server, "prisma_client", _NoMembershipRowPrisma())

    request = Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    request._url = URL(url="/chat/completions")

    test_user_cache = getattr(litellm.proxy.proxy_server, "user_api_key_cache")

    assert test_user_cache.get_cache(key=hash_token(user_key), model_type=UserAPIKeyAuth) == valid_token

    if key_ownership == "user_key":
        with pytest.raises(ProxyException) as exc_info:
            await user_api_key_auth(request=request, api_key="Bearer " + user_key)
        assert exc_info.value.type == ProxyErrorTypes.budget_exceeded
    else:
        await user_api_key_auth(request=request, api_key="Bearer " + user_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("spend,expect_refusal", [(5.0, True), (0.5, False)], ids=["over_budget", "under_budget"])
async def test_virtual_key_max_budget_decides_a_queued_chat_call(monkeypatch, spend, expect_refusal):
    import time

    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    route: Final = "/queue/chat/completions"
    key: Final = "sk-queued-chat-budget"
    user_api_key_cache.set_cache(
        key=hash_token(key),
        value=UserAPIKeyAuth(
            token=hash_token(key),
            last_refreshed_at=time.time(),
            allowed_routes=[route],
            max_budget=1.0,
            spend=spend,
        ),
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    monkeypatch.setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", _NoMembershipRowPrisma())

    request: Final = Request(scope={"type": "http", "method": "POST", "path": route, "headers": []})
    request._url = URL(url=route)

    if not expect_refusal:
        served: Final = await user_api_key_auth(request=request, api_key="Bearer " + key)
        assert served.token == hash_token(key)
        return
    with pytest.raises(ProxyException) as exc_info:
        await user_api_key_auth(request=request, api_key="Bearer " + key)
    assert exc_info.value.type == ProxyErrorTypes.budget_exceeded


@pytest.mark.asyncio
@pytest.mark.parametrize("prohibited_param", ["api_base", "base_url"])
async def test_user_api_key_auth_fails_with_prohibited_params(prohibited_param):
    """
    Relevant issue: https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997
    """
    import json

    from fastapi import Request

    # Setup
    user_key = MASTER_KEY

    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)

    # Create request with prohibited parameter in body
    request = Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    request._url = URL(url="/chat/completions")

    async def return_body():
        body = {prohibited_param: "https://custom-api.com"}
        return bytes(json.dumps(body), "utf-8")

    request.body = return_body
    with pytest.raises(Exception, match="is not allowed in request body") as exc_info:
        await user_api_key_auth(request=request, api_key="Bearer " + user_key)
    error_message = str(exc_info.value.message)
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
    user_key = MASTER_KEY

    general_settings = {"allowed_routes": ["/embeddings"]}
    from fastapi import Request

    from litellm.proxy import proxy_server

    initial_general_settings = getattr(proxy_server, "general_settings")

    setattr(proxy_server, "master_key", MASTER_KEY)
    setattr(proxy_server, "general_settings", general_settings)

    request = Request(scope={"type": "http", "method": "POST", "path": route, "headers": []})
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
    "route, user_role, should_be_allowed",
    [
        # Admin can access everything
        ("/config/update", "proxy_admin", True),
        ("/global/spend/logs", "proxy_admin", True),
        ("/global/activity/cache_hits", "proxy_admin", True),
        # Internal User - allowed read-only routes
        ("/spend/logs/ui", "internal_user", True),
        ("/global/activity/cache_hits", "internal_user", True),
        ("/health/services", "internal_user", True),
        # Internal User - BLOCKED from admin routes (security fix)
        ("/config/update", "internal_user", False),
        ("/config/pass_through_endpoint", "internal_user", False),
        ("/config/field/update", "internal_user", False),
        ("/organization/member_add", "internal_user", False),
        # Internal User - BLOCKED from proxy-wide spend routes
        ("/global/spend/logs", "internal_user", False),
        # Internal User Viewer - allowed spend routes only
        ("/spend/logs/ui", "internal_user_viewer", True),
        # Internal User Viewer - BLOCKED from proxy-wide spend routes
        ("/global/spend/all_tag_names", "internal_user_viewer", False),
        # Internal User Viewer - blocked from admin routes
        ("/config/update", "internal_user_viewer", False),
        ("/key/generate", "internal_user_viewer", False),
    ],
)
def test_ui_token_route_access(route, user_role, should_be_allowed):
    """
    Verify that UI tokens (team_id=litellm-dashboard) go through the same
    RBAC checks as API tokens. Non-admin dashboard users must not be able
    to access admin-only routes like /config/update.
    """
    from litellm.proxy.auth.auth_checks import _is_api_route_allowed
    from litellm.proxy._types import LiteLLM_UserTable, UserAPIKeyAuth

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

    valid_token = UserAPIKeyAuth(
        user_id="3b803c0e-666e-4e99-bd5c-6e534c07e297",
        team_id="litellm-dashboard",
        user_role=user_role,
    )

    from starlette.datastructures import URL
    from fastapi import Request

    request = Request(scope={"type": "http", "method": "POST", "path": route, "headers": []})
    request._url = URL(url=route)

    if should_be_allowed:
        result = _is_api_route_allowed(
            route=route,
            request=request,
            request_data={},
            valid_token=valid_token,
            user_obj=user_obj,
        )
        assert result is True
    else:
        with pytest.raises(Exception, match="Only proxy admin can be used to generate"):
            _is_api_route_allowed(
                route=route,
                request=request,
                request_data={},
                valid_token=valid_token,
                user_obj=user_obj,
            )


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

    user_key = "sk-98765678"

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    setattr(litellm.proxy.proxy_server, "prisma_client", None)
    setattr(
        litellm.proxy.proxy_server,
        "general_settings",
        {"allow_requests_on_db_unavailable": True},
    )

    request = Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    request._url = URL(url="/chat/completions")

    valid_token = await user_api_key_auth(request=request, api_key="Bearer " + user_key)
    print("got valid token", valid_token)
    assert valid_token.key_name == "failed-to-connect-to-db"
    assert valid_token.token == "failed-to-connect-to-db"


def _assert_api_key_from_custom_header(headers, custom_header_name, expected_api_key):
    verbose_proxy_logger.setLevel(logging.DEBUG)
    request = MagicMock(spec=Request)
    request.headers = headers
    api_key = get_api_key_from_custom_header(request=request, custom_litellm_key_header_name=custom_header_name)
    assert api_key == expected_api_key


def test_get_api_key_from_custom_header_bearer_token():
    token = "sk-" + "1" * 8
    _assert_api_key_from_custom_header(
        headers={"x-custom-api-key": f"Bearer {token}"},
        custom_header_name="x-custom-api-key",
        expected_api_key=token,
    )


def test_get_api_key_from_custom_header_empty_value():
    _assert_api_key_from_custom_header(
        headers={"x-custom-api-key": ""},
        custom_header_name="x-custom-api-key",
        expected_api_key="",
    )


def test_get_api_key_from_custom_header_missing_header():
    _assert_api_key_from_custom_header(
        headers={},
        custom_header_name="X-Custom-API-Key",
        expected_api_key="",
    )


def test_get_api_key_from_custom_header_different_casing():
    token = "sk-" + "1" * 8
    _assert_api_key_from_custom_header(
        headers={"X-CUSTOM-API-KEY": f"Bearer {token}"},
        custom_header_name="X-Custom-API-Key",
        expected_api_key=token,
    )




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
def test_allowed_route_inside_route(user_role, auth_user_id, requested_user_id, expected_result):
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
    from litellm.proxy.common_utils.http_parsing_utils import read_request_body
    from fastapi import Request

    payload = "()" * 1000000
    request = Request(scope={"type": "http"})

    async def return_body():
        return payload

    request.body = return_body
    result = read_request_body(request)
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
    user_key = "sk-98765678"

    # Create a virtual key with a specific model
    valid_token = UserAPIKeyAuth(
        token=hash_token(user_key),
        models=["gpt-4"],
    )

    # Store the virtual key in cache
    user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
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
    user_key = "sk-98765"
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
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    setattr(litellm.proxy.proxy_server, "prisma_client", AsyncMock())

    # Create request
    request = Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
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
        response = await user_api_key_auth(request=request, api_key="Bearer " + user_key)

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
    from litellm.proxy.auth.auth_checks import _is_api_route_allowed
    from litellm.proxy._types import UserAPIKeyAuth
    import datetime

    request = MagicMock()

    args = {
        "route": "/embeddings",
        "request": request,
        "request_data": {"input": ["hello world"], "model": "embedding-small"},
        "valid_token": UserAPIKeyAuth(
            token="sk-test-mock-token-101",
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

    assert _is_api_route_allowed(**args)


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
    from litellm.proxy.auth.auth_checks import is_user_proxy_admin

    assert is_user_proxy_admin(user_obj) == expected_result


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
    from litellm.proxy.auth.auth_checks import get_user_role

    assert get_user_role(user_obj) == expected_role


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
    with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", autospec=True) as mock_user_api_key_auth:
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

        assert mock_user_api_key_auth.call_args.kwargs["api_key"] == "Bearer some_api_key"


@pytest.mark.asyncio
async def test_user_api_key_auth_websocket_carries_asgi_path():
    """
    The synthetic Request must carry the ASGI scope's ``path`` so
    ``get_request_route`` returns the real WebSocket path, not a value
    reconstructed from the (Host-poisonable) ``websocket.url``.
    """
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_websocket

    mock_websocket = MagicMock(spec=WebSocket)
    mock_websocket.query_params = {"model": "some_model"}
    mock_websocket.headers = {"authorization": "Bearer some_api_key"}
    mock_websocket.scope = {
        "type": "websocket",
        "path": "/v1/realtime",
        "root_path": "",
        "headers": [(b"authorization", b"Bearer some_api_key")],
    }
    mock_websocket.url = URL(url="/v1/realtime")

    with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", autospec=True) as mock_user_api_key_auth:
        await user_api_key_auth_websocket(mock_websocket)

        request_arg = mock_user_api_key_auth.call_args.kwargs["request"]
        assert request_arg.scope.get("path") == "/v1/realtime"
        assert request_arg.scope.get("root_path") == ""


@pytest.mark.asyncio
async def test_user_api_key_auth_live_websocket_ignores_query_model():
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_live_websocket

    mock_websocket = MagicMock(spec=WebSocket)
    mock_websocket.query_params = {"model": "gpt-realtime"}
    mock_websocket.headers = {"authorization": "Bearer some_api_key"}
    mock_websocket.scope = {"headers": [(b"authorization", b"Bearer some_api_key")]}
    mock_websocket.url = URL(url="/v1/live/sessions?model=gpt-realtime")

    with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", autospec=True) as mock_user_api_key_auth:
        await user_api_key_auth_live_websocket(mock_websocket)

    request_arg = mock_user_api_key_auth.call_args.kwargs["request"]
    assert json.loads(await request_arg.body()) == {"model": "", "disable_fallbacks": True}
    mock_user_api_key_auth.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth_name", "path"),
    [("user_api_key_auth_websocket", "/v1/responses"), ("user_api_key_auth_realtime_websocket", "/v1/realtime")],
)
async def test_websocket_without_a_model_is_refused_for_a_role_limited_to_other_models(
    auth_name: str, path: str
) -> None:
    from litellm.proxy._types import RoleBasedPermissions, UserAPIKeyAuth
    from litellm.proxy.auth import user_api_key_auth as auth_module
    from litellm.proxy.auth.handle_jwt import JWTAuthManager

    async def role_limited_auth(request, api_key: str) -> UserAPIKeyAuth:
        JWTAuthManager.can_rbac_role_call_model(
            rbac_role=LitellmUserRoles.INTERNAL_USER,
            general_settings={
                "role_permissions": [RoleBasedPermissions(role=LitellmUserRoles.INTERNAL_USER, models=["chat"])]
            },
            model=json.loads(await request.body())["model"],
        )
        return UserAPIKeyAuth(api_key=api_key)

    websocket: Final = MagicMock(spec=WebSocket)
    websocket.query_params = {}
    websocket.headers = {"authorization": "Bearer some_api_key"}
    websocket.scope = {"type": "websocket", "path": path, "headers": [(b"authorization", b"Bearer some_api_key")]}
    websocket.url = URL(url=path)
    websocket.close = AsyncMock()

    with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", side_effect=role_limited_auth):
        with pytest.raises(HTTPException) as raised:
            await getattr(auth_module, auth_name)(websocket)

    assert raised.value.status_code == 403
    websocket.close.assert_awaited_once_with(code=status.WS_1008_POLICY_VIOLATION)


@pytest.mark.asyncio
async def test_accepted_websocket_reauthorizes_the_frame_model_without_closing():
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_accepted_websocket_for_model

    mock_websocket = MagicMock(spec=WebSocket)
    mock_websocket.query_params = {}
    mock_websocket.headers = {"authorization": "Bearer some_api_key"}
    mock_websocket.scope = {
        "type": "websocket",
        "path": "/v1/live/sessions",
        "headers": [(b"authorization", b"Bearer some_api_key")],
    }
    mock_websocket.url = URL(url="/v1/live/sessions")
    denial = ProxyException("denied", "user_model_access_denied", "model", 403)

    with patch(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth", autospec=True, side_effect=denial
    ) as mock_user_api_key_auth:
        with pytest.raises(ProxyException) as raised:
            await user_api_key_auth_accepted_websocket_for_model(mock_websocket, "gpt-live-1")

    assert raised.value is denial
    request_arg = mock_user_api_key_auth.call_args.kwargs["request"]
    assert json.loads(await request_arg.body()) == {"model": "gpt-live-1", "disable_fallbacks": True}
    assert request_arg.scope.get("path") == "/v1/live/sessions"
    assert mock_user_api_key_auth.call_args.kwargs["api_key"] == "Bearer some_api_key"
    mock_websocket.close.assert_not_called()


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
        "api_key": MASTER_KEY,
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
@pytest.mark.parametrize(
    "route_template",
    [
        "/v1beta/models/{model}:generateContent",
        "/models/{model}:streamGenerateContent",
        "/openai/deployments/{model}/images/generations",
        "/openai/deployments/{model}/images/edits",
    ],
)
@pytest.mark.parametrize("body", [{"model": "cheap-model"}, {"target_model_names": ["cheap-model"]}])
async def test_key_model_allowlist_applies_to_the_model_selected_by_the_url(
    monkeypatch: pytest.MonkeyPatch, route_template: str, body: dict[str, object]
) -> None:
    from litellm.proxy._types import ProxyErrorTypes
    from litellm.proxy.auth.user_api_key_auth import enforce_key_and_fallback_model_access

    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm.proxy.proxy_server, "general_settings", {})
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_model", None)
    router: Final = litellm.Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": f"openai/{name}", "api_key": "sk-fake"}}
            for name in ("cheap-model", "premium-model")
        ]
    )
    token: Final = UserAPIKeyAuth(models=["cheap-model"])

    async def authorize(url_model: str) -> None:
        route: Final = route_template.format(model=url_model)
        request: Final = StarletteRequest(
            {"type": "http", "method": "POST", "path": route, "headers": [], "query_string": b""}
        )
        await enforce_key_and_fallback_model_access(
            valid_token=token,
            request_data=dict(body),
            route=route,
            request=request,
            llm_model_list=list(router.model_list),
            llm_router=router,
        )

    await authorize("cheap-model")
    with pytest.raises(ProxyException) as exc_info:
        await authorize("premium-model")
    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied


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
        "jwt_claims": {},
        "team_id": None,
        "team_object": None,
        "user_id": None,
        "user_email": None,
        "user_object": None,
        "org_id": None,
        "org_object": None,
        "end_user_id": None,
        "end_user_object": None,
        "token": "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJmR09YQTNhbHFObjByRzJ6OHJQT1FLZVVMSWxCNDFnVWl4VDJ5WE1QVG1ZIn0.eyJleHAiOjE3NDI2MDAzODIsImlhdCI6MTc0MjYwMDA4MiwianRpIjoiODRhNjZmZjAtMTE5OC00YmRkLTk1NzAtNWZhMjNhZjYxMmQyIiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo4MDgwL3JlYWxtcy9saXRlbGxtLXJlYWxtIiwiYXVkIjoiYWNjb3VudCIsInN1YiI6ImZmMGZjOGNiLWUyMjktNDkyYy05NzYwLWNlYzVhMDYxNmI2MyIsInR5cCI6IkJlYXJlciIsImF6cCI6ImxpdGVsbG0tdGVzdC1jbGllbnQtaWQiLCJzaWQiOiI4MTYwNjIxOC0yNmZmLTQwMjAtOWQxNy05Zjc0YmFlNTBkODUiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbImh0dHA6Ly9sb2NhbGhvc3Q6NDAwMC8qIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsImRlZmF1bHQtcm9sZXMtbGl0ZWxsbS1yZWFsbSIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGdyb3Vwcy1zY29wZSBlbWFpbCBsaXRlbGxtLmFwaS5jb25zdW1lciIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiS3Jpc2ggRGhvbGFraWEiLCJncm91cHMiOlsiL28zX21pbmlfYWNjZXNzIl0sInByZWZlcnJlZF91c2VybmFtZSI6ImtycmlzaGRoMiIsImdpdmVuX25hbWUiOiJLcmlzaCIsImZhbWlseV9uYW1lIjoiRGhvbGFraWEiLCJlbWFpbCI6ImtycmlzaGRob2xha2lhMkBnbWFpbC5jb20ifQ.Fu2ErZhnfez-bhn_XmjkywcFdZHcFUSvzIzfdNiEowdA0soLmCyqf9731amP6m68shd9qk11e0mQhxFIAIxZPojViC1Csc9TBXLRRQ8ESMd6gPIj-DBkKVkQSZLJ1uibsh4Oo2RViGtqWVcEt32T8U_xhGdtdzNkJ8qy_e0fdNDsUnhmSaTQvmZJYarW0roIrkC-zYZrX3fftzbQfavSu9eqdfPf6wUttIrkaWThWUuORy-xaeZfSmvsGbEg027hh6QwlChiZTSF8R6bRxoqfPN3ZaGFFgbBXNRYZA_eYi2IevhIwJHi_r4o1UvtKAJyfPefm-M6hCfkN_6da4zsog",
    }

    # Create request
    request = Request(
        scope={
            "type": "http",
            "headers": [(b"authorization", b"Bearer fake.jwt.token")],
        }
    )
    request._url = URL(url="/team/new")

    monkeypatch.setattr(litellm.proxy.proxy_server, "general_settings", {"enable_jwt_auth": True})

    # Initialize jwt_handler with a default LiteLLM_JWTAuth so that the
    # virtual_key_claim_field check in user_api_key_auth doesn't fail with
    # "JWTHandler has no attribute 'litellm_jwtauth'"
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.caching.dual_cache import DualCache

    litellm.proxy.proxy_server.jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=DualCache(),
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )

    # Mock enterprise license check and JWTAuthManager.auth_builder
    # License check must be mocked to avoid environment variable pollution
    # in parallel test execution
    with (
        patch(
            "litellm.proxy.proxy_server.premium_user",
            True,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.JWTAuthManager.auth_builder",
            return_value=mock_jwt_response,
        ),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await user_api_key_auth(request=request, api_key="Bearer fake.jwt.token")
        assert "Only proxy admin can be used to generate" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_x_litellm_api_key():
    """
    Check if auth can pick up x-litellm-api-key header, even if Bearer token is provided.

    On a master-key match, ``UserAPIKeyAuth.api_key`` (and the derived
    ``token``) are now the stable alias ``LITELLM_PROXY_MASTER_KEY_ALIAS``
    rather than ``hash_token(master_key)`` — the master key (or its hash)
    must not propagate into spend logs / metrics / audit trails.
    """
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
    from litellm.proxy._types import (
        LiteLLM_TeamTable,
        LiteLLM_TeamTableCachedObj,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    master_key = MASTER_KEY

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    setattr(litellm.proxy.proxy_server, "prisma_client", "hello-world")

    ignored_key = "aj12445"

    # Create request with headers as bytes
    request = Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    request._url = URL(url="/chat/completions")

    valid_token = await user_api_key_auth(
        request=request,
        api_key="Bearer " + ignored_key,
        custom_litellm_key_header=master_key,
    )
    assert valid_token.token == LITELLM_PROXY_MASTER_KEY_ALIAS
    assert valid_token.token != hash_token(master_key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "api_key", "custom_litellm_key_header"),
    [
        ({"Authorization": "Bearer sk-lit9211-master"}, "Bearer sk-lit9211-master", None),
        ({"x-litellm-api-key": "Bearer sk-lit9211-master"}, "", "Bearer sk-lit9211-master"),
        (
            {"X-Custom-Key": "Bearer sk-lit9211-master", "Authorization": "Bearer sk-wrong"},
            "Bearer sk-wrong",
            None,
        ),
    ],
)
async def test_auth_custom_key_header_precedence_and_fallback(
    headers: dict[str, str],
    api_key: str,
    custom_litellm_key_header: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "sk-lit9211-master")
    monkeypatch.setattr(proxy_server, "general_settings", {"litellm_key_header_name": "X-Custom-Key"})

    request_headers = [(name.lower().encode(), value.encode()) for name, value in headers.items()]
    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": request_headers,
        },
        receive=AsyncMock(return_value={"type": "http.request", "body": b"", "more_body": False}),
    )
    request._url = URL(url="/chat/completions")

    valid_token = await user_api_key_auth(
        request=request,
        api_key=api_key,
        custom_litellm_key_header=custom_litellm_key_header,
    )

    assert valid_token.user_role == LitellmUserRoles.PROXY_ADMIN
    assert valid_token.api_key == LITELLM_PROXY_MASTER_KEY_ALIAS


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_header_value", [b"Bearer sk-wrong", b""])
async def test_auth_rejects_wrong_or_empty_present_custom_header(
    custom_header_value: bytes, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy import proxy_server
    from litellm.proxy._types import ProxyException

    monkeypatch.setattr(proxy_server, "master_key", "sk-lit9211-master")
    monkeypatch.setattr(proxy_server, "general_settings", {"litellm_key_header_name": "X-Custom-Key"})
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock(get_data=AsyncMock(return_value=None)))

    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [
                (b"x-custom-key", custom_header_value),
                (b"authorization", b"Bearer sk-lit9211-master"),
            ],
        },
        receive=AsyncMock(return_value={"type": "http.request", "body": b"", "more_body": False}),
    )
    request._url = URL(url="/chat/completions")

    with pytest.raises(ProxyException) as exc_info:
        await user_api_key_auth(request=request, api_key="Bearer sk-lit9211-master")

    assert exc_info.value.code == "401"


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [None, ""], ids=["no_key", "no_bearer_prefix"])
async def test_auth_rejection_without_a_usable_key_has_a_json_null_param(
    api_key: str | None, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import Request

    from litellm.proxy import proxy_server
    from litellm.proxy._types import ProxyErrorTypes, ProxyException

    monkeypatch.setattr(proxy_server, "master_key", "sk-test-master")
    monkeypatch.setattr(proxy_server, "general_settings", {})
    request = Request(
        scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []},
        receive=AsyncMock(return_value={"type": "http.request", "body": b"", "more_body": False}),
    )
    request._url = URL(url="/chat/completions")

    with pytest.raises(ProxyException) as exc_info:
        await user_api_key_auth(request=request, api_key=api_key)

    body = json.loads(json.dumps({"error": exc_info.value.to_dict()}))
    assert body["error"]["code"] == "401"
    assert body["error"]["type"] == ProxyErrorTypes.auth_error.value
    assert body["error"]["param"] is None


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
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_id,user_model_max_budget,expected_calls",
    [
        ("u-1", {"gpt-4": {"budget_limit": 1.0, "time_period": "1mo"}}, 1),
        ("u-1", {}, 0),
        ("u-1", None, 0),
        (None, {"gpt-4": {"budget_limit": 1.0, "time_period": "1mo"}}, 0),
    ],
    ids=["enforced", "empty_budget", "no_budget", "no_user_id"],
)
async def test_check_user_model_budget(user_id, user_model_max_budget, expected_calls):
    """
    An internal user's model_max_budget must reach the limiter. Before this it was
    stored on LiteLLM_UserTable, accepted by /user/new and /user/update, and read
    by nothing, so a user-level per-model budget never blocked anything.
    """
    from litellm.proxy.auth.user_api_key_auth import _check_user_model_budget

    calls = []

    class _Limiter:
        async def is_user_within_model_budget(
            self, user_id: str, user_model_max_budget: Mapping[str, object], model: str, team_id: str | None = None
        ) -> bool:
            calls.append((user_id, user_model_max_budget, model))
            return True

    valid_token = UserAPIKeyAuth(
        token="hash",
        user_id=user_id,
        user_model_max_budget=user_model_max_budget,
    )
    await _check_user_model_budget(
        valid_token=valid_token,
        model_max_budget_limiter=_Limiter(),
        models=["gpt-4"],
    )
    assert len(calls) == expected_calls
    if expected_calls:
        assert calls[0] == ("u-1", user_model_max_budget, "gpt-4")


@pytest.mark.asyncio
async def test_user_model_max_budget_is_threaded_onto_the_auth_object():
    """
    The limiter can only enforce what auth carries. Regression for the user row's
    model_max_budget being dropped on the way into UserAPIKeyAuth.
    """
    from datetime import datetime

    from litellm.proxy.auth.user_api_key_auth import _return_user_api_key_auth_obj

    budget = {"gpt-4": {"budget_limit": 1.0, "time_period": "1mo"}}
    user_obj = LiteLLM_UserTable(
        user_id="u-1",
        max_budget=None,
        spend=0.0,
        user_email=None,
        models=[],
        model_max_budget=budget,
    )

    auth_obj = await _return_user_api_key_auth_obj(
        user_obj=user_obj,
        api_key=MASTER_KEY,
        parent_otel_span=None,
        valid_token_dict={"token": "hash"},
        route="/chat/completions",
        start_time=datetime.now(),
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    assert auth_obj.user_model_max_budget == budget


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "over_budget,expect_refusal",
    [(True, True), (False, False)],
    ids=["over_budget_is_refused", "under_budget_is_served"],
)
async def test_user_model_budget_is_enforced_through_user_api_key_auth(over_budget, expect_refusal):
    """
    Drive the real auth entry point, not the helper.

    The user's model_max_budget lives on the user row, and the joint
    verification-token view auth builds its token from does not carry it. A test
    that only exercises the helper passes while the whole path is inert, so this
    one goes through user_api_key_auth with a key that has no per-model budget of
    its own and asserts the USER's budget decides the outcome.
    """
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy._types import LiteLLM_UserTable, Litellm_EntityType, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.hooks.model_max_budget_limiter import model_budget_spend_cache_key
    from litellm.proxy.proxy_server import (
        hash_token,
        model_max_budget_limiter,
        user_api_key_cache,
    )

    user_id = "user-model-budget"
    model = "gpt-4o"
    key = "sk-user-model-budget"
    hashed = hash_token(key)
    user_model_max_budget = {model: {"budget_limit": 1.0, "time_period": "1mo"}}

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    setattr(litellm.proxy.proxy_server, "prisma_client", "present")

    await user_api_key_cache.async_set_cache(
        key=hashed,
        value=UserAPIKeyAuth(token=hashed, user_id=user_id, models=[], model_max_budget={}),
        model_type=UserAPIKeyAuth,
    )
    await model_max_budget_limiter.dual_cache.async_set_cache(
        key=model_budget_spend_cache_key(
            entity_type=Litellm_EntityType.USER,
            entity_id=user_id,
            budget_model=model,
            budget_duration="1mo",
        ),
        value=5.0 if over_budget else 0.25,
        ttl=600,
    )

    request = Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    request._url = URL(url="/chat/completions")

    async def return_body():
        return f'{{"model": "{model}"}}'.encode()

    request.body = return_body

    async def fake_get_user_object(**kwargs):
        return LiteLLM_UserTable(
            user_id=user_id,
            max_budget=None,
            spend=0.0,
            user_email=None,
            models=[],
            model_max_budget=user_model_max_budget,
        )

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_user_object",
        new=fake_get_user_object,
    ):
        if expect_refusal:
            with pytest.raises(Exception, match=r"(?i)budget") as exc:
                await user_api_key_auth(request=request, api_key="Bearer " + key)
            assert user_id in str(exc.value)
        else:
            result = await user_api_key_auth(request=request, api_key="Bearer " + key)
            # The budget must also reach the token, or the post-call increment
            # has nothing to charge and the counter never grows.
            assert result.user_model_max_budget == user_model_max_budget


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "over_budget,expect_refusal",
    [(True, True), (False, False)],
    ids=["over_budget_is_refused", "under_budget_is_served"],
)
async def test_jwt_user_model_budget_is_enforced_before_the_jwt_path_returns(over_budget, expect_refusal):
    """
    JWT auth returns its own token instead of falling through to the
    virtual-key budget checks, so the user's per-model budget has to be enforced
    on that path explicitly.

    The dangerous shape is not "no tracking": the post-call increment charges the
    JWT user's counter either way, so without this check the counter grows and
    nothing ever reads it, which looks enforced and is not.
    """
    from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import _check_user_model_budget
    from litellm.proxy.hooks.model_max_budget_limiter import model_budget_spend_cache_key
    from litellm.proxy.proxy_server import model_max_budget_limiter

    user_id = "jwt-user-model-budget"
    model = "gpt-4o"
    user_model_max_budget = {model: {"budget_limit": 1.0, "time_period": "1mo"}}

    await model_max_budget_limiter.dual_cache.async_set_cache(
        key=model_budget_spend_cache_key(
            entity_type=Litellm_EntityType.USER,
            entity_id=user_id,
            budget_model=model,
            budget_duration="1mo",
        ),
        value=5.0 if over_budget else 0.25,
        ttl=600,
    )

    # The token the JWT branch builds and returns.
    valid_token = UserAPIKeyAuth(
        api_key=None,
        user_id=user_id,
        user_model_max_budget=user_model_max_budget,
    )

    if expect_refusal:
        with pytest.raises(litellm.BudgetExceededError) as exc:
            await _check_user_model_budget(
                valid_token=valid_token,
                model_max_budget_limiter=model_max_budget_limiter,
                models=[model],
            )
        assert exc.value.entity_type == Litellm_EntityType.USER.value
    else:
        await _check_user_model_budget(
            valid_token=valid_token,
            model_max_budget_limiter=model_max_budget_limiter,
            models=[model],
        )


def test_jwt_path_enforces_the_user_model_budget_before_returning():
    """
    The JWT branch returns early, so the enforcement call has to sit before that
    return rather than in the virtual-key block. Assert on the call graph, since
    a helper-level test passes whether or not the JWT path ever calls it.
    """
    import ast
    import inspect
    import textwrap

    from litellm.proxy.auth import user_api_key_auth as auth_module

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(auth_module._user_api_key_auth_builder))
        + "\n"
        + textwrap.dedent(inspect.getsource(auth_module.validate_resolved_virtual_key))
    )

    def calls_before_each_return(node):
        seen_check = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                fn = child.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name == "_check_user_model_budget":
                    seen_check.append(child.lineno)
        return seen_check

    check_lines = calls_before_each_return(tree)
    assert check_lines, "_user_api_key_auth_builder never enforces the user model budget"

    jwt_returns = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", None) == "cast"
    ]
    assert jwt_returns, "expected the JWT branch's `return cast(UserAPIKeyAuth, valid_token)`"
    assert any(check < jwt_return for check in check_lines for jwt_return in jwt_returns), (
        "the user model-budget check must run before the JWT branch returns"
    )


def test_every_jwt_branch_carries_the_user_model_budget():
    """
    Each JWT branch that builds or replaces `valid_token` has to put the user's
    model budget on it, or the enforcement call a few lines later has nothing to
    read and silently admits the request.

    The auto-register branch is the one that regressed: it REPLACES the token
    built above it with a key-scoped one whose columns carry no user budget.
    """
    import ast
    import inspect
    import textwrap

    from litellm.proxy.auth import user_api_key_auth as auth_module

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(auth_module._user_api_key_auth_builder))
        + "\n"
        + textwrap.dedent(inspect.getsource(auth_module.validate_resolved_virtual_key))
    )

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "user_model_max_budget" for t in node.targets)
    ]
    targets = {
        t.value.id
        for node in assignments
        for t in node.targets
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
    }
    assert "auto_registered" in targets, (
        f"the auto-registered JWT token must carry the user's model budget; only these are populated: {sorted(targets)}"
    )
    assert "valid_token" in targets, "the virtual-key path must carry the user's model budget"


@pytest.mark.asyncio
async def test_user_budget_lookup_tolerates_an_unreadable_user():
    """
    `get_user_object(user_id_upsert=False)` raises a bare Exception when the row
    is simply ABSENT, which is the ordinary state for a custom-auth deployment
    that never writes users to the proxy DB. Refusing on that exception would
    turn "no user row" into a 4xx for every such request, and a transient DB
    blip into a full outage.

    The virtual-key path makes the same call and swallows the same exception
    ("Unable to get user from db/cache. Setting user_obj to None"), so this is
    the established contract, not a shortcut. There is also nothing to enforce:
    the budget being looked up lives on the row that could not be read.
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.auth.user_api_key_auth import _read_user_model_max_budget

    prisma_client = MagicMock()

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_user_object",
        new=AsyncMock(side_effect=Exception("No user table row")),
    ):
        budget = await _read_user_model_max_budget(
            user_id="user-with-no-row",
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    assert budget is None


@pytest.mark.asyncio
async def test_user_budget_lookup_is_also_unenforced_when_the_database_is_down():
    """
    KNOWN LIMITATION, pinned deliberately rather than discovered later.

    `get_user_object` lets a connection-level outage propagate as-is and rewrites
    every other read failure (a query-level Prisma error, a malformed row) into
    the same `ValueError("User doesn't exist in db...")` as an absent row, and
    `_read_user_model_max_budget` swallows every exception either way.

    So tolerating the absent case, which the test above requires, also
    tolerates an outage, and a user who DOES have a per-model budget goes
    unenforced while the DB is unreachable. This is pre-existing behaviour of
    the virtual-key path; it is not introduced here. Distinguishing them needs
    `_read_user_model_max_budget` to let an outage through the way the JWT
    path does.
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.auth.user_api_key_auth import _read_user_model_max_budget

    db_down = ValueError("User doesn't exist in db. 'user_id'=u-1. Got error - Connection refused")

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_user_object",
        new=AsyncMock(side_effect=db_down),
    ):
        budget = await _read_user_model_max_budget(
            user_id="u-1",
            prisma_client=MagicMock(),
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    assert budget is None


@pytest.mark.asyncio
async def test_user_budget_lookup_returns_the_budget_when_the_row_reads():
    """Positive control: the tolerance above must not be swallowing every result."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.auth.user_api_key_auth import _read_user_model_max_budget

    stored = {"claude-opus-4-8": {"budget_limit": 1.0, "time_period": "18h"}}
    user_obj = MagicMock()
    user_obj.model_max_budget = stored

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_user_object",
        new=AsyncMock(return_value=user_obj),
    ):
        budget = await _read_user_model_max_budget(
            user_id="user-1",
            prisma_client=MagicMock(),
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    assert budget == stored


def test_zero_cost_models_skip_the_user_budget_check_on_every_path():
    """
    `skip_budget_checks` is computed per request for zero-cost models, and the
    JWT branch logs "Skipping all budget checks" when it is set. Any enforcement
    call that ignores it makes the same request behave differently depending on
    whether the caller used a JWT or a virtual key, and makes that log a lie.

    Structural rather than behavioural on purpose: the defect is a call site
    sitting outside a guard, and driving both auth paths to a zero-cost model
    would prove it for the two requests exercised rather than for every site.
    """
    import ast
    import inspect
    import textwrap

    from litellm.proxy.auth import user_api_key_auth as auth_module

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(auth_module._user_api_key_auth_builder))
        + "\n"
        + textwrap.dedent(inspect.getsource(auth_module.validate_resolved_virtual_key))
    )

    def guarded_by_skip(node: ast.AST, target: ast.AST) -> bool:
        for parent in ast.walk(node):
            if not isinstance(parent, ast.If):
                continue
            test = parent.test
            is_skip_guard = (
                isinstance(test, ast.UnaryOp)
                and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Name)
                and test.operand.id == "skip_budget_checks"
            )
            if is_skip_guard and any(sub is target for sub in ast.walk(parent)):
                return True
        return False

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_check_user_model_budget"
    ]
    assert len(calls) == 2, f"expected the JWT and virtual-key call sites, found {len(calls)}"

    unguarded = [c for c in calls if not guarded_by_skip(tree, c)]
    assert not unguarded, (
        f"{len(unguarded)} _check_user_model_budget call(s) run even when "
        "skip_budget_checks is set, so a zero-cost model is enforced on one auth path and not the other"
    )


def test_custom_auth_also_skips_budget_checks_for_zero_cost_models():
    """
    The custom-auth helper runs its own key, user and end-user per-model budget
    checks. If it does not honour the zero-cost skip that the JWT and
    virtual-key paths honour, the same free request is refused under one auth
    method and served under the others.

    Asserted structurally, on the same reasoning as the sibling test: the defect
    is a check sitting outside a guard, and it must hold for checks added later
    rather than only for whichever request a behavioural test happened to drive.
    """
    import ast
    import inspect
    import textwrap

    from litellm.proxy.auth import user_api_key_auth as auth_module

    src = textwrap.dedent(inspect.getsource(auth_module._run_post_custom_auth_checks))
    tree = ast.parse(src)

    assert "skip_budget_checks" in src, "the custom-auth path never computes the zero-cost skip flag"

    budget_calls = (
        "_check_key_model_budget_with_fallback",
        "_check_user_model_budget",
        "is_end_user_within_model_budget",
    )

    def guarding_ifs(target: ast.AST) -> list[ast.If]:
        return [
            node for node in ast.walk(tree) if isinstance(node, ast.If) and any(sub is target for sub in ast.walk(node))
        ]

    def mentions_skip(node: ast.If) -> bool:
        return any(isinstance(sub, ast.Name) and sub.id == "skip_budget_checks" for sub in ast.walk(node.test))

    for call_name in budget_calls:
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == call_name)
                or (isinstance(node.func, ast.Attribute) and node.func.attr == call_name)
            )
        ]
        assert calls, f"{call_name} is no longer called here; update this invariant"
        for call in calls:
            assert any(mentions_skip(node) for node in guarding_ifs(call)), (
                f"{call_name} runs even for a zero-cost model, so custom auth refuses "
                "requests the JWT and virtual-key paths serve"
            )


def test_custom_auth_attaches_the_user_budget_even_when_it_does_not_enforce():
    """
    The post-call spend hook reads `user_model_max_budget` off the token, so the
    attach has to happen whether or not THIS request was enforceable. Gating it
    on the same condition as the check leaves the user's counter uncharged for
    every request with no resolvable model or a zero-cost one, which is exactly
    the untracked-spend defect this PR fixes.

    Structural, because the failure is an assignment sitting inside a guard.
    """
    import ast
    import inspect
    import textwrap

    from litellm.proxy.auth import user_api_key_auth as auth_module

    tree = ast.parse(textwrap.dedent(inspect.getsource(auth_module._run_post_custom_auth_checks)))

    attaches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "user_model_max_budget" for t in node.targets)
    ]
    assert attaches, "custom auth no longer attaches the user budget at all"

    for attach in attaches:
        enclosing_ifs = [
            node for node in ast.walk(tree) if isinstance(node, ast.If) and any(sub is attach for sub in ast.walk(node))
        ]
        assert not enclosing_ifs, (
            "the user budget is attached inside a conditional, so the spend hook "
            "cannot charge the user counter whenever that condition is false"
        )


def test_mapped_key_jwt_falls_through_to_the_shared_user_budget_attach():
    """
    A JWT that maps to an existing virtual key resolves through the resolver
    store, which builds the token from the KEY row alone and therefore carries
    no user-level per-model budget. That branch sets `do_standard_jwt_auth =
    False` precisely so it falls through to the shared virtual-key checks, where
    the user row is loaded and its budget copied onto the token.

    Reviewed as a bypass three times, so the two halves it depends on are pinned
    here: the branch must not return before the shared block, and the shared
    block must copy the user row's budget onto the token. Structural on purpose,
    because the claim is about control flow reaching a statement, and it has to
    hold for branches added later rather than for one mocked request.
    """
    import ast
    import inspect
    import textwrap

    from litellm.proxy.auth import user_api_key_auth as auth_module

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(auth_module._user_api_key_auth_builder))
        + "\n"
        + textwrap.dedent(inspect.getsource(auth_module.validate_resolved_virtual_key))
    )

    # Half one: the shared block copies the user row's budget onto the token.
    copies_user_row = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "user_model_max_budget" for t in node.targets)
        and any(isinstance(v, ast.Attribute) and v.attr == "model_max_budget" for v in ast.walk(node.value))
    ]
    assert copies_user_row, (
        "nothing copies the user row's model_max_budget onto the token, so a mapped-key "
        "JWT reaches enforcement carrying the key's columns only"
    )

    # Half two: the mapped-key branch does not return before reaching it.
    disables_standard_auth = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "do_standard_jwt_auth" for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
    ]
    assert len(disables_standard_auth) == 1, "expected exactly one mapped-key branch"
    marker = disables_standard_auth[0]

    enclosing = [
        node for node in ast.walk(tree) if isinstance(node, ast.If) and any(sub is marker for sub in node.body)
    ]
    assert enclosing, "could not locate the mapped-key branch body"

    returns_after = [
        node for node in ast.walk(enclosing[0]) if isinstance(node, ast.Return) and node.lineno > marker.lineno
    ]
    assert not returns_after, (
        "the mapped-key branch returns before the shared virtual-key checks, so the "
        "user's per-model budget is never attached and never enforced"
    )


def _rejected_websocket(sent: list[Message], bearer: str) -> WebSocket:
    async def receive() -> Message:
        return {"type": "websocket.connect"}

    async def send(message: Message) -> None:
        sent.append(message)

    return WebSocket(
        {
            "type": "websocket",
            "path": "/v1/responses",
            "query_string": b"model=gpt-5.4",
            "headers": [(b"authorization", f"Bearer {bearer}".encode())],
        },
        receive,
        send,
    )


def _serve_virtual_key(monkeypatch: pytest.MonkeyPatch, user_key: str, token: UserAPIKeyAuth) -> None:
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    user_api_key_cache.set_cache(key=hash_token(user_key), value=token)
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    monkeypatch.setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", "connected")
    monkeypatch.setattr(litellm, "log_client_error_tracebacks", False)
    monkeypatch.setattr(verbose_proxy_logger, "propagate", True)


@pytest.mark.asyncio
async def test_user_api_key_auth_websocket_logs_a_model_access_denial_as_one_warning_without_a_traceback(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_websocket
    from litellm.proxy.proxy_server import hash_token

    user_key: Final = "sk-websocket-key-limited-to-mini"
    _serve_virtual_key(
        monkeypatch,
        user_key,
        UserAPIKeyAuth(token=hash_token(user_key), models=["gpt-5.4-mini"]),
    )
    sent: Final[list[Message]] = []
    with (
        caplog.at_level(logging.WARNING, logger=verbose_proxy_logger.name),
        pytest.raises(HTTPException) as rejection,
    ):
        await user_api_key_auth_websocket(_rejected_websocket(sent, user_key))

    assert rejection.value.status_code == 403
    assert sent == [{"type": "websocket.close", "code": status.WS_1008_POLICY_VIOLATION, "reason": ""}]
    proxy_records: Final = [record for record in caplog.records if record.name == verbose_proxy_logger.name]
    assert [record.getMessage() for record in proxy_records if record.exc_info is not None] == []
    assert [record.getMessage() for record in proxy_records if record.levelno == logging.WARNING] == [
        "key not allowed to access model. This key can only access models=['gpt-5.4-mini']. Tried to access gpt-5.4"
    ]


@pytest.mark.asyncio
async def test_user_api_key_auth_websocket_rejection_adds_no_traceback_for_other_auth_errors(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_websocket
    from litellm.proxy.proxy_server import hash_token

    user_key: Final = "sk-websocket-key-expired-in-2020"
    expired_in_2020: Final = "2020-01-01T00:00:00+00:00"
    _serve_virtual_key(
        monkeypatch,
        user_key,
        UserAPIKeyAuth(token=hash_token(user_key), expires=expired_in_2020),
    )
    sent: Final[list[Message]] = []
    with (
        caplog.at_level(logging.DEBUG, logger=verbose_proxy_logger.name),
        pytest.raises(HTTPException) as rejection,
    ):
        await user_api_key_auth_websocket(_rejected_websocket(sent, user_key))

    assert rejection.value.status_code == 403
    assert "expired key" in str(rejection.value.detail).lower()
    assert sent == [{"type": "websocket.close", "code": status.WS_1008_POLICY_VIOLATION, "reason": ""}]
    proxy_records: Final = [record for record in caplog.records if record.name.startswith(verbose_proxy_logger.name)]
    assert [record.getMessage() for record in proxy_records if record.exc_info is not None] == []
    assert [record.levelno for record in proxy_records if record.levelno >= logging.WARNING] == [logging.WARNING]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_data, expect_user_model_budget_check",
    [
        ({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, False),
        ({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, False),
        ({"jsonrpc": "2.0", "id": 1, "method": "tools/call"}, True),
    ],
)
async def test_jwt_path_skips_user_model_budget_only_for_zero_spend_mcp_methods(
    monkeypatch: pytest.MonkeyPatch,
    request_data: dict[str, object],
    expect_user_model_budget_check: bool,
) -> None:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_builder

    jwt_response: Final = {
        "is_proxy_admin": False,
        "jwt_claims": {},
        "team_id": None,
        "team_object": None,
        "user_id": "jwt-user",
        "user_email": None,
        "user_object": LiteLLM_UserTable(
            user_id="jwt-user",
            model_max_budget={"gpt-4o": {"budget_limit": 1.0, "time_period": "1d"}},
        ),
        "org_id": None,
        "org_object": None,
        "end_user_id": None,
        "end_user_object": None,
        "token": "fake.jwt.token",
    }
    mock_request: Final = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.method = "POST"
    mock_request.headers = {"authorization": "Bearer fake.jwt.token"}
    mock_request.query_params = {}

    monkeypatch.setattr(litellm.proxy.proxy_server, "general_settings", {"enable_jwt_auth": True})
    litellm.proxy.proxy_server.jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=DualCache(),
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )
    user_model_budget_check: Final = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.auth.handle_jwt.JWTAuthManager.auth_builder", return_value=jwt_response),
        patch("litellm.proxy.auth.user_api_key_auth._check_user_model_budget", user_model_budget_check),
    ):
        result: Final = await user_api_key_auth_builder(
            request=mock_request,
            api_key="Bearer fake.jwt.token",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data=request_data,
        )

    assert result.user_id == "jwt-user"
    assert user_model_budget_check.await_count == (1 if expect_user_model_budget_check else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, path, expect_rejected",
    [
        ("GET", "/v1/videos/{video_id}", False),
        ("GET", "/videos/{video_id}/content", False),
        ("POST", "/v1/videos", True),
        ("POST", "/v1/videos/{video_id}/remix", True),
    ],
)
async def test_jwt_path_spares_only_video_retrieval_from_an_exhausted_user_model_budget(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str, expect_rejected: bool
) -> None:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import JWTAuthBuilderResult, LiteLLM_JWTAuth, Litellm_EntityType, ProxyErrorTypes
    from litellm.proxy.auth import user_api_key_auth as user_api_key_auth_module
    from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_builder
    from litellm.proxy.hooks.model_max_budget_limiter import (
        PROXY_VirtualKeyModelMaxBudgetLimiter,
        model_budget_spend_cache_key,
    )
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id: Final = encode_video_id_with_provider(video_id="video_raw", provider="openai", model_id="sora-2")
    jwt_result: Final = JWTAuthBuilderResult(
        is_proxy_admin=False,
        team_object=None,
        user_object=LiteLLM_UserTable(
            user_id="jwt-user",
            model_max_budget={"sora-2": {"budget_limit": 0.3, "time_period": "1d"}},
        ),
        end_user_object=None,
        org_object=None,
        token="fake.jwt.token",
        team_id=None,
        user_id="jwt-user",
        user_email=None,
        end_user_id=None,
        org_id=None,
        team_membership=None,
        jwt_claims={},
        agent_id=None,
    )

    class _VerifiedJWTAuthManager(JWTAuthManager):
        @staticmethod
        async def auth_builder(
            api_key: object,
            jwt_handler: object,
            request_data: object,
            general_settings: object,
            route: object,
            prisma_client: object,
            user_api_key_cache: object,
            parent_otel_span: object,
            proxy_logging_obj: object,
            request_headers: object = None,
            request_method: object = None,
        ) -> JWTAuthBuilderResult:
            return jwt_result

    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    await limiter.dual_cache.async_set_cache(
        key=model_budget_spend_cache_key(
            entity_type=Litellm_EntityType.USER, entity_id="jwt-user", budget_model="sora-2", budget_duration="1d"
        ),
        value=0.4,
    )
    jwt_handler: Final = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=DualCache(),
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )
    mock_request: Final = MagicMock()
    mock_request.url.path = path.format(video_id=video_id)
    mock_request.method = method
    mock_request.headers = {"authorization": "Bearer fake.jwt.token"}
    mock_request.query_params = {}
    request_data: Final[dict[str, object]] = {"model": "sora-2"} if path == "/v1/videos" else {"video_id": video_id}

    monkeypatch.setattr(litellm.proxy.proxy_server, "general_settings", {"enable_jwt_auth": True})
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm.proxy.proxy_server, "jwt_handler", jwt_handler)
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm.proxy.proxy_server, "model_max_budget_limiter", limiter)
    monkeypatch.setattr(user_api_key_auth_module, "JWTAuthManager", _VerifiedJWTAuthManager)

    async def authenticate() -> str | None:
        result: Final = await user_api_key_auth_builder(
            request=mock_request,
            api_key="Bearer fake.jwt.token",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data=request_data,
        )
        return result.user_id

    if not expect_rejected:
        assert await authenticate() == "jwt-user"
        return
    with pytest.raises(ProxyException) as exc_info:
        await authenticate()
    assert exc_info.value.type == ProxyErrorTypes.budget_exceeded


def _qa_websocket(messages):
    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        messages.append(message)

    return WebSocket(
        {
            "type": "websocket",
            "path": "/v1/responses",
            "scheme": "ws",
            "server": ("localhost", 4000),
            "query_string": b"model=qa-model",
            "headers": [(b"authorization", b"Bearer qa-key")],
        },
        receive=receive,
        send=send,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("close_on_auth_error", [True, False])
async def test_websocket_inference_is_denied_when_a_required_baggage_mapping_is_configured(close_on_auth_error):
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_websocket_for_model

    messages = []
    with (
        patch.dict(
            litellm.proxy.proxy_server.general_settings,
            {
                "request_tag_mappings": [
                    {"key": "region", "tag_prefix": "region:"},
                    {"key": "service", "tag_prefix": "app:", "required": True},
                ]
            },
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(return_value=UserAPIKeyAuth(api_key="qa-key")),
        ),
    ):
        with pytest.raises(HTTPException, match="WebSocket inference is disabled") as error:
            await user_api_key_auth_websocket_for_model(
                _qa_websocket(messages), "qa-model", close_on_auth_error=close_on_auth_error
            )
    assert error.value.status_code == 403
    assert messages == ([{"type": "websocket.close", "code": 1008, "reason": ""}] if close_on_auth_error else [])


@pytest.mark.asyncio
async def test_websocket_invalid_key_is_rejected_before_the_required_baggage_mapping_denial():
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_websocket_for_model

    with (
        patch.dict(
            litellm.proxy.proxy_server.general_settings,
            {"request_tag_mappings": [{"key": "service", "tag_prefix": "app:", "required": True}]},
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(side_effect=HTTPException(status_code=401, detail="Invalid proxy server token passed")),
        ),
    ):
        with pytest.raises(HTTPException) as error:
            await user_api_key_auth_websocket_for_model(_qa_websocket([]), "qa-model", close_on_auth_error=False)
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_websocket_inference_is_allowed_when_no_baggage_mapping_is_required():
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_websocket_for_model

    messages = []
    authorized = UserAPIKeyAuth(api_key="qa-key")
    with (
        patch.dict(
            litellm.proxy.proxy_server.general_settings,
            {"request_tag_mappings": [{"key": "service", "tag_prefix": "app:", "require_budgeted_tag": True}]},
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth", new=AsyncMock(return_value=authorized)
        ) as http_auth,
    ):
        result = await user_api_key_auth_websocket_for_model(_qa_websocket(messages), "qa-model")
    assert result is authorized
    assert http_auth.await_args.kwargs["api_key"] == "Bearer qa-key"
    assert messages == []


async def _auth_then_serve_chat(monkeypatch, general_settings, body, headers):
    import copy

    from fastapi import Request as FastAPIRequest
    from fastapi import Response

    from litellm.proxy.common_utils.http_parsing_utils import read_request_body
    from litellm.proxy.proxy_server import chat_completion

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4o", "tags": ["app:a"], "mock_response": "hi"},
                "model_info": {"id": "app-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4o-mini", "tags": ["default"], "mock_response": "hi"},
                "model_info": {"id": "default-deployment"},
            },
        ],
        enable_tag_filtering=True,
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", None)
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    monkeypatch.setattr(litellm.proxy.proxy_server, "general_settings", general_settings)

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    request = FastAPIRequest(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "scheme": "http",
            "server": ("localhost", 4000),
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), *headers],
        },
        receive,
    )
    auth = await user_api_key_auth(request=request, api_key=f"Bearer {MASTER_KEY}")
    cached_body = copy.deepcopy(await read_request_body(request))
    response = Response()
    await chat_completion(request=request, fastapi_response=response, user_api_key_dict=auth)
    return cached_body, response.headers["x-litellm-model-id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "general_settings",
    [{}, {"request_tag_mappings": [{"key": "region", "tag_prefix": "region:"}]}],
    ids=["no_mappings", "mapped_baggage"],
)
async def test_auth_header_tag_merge_keeps_json_string_metadata_tags_routable(monkeypatch, general_settings):
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": json.dumps({"tags": ["app:a"]}),
    }
    cached_body, served_by = await _auth_then_serve_chat(
        monkeypatch, general_settings, body, [(b"x-litellm-tags", b"other"), (b"baggage", b"region=sg")]
    )
    assert served_by == "app-deployment"
    assert cached_body["metadata"] == {"tags": ["app:a", "other"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("client_metadata", ["run-123", 42, ["x"]])
@pytest.mark.parametrize(
    "baggage",
    [[], [(b"baggage", b"region=sg")], [(b"baggage", b"service=a")]],
    ids=["no_baggage", "unmapped_baggage", "mapped_baggage"],
)
async def test_non_object_metadata_is_served_like_unset_mappings_whatever_the_baggage_maps(
    monkeypatch, client_metadata, baggage
):
    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "metadata": client_metadata}
    headers = [(b"x-litellm-tags", b"app:a"), *baggage]
    unset = await _auth_then_serve_chat(monkeypatch, {}, body, headers)
    mapped = await _auth_then_serve_chat(
        monkeypatch, {"request_tag_mappings": [{"key": "service", "tag_prefix": "app:"}]}, body, headers
    )
    assert mapped == unset
    assert mapped[1] == "app-deployment"


@pytest.mark.asyncio
async def test_customer_hierarchy_survives_virtual_key_auth_without_legacy_end_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.auth import customer_hierarchy_auth as hierarchy_module
    from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth

    server: Final = litellm.proxy.proxy_server
    key: Final = "sk-hierarchy-member"
    stored: Final = UserAPIKeyAuth(token=server.hash_token(key))
    hierarchy: Final = CustomerHierarchyAuth(
        customer_member_id="m1",
        customer_team_id="t1",
        customer_org_id="o1",
        scopes=(CustomerScopeAuth(key="customer_member", entity_id="m1", rpm_limit=1),),
    )
    loader: Final = AsyncMock(return_value=hierarchy)
    monkeypatch.setattr(hierarchy_module, "get_customer_hierarchy", loader)
    monkeypatch.setattr(server, "master_key", MASTER_KEY)
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(server, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(server, "prisma_client", MagicMock(get_data=AsyncMock(return_value=stored)))
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "user_custom_auth", None)
    request: Final = StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [(b"x-litellm-customer-member-id", b"m1")],
        },
        receive=AsyncMock(
            return_value={
                "type": "http.request",
                "body": json.dumps({"model": "gpt-4o-mini", "messages": []}).encode(),
                "more_body": False,
            }
        ),
    )
    identity: Final = await user_api_key_auth(request=request, api_key=f"Bearer {key}")
    assert identity.customer_hierarchy == hierarchy
    assert identity.end_user_id is None
    assert loader.await_args is not None and loader.await_args.args[0] == "m1"


@pytest.mark.asyncio
@pytest.mark.parametrize("key_org,allowed", (("o1", True), ("o2", False), (None, True)))
async def test_a_key_pinned_to_a_customer_org_only_serves_that_orgs_members(
    monkeypatch: pytest.MonkeyPatch, key_org: str | None, allowed: bool
) -> None:
    from litellm.proxy.auth import customer_hierarchy_auth as hierarchy_module
    from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth

    server: Final = litellm.proxy.proxy_server
    key: Final = "sk-hierarchy-scoped"
    metadata: Final = {"customer_org_id": key_org} if key_org is not None else {}
    stored: Final = UserAPIKeyAuth(token=server.hash_token(key), metadata=metadata)
    hierarchy: Final = CustomerHierarchyAuth(
        customer_member_id="m1",
        customer_team_id="t1",
        customer_org_id="o1",
        scopes=(CustomerScopeAuth(key="customer_member", entity_id="m1"),),
    )
    monkeypatch.setattr(hierarchy_module, "get_customer_hierarchy", AsyncMock(return_value=hierarchy))
    monkeypatch.setattr(server, "master_key", MASTER_KEY)
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(server, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(server, "prisma_client", MagicMock(get_data=AsyncMock(return_value=stored)))
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "user_custom_auth", None)
    request: Final = StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [(b"x-litellm-customer-member-id", b"m1")],
        },
        receive=AsyncMock(
            return_value={
                "type": "http.request",
                "body": json.dumps({"model": "gpt-4o-mini", "messages": []}).encode(),
                "more_body": False,
            }
        ),
    )
    if allowed:
        identity: Final = await user_api_key_auth(request=request, api_key=f"Bearer {key}")
        assert identity.customer_hierarchy == hierarchy
    else:
        with pytest.raises(ProxyException) as denied:
            await user_api_key_auth(request=request, api_key=f"Bearer {key}")
        assert denied.value.code == "403"


@pytest.mark.asyncio
async def test_customer_model_access_is_checked_against_the_model_the_route_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.auth import customer_hierarchy_auth as hierarchy_module
    from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth

    server: Final = litellm.proxy.proxy_server
    key: Final = "sk-hierarchy-route-model"
    stored: Final = UserAPIKeyAuth(token=server.hash_token(key))
    hierarchy: Final = CustomerHierarchyAuth(
        customer_member_id="m1",
        customer_team_id="t1",
        customer_org_id="o1",
        scopes=(CustomerScopeAuth(key="customer_member", entity_id="m1", allowed_models=("allowed",)),),
    )
    monkeypatch.setattr(hierarchy_module, "get_customer_hierarchy", AsyncMock(return_value=hierarchy))
    monkeypatch.setattr(server, "master_key", MASTER_KEY)
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(server, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(server, "prisma_client", MagicMock(get_data=AsyncMock(return_value=stored)))
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "user_custom_auth", None)
    request: Final = StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai/deployments/blocked/chat/completions",
            "headers": [(b"x-litellm-customer-member-id", b"m1")],
        },
        receive=AsyncMock(
            return_value={
                "type": "http.request",
                "body": json.dumps({"messages": []}).encode(),
                "more_body": False,
            }
        ),
    )
    with pytest.raises(ProxyException) as denied:
        await user_api_key_auth(request=request, api_key=f"Bearer {key}")
    assert denied.value.code == "403"
    assert "blocked" in denied.value.message
