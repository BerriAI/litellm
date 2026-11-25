# What is this?
## Tests if 'get_end_user_object' works as expected

import asyncio
import hashlib
import json
import uuid

import traceback
from dotenv import load_dotenv

load_dotenv()

from collections import UserDict
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from functools import reduce
from types import SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, status

import litellm
from tests.unit.redis.stores.auth_cache_double import FakeAuthCache, shared_cache
from tests.unit.redis.stores.cache_invalidation_double import FakeChannel
from litellm.redis.stores.cache_invalidation import CacheInvalidation
import httpx
from prisma import Prisma
from litellm._service_logger import ServiceTypes
from tests.unit.proxy.rust_auth_stores import RustAuthStores
from litellm.caching.caching import DualCache
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_TeamMembership,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_EndUserTable,
    Litellm_EntityType,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.utils import PrismaClient
from litellm.proxy.auth.auth_checks import (
    _team_soft_budget_check,
    can_team_access_model,
    check_opaque_inference_controls,
    common_checks,
    delete_cache_key_objects,
    get_end_user_object,
    get_org_object,
    get_user_object,
    is_model_cost_zero,
    virtual_key_soft_budget_check,
)
from litellm.proxy.utils import CallInfo, InternalUsageCache, ProxyLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import LITELLM_PASS_THROUGH_ENDPOINT_MARKER


@pytest.mark.parametrize("container", ["mapping", "list"])
@pytest.mark.parametrize("at_limit", [False, True])
def test_opaque_controls_reject_unchecked_nesting(container: str, at_limit: bool) -> None:
    from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

    def wrap(value: object, _: int) -> object:
        return {"metadata": value} if container == "mapping" else [value]

    depth: Final = DEFAULT_MAX_RECURSE_DEPTH if at_limit else DEFAULT_MAX_RECURSE_DEPTH - 1
    controls: Final = reduce(wrap, range(depth), "unrestricted")
    if not at_limit:
        assert check_opaque_inference_controls(controls) is None
        return
    with pytest.raises(HTTPException) as denied:
        check_opaque_inference_controls(controls)
    assert denied.value.status_code == 403
    assert "nesting" in denied.value.detail


@pytest.mark.parametrize(
    "control", [{"models": ["restricted-model"]}, {"guardrails": ["required-guardrail"]}, {"max_budget": 0}]
)
def test_opaque_controls_reject_restrictions_nested_in_lists_and_mappings(control: dict[str, object]) -> None:
    with pytest.raises(HTTPException) as denied:
        check_opaque_inference_controls({"metadata": [{"budget": control}]})
    assert denied.value.status_code == 403
    assert "cannot enforce" in denied.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,expected",
    [
        ("project", "model restrictions"),
        ("user", "model restrictions"),
        ("member", "model restrictions"),
        ("missing-user", "owner"),
        ("missing-project", "inherited policies"),
        ("personal-key", "personal-key"),
        ("member-model-budget", "model_max_budget"),
    ],
)
async def test_cursor_opaque_admission_rejects_inherited_model_scopes(
    monkeypatch: pytest.MonkeyPatch, scope: str, expected: str
) -> None:
    from litellm.proxy import proxy_server

    cache: Final = UserApiKeyCache()
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", scope == "personal-key")
    membership: Final = LiteLLM_TeamMembership(
        user_id="opaque-user",
        team_id="opaque-team",
        litellm_budget_table=LiteLLM_BudgetTable(budget_id="opaque-budget", allowed_models=["scoped-model"]),
    )
    await cache.async_set_cache(
        key=AuthObject.team_membership_reservation(user_id="opaque-user", team_id="opaque-team"),
        value=membership,
        model_type=LiteLLM_TeamMembership,
    )
    route: Final = "/cursor-cli/agent.v1.AgentService/RunSSE"
    request: Final = Request({"type": "http", "method": "POST", "path": route, "headers": [], "query_string": b""})
    project: Final = (
        LiteLLM_ProjectTableCachedObj(project_id="opaque-project", models=["scoped-model"])
        if scope == "project"
        else None
    )
    user: Final = LiteLLM_UserTable(
        user_id="opaque-user",
        models=["scoped-model"] if scope == "user" else [],
        teams=["restricted-owner-team"] if scope == "personal-key" else [],
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    identity: Final = UserAPIKeyAuth(
        user_id="opaque-user" if scope in {"member", "missing-user", "personal-key"} else None,
        project_id="missing-project" if scope == "missing-project" else None,
        team_member_model_max_budget={"restricted-model": {"max_budget": 10}}
        if scope == "member-model-budget"
        else None,
    )
    with pytest.raises(HTTPException) as denied:
        await common_checks(
            request_body={},
            team_object=LiteLLM_TeamTable(team_id="opaque-team") if scope == "member" else None,
            user_object=None if scope == "missing-user" else user,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route=route,
            llm_router=None,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=cache),
            valid_token=identity,
            request=request,
            skip_budget_checks=True,
            project_object=project,
        )
    assert denied.value.status_code == 403
    assert expected in denied.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        "owner-budget",
        "customer-model",
        "customer-tokens",
        "customer-budget",
        "customer_org_id",
        "customer_team_id",
        "customer_member_id",
        "resolved-customer-model",
    ],
)
async def test_opaque_admission_checks_tuple_backed_inherited_controls(scope: str) -> None:
    from litellm.models.budget import BudgetLimit
    from litellm.models.team import BudgetLimitEntry
    from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth

    customer_scope: Final = CustomerScopeAuth(
        key="customer_org",
        entity_id="restricted-customer",
        allowed_models=("restricted-model",) if scope == "customer-model" else None,
        tpm_limit=1000 if scope == "customer-tokens" else None,
        budget_limits=(BudgetLimit(budget_duration="1mo", max_budget=10),) if scope == "customer-budget" else (),
    )
    identity: Final = UserAPIKeyAuth(
        customer_hierarchy=CustomerHierarchyAuth(customer_org_id="restricted-customer", scopes=(customer_scope,)),
        metadata={scope: "bound-customer"} if scope.endswith("_id") else {},
    ).with_customer_model_scopes((("restricted-model",),) if scope == "resolved-customer-model" else ())
    user: Final = LiteLLM_UserTable(
        user_id="restricted-owner",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        budget_limits=(BudgetLimitEntry(budget_duration="1d", max_budget=10),) if scope == "owner-budget" else None,
    )
    route: Final = "/cursor-cli/agent.v1.AgentService/RunSSE"
    with pytest.raises(HTTPException) as denied:
        await common_checks(
            request_body={},
            team_object=None,
            user_object=user,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route=route,
            llm_router=None,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=UserApiKeyCache()),
            valid_token=identity,
            request=Request({"type": "http", "method": "POST", "path": route, "headers": [], "query_string": b""}),
            skip_budget_checks=True,
        )
    assert denied.value.status_code == 403
    assert "cannot enforce" in denied.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup_error", [False, True])
async def test_opaque_member_budget_lookup_fails_closed_when_budget_checks_are_skipped(
    monkeypatch: pytest.MonkeyPatch, lookup_error: bool
) -> None:
    from litellm.proxy import proxy_server

    database: Final = MagicMock()
    database.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=None, side_effect=ConnectionError("budget unavailable") if lookup_error else None
    )
    cache: Final = UserApiKeyCache()
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    await cache.async_set_cache(
        key=AuthObject.team_membership_reservation(user_id="member", team_id="team"),
        value=LiteLLM_TeamMembership(user_id="member", team_id="team"),
        model_type=LiteLLM_TeamMembership,
    )
    route: Final = "/cursor-cli/agent.v1.AgentService/RunSSE"
    with pytest.raises(HTTPException) as denied:
        await common_checks(
            request_body={},
            team_object=LiteLLM_TeamTable(team_id="team", metadata={"team_member_budget_id": "member-budget"}),
            user_object=LiteLLM_UserTable(user_id="member", user_role=LitellmUserRoles.PROXY_ADMIN),
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route=route,
            llm_router=None,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=cache),
            valid_token=UserAPIKeyAuth(user_id="member", team_id="team"),
            request=Request({"type": "http", "method": "POST", "path": route, "headers": [], "query_string": b""}),
            skip_budget_checks=True,
        )
    assert denied.value.status_code == 403
    assert "member" in denied.value.detail


async def _run_team_trace_id_check(
    *,
    route: str,
    method: str = "POST",
    request_body: dict[str, object] | None = None,
    headers: Mapping[str, str] | None = None,
    team_metadata: Mapping[str, object] | None = None,
    team_object: LiteLLM_TeamTable | None = None,
    registered_pass_through_endpoint: bool = False,
) -> bool:
    request_payload: Final[dict[str, object]] = {} if request_body is None else request_body
    raw_headers: Final = tuple(
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (() if headers is None else headers.items())
    )
    request: Final = Request(
        {
            "type": "http",
            "method": method,
            "path": route,
            "headers": raw_headers,
            "query_string": b"",
        }
    )
    if registered_pass_through_endpoint:
        endpoint: Final = SimpleNamespace()
        setattr(endpoint, LITELLM_PASS_THROUGH_ENDPOINT_MARKER, True)
        request.scope["endpoint"] = endpoint
    team: Final = (
        team_object
        if team_object is not None
        else LiteLLM_TeamTable(
            team_id="trace-id-team",
            metadata=None if team_metadata is None else dict(team_metadata),
        )
    )
    valid_token: Final = UserAPIKeyAuth(token="test-token", team_id=team.team_id)
    user: Final = LiteLLM_UserTable(user_id="trace-id-admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    return await common_checks(
        request_body=request_payload,
        team_object=team,
        user_object=user,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings={},
        route=route,
        llm_router=None,
        proxy_logging_obj=MagicMock(spec=ProxyLogging),
        valid_token=valid_token,
        request=request,
        skip_budget_checks=True,
    )


@pytest.mark.parametrize(
    "route",
    ["/chat/completions", "/queue/chat/completions", "/v1/messages", "/mcp/", "/a2a/my-agent"],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_rejects_llm_mcp_and_agent_requests(route: str) -> None:
    request_body: Final[dict[str, object]] = {"metadata": {"source": "unit-test"}}
    original_body: Final = deepcopy(request_body)
    exc_info: Final[pytest.ExceptionInfo[ProxyException]]
    with pytest.raises(ProxyException) as exc_info:
        await _run_team_trace_id_check(
            route=route,
            request_body=request_body,
            team_metadata={"require_trace_id": True},
        )
    assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
    assert exc_info.value.param == "trace_id"
    assert exc_info.value.message == (
        "Team 'trace-id-team' requires a trace ID on every LLM, MCP and agent request. "
        "Send an x-litellm-trace-id header. "
        "LLM requests can also send a W3C traceparent header or set metadata.trace_id in the body."
    )
    assert request_body == original_body


@pytest.mark.parametrize(
    ("headers", "request_body"),
    [
        ({"x-litellm-trace-id": "trace-123"}, {}),
        ({"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}, {}),
        ({}, {"metadata": {"trace_id": "trace-123"}}),
        ({}, {"litellm_metadata": {"trace_id": "trace-123"}}),
        ({}, {"litellm_trace_id": "trace-123"}),
        ({}, {"litellm_trace_id": "", "metadata": {"trace_id": "trace-123"}}),
        ({}, {"litellm_trace_id": None, "metadata": {"trace_id": "trace-123"}}),
    ],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_accepts_supported_trace_carriers(
    headers: Mapping[str, str],
    request_body: dict[str, object],
) -> None:
    original_body: Final = deepcopy(request_body)
    await _run_team_trace_id_check(
        route="/chat/completions",
        headers=headers,
        request_body=request_body,
        team_metadata={"require_trace_id": True},
    )
    assert request_body == original_body


@pytest.mark.parametrize("route", ["/v1/traces", "/v1/logs", "/v1/traces/query"])
@pytest.mark.asyncio
async def test_team_require_trace_id_accepts_trace_telemetry_routes(route: str) -> None:
    assert (
        await _run_team_trace_id_check(
            route=route,
            method="POST",
            request_body={},
            team_metadata={"require_trace_id": True},
        )
        is True
    )


@pytest.mark.asyncio
async def test_team_require_trace_id_rejects_overridden_body_metadata_on_chat() -> None:
    request_body: Final[dict[str, object]] = {
        "metadata": {"trace_id": ""},
        "litellm_metadata": {"trace_id": "trace-123"},
    }
    exc_info: Final[pytest.ExceptionInfo[ProxyException]]
    with pytest.raises(ProxyException) as exc_info:
        await _run_team_trace_id_check(
            route="/chat/completions",
            request_body=request_body,
            team_metadata={"require_trace_id": True},
        )
    assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
    assert exc_info.value.param == "trace_id"


@pytest.mark.parametrize(
    "request_body",
    [
        {"litellm_metadata": {"trace_id": "trace-123"}},
        {"metadata": {"trace_id": "trace-123"}},
    ],
    ids=["selected-litellm-metadata", "promoted-requester-metadata"],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_accepts_effective_metadata_on_responses_route(
    request_body: dict[str, object],
) -> None:
    assert (
        await _run_team_trace_id_check(
            route="/v1/responses",
            request_body=request_body,
            team_metadata={"require_trace_id": True},
        )
        is True
    )


@pytest.mark.parametrize(
    ("headers", "request_body"),
    [
        (
            {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
            {"litellm_trace_id": ""},
        ),
        (
            {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
            {"litellm_trace_id": None},
        ),
    ],
    ids=[
        "traceparent-with-empty-top-level",
        "traceparent-with-null-top-level",
    ],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_rejects_fallback_carriers_when_litellm_trace_id_is_explicit(
    headers: Mapping[str, str],
    request_body: dict[str, object],
) -> None:
    exc_info: Final[pytest.ExceptionInfo[ProxyException]]
    with pytest.raises(ProxyException) as exc_info:
        await _run_team_trace_id_check(
            route="/chat/completions",
            headers=headers,
            request_body=request_body,
            team_metadata={"require_trace_id": True},
        )
    assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
    assert exc_info.value.param == "trace_id"


@pytest.mark.asyncio
async def test_team_require_trace_id_accepts_header_on_mcp_with_empty_body() -> None:
    assert (
        await _run_team_trace_id_check(
            route="/mcp/",
            request_body={},
            headers={"x-litellm-trace-id": "trace-mcp-123"},
            team_metadata={"require_trace_id": True},
        )
        is True
    )


@pytest.mark.parametrize(
    ("headers", "request_body"),
    [
        (
            {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
            {},
        ),
        ({}, {"metadata": {"trace_id": "trace-body-123"}}),
    ],
    ids=["traceparent", "body-metadata"],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_rejects_nonheader_carriers_on_pass_through_routes(
    headers: Mapping[str, str],
    request_body: dict[str, object],
) -> None:
    exc_info: Final[pytest.ExceptionInfo[ProxyException]]
    with pytest.raises(ProxyException) as exc_info:
        await _run_team_trace_id_check(
            route="/anthropic/v1/messages",
            headers=headers,
            request_body=request_body,
            team_metadata={"require_trace_id": True},
        )
    assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
    assert exc_info.value.param == "trace_id"


@pytest.mark.asyncio
async def test_team_require_trace_id_accepts_header_on_pass_through_route() -> None:
    assert (
        await _run_team_trace_id_check(
            route="/anthropic/v1/messages",
            request_body={},
            headers={"x-litellm-trace-id": "trace-provider-123"},
            team_metadata={"require_trace_id": True},
        )
        is True
    )


@pytest.mark.parametrize(
    ("headers", "request_body"),
    [
        (
            {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
            {},
        ),
        ({}, {"metadata": {"trace_id": "trace-body-123"}}),
    ],
    ids=["traceparent", "body-metadata"],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_rejects_nonheader_carriers_on_registered_pass_through_routes(
    headers: Mapping[str, str],
    request_body: dict[str, object],
) -> None:
    exc_info: Final[pytest.ExceptionInfo[ProxyException]]
    with pytest.raises(ProxyException) as exc_info:
        await _run_team_trace_id_check(
            route="/configured-passthrough/infer",
            headers=headers,
            request_body=request_body,
            team_metadata={"require_trace_id": True},
            registered_pass_through_endpoint=True,
        )
    assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
    assert exc_info.value.param == "trace_id"


@pytest.mark.parametrize(
    ("route", "request_body"),
    [
        ("/mcp/", {}),
        (
            "/mcp-rest/tools/call",
            {"server_id": "math", "name": "add", "arguments": {"left": 2, "right": 3}},
        ),
        (
            "/a2a/my-agent",
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {
                    "message": {"messageId": "msg-1", "role": "user", "parts": [{"type": "text", "text": "hi"}]}
                },
            },
        ),
    ],
    ids=["streamable-mcp", "mcp-rest", "a2a"],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_rejects_traceparent_on_mcp_and_agent_routes(
    route: str,
    request_body: dict[str, object],
) -> None:
    exc_info: Final[pytest.ExceptionInfo[ProxyException]]
    with pytest.raises(ProxyException) as exc_info:
        await _run_team_trace_id_check(
            route=route,
            headers={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
            request_body=request_body,
            team_metadata={"require_trace_id": True},
        )
    assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
    assert exc_info.value.param == "trace_id"


@pytest.mark.parametrize(
    ("route", "request_body"),
    [
        (
            "/mcp-rest/tools/call",
            {
                "server_id": "math",
                "name": "add",
                "arguments": {"left": 2, "right": 3},
                "metadata": {"trace_id": "trace-body-123"},
            },
        ),
        (
            "/a2a/my-agent",
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {
                    "message": {"messageId": "msg-1", "role": "user", "parts": [{"type": "text", "text": "hi"}]}
                },
                "metadata": {"trace_id": "trace-body-123"},
            },
        ),
    ],
    ids=["mcp-rest", "a2a"],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_rejects_body_metadata_on_mcp_and_agent_routes(
    route: str,
    request_body: dict[str, object],
) -> None:
    exc_info: Final[pytest.ExceptionInfo[ProxyException]]
    with pytest.raises(ProxyException) as exc_info:
        await _run_team_trace_id_check(
            route=route,
            request_body=request_body,
            team_metadata={"require_trace_id": True},
        )
    assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
    assert exc_info.value.param == "trace_id"


@pytest.mark.parametrize(
    ("headers", "request_body"),
    [
        ({"traceparent": "invalid"}, {}),
        ({}, {"metadata": {"trace_id": ""}}),
    ],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_rejects_invalid_trace_carriers(
    headers: Mapping[str, str],
    request_body: dict[str, object],
) -> None:
    original_body: Final = deepcopy(request_body)
    exc_info: Final[pytest.ExceptionInfo[ProxyException]]
    with pytest.raises(ProxyException) as exc_info:
        await _run_team_trace_id_check(
            route="/chat/completions",
            headers=headers,
            request_body=request_body,
            team_metadata={"require_trace_id": True},
        )
    assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
    assert exc_info.value.param == "trace_id"
    assert request_body == original_body


@pytest.mark.parametrize(
    ("route", "method", "team_metadata"),
    [
        ("/v1/models", "GET", {"require_trace_id": True}),
        ("/chat/completions", "POST", None),
        ("/chat/completions", "POST", {"require_trace_id": False}),
        ("/key/generate", "POST", {"require_trace_id": True}),
    ],
)
@pytest.mark.asyncio
async def test_team_require_trace_id_exempts_methods_routes_and_disabled_teams(
    route: str,
    method: str,
    team_metadata: Mapping[str, object] | None,
) -> None:
    assert await _run_team_trace_id_check(route=route, method=method, team_metadata=team_metadata) is True


@pytest.mark.asyncio
async def test_team_require_trace_id_skips_non_dict_team_metadata() -> None:
    team: Final[LiteLLM_TeamTable] = LiteLLM_TeamTable.model_construct(
        team_id="trace-id-team",
        metadata="not a mapping",
    )
    assert await _run_team_trace_id_check(route="/v1/chat/completions", team_object=team) is True


@pytest.mark.parametrize(
    ("route", "registered_pass_through_endpoint"),
    [("/anthropic/v1/messages", False), ("/configured-passthrough/infer", True)],
    ids=["provider-pass-through", "registered-pass-through"],
)
@pytest.mark.parametrize("with_header", [False, True])
@pytest.mark.asyncio
async def test_team_required_customer_headers_apply_to_pass_through_routes(
    route: str, registered_pass_through_endpoint: bool, with_header: bool
) -> None:
    run: Final = _run_team_trace_id_check(
        route=route,
        request_body={},
        headers={"X-Tenant-ID": "tenant"} if with_header else {},
        team_metadata={"required_customer_headers": ["X-Tenant-ID"]},
        registered_pass_through_endpoint=registered_pass_through_endpoint,
    )
    if with_header:
        assert await run is True
        return
    exc_info: Final[pytest.ExceptionInfo[ProxyException]]
    with pytest.raises(ProxyException) as exc_info:
        await run
    assert (exc_info.value.code, exc_info.value.param) == (str(status.HTTP_403_FORBIDDEN), "X-Tenant-ID")


class _ReadCountingHeaders(UserDict[str, str]):
    reads = 0

    def __getitem__(self, key: str) -> str:
        self.reads += 1
        return super().__getitem__(key)


def test_duplicate_required_customer_headers_do_not_multiply_header_reads_or_admit_a_blank_header() -> None:
    from litellm.proxy.auth.auth_checks import _missing_customer_header

    headers: Final = _ReadCountingHeaders(
        {**{f"x-extra-{index}": "value" for index in range(100)}, "hOsT": "gateway", "X-Tenant-ID": " \t "}
    )
    assert _missing_customer_header(headers, (*("Host" for _ in range(1000)), "X-Tenant-ID")) == "X-Tenant-ID"
    assert headers.reads <= len(headers)


@pytest.mark.parametrize("route", ("/lens/datasets", "/lens/tracing/keys"))
def test_lens_management_does_not_require_an_inference_user_parameter(route: str) -> None:
    from starlette.requests import Request

    from litellm.proxy.auth.auth_checks import _enforce_user_param_check

    request: Final = Request({"type": "http", "method": "POST", "path": route})
    _enforce_user_param_check({"enforce_user_param": True}, request, {}, route)
    with pytest.raises(Exception, match="'user' param not passed"):
        _enforce_user_param_check({"enforce_user_param": True}, request, {}, "/v1/chat/completions")


@pytest.mark.parametrize("customer_spend, customer_budget", [(0, 10), (10, 0)])
@pytest.mark.asyncio
async def test_get_end_user_object(customer_spend, customer_budget):
    """
    Scenario 1: normal - get_end_user_object returns the cached user
    Scenario 2: user over budget - NOTE: budget enforcement now happens in
                common_checks() via _check_end_user_budget(), not in get_end_user_object()

    This test verifies that get_end_user_object correctly retrieves the end user
    from cache. Budget enforcement is tested separately in test_check_end_user_budget().
    """
    end_user_id = "my-test-customer"
    _budget = LiteLLM_BudgetTable(max_budget=customer_budget)
    end_user_obj = LiteLLM_EndUserTable(
        user_id=end_user_id,
        spend=customer_spend,
        litellm_budget_table=_budget,
        blocked=False,
    )
    # UserApiKeyCache applies model_type on get/set; plain DualCache returns raw dicts
    # and breaks get_end_user_object's typed async_get_cache path.
    _cache = UserApiKeyCache()
    _key = f"end_user_id:{end_user_id}"
    await _cache.async_set_cache(
        key=AuthObject.end_user(end_user_id),
        value=end_user_obj,
        model_type=LiteLLM_EndUserTable,
    )
    # get_end_user_object only fetches data - it no longer enforces budget
    # Budget enforcement happens in common_checks() via _check_end_user_budget()
    result = await get_end_user_object(
        end_user_id=end_user_id,
        prisma_client="RANDOM VALUE",  # type: ignore
        user_api_key_cache=_cache,
        route="/v1/chat/completions",
    )
    assert result is not None
    assert result.user_id == end_user_id


@pytest.mark.parametrize("customer_spend, customer_budget", [(0, 10), (10, 0)])
@pytest.mark.asyncio
async def test_check_end_user_budget(customer_spend, customer_budget):
    """
    Test _check_end_user_budget enforcement:
    - Scenario 1: customer_spend=0, customer_budget=10 - should pass (under budget)
    - Scenario 2: customer_spend=10, customer_budget=0 - should fail (over budget)

    Note: Budget enforcement for end users happens in common_checks() via
    _check_end_user_budget(), not in get_end_user_object().
    """
    from litellm.proxy.auth.auth_checks import check_end_user_budget

    _budget = LiteLLM_BudgetTable(max_budget=customer_budget)
    end_user_obj = LiteLLM_EndUserTable(
        user_id="my-test-customer",
        spend=customer_spend,
        litellm_budget_table=_budget,
        blocked=False,
    )

    should_exceed = customer_spend > customer_budget

    if not should_exceed:
        await check_end_user_budget(
            end_user_obj=end_user_obj,
            route="/v1/chat/completions",
        )
        return

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await check_end_user_budget(
            end_user_obj=end_user_obj,
            route="/v1/chat/completions",
        )
    # Verify the error has correct info
    assert exc_info.value.current_cost == customer_spend
    assert exc_info.value.max_budget == customer_budget


@pytest.mark.parametrize(
    "model, expect_to_work",
    [
        ("openai/gpt-4o-mini", True),
        ("openai/gpt-4o", False),
    ],
)
@pytest.mark.asyncio
async def test_can_key_call_model(model, expect_to_work):
    """
    If wildcard model + specific model is used, choose the specific model settings
    """

    from litellm.proxy.auth.auth_checks import can_key_call_model

    llm_model_list = [
        {
            "model_name": "openai/*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
                "access_groups": ["public-openai-models"],
            },
        },
        {
            "model_name": "openai/gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "0cfcd87f2cb12a783a466888d05c6c89df66db23e01cecd75ec0b83aed73c9ad",
                "db_model": False,
                "access_groups": ["private-openai-models"],
            },
        },
    ]
    router = litellm.Router(model_list=llm_model_list)
    args = {
        "model": model,
        "llm_model_list": llm_model_list,
        "valid_token": UserAPIKeyAuth(
            models=["public-openai-models"],
        ),
        "llm_router": router,
    }
    if expect_to_work:
        await can_key_call_model(**args)
    else:
        with pytest.raises(Exception, match="is not available for this API key") as e:
            await can_key_call_model(**args)

        print(e)


@pytest.mark.parametrize(
    "model, expect_to_work",
    [("openai/gpt-4o", False), ("openai/gpt-4o-mini", True)],
)
@pytest.mark.asyncio
async def test_can_team_call_model(model, expect_to_work):

    from litellm.proxy.auth.auth_checks import model_in_access_group

    llm_model_list = [
        {
            "model_name": "openai/*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
                "access_groups": ["public-openai-models"],
            },
        },
        {
            "model_name": "openai/gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "0cfcd87f2cb12a783a466888d05c6c89df66db23e01cecd75ec0b83aed73c9ad",
                "db_model": False,
                "access_groups": ["private-openai-models"],
            },
        },
    ]
    router = litellm.Router(model_list=llm_model_list)

    args = {
        "model": model,
        "team_models": ["public-openai-models"],
        "llm_router": router,
    }
    if expect_to_work:
        assert model_in_access_group(**args)
    else:
        assert not model_in_access_group(**args)


@pytest.mark.parametrize(
    "key_models, model, expect_to_work",
    [
        (["openai/*"], "openai/gpt-4o", True),
        (["openai/*"], "openai/gpt-4o-mini", True),
        (["openai/*"], "openaiz/gpt-4o-mini", False),
        (["bedrock/*"], "bedrock/anthropic.claude-3-5-sonnet-20240620", True),
        (["bedrock/*"], "bedrockz/anthropic.claude-3-5-sonnet-20240620", False),
        (["bedrock/us.*"], "bedrock/us.amazon.nova-micro-v1:0", True),
        (["openai/*"], "ft:gpt-4-0613", True),
        (["openai/*"], "bedrockz/ft:gpt-4-0613", False),
    ],
)
@pytest.mark.asyncio
async def test_can_key_call_model_wildcard_access(key_models, model, expect_to_work):
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.auth_checks import can_key_call_model

    llm_model_list = [
        {
            "model_name": "openai/*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
            },
        },
        {
            "model_name": "bedrock/*",
            "litellm_params": {
                "model": "bedrock/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
            },
        },
        {
            "model_name": "openai/gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "0cfcd87f2cb12a783a466888d05c6c89df66db23e01cecd75ec0b83aed73c9ad",
                "db_model": False,
            },
        },
    ]
    router = litellm.Router(model_list=llm_model_list)

    user_api_key_object = UserAPIKeyAuth(
        models=key_models,
    )

    if expect_to_work:
        await can_key_call_model(
            model=model,
            llm_model_list=llm_model_list,
            valid_token=user_api_key_object,
            llm_router=router,
        )
    else:
        with pytest.raises(ProxyException):
            await can_key_call_model(
                model=model,
                llm_model_list=llm_model_list,
                valid_token=user_api_key_object,
                llm_router=router,
            )


@pytest.fixture
def shipped_routing_rules() -> Iterator[None]:
    from litellm.litellm_core_utils.fallback_generalizations import (
        get_fallback_generalization_rules,
        set_fallback_generalizations,
    )
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    previous_rules: Final = list(get_fallback_generalization_rules())
    set_fallback_generalizations(GetModelCostMap.load_local_model_cost_map()["fallback_generalizations"]["rules"])
    yield
    set_fallback_generalizations(previous_rules)


@pytest.mark.parametrize("object_type", ["team", "project"])
@pytest.mark.parametrize(
    "allowed_models, model, expect_to_work",
    [
        (["typesafe/*"], "jev-latest", True),
        (["typesafe/*"], "jev-1.14.0", True),
        (["jev-latest"], "jev-latest", True),
        (["openai/*"], "jev-latest", False),
        (["typesafe/*"], "foo/jev-latest", False),
        (["typesafe/*"], "jev-internal-router", False),
        (["typesafe/*"], "jev-2.0-internal", False),
        (["typesafe/*"], "jev-1.0", False),
    ],
)
def test_typesafe_provider_wildcard_covers_bare_jev_model(
    shipped_routing_rules: None,
    object_type: Literal["team", "project"],
    allowed_models: list[str],
    model: str,
    expect_to_work: bool,
) -> None:
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.auth_checks import can_object_call_model

    if expect_to_work:
        assert can_object_call_model(model=model, llm_router=None, models=allowed_models, object_type=object_type)
        return
    with pytest.raises(ProxyException):
        can_object_call_model(model=model, llm_router=None, models=allowed_models, object_type=object_type)


@pytest.mark.parametrize(
    "key_models, model, expect_to_work",
    [
        # After a cost-map reload, add_known_models() updates anthropic_models so
        # the anthropic/* wildcard can match a newly-added Anthropic model.
        (["anthropic/*"], "claude-brand-new-model-reload-test", True),
        # Wrong provider wildcard must still be denied even after reload.
        (["openai/*"], "claude-brand-new-model-reload-test", False),
    ],
)
@pytest.mark.asyncio
async def test_wildcard_access_after_cost_map_reload(key_models, model, expect_to_work):
    """
    Regression test: after a cost-map hot-reload, calling
    add_known_models(model_cost_map=new_map) must update litellm.anthropic_models
    so that the anthropic/* wildcard correctly grants (or denies) access to
    newly-added models.

    Root cause: both reload paths in proxy_server.py only updated
    litellm.model_cost but never re-ran add_known_models(), so the provider sets
    stayed stale and wildcard matching failed for new models.

    Fix: each reload now calls litellm.add_known_models(model_cost_map=new_map)
    with the fetched map passed explicitly to avoid any reference ambiguity.
    """
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.auth_checks import can_key_call_model

    # Build a new cost map that includes the brand-new model — exactly what
    # proxy_server.py receives from get_model_cost_map() during a reload.
    new_cost_map = dict(litellm.model_cost)
    new_cost_map[model] = {
        "litellm_provider": "anthropic",
        "max_tokens": 8192,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    }

    original_model_cost = litellm.model_cost
    litellm.model_cost = new_cost_map

    # Confirm the model is NOT yet in the provider set before reload propagation.
    assert model not in litellm.anthropic_models

    # Simulate what proxy_server.py now does after every reload.
    litellm.add_known_models(model_cost_map=new_cost_map)

    # After add_known_models(), the model must be in the set.
    assert model in litellm.anthropic_models

    llm_model_list = [
        {
            "model_name": "anthropic/*",
            "litellm_params": {"model": "anthropic/*", "api_key": "test-api-key"},
            "model_info": {"id": "test-id-anthropic-wildcard", "db_model": False},
        },
        {
            "model_name": "openai/*",
            "litellm_params": {"model": "openai/*", "api_key": "test-api-key"},
            "model_info": {"id": "test-id-openai-wildcard", "db_model": False},
        },
    ]
    router = litellm.Router(model_list=llm_model_list)
    user_api_key_object = UserAPIKeyAuth(models=key_models)

    try:
        if expect_to_work:
            await can_key_call_model(
                model=model,
                llm_model_list=llm_model_list,
                valid_token=user_api_key_object,
                llm_router=router,
            )
        else:
            with pytest.raises(ProxyException):
                await can_key_call_model(
                    model=model,
                    llm_model_list=llm_model_list,
                    valid_token=user_api_key_object,
                    llm_router=router,
                )
    finally:
        litellm.model_cost = original_model_cost
        litellm.anthropic_models.discard(model)


@pytest.mark.asyncio
async def test_add_known_models_explicit_map_updates_provider_sets():
    """
    Regression test: after a cost-map hot-reload, calling
    add_known_models(model_cost_map=new_map) with the new map passed explicitly
    must add any new provider models to the correct provider sets so that
    wildcard access checks (anthropic/*, openai/*, …) work immediately.

    This covers the proxy_server.py fix where both reload paths now call
    litellm.add_known_models(model_cost_map=new_model_cost_map) instead of
    relying on the module-level model_cost being up to date.
    """
    fake_new_model = "claude-brand-new-explicit-map-test"

    # Baseline: the model must not be in the sets before we do anything.
    assert fake_new_model not in litellm.anthropic_models

    new_cost_map = dict(litellm.model_cost)
    new_cost_map[fake_new_model] = {
        "litellm_provider": "anthropic",
        "max_tokens": 8192,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    }

    # Simulate what proxy_server.py does on reload.
    original_model_cost = litellm.model_cost
    litellm.model_cost = new_cost_map
    litellm.add_known_models(model_cost_map=new_cost_map)

    try:
        assert fake_new_model in litellm.anthropic_models, (
            "add_known_models(model_cost_map=...) did not add the new model to "
            "litellm.anthropic_models — wildcard access checks would fail."
        )
    finally:
        # Clean up: restore original state.
        litellm.model_cost = original_model_cost
        litellm.anthropic_models.discard(fake_new_model)


@pytest.mark.asyncio
async def test_is_valid_fallback_model():
    from litellm import Router
    from litellm.proxy.auth.auth_checks import is_valid_fallback_model

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "openai/gpt-3.5-turbo"},
            }
        ]
    )

    try:
        await is_valid_fallback_model(model="gpt-3.5-turbo", llm_router=router, user_model=None)
    except Exception as e:
        pytest.fail(f"Expected is_valid_fallback_model to work, got exception: {e}")

    with pytest.raises(Exception, match="Invalid") as exc_info:
        await is_valid_fallback_model(model="gpt-4o", llm_router=router, user_model=None)
    e = exc_info.value
    assert "Invalid" in str(e)


@pytest.mark.parametrize(
    "token_spend, max_budget, expect_budget_error",
    [
        (5.0, 10.0, False),  # Under budget
        (10.0, 10.0, True),  # At budget limit
        (15.0, 10.0, True),  # Over budget
    ],
)
@pytest.mark.asyncio
async def test_virtual_key_max_budget_check(token_spend, max_budget, expect_budget_error):
    """
    Test if virtual key budget checks work as expected:
    1. Triggers budget alert for all cases
    2. Raises BudgetExceededError when spend >= max_budget
    """
    from litellm.proxy.auth.auth_checks import virtual_key_max_budget_check

    # Setup test data
    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=token_spend,
        max_budget=max_budget,
        user_id="test-user",
        key_alias="test-key",
    )

    user_obj = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@email.com",
        max_budget=None,
    )

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=None,
    )

    # Track if budget alert was called
    alert_called = False

    async def mock_budget_alert(*args, **kwargs):
        nonlocal alert_called
        alert_called = True

    proxy_logging_obj.budget_alerts = mock_budget_alert

    if expect_budget_error:
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=proxy_logging_obj,
                user_obj=user_obj,
            )
        assert exc_info.value.current_cost == token_spend
        assert exc_info.value.max_budget == max_budget
    else:
        await virtual_key_max_budget_check(
            valid_token=valid_token,
            proxy_logging_obj=proxy_logging_obj,
            user_obj=user_obj,
        )

    await asyncio.sleep(1)

    # Verify budget alert was triggered
    assert alert_called, "Budget alert should be triggered"


@pytest.mark.parametrize(
    "model, team_models, expect_to_work",
    [
        ("gpt-4", ["gpt-4"], True),  # exact match
        ("gpt-4", ["all-proxy-models"], True),  # all-proxy-models access
        ("gpt-4", ["*"], True),  # wildcard access
        ("gpt-4", ["openai/*"], True),  # openai wildcard access
        (
            "bedrock/anthropic.claude-3-5-sonnet-20240620",
            ["bedrock/*"],
            True,
        ),  # wildcard access
        (
            "bedrockz/anthropic.claude-3-5-sonnet-20240620",
            ["bedrock/*"],
            False,
        ),  # non-match wildcard access
        ("bedrock/very_new_model", ["bedrock/*"], True),  # bedrock wildcard access
        (
            "bedrock/claude-3-5-sonnet-20240620",
            ["bedrock/claude-*"],
            True,
        ),  # match on pattern
        (
            "bedrock/claude-3-6-sonnet-20240620",
            ["bedrock/claude-3-5-*"],
            False,
        ),  # don't match on pattern
        ("openai/gpt-4o", ["openai/*"], True),  # openai wildcard access
        ("gpt-4", ["gpt-3.5-turbo"], False),  # model not in allowed list
        ("claude-3", [], True),  # empty model list (allows all)
    ],
)
@pytest.mark.asyncio
async def test_can_team_access_model(model, team_models, expect_to_work):
    """
    Test cases for can_team_access_model:
    1. Exact model match
    2. all-proxy-models access
    3. Wildcard (*) access
    4. OpenAI wildcard access
    5. Model not in allowed list
    6. Empty model list
    7. None model list
    """
    try:
        team_object = LiteLLM_TeamTable(
            team_id="test-team",
            models=team_models,
        )
        result = await can_team_access_model(
            model=model,
            team_object=team_object,
            llm_router=None,
            team_model_aliases=None,
        )
        if not expect_to_work:
            pytest.fail(f"Expected model access check to fail for model={model}, team_models={team_models}")
    except Exception as e:
        if expect_to_work:
            pytest.fail(
                f"Expected model access check to work for model={model}, team_models={team_models}. Got error: {e!s}"
            )


@pytest.mark.parametrize(
    "spend, soft_budget, expect_alert",
    [
        (100, 50, True),  # Over soft budget
        (50, 50, True),  # At soft budget
        (25, 50, False),  # Under soft budget
        (100, None, False),  # No soft budget set
    ],
)
@pytest.mark.asyncio
async def test_virtual_key_soft_budget_check(spend, soft_budget, expect_alert):
    """
    Test cases for _virtual_key_soft_budget_check:
    1. Spend over soft budget
    2. Spend at soft budget
    3. Spend under soft budget
    4. No soft budget set
    """
    alert_triggered = False

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered
            alert_triggered = True
            assert type == "soft_budget"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=spend,
        soft_budget=soft_budget,
        user_id="test-user",
        team_id="test-team",
        key_alias="test-key",
    )

    proxy_logging_obj = MockProxyLogging()

    await virtual_key_soft_budget_check(
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
    )

    await asyncio.sleep(0.1)  # Allow time for the alert task to complete

    assert alert_triggered == expect_alert, (
        f"Expected alert_triggered to be {expect_alert} for spend={spend}, soft_budget={soft_budget}"
    )


@pytest.mark.parametrize(
    "spend, soft_budget, expect_alert, metadata, expected_alert_emails",
    [
        (
            100,
            50,
            False,
            None,
            None,
        ),  # Over soft budget, no metadata - no alert_emails configured, so no alert
        (
            50,
            50,
            False,
            None,
            None,
        ),  # At soft budget, no metadata - no alert_emails configured, so no alert
        (25, 50, False, None, None),  # Under soft budget
        (100, None, False, None, None),  # No soft budget set
        (
            100,
            50,
            True,
            {"soft_budget_alerting_emails": ["team1@example.com", "team2@example.com"]},
            ["team1@example.com", "team2@example.com"],
        ),  # Over soft budget with list of emails
        (
            100,
            50,
            True,
            {"soft_budget_alerting_emails": "team1@example.com,team2@example.com"},
            ["team1@example.com", "team2@example.com"],
        ),  # Over soft budget with comma-separated emails
        (
            100,
            50,
            True,
            {
                "soft_budget_alerting_emails": [
                    "team1@example.com",
                    "",
                    "  ",
                    "team2@example.com",
                ]
            },
            ["team1@example.com", "team2@example.com"],
        ),  # Over soft budget with empty strings filtered
    ],
)
@pytest.mark.asyncio
async def test_team_soft_budget_check(spend, soft_budget, expect_alert, metadata, expected_alert_emails):
    """
    Test cases for _team_soft_budget_check:
    1. Spend over soft budget, no alert_emails configured - should NOT trigger alert (alerts only sent when alert_emails configured)
    2. Spend at soft budget, no alert_emails configured - should NOT trigger alert (alerts only sent when alert_emails configured)
    3. Spend under soft budget - should not trigger alert
    4. No soft budget set - should not trigger alert
    5. Team with alert emails in metadata (list) - should include alert_emails in CallInfo
    6. Team with alert emails in metadata (comma-separated string) - should parse and include alert_emails
    7. Team with alert emails containing empty strings - should filter them out
    """
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info
            assert type == "soft_budget"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
        team_alias="test-team-alias",
        key_alias="test-key",
    )

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        spend=spend,
        soft_budget=soft_budget,
        max_budget=100.0,
        metadata=metadata,
    )

    proxy_logging_obj = MockProxyLogging()

    await _team_soft_budget_check(
        team_object=team_object,
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
    )

    await asyncio.sleep(0.1)  # Allow time for the alert task to complete

    assert alert_triggered == expect_alert, (
        f"Expected alert_triggered to be {expect_alert} for spend={spend}, soft_budget={soft_budget}"
    )

    if expect_alert:
        assert captured_call_info is not None
        assert captured_call_info.team_id == "test-team"
        assert captured_call_info.spend == spend
        assert captured_call_info.soft_budget == soft_budget
        assert captured_call_info.event_group == Litellm_EntityType.TEAM
        # Verify alert_emails if expected
        if expected_alert_emails is not None:
            assert captured_call_info.alert_emails == expected_alert_emails
        else:
            assert captured_call_info.alert_emails is None or captured_call_info.alert_emails == []


@pytest.mark.asyncio
async def test_can_user_call_model():
    from litellm import Router
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.auth_checks import can_user_call_model

    router = Router(
        model_list=[
            {
                "model_name": "anthropic-claude",
                "litellm_params": {"model": "anthropic/anthropic-claude"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test-api-key"},
            },
        ]
    )

    args = {
        "model": "anthropic-claude",
        "llm_router": router,
        "user_object": LiteLLM_UserTable(
            user_id="testuser21@mycompany.com",
            max_budget=None,
            spend=0.0042295,
            model_max_budget={},
            model_spend={},
            user_email="testuser@mycompany.com",
            models=["gpt-3.5-turbo"],
        ),
    }

    with pytest.raises(ProxyException) as e:
        await can_user_call_model(**args)

    args["model"] = "gpt-3.5-turbo"
    await can_user_call_model(**args)


@pytest.mark.asyncio
async def test_can_user_call_model_with_no_default_models():
    from unittest.mock import MagicMock

    from litellm.proxy._types import ProxyException, SpecialModelNames
    from litellm.proxy.auth.auth_checks import can_user_call_model

    args = {
        "model": "anthropic-claude",
        "llm_router": MagicMock(),
        "user_object": LiteLLM_UserTable(
            user_id="testuser21@mycompany.com",
            max_budget=None,
            spend=0.0042295,
            model_max_budget={},
            model_spend={},
            user_email="testuser@mycompany.com",
            models=[SpecialModelNames.no_default_models.value],
        ),
    }

    with pytest.raises(ProxyException) as e:
        await can_user_call_model(**args)


@pytest.mark.parametrize(
    "model, alias_map, expect_to_work",
    [
        ("gpt-4", {"gpt-4": "gpt-4-team1"}, True),  # model matches alias value
        ("gpt-5", {"gpt-4": "gpt-4-team1"}, False),
    ],
)
@pytest.mark.asyncio
async def test_can_key_call_model_with_aliases(model, alias_map, expect_to_work):
    """
    Test if can_key_call_model correctly handles model aliases in the token
    """
    from litellm.proxy.auth.auth_checks import can_key_call_model

    llm_model_list = [
        {
            "model_name": "gpt-4-team1",
            "litellm_params": {
                "model": "gpt-4",
                "api_key": "test-api-key",
            },
        }
    ]
    router = litellm.Router(model_list=llm_model_list)

    user_api_key_object = UserAPIKeyAuth(
        models=[
            "gpt-4-team1",
        ],
        team_model_aliases=alias_map,
    )

    if expect_to_work:
        await can_key_call_model(
            model=model,
            llm_model_list=llm_model_list,
            valid_token=user_api_key_object,
            llm_router=router,
        )
    else:
        with pytest.raises(Exception, match="is not available for this API key") as e:
            await can_key_call_model(
                model=model,
                llm_model_list=llm_model_list,
                valid_token=user_api_key_object,
                llm_router=router,
            )


# ---------------------------------------------------------------------------
# Access group cache helpers (_cache_access_object, _delete_cache_access_object)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_access_object():
    """Test _cache_access_object stores access group in cache with correct key."""
    from litellm.proxy._types import LiteLLM_AccessGroupTable
    from litellm.proxy.auth.auth_checks import cache_access_object

    cache = UserApiKeyCache()
    ag_id = "ag-test-123"
    ag_table = LiteLLM_AccessGroupTable(
        access_group_id=ag_id,
        access_group_name="test-group",
        access_model_names=["gpt-4"],
    )
    await cache_access_object(
        access_group_id=ag_id,
        access_group_table=ag_table,
        user_api_key_cache=cache,
    )
    cached = await cache.async_get_cache(key=AuthObject.access_group(ag_id))
    assert cached is not None
    if isinstance(cached, dict):
        assert cached.get("access_group_id") == ag_id
        assert cached.get("access_group_name") == "test-group"
    else:
        assert cached.access_group_id == ag_id
        assert cached.access_group_name == "test-group"


@pytest.mark.asyncio
async def test_delete_cache_access_object():
    """Test _delete_cache_access_object removes access group from in-memory cache."""
    from litellm.proxy._types import LiteLLM_AccessGroupTable
    from litellm.proxy.auth.auth_checks import delete_cache_access_object

    cache = UserApiKeyCache()
    ag_id = "ag-delete-test"
    ag_table = LiteLLM_AccessGroupTable(
        access_group_id=ag_id,
        access_group_name="to-delete",
    )
    await cache.async_set_cache(key=AuthObject.access_group(ag_id), value=ag_table, ttl=60)
    await delete_cache_access_object(access_group_id=ag_id, user_api_key_cache=cache)
    cached = await cache.async_get_cache(key=AuthObject.access_group(ag_id))
    assert cached is None


# ---------------------------------------------------------------------------
# Access group resource fetchers (_get_models_from_access_groups, _get_agent_ids_from_access_groups)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resource_field, access_group_data, expected",
    [
        (
            "access_model_names",
            {"access_group_id": "ag-1", "access_model_names": ["gpt-4", "claude-3"]},
            ["gpt-4", "claude-3"],
        ),
        (
            "access_agent_ids",
            {"access_group_id": "ag-2", "access_agent_ids": ["agent-a", "agent-b"]},
            ["agent-a", "agent-b"],
        ),
        (
            "access_model_names",
            {"access_group_id": "ag-3", "access_model_names": []},
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_resources_from_access_groups(resource_field, access_group_data, expected):
    """Test _get_resources_from_access_groups returns correct resource list from access groups."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy._types import LiteLLM_AccessGroupTable
    from litellm.proxy.auth.auth_checks import (
        get_agent_ids_from_access_groups,
        get_models_from_access_groups,
    )

    ag_table = LiteLLM_AccessGroupTable(
        access_group_id=access_group_data["access_group_id"],
        access_group_name="test",
        access_model_names=access_group_data.get("access_model_names", []),
        access_agent_ids=access_group_data.get("access_agent_ids", []),
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_access_object",
        new_callable=AsyncMock,
        return_value=ag_table,
    ):
        if resource_field == "access_model_names":
            result = await get_models_from_access_groups(
                access_group_ids=[access_group_data["access_group_id"]],
                prisma_client=MagicMock(),
                user_api_key_cache=UserApiKeyCache(),
            )
        else:
            result = await get_agent_ids_from_access_groups(
                access_group_ids=[access_group_data["access_group_id"]],
                prisma_client=MagicMock(),
                user_api_key_cache=UserApiKeyCache(),
            )
        assert sorted(result) == sorted(expected)


@pytest.mark.asyncio
async def test_get_models_from_access_groups_empty_ids():
    """Test _get_models_from_access_groups returns empty list when access_group_ids is empty."""
    from litellm.proxy.auth.auth_checks import get_models_from_access_groups

    result = await get_models_from_access_groups(access_group_ids=[])
    assert result == []


# ---------------------------------------------------------------------------
# can_team_access_model with access_group_ids fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_can_team_access_model_via_access_group_ids():
    """Test can_team_access_model allows access when team has access_group_ids granting model access."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import can_team_access_model

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        models=[],
        access_group_ids=["ag-with-gpt4"],
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["gpt-4"],
    ):
        result = await can_team_access_model(
            model="gpt-4",
            team_object=team_object,
            llm_router=None,
            team_model_aliases=None,
        )
        assert result is True


@pytest.mark.asyncio
async def test_can_team_access_model_access_group_ids_denied():
    """Test can_team_access_model denies when neither team models nor access_group_ids grant access."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.auth_checks import can_team_access_model

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-3.5-turbo"],
        access_group_ids=["ag-other"],
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["claude-3"],
    ):
        with pytest.raises(ProxyException):
            await can_team_access_model(
                model="gpt-4",
                team_object=team_object,
                llm_router=None,
                team_model_aliases=None,
            )


# ---------------------------------------------------------------------------
# can_key_call_model with access_group_ids fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_can_key_call_model_via_access_group_ids():
    """Test can_key_call_model allows access when key has access_group_ids granting model access."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import can_key_call_model

    user_api_key_object = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=["ag-with-gpt4"],
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "test"},
            }
        ]
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["gpt-4"],
    ):
        await can_key_call_model(
            model="gpt-4",
            llm_model_list=[],
            valid_token=user_api_key_object,
            llm_router=router,
        )


# ---------------------------------------------------------------------------
# _key_access_group_grants_model (key access group overriding team restriction)
# ---------------------------------------------------------------------------


def _patch_proxy_server_globals():
    """Patch proxy_server's prisma_client and user_api_key_cache to non-None mocks
    so the helper's None-guard doesn't short-circuit. The actual values don't
    matter because get_access_object is patched separately to return fixtures."""
    from unittest.mock import MagicMock, patch

    return [
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
    ]


def _fake_access_group(
    access_group_id: str,
    access_model_names=None,
    assigned_team_ids=None,
    assigned_key_ids=None,
):
    from litellm.proxy._types import LiteLLM_AccessGroupTable

    return LiteLLM_AccessGroupTable(
        access_group_id=access_group_id,
        access_group_name=access_group_id,
        access_model_names=access_model_names or [],
        assigned_team_ids=assigned_team_ids or [],
        assigned_key_ids=assigned_key_ids or [],
    )


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_team_authorized():
    """Group's assigned_team_ids includes the key's team and grants the model → True.

    This is the happy path equivalent of Andres's report: admin creates an
    access group with assigned_team_ids=[team-a], grants claude-haiku-4-5,
    attaches it to a key on team-a. Override fires.
    """
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=["premium-group"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],  # deliberately not synced — the access group itself authorizes
    )

    fake_ag = _fake_access_group(
        access_group_id="premium-group",
        access_model_names=["claude-haiku-4-5"],
        assigned_team_ids=["team-a"],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-haiku-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is True
        )
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_key_directly_authorized():
    """Group's assigned_key_ids includes the key's token and grants the model → True.

    Per-key authorization path: an admin scopes a group directly to a key
    (assigned_key_ids) without listing the team.
    """
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token-hashed",
        models=[],
        access_group_ids=["per-key-group"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],
    )

    fake_ag = _fake_access_group(
        access_group_id="per-key-group",
        access_model_names=["claude-haiku-4-5"],
        assigned_team_ids=[],
        assigned_key_ids=["test-token-hashed"],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-haiku-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is True
        )
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_key_has_no_groups():
    """Key with no access_group_ids → False (early return, no DB read)."""
    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=[],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=["any-group"],
    )
    assert (
        await key_access_group_grants_model(
            model="claude-haiku-4-5",
            valid_token=valid_token,
            team_object=team_object,
            llm_router=None,
        )
        is False
    )


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_group_does_not_cover_model():
    """Group authorizes the team but does not grant the requested model → False."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=["basic-group"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],
    )

    fake_ag = _fake_access_group(
        access_group_id="basic-group",
        access_model_names=["gpt-4o-mini"],
        assigned_team_ids=["team-a"],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-haiku-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is False
        )
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_group_authorizes_neither():
    """
    Bypass regression test: a team member sets a foreign access group on their
    key. The group grants the requested model but its assigned_team_ids /
    assigned_key_ids do not include this caller's team or token. Override is
    denied — the team's 401 propagates.
    """
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="team-a-token",
        models=[],
        access_group_ids=["team-b-premium"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],
    )

    fake_ag = _fake_access_group(
        access_group_id="team-b-premium",
        access_model_names=["claude-opus-4-5"],
        assigned_team_ids=["team-b"],
        assigned_key_ids=["team-b-token"],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-opus-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is False
        )
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_get_access_object_raises():
    """Group lookup failure (404, network, etc.) is treated as no authorization."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=["missing-group"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            side_effect=Exception("not found"),
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-haiku-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is False
        )
    finally:
        for p in patches:
            p.stop()


@pytest.fixture
def db_success_hook() -> Iterator[AsyncMock]:
    hook: Final = AsyncMock()
    with patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        MagicMock(service_logging_obj=MagicMock(async_service_success_hook=hook)),
    ):
        yield hook


async def _db_service_call_types(hook: AsyncMock) -> tuple[str, ...]:
    await asyncio.sleep(0)
    return tuple(call.kwargs["call_type"] for call in hook.await_args_list if call.kwargs["service"] == ServiceTypes.DB)


@pytest.mark.asyncio
async def test_get_user_object_cache_hit_emits_no_postgres_service_event(db_success_hook: AsyncMock) -> None:
    user_id: Final = f"cached-user-{uuid.uuid4()}"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=AuthObject.user(user_id), value=LiteLLM_UserTable(user_id=user_id, user_role="internal_user")
    )

    result: Final = await get_user_object(
        user_id=user_id,
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
        user_id_upsert=False,
        parent_otel_span="auth-span",
    )

    assert result is not None and result.user_id == user_id
    assert await _db_service_call_types(db_success_hook) == ()


@pytest.mark.asyncio
async def test_get_org_object_cache_hit_emits_no_postgres_service_event(db_success_hook: AsyncMock) -> None:
    org_id: Final = f"cached-org-{uuid.uuid4()}"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=AuthObject.org(org_id),
        value=LiteLLM_OrganizationTable(
            organization_id=org_id, budget_id="b", models=[], created_by="t", updated_by="t"
        ),
    )

    result: Final = await get_org_object(
        org_id=org_id,
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
        parent_otel_span="auth-span",
    )

    assert result is not None and result.organization_id == org_id
    assert await _db_service_call_types(db_success_hook) == ()


@pytest.mark.asyncio
async def test_get_user_object_cache_miss_reads_the_rust_store_which_reports_its_own_db_event(
    db_success_hook: AsyncMock, rust_auth_stores: RustAuthStores
) -> None:
    user_id: Final = f"db-user-{uuid.uuid4()}"
    rust_auth_stores.user.rows = {user_id: LiteLLM_UserTable(user_id=user_id, user_role="internal_user").model_dump()}

    result: Final = await get_user_object(
        user_id=user_id,
        prisma_client=MagicMock(),
        user_api_key_cache=UserApiKeyCache(),
        user_id_upsert=False,
        parent_otel_span="auth-span",
    )

    assert result is not None and result.user_id == user_id
    assert [sign_in["user_id"] for sign_in, _ in rust_auth_stores.user.sign_ins] == [user_id]
    assert await _db_service_call_types(db_success_hook) == ()


@pytest.mark.parametrize("entry_first", [True, False])
def test_is_model_cost_zero_judges_an_alias_chain_by_the_deployment_its_entry_routes_to(
    monkeypatch: pytest.MonkeyPatch, entry_first: bool
) -> None:
    """chain-entry resolves one hop to local-free and is served by local-free's own free
    deployment, so an over-budget key is waived for it; local-free by name resolves to paid-gpt
    and stays enforced. local-free's alias is a hop the router never takes for chain-entry, and
    the per-name verdict cache must not let either name's verdict leak into the other's."""
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "local-free",
                "litellm_params": {
                    "model": "ollama/qwen3:0.6b",
                    "api_base": "http://localhost:11434",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
            },
            {
                "model_name": "paid-gpt",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "fake",
                    "input_cost_per_token": 3e-06,
                    "output_cost_per_token": 1.5e-05,
                },
            },
        ],
        model_group_alias={"chain-entry": "local-free", "local-free": "paid-gpt"},
    )
    expected: Final = {"chain-entry": True, "local-free": False, "paid-gpt": False}
    order: Final = (
        ("chain-entry", "local-free", "paid-gpt") if entry_first else ("local-free", "paid-gpt", "chain-entry")
    )

    verdicts: Final = {name: is_model_cost_zero(model=name, llm_router=router) for name in order}

    assert verdicts == expected
    assert {name: is_model_cost_zero(model=name, llm_router=router) for name in order} == expected


MCP_DISCOVERY_REST_ROUTES = [
    "/mcp-rest/tools/list",
    "/v1/mcp/tools",
]


@pytest.mark.parametrize("route", MCP_DISCOVERY_REST_ROUTES)
@pytest.mark.asyncio
async def test_mcp_tool_discovery_route_bypasses_team_budget(route):
    from litellm.proxy.auth.auth_checks import common_checks

    team_object = LiteLLM_TeamTable(team_id="test-team", spend=150.0, max_budget=100.0)

    result = await common_checks(
        request_body={},
        team_object=team_object,
        user_object=None,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings={},
        route=route,
        llm_router=None,
        proxy_logging_obj=AsyncMock(),
        valid_token=UserAPIKeyAuth(token="test-token", team_id="test-team"),
        request=MagicMock(),
    )

    assert result is True


@pytest.mark.parametrize("method", ["initialize", "notifications/initialized", "ping", "tools/list"])
@pytest.mark.parametrize("route", ["/mcp", "/mcp/some-server"])
@pytest.mark.asyncio
async def test_mcp_jsonrpc_zero_spend_method_bypasses_team_budget(route, method):
    from litellm.proxy.auth.auth_checks import common_checks

    team_object = LiteLLM_TeamTable(team_id="test-team", spend=150.0, max_budget=100.0)

    result = await common_checks(
        request_body={"jsonrpc": "2.0", "id": 1, "method": method, "params": {}},
        team_object=team_object,
        user_object=None,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings={},
        route=route,
        llm_router=None,
        proxy_logging_obj=AsyncMock(),
        valid_token=UserAPIKeyAuth(token="test-token", team_id="test-team"),
        request=MagicMock(),
    )

    assert result is True


@pytest.mark.parametrize(
    "route,request_body",
    [
        ("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add"}}),
        ("/mcp/some-server", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add"}}),
        ("/mcp-rest/tools/call", {"name": "add", "arguments": {}}),
        ("/mcp/tools/call", {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        ("/mcp/tools", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add"}}),
        ("/mcp/tools/list", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add"}}),
    ],
)
@pytest.mark.asyncio
async def test_mcp_tool_call_route_still_enforces_team_budget(monkeypatch, route, request_body):
    from litellm.proxy.auth.auth_checks import common_checks
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    monkeypatch.setattr(  # test-quality-ok: the registry is a module singleton read by the budget check; no injection seam
        global_mcp_server_manager, "config_mcp_servers", {_PAID_SERVER[0]: _mcp_server(*_PAID_SERVER)}
    )

    team_object = LiteLLM_TeamTable(team_id="test-team", spend=150.0, max_budget=100.0)

    with pytest.raises(litellm.BudgetExceededError):
        await common_checks(
            request_body=request_body,
            team_object=team_object,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route=route,
            llm_router=None,
            proxy_logging_obj=AsyncMock(),
            valid_token=UserAPIKeyAuth(token="test-token", team_id="test-team"),
            request=MagicMock(),
        )


@pytest.mark.parametrize(
    "route,request_body,expected",
    [
        ("/mcp-rest/tools/list", {}, True),
        ("/v1/mcp/tools", {}, True),
        ("/mcp/tools", {}, False),
        ("/mcp/tools/list", {}, False),
        ("/mcp/tools", {"method": "tools/call"}, False),
        ("/mcp/tools", {"method": "tools/list"}, True),
        ("/mcp", {"method": "initialize"}, True),
        ("/mcp", {"method": "notifications/initialized"}, True),
        ("/mcp", {"method": "ping"}, True),
        ("/mcp", {"method": "tools/list"}, True),
        ("/mcp/some-server", {"method": "tools/list"}, True),
        ("/mcp", {"method": "tools/call"}, False),
        ("/mcp/some-server", {"method": "tools/call"}, False),
        ("/mcp", {}, False),
        ("/mcp", {"method": ["tools/list"]}, False),
        ("/mcp", {"method": {"name": "ping"}}, False),
        ("/mcp-rest/tools/call", {"method": "initialize"}, False),
        ("/mcp/tools/call", {"method": "initialize"}, False),
        ("/v1/chat/completions", {"method": "initialize"}, False),
        ("/mcp-rest/other", {"method": "tools/list"}, False),
    ],
)
def test_is_mcp_discovery_request(route, request_body, expected):
    from litellm.proxy.auth.auth_checks import is_mcp_discovery_request

    assert is_mcp_discovery_request(route=route, request_body=request_body) is expected


def _mcp_server(
    server_id: str, name: str, cost_info: dict[str, object] | None, display_names: dict[str, str] | None = None
) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=name,
        alias=name,
        url="https://example.invalid/mcp",
        transport=MCPTransport.http,
        tool_name_to_display_name=display_names,
        mcp_info=None if cost_info is None else {"server_name": name, "mcp_server_cost_info": cost_info},
    )


_UNPRICED_SERVER = ("unpriced-id", "unpriced_tools", None)
_RENAMED_PAID_TOOL_SERVER = (
    "renamed-id",
    "renamed_tools",
    {"default_cost_per_query": 0.0, "tool_name_to_cost_per_query": {"premium": 0.5, "other_premium": 0.5}},
    {"premium": "friendly", "basic": "plain", "other_premium": "unpriced_tools-echo"},
)
_FREE_SERVER_SHOWING_A_TOOL_AS_PREMIUM = (
    "shows-id",
    "shows_tools",
    None,
    {"basic": "premium", "plain": "mcp_tool_call"},
)
_PAID_DEFAULT_FREE_RENAMED_TOOL_SERVER = (
    "paid-default-id",
    "paid_default_tools",
    {"default_cost_per_query": 0.5, "tool_name_to_cost_per_query": {"basic": 0.0}},
    {"basic": "plain"},
)
_SERVER_SHOWING_A_FREE_TOOL_AS_A_PAID_ONE = (
    "swap-id",
    "swap_tools",
    {"default_cost_per_query": 0.5, "tool_name_to_cost_per_query": {"basic": 0.0}},
    {"basic": "swap_tools-premium", "premium": "swap_tools-basic"},
)
_SERVER_SHOWING_A_PAID_TOOL_AS_A_FREE_ONES_BARE_NAME = (
    "bare-id",
    "bare_tools",
    {"default_cost_per_query": 0.5, "tool_name_to_cost_per_query": {"basic": 0.0}},
    {"basic": "friendly", "premium": "basic"},
)
_PAID_SERVER_SHOWING_A_TOOL_AS_AN_UNPRICED_SERVERS_PREFIXED_NAME = (
    "cross-id",
    "cross_tools",
    {"default_cost_per_query": 0.5},
    {"premium": "unpriced_tools-echo"},
)
_ZERO_PRICED_SERVER = ("zero-id", "zero_tools", {"default_cost_per_query": 0.0})
_PAID_SERVER = (
    "paid-id",
    "paid_tools",
    {"default_cost_per_query": 0.5, "tool_name_to_cost_per_query": {"gratis": 0.0}},
)
_FREE_DEFAULT_PAID_TOOL_SERVER = ("mixed-id", "mixed_tools", {"tool_name_to_cost_per_query": {"premium": 0.25}})


@pytest.mark.parametrize(
    "servers,route,request_body,expected",
    [
        pytest.param(
            (_UNPRICED_SERVER, _ZERO_PRICED_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "unpriced_tools-echo"}},
            True,
            id="jsonrpc-call-when-no-server-charges",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _ZERO_PRICED_SERVER),
            "/mcp",
            {"method": "tools/call"},
            True,
            id="jsonrpc-call-without-a-readable-name-when-no-server-charges",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _ZERO_PRICED_SERVER),
            "/mcp-rest/tools/call",
            {"name": "echo", "server_id": "zero-id"},
            True,
            id="rest-call-when-no-server-charges",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "unpriced_tools-echo"}},
            True,
            id="jsonrpc-call-prefixed-with-a-free-server",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "paid_tools-echo"}},
            False,
            id="jsonrpc-call-prefixed-with-a-paid-server",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "paid_tools-gratis"}},
            True,
            id="jsonrpc-call-to-a-zero-priced-tool-on-a-paid-server",
        ),
        pytest.param(
            (_FREE_DEFAULT_PAID_TOOL_SERVER,),
            "/mcp",
            {"method": "tools/call", "params": {"name": "mixed_tools-premium"}},
            False,
            id="jsonrpc-call-to-a-priced-tool-on-a-server-with-no-default-cost",
        ),
        pytest.param(
            (_FREE_DEFAULT_PAID_TOOL_SERVER,),
            "/mcp",
            {"method": "tools/call", "params": {"name": "mixed_tools-echo"}},
            True,
            id="jsonrpc-call-to-an-unpriced-tool-beside-a-priced-one",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "echo"}},
            False,
            id="jsonrpc-bare-name-is-not-trusted-when-a-server-charges",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp",
            {"method": "tools/call"},
            False,
            id="jsonrpc-call-without-a-readable-name-when-a-server-charges",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp-rest/tools/call",
            {"name": "echo", "server_id": "unpriced-id"},
            True,
            id="rest-bare-name-with-a-free-server-id",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp-rest/tools/call",
            {"name": "echo", "server_id": "unpriced_tools"},
            True,
            id="rest-bare-name-with-a-free-server-name",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp-rest/tools/call",
            {"name": "echo", "server_id": "paid-id"},
            False,
            id="rest-bare-name-with-a-paid-server-id",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp-rest/tools/call",
            {"name": "unpriced_tools-echo", "server_id": "paid-id"},
            False,
            id="rest-free-prefix-with-a-paid-server-id",
        ),
        pytest.param(
            (_RENAMED_PAID_TOOL_SERVER,),
            "/mcp-rest/tools/call",
            {"name": "friendly", "server_id": "renamed-id"},
            False,
            id="rest-call-by-the-display-name-of-a-paid-tool",
        ),
        pytest.param(
            (_RENAMED_PAID_TOOL_SERVER,),
            "/mcp-rest/tools/call",
            {"name": "plain", "server_id": "renamed-id"},
            False,
            id="rest-call-by-a-display-name-stays-checked-even-when-its-tool-is-free",
        ),
        pytest.param(
            (_RENAMED_PAID_TOOL_SERVER,),
            "/mcp",
            {"method": "tools/call", "params": {"name": "plain"}},
            False,
            id="jsonrpc-call-by-a-bare-display-name-stays-checked",
        ),
        pytest.param(
            (_FREE_SERVER_SHOWING_A_TOOL_AS_PREMIUM, _RENAMED_PAID_TOOL_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "premium"}},
            False,
            id="jsonrpc-call-by-a-free-display-name-that-is-a-paid-native-tool-elsewhere",
        ),
        pytest.param(
            (_FREE_SERVER_SHOWING_A_TOOL_AS_PREMIUM, _RENAMED_PAID_TOOL_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "mcp_tool_call"}},
            False,
            id="jsonrpc-call-by-a-free-display-name-that-is-a-reserved-tool-name",
        ),
        pytest.param(
            (_PAID_DEFAULT_FREE_RENAMED_TOOL_SERVER,),
            "/mcp-rest/tools/call",
            {"name": "plain", "server_id": "paid-default-id"},
            False,
            id="rest-call-by-a-display-name-on-a-server-with-a-paid-default-stays-checked",
        ),
        pytest.param(
            (_SERVER_SHOWING_A_FREE_TOOL_AS_A_PAID_ONE,),
            "/mcp",
            {"method": "tools/call", "params": {"name": "swap_tools-premium"}},
            False,
            id="jsonrpc-call-by-a-display-name-that-resolves-on-to-a-paid-tool",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp-rest/tools/call",
            {"name": "mcp_tool_call", "server_id": "unpriced-id"},
            False,
            id="rest-call-of-a-reserved-tool-name-with-a-free-server-id",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "unpriced_tools-echo"}, "mcp_servers": ["paid_tools"]},
            False,
            id="jsonrpc-call-of-a-free-servers-prefixed-name-scoped-to-a-paid-server-that-would-run-it",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "unpriced_tools-echo"}, "mcp_servers": ["unpriced_tools"]},
            True,
            id="jsonrpc-call-scoped-to-the-free-server-that-owns-the-name",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "unpriced_tools-echo"}, "mcp_servers": ["some-access-group"]},
            False,
            id="jsonrpc-call-scoped-to-a-name-that-is-not-one-server",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _PAID_SERVER_SHOWING_A_TOOL_AS_AN_UNPRICED_SERVERS_PREFIXED_NAME),
            "/mcp",
            {"method": "tools/call", "params": {"name": "unpriced_tools-echo"}},
            False,
            id="prefixed-name-of-a-free-server-that-another-server-shows-a-tool-under-stays-checked",
        ),
        pytest.param(
            (_SERVER_SHOWING_A_PAID_TOOL_AS_A_FREE_ONES_BARE_NAME,),
            "/mcp-rest/tools/call",
            {"name": "friendly", "server_id": "bare-id"},
            False,
            id="rest-call-on-a-server-with-display-names-stays-checked",
        ),
        pytest.param(
            (_UNPRICED_SERVER, _RENAMED_PAID_TOOL_SERVER),
            "/mcp",
            {"method": "tools/call", "params": {"name": "unpriced_tools-echo"}},
            False,
            id="jsonrpc-call-whose-free-looking-name-is-the-display-name-of-a-paid-tool",
        ),
        pytest.param(
            (_UNPRICED_SERVER,),
            "/mcp",
            {"method": "resources/read", "params": {"name": "unpriced_tools-echo"}},
            False,
            id="other-jsonrpc-methods-stay-budget-checked",
        ),
        pytest.param(
            (_UNPRICED_SERVER,),
            "/v1/chat/completions",
            {"method": "tools/call", "params": {"name": "unpriced_tools-echo"}},
            False,
            id="non-mcp-route-with-a-tool-call-shaped-body",
        ),
    ],
)
def test_request_skips_budget_checks_for_mcp_tool_calls_that_cost_nothing(
    monkeypatch, servers, route, request_body, expected
):
    from litellm.proxy.auth.auth_checks import request_skips_budget_checks
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    monkeypatch.setattr(  # test-quality-ok: the registry is a module singleton read by the budget check; no injection seam
        global_mcp_server_manager,
        "config_mcp_servers",
        {server[0]: _mcp_server(*server) for server in servers},
    )

    assert (
        request_skips_budget_checks(route=route, model=None, llm_router=None, request_body=request_body) is expected
    ), f"{route} {request_body}"


@pytest.mark.asyncio
async def test_get_skill_names_from_access_groups_reads_cached_group_rows():
    from litellm.proxy._types import LiteLLM_AccessGroupTable
    from litellm.proxy.auth.auth_checks import cache_access_object, get_skill_names_from_access_groups
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    cache = UserApiKeyCache()
    for group_id, skills in (("group-a", ["skill-a", "shared"]), ("group-b", ["skill-b", "shared"])):
        await cache_access_object(
            access_group_id=group_id,
            access_group_table=LiteLLM_AccessGroupTable(
                access_group_id=group_id, access_group_name=group_id, access_skill_names=skills
            ),
            user_api_key_cache=cache,
        )
    untouched_prisma = MagicMock()
    untouched_prisma.db.litellm_accessgrouptable.find_unique = AsyncMock(
        side_effect=AssertionError("cache hit must not reach the DB")
    )

    assert await get_skill_names_from_access_groups(
        access_group_ids=("group-a", "group-b"),
        prisma_client=untouched_prisma,
        user_api_key_cache=cache,
        proxy_logging_obj=MagicMock(),
    ) == frozenset({"skill-a", "skill-b", "shared"})


class _GatedChannel(FakeChannel):
    """Holds every bulk broadcast until ``release`` is set; ``sent`` is set once one went out."""

    release: asyncio.Event
    sent: asyncio.Event

    def __init__(self, released: bool = False) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.sent = asyncio.Event()
        if released:
            self.release.set()

    async def broadcast_evictions(self, cache_keys: Sequence[str]) -> None:
        await self.release.wait()
        await super().broadcast_evictions(cache_keys)
        self.sent.set()


def _install_channel(monkeypatch: pytest.MonkeyPatch, handle: FakeChannel) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server.proxy_config, "cache_invalidation", CacheInvalidation(handle))


@pytest.mark.asyncio
async def test_evicting_more_keys_than_the_publisher_holds_pending_broadcasts_every_one_in_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle: Final = FakeChannel()
    _install_channel(monkeypatch, handle)
    hashed_tokens: Final = tuple(f"token-{index}" for index in range(1500))

    await delete_cache_key_objects(
        hashed_tokens=hashed_tokens, user_api_key_cache=UserApiKeyCache(), proxy_logging_obj=None
    )

    assert handle.bulk == [
        (*hashed_tokens, *(AuthObject.customer_app_key_resource(token).key for token in hashed_tokens))
    ]


@pytest.mark.asyncio
async def test_bulk_eviction_clears_this_worker_before_waiting_on_a_wedged_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle: Final = _GatedChannel()
    _install_channel(monkeypatch, handle)
    cache: Final = UserApiKeyCache(max_size_in_memory=2_000)
    hashed_tokens: Final = tuple(hashlib.sha256(f"token-{index}".encode()).hexdigest() for index in range(600))
    for hashed_token in hashed_tokens:
        cache.set_local(AuthObject.virtual_key(hashed_token), {"token": hashed_token}, None)
    seeded: Final = tuple(
        token for token in hashed_tokens if cache.get_local(AuthObject.virtual_key(token)) is not None
    )

    eviction: Final = asyncio.create_task(
        delete_cache_key_objects(hashed_tokens=hashed_tokens, user_api_key_cache=cache, proxy_logging_obj=None)
    )
    await asyncio.sleep(0.05)
    still_cached: Final = tuple(token for token in seeded if cache.get_local(AuthObject.virtual_key(token)) is not None)
    handle.release.set()
    await eviction

    assert seeded == hashed_tokens
    assert still_cached == ()
    assert handle.bulk == [
        (*hashed_tokens, *(AuthObject.customer_app_key_resource(token).key for token in hashed_tokens))
    ]


class _AuthRedisCache:
    namespace = None

    def __init__(self, entries: dict[str, object]) -> None:
        self.entries = dict(entries)

    async def async_get_cache(self, key: str, **kwargs: object) -> object | None:
        return self.entries.get(key)

    def set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.entries[key] = value

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.entries[key] = value

    def delete_cache(self, key: str) -> None:
        self.entries.pop(key, None)

    async def async_delete_cache(self, key: str) -> None:
        self.entries.pop(key, None)

    async def delete_cache_keys(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.entries.pop(key, None)


@pytest.mark.asyncio
async def test_bulk_eviction_clears_the_redis_copies_before_waiting_on_a_wedged_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle: Final = _GatedChannel()
    _install_channel(monkeypatch, handle)
    hashed_tokens: Final = tuple(hashlib.sha256(f"token-{index}".encode()).hexdigest() for index in range(600))
    cache, redis = shared_cache(FakeAuthCache(redis={token: {"token": token} for token in hashed_tokens}))
    for hashed_token in hashed_tokens:
        cache.set_local(AuthObject.virtual_key(hashed_token), {"token": hashed_token}, None)

    eviction: Final = asyncio.create_task(
        delete_cache_key_objects(hashed_tokens=hashed_tokens, user_api_key_cache=cache, proxy_logging_obj=None)
    )
    await asyncio.sleep(0.05)
    refilled: Final = [
        token for token in hashed_tokens if await cache.async_get_cache(key=AuthObject.virtual_key(token)) is not None
    ]
    handle.release.set()
    await eviction

    assert refilled == []
    assert redis.redis == {}


class _DeleteAfterBroadcastRedisCache(_AuthRedisCache):
    def __init__(self, broadcast_sent: asyncio.Event) -> None:
        super().__init__({})
        self._broadcast_sent = broadcast_sent
        self.finished = False

    async def delete_cache_keys(self, keys: Sequence[str]) -> None:
        await self._broadcast_sent.wait()
        self.finished = True


@pytest.mark.asyncio
async def test_bulk_eviction_broadcasts_while_the_usage_cache_redis_is_wedged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle: Final = _GatedChannel(released=True)
    _install_channel(monkeypatch, handle)
    usage_redis: Final = _DeleteAfterBroadcastRedisCache(broadcast_sent=handle.sent)
    proxy_logging_obj: Final = ProxyLogging(user_api_key_cache=None)
    proxy_logging_obj.internal_usage_cache = InternalUsageCache(
        dual_cache=DualCache(redis_cache=usage_redis)  # pyright: ignore[reportArgumentType]  # a dict-backed fake of the Redis layer
    )
    hashed_tokens: Final = ("token-a", "token-b")

    await delete_cache_key_objects(
        hashed_tokens=hashed_tokens, user_api_key_cache=UserApiKeyCache(), proxy_logging_obj=proxy_logging_obj
    )

    assert usage_redis.finished
    assert handle.bulk == [
        (*hashed_tokens, *(AuthObject.customer_app_key_resource(token).key for token in hashed_tokens))
    ]


@pytest.mark.parametrize("route", ("/v1/chat/completions", "/queue/chat/completions"))
def test_chat_routes_require_the_user_param_when_enforced(route: str) -> None:
    from starlette.requests import Request

    from litellm.proxy.auth.auth_checks import _enforce_user_param_check

    request: Final = Request({"type": "http", "method": "POST", "path": route})
    with pytest.raises(Exception, match="'user' param not passed"):
        _enforce_user_param_check({"enforce_user_param": True}, request, {}, route)
    _enforce_user_param_check({"enforce_user_param": True}, request, {"user": "u"}, route)


@pytest.mark.parametrize("route", ("/v1/chat/completions", "/queue/chat/completions"))
def test_spent_proxy_budget_refuses_chat_routes(monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    from litellm.proxy.auth.auth_checks import _global_proxy_budget_check

    monkeypatch.setattr(litellm, "max_budget", 0.5)
    _global_proxy_budget_check(global_proxy_spend=0.4, skip_budget_checks=False, route=route)
    with pytest.raises(litellm.BudgetExceededError):
        _global_proxy_budget_check(global_proxy_spend=1.0, skip_budget_checks=False, route=route)


@pytest.mark.parametrize(
    "authenticated_by_custom_auth, route, spend, expect_budget_error",
    [
        pytest.param(True, "/chat/completions", 15.0, True, id="custom-auth-over-budget-llm-route"),
        pytest.param(True, "/queue/chat/completions", 15.0, True, id="custom-auth-over-budget-queued-route"),
        pytest.param(True, "/chat/completions", 5.0, False, id="custom-auth-under-budget-llm-route"),
        pytest.param(True, "/key/info", 15.0, False, id="custom-auth-over-budget-management-route"),
        pytest.param(False, "/chat/completions", 15.0, False, id="virtual-key-path-checks-before-common-checks"),
    ],
)
@pytest.mark.asyncio
async def test_common_checks_enforces_key_max_budget_for_custom_auth_keys(
    authenticated_by_custom_auth: bool, route: str, spend: float, expect_budget_error: bool
) -> None:
    from litellm.proxy.auth.auth_checks import common_checks

    max_budget: Final = 10.0
    valid_token: Final = UserAPIKeyAuth(token="test-token", spend=spend, max_budget=max_budget).model_copy(
        update={"authenticated_by_custom_auth": authenticated_by_custom_auth}
    )

    async def run() -> bool:
        return await common_checks(
            request_body={},
            team_object=None,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route=route,
            llm_router=None,
            proxy_logging_obj=AsyncMock(),
            valid_token=valid_token,
            request=MagicMock(),
        )

    if expect_budget_error:
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await run()
        assert (exc_info.value.current_cost, exc_info.value.max_budget) == (spend, max_budget)
    else:
        assert await run() is True


def _latest_alias_router() -> tuple[litellm.Router, str, str]:
    from litellm.litellm_core_utils.model_latest_aliases import get_model_latest_aliases

    alias, target = next(iter(get_model_latest_aliases()["openai"].items()))
    router: Final = litellm.Router(
        model_list=[
            {"model_name": "latest", "litellm_params": {"model": f"openai/{alias}", "api_key": "test-api-key"}},
            {
                "model_name": "pinned",
                "litellm_params": {"model": f"openai/{target}", "api_key": "test-api-key"},
                "model_info": {"access_groups": ["pinned-group"]},
            },
            {
                "model_name": "unrelated",
                "litellm_params": {"model": "openai/unrelated-model", "api_key": "test-api-key"},
            },
        ]
    )
    return router, alias, target


@pytest.mark.parametrize(
    ("requested", "allowed_models", "allowed"),
    [
        ("latest", ["latest"], True),
        ("latest", ["pinned"], True),
        ("latest", ["pinned-group"], True),
        ("latest", ["unrelated"], False),
        ("pinned", ["latest"], True),
        ("pinned", ["unrelated"], False),
        ("unrelated", ["pinned"], False),
        ("unrelated", ["latest"], False),
    ],
)
def test_a_latest_alias_is_callable_by_whoever_may_call_the_model_it_points_to(
    requested: str, allowed_models: list[str], allowed: bool
) -> None:
    from litellm.proxy.auth.auth_checks import can_object_call_model

    router, _, _ = _latest_alias_router()
    if allowed:
        assert can_object_call_model(model=requested, llm_router=router, models=allowed_models, object_type="key")
    else:
        with pytest.raises(ProxyException):
            can_object_call_model(model=requested, llm_router=router, models=allowed_models, object_type="key")


@pytest.mark.parametrize("with_router", [True, False])
def test_a_latest_alias_requested_by_its_provider_name_passes_on_its_versioned_name(with_router: bool) -> None:
    from litellm.proxy.auth.auth_checks import can_object_call_model

    router, alias, target = _latest_alias_router()
    llm_router: Final = router if with_router else None
    assert can_object_call_model(
        model=f"openai/{alias}", llm_router=llm_router, models=[f"openai/{target}"], object_type="key"
    )
    with pytest.raises(ProxyException):
        can_object_call_model(
            model=f"openai/{target}", llm_router=llm_router, models=[f"openai/{alias}"], object_type="key"
        )


@pytest.mark.parametrize(("requested", "served_by"), [("latest", "pinned"), ("pinned", None), ("unrelated", None)])
def test_a_latest_alias_is_served_by_the_model_group_of_the_model_it_points_to(
    requested: str, served_by: str | None
) -> None:
    from litellm.proxy.common_utils.latest_alias_model_group import latest_alias_versioned_model_group

    router, _, _ = _latest_alias_router()
    assert latest_alias_versioned_model_group(requested, router) == served_by


def test_a_latest_alias_keeps_its_own_deployment_when_no_model_group_serves_the_model_it_points_to() -> None:
    from litellm.litellm_core_utils.model_latest_aliases import get_model_latest_aliases
    from litellm.proxy.common_utils.latest_alias_model_group import latest_alias_versioned_model_group

    alias: Final = next(iter(get_model_latest_aliases()["openai"]))
    router: Final = litellm.Router(
        model_list=[{"model_name": "latest", "litellm_params": {"model": f"openai/{alias}", "api_key": "test-api-key"}}]
    )
    assert latest_alias_versioned_model_group("latest", router) is None
