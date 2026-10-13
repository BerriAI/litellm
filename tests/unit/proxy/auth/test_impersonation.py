from collections.abc import Mapping
from typing import Final

import pytest
from fastapi import HTTPException, Request

from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, ProxyException
from litellm.proxy.auth.impersonation import (
    IMPERSONATE_EMAIL_HEADER,
    IMPERSONATE_ENCODING_HEADER,
    IMPERSONATE_USER_HEADER,
    ImpersonatedUserNotFound,
    ImpersonationForbidden,
    ImpersonationRequest,
    UserLookup,
    read_impersonation_request,
)
from litellm.proxy.auth.impersonation import (
    impersonate as _impersonate,
)
from litellm.proxy.auth.resolvers.models import PrincipalType
from litellm.proxy.auth.user_api_key_auth import _impersonate_or_raise, _resolve_request_principal
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


async def impersonate(
    actor: UserAPIKeyAuth,
    request: ImpersonationRequest,
    lookup: UserLookup,
) -> UserAPIKeyAuth | ImpersonationForbidden | ImpersonatedUserNotFound:
    return await _impersonate(actor, request, lookup, actor_is_master_key=False)


ALICE: Final = LiteLLM_UserTable(
    user_id="alice", user_email="alice@x.test", user_role=LitellmUserRoles.PROXY_ADMIN.value
)


class _BudgetLimiter:
    async def is_user_within_model_budget(
        self,
        user_id: str,
        user_model_max_budget: Mapping[str, object],
        model: str,
        team_id: str | None = None,
    ) -> bool:
        raise RuntimeError("subject model budget exceeded")


def _service_key(permissions: dict[str, object]) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed",
        token="hashed",
        key_alias="control-plane",
        user_id="svc",
        team_id="platform",
        permissions=permissions,
    )


async def _directory(user_id: str) -> LiteLLM_UserTable | None:
    return ALICE if user_id == "alice" else None


def test_headers_without_a_user_are_not_an_impersonation_request() -> None:
    assert read_impersonation_request({}) is None
    assert read_impersonation_request({IMPERSONATE_USER_HEADER: "  ", IMPERSONATE_EMAIL_HEADER: "a@x"}) is None


def test_headers_parse_user_and_optional_email() -> None:
    assert read_impersonation_request({IMPERSONATE_USER_HEADER: "alice"}) == ImpersonationRequest("alice", None)
    assert read_impersonation_request({IMPERSONATE_USER_HEADER: "alice", IMPERSONATE_EMAIL_HEADER: "a@x"}) == (
        ImpersonationRequest("alice", "a@x")
    )


@pytest.mark.parametrize("encoding,value", [("unknown", "alice"), ("base64-utf8", "!!!!"), ("base64-utf8", "/w==")])
def test_invalid_identity_encoding_is_a_bad_request(encoding: str, value: str) -> None:
    with pytest.raises(HTTPException) as denied:
        read_impersonation_request({IMPERSONATE_USER_HEADER: value, IMPERSONATE_ENCODING_HEADER: encoding})
    assert denied.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("permissions", [{}, {"impersonate": False}, {"impersonate": "true"}])
async def test_key_without_the_permission_is_forbidden(permissions: dict[str, object]) -> None:
    result: Final = await impersonate(_service_key(permissions), ImpersonationRequest("alice", None), _directory)
    assert isinstance(result, ImpersonationForbidden)
    assert "control-plane" in result.detail


@pytest.mark.asyncio
async def test_key_without_permission_is_rejected_with_http_403() -> None:
    with pytest.raises(ProxyException) as exc:
        await _impersonate_or_raise(
            actor=_service_key({}),
            impersonation=ImpersonationRequest("alice", None),
            lookup=_directory,
            request=Request(scope={"type": "http", "method": "GET", "path": "/key/info", "headers": []}),
            request_data={},
            llm_router=None,
            model_max_budget_limiter=_BudgetLimiter(),
            route="/key/info",
        )
    assert exc.value.code == "403"


@pytest.mark.asyncio
async def test_blocked_team_key_is_rejected_with_http_403() -> None:
    actor: Final = _service_key({"impersonate": True}).model_copy(update={"team_blocked": True})
    with pytest.raises(ProxyException) as exc:
        await _impersonate_or_raise(
            actor=actor,
            impersonation=ImpersonationRequest("alice", None),
            lookup=_directory,
            request=Request(scope={"type": "http", "method": "GET", "path": "/key/info", "headers": []}),
            request_data={},
            llm_router=None,
            model_max_budget_limiter=_BudgetLimiter(),
            route="/key/info",
        )
    assert exc.value.code == "403"


@pytest.mark.asyncio
async def test_unknown_subject_is_not_found_not_created() -> None:
    result: Final = await impersonate(
        _service_key({"impersonate": True}), ImpersonationRequest("ghost", None), _directory
    )
    assert result == ImpersonatedUserNotFound("ghost")


@pytest.mark.asyncio
async def test_permitted_key_becomes_the_subject_and_records_the_actor() -> None:
    result: Final = await impersonate(
        _service_key({"impersonate": True}), ImpersonationRequest("alice", None), _directory
    )
    assert isinstance(result, UserAPIKeyAuth)
    assert (result.user_id, result.user_email, result.user_role) == (
        "alice",
        "alice@x.test",
        LitellmUserRoles.PROXY_ADMIN,
    )
    assert result.impersonated_by == "control-plane"
    assert result.api_key is None
    assert result.token == "hashed"


@pytest.mark.asyncio
async def test_null_permissions_and_subject_grants_use_safe_defaults() -> None:
    actor: Final = _service_key({"impersonate": True}).model_copy(update={"permissions": None})
    denied: Final = await impersonate(actor, ImpersonationRequest("alice", None), _directory)
    assert isinstance(denied, ImpersonationForbidden)

    subject: Final = ALICE.model_copy(update={"models": None, "model_max_budget": None, "metadata": None})
    result: Final = await _impersonate(
        _service_key({"impersonate": True}),
        ImpersonationRequest("alice", None),
        lambda user_id: _return_subject(subject, user_id),
        actor_is_master_key=False,
    )
    assert isinstance(result, UserAPIKeyAuth)
    assert result.models == []


async def _return_subject(subject: LiteLLM_UserTable, user_id: str) -> LiteLLM_UserTable | None:
    return subject if user_id == subject.user_id else None


@pytest.mark.asyncio
async def test_impersonated_subject_principal_keeps_the_presented_service_token() -> None:
    result: Final = await impersonate(
        _service_key({"impersonate": True}), ImpersonationRequest("alice", None), _directory
    )
    assert isinstance(result, UserAPIKeyAuth)
    request: Final = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "path": "/user/info",
            "raw_path": b"/user/info",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )
    principal: Final = _resolve_request_principal(request, result)
    assert principal.principal_type == PrincipalType.HUMAN
    assert principal.credential_ref.token_id == "hashed"


@pytest.mark.asyncio
async def test_subject_does_not_inherit_the_service_keys_team_models_or_permissions() -> None:
    key: Final = _service_key({"impersonate": True}).model_copy(
        update={"models": ["gpt-4o"], "max_budget": 1000.0, "metadata": {"tier": "platform"}}
    )
    result: Final = await impersonate(key, ImpersonationRequest("alice", None), _directory)
    assert isinstance(result, UserAPIKeyAuth)
    assert result.model_dump(include={"team_id", "models", "max_budget", "permissions", "metadata"}) == {
        "team_id": None,
        "models": [],
        "max_budget": None,
        "permissions": {},
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_email_header_naming_another_user_is_not_found() -> None:
    mismatch: Final = await impersonate(
        _service_key({"impersonate": True}), ImpersonationRequest("alice", "bob@x.test"), _directory
    )
    assert mismatch == ImpersonatedUserNotFound("alice")
    same: Final = await impersonate(
        _service_key({"impersonate": True}), ImpersonationRequest("alice", "ALICE@x.test"), _directory
    )
    assert isinstance(same, UserAPIKeyAuth)


@pytest.mark.asyncio
async def test_subject_budget_is_copied_to_user_budget_fields_only() -> None:
    budgeted: Final = ALICE.model_copy(update={"max_budget": 5.0, "spend": 4.0})

    async def directory(user_id: str) -> LiteLLM_UserTable | None:
        return budgeted

    result: Final = await impersonate(
        _service_key({"impersonate": True}), ImpersonationRequest("alice", None), directory
    )
    assert isinstance(result, UserAPIKeyAuth)
    assert (result.max_budget, result.spend) == (None, 0.0)
    assert (result.user_max_budget, result.user_spend) == (5.0, 4.0)
    assert result.impersonated_by == "control-plane"


def test_impersonated_by_is_not_set_from_auth_input() -> None:
    auth: Final = UserAPIKeyAuth(impersonated_by="forged")
    assert auth.impersonated_by is None


@pytest.mark.asyncio
async def test_subject_does_not_inherit_the_service_keys_route_allowlist() -> None:
    key: Final = _service_key({"impersonate": True}).model_copy(update={"allowed_routes": ["/user/info", "/key/list"]})
    result: Final = await impersonate(key, ImpersonationRequest("alice", None), _directory)
    assert isinstance(result, UserAPIKeyAuth)
    assert result.allowed_routes is None


@pytest.mark.asyncio
async def test_scim_deactivated_subject_is_forbidden() -> None:
    inactive: Final = ALICE.model_copy(update={"metadata": {"scim_active": False}})

    async def directory(user_id: str) -> LiteLLM_UserTable | None:
        return inactive if user_id == "alice" else None

    result: Final = await impersonate(
        _service_key({"impersonate": True}), ImpersonationRequest("alice", None), directory
    )
    assert isinstance(result, ImpersonationForbidden)
    assert "deactivated via SCIM" in result.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("master", [False, True])
async def test_view_only_subject_cannot_use_a_service_keys_write_allowlist(master: bool) -> None:
    subject: Final = LiteLLM_UserTable(
        user_id="alice",
        user_email="alice@x.test",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    actor: Final = _service_key({"impersonate": True}).model_copy(
        update={"allowed_routes": ["/user/new"], "user_role": LitellmUserRoles.PROXY_ADMIN}
    )

    async def directory(user_id: str) -> LiteLLM_UserTable | None:
        return subject if user_id == "alice" else None

    with pytest.raises(HTTPException) as denied:
        await _impersonate_or_raise(
            actor=actor,
            impersonation=ImpersonationRequest("alice", None),
            lookup=directory,
            route="/user/new",
            request=Request(scope={"type": "http", "method": "POST", "path": "/user/new", "headers": []}),
            request_data={},
            llm_router=None,
            model_max_budget_limiter=_BudgetLimiter(),
            actor_is_master_key=master,
        )
    assert denied.value.status_code == 403


@pytest.mark.asyncio
async def test_admin_subject_stays_within_the_service_keys_route_ceiling() -> None:
    actor: Final = _service_key({"impersonate": True}).model_copy(update={"allowed_routes": ["/user/info"]})

    with pytest.raises(HTTPException) as exc:
        await _impersonate_or_raise(
            actor=actor,
            impersonation=ImpersonationRequest("alice", None),
            lookup=_directory,
            request=Request(scope={"type": "http", "method": "POST", "path": "/user/new", "headers": []}),
            request_data={},
            llm_router=None,
            model_max_budget_limiter=_BudgetLimiter(),
            route="/user/new",
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_impersonation_is_forbidden_on_chat_completions() -> None:
    with pytest.raises(ProxyException) as exc:
        await _impersonate_or_raise(
            actor=_service_key({"impersonate": True}),
            impersonation=ImpersonationRequest("alice", None),
            lookup=_directory,
            request=Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []}),
            request_data={},
            llm_router=None,
            model_max_budget_limiter=_BudgetLimiter(),
            route="/chat/completions",
        )
    assert exc.value.code == "403"


@pytest.mark.asyncio
async def test_impersonation_remains_available_on_key_info() -> None:
    result: Final = await _impersonate_or_raise(
        actor=_service_key({"impersonate": True}),
        impersonation=ImpersonationRequest("alice", None),
        lookup=_directory,
        request=Request(scope={"type": "http", "method": "GET", "path": "/key/info", "headers": []}),
        request_data={},
        llm_router=None,
        model_max_budget_limiter=_BudgetLimiter(),
        route="/key/info",
    )
    assert result.user_id == "alice"


@pytest.mark.asyncio
async def test_llm_route_group_does_not_allow_impersonation() -> None:
    actor: Final = _service_key({"impersonate": True}).model_copy(update={"allowed_routes": ["llm_api_routes"]})
    request: Final = Request(scope={"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})

    with pytest.raises(ProxyException) as exc:
        await _impersonate_or_raise(
            actor=actor,
            impersonation=ImpersonationRequest("alice", None),
            lookup=_directory,
            request=request,
            request_data={"model": "gpt-4o"},
            llm_router=None,
            model_max_budget_limiter=_BudgetLimiter(),
            route="/v1/chat/completions",
        )
    assert exc.value.code == "403"


@pytest.mark.asyncio
async def test_impersonation_is_denied_before_subject_budget_check() -> None:
    subject: Final = ALICE.model_copy(
        update={"model_max_budget": {"gpt-4o": {"budget_limit": 1.0, "time_period": "1mo"}}}
    )

    async def directory(user_id: str) -> LiteLLM_UserTable | None:
        return subject if user_id == "alice" else None

    with pytest.raises(ProxyException) as exc:
        await _impersonate_or_raise(
            actor=_service_key({"impersonate": True}),
            impersonation=ImpersonationRequest("alice", None),
            lookup=directory,
            request=Request(scope={"type": "http", "method": "POST", "path": "/chat/completions", "headers": []}),
            request_data={"model": "gpt-4o"},
            llm_router=None,
            model_max_budget_limiter=_BudgetLimiter(),
            route="/chat/completions",
        )
    assert exc.value.code == "403"
