import pytest

from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles
from litellm.proxy.auth.impersonation import (
    IMPERSONATE_EMAIL_HEADER,
    IMPERSONATE_USER_HEADER,
    ImpersonatedUserNotFound,
    ImpersonationForbidden,
    ImpersonationRequest,
    impersonate,
    read_impersonation_request,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

ALICE = LiteLLM_UserTable(user_id="alice", user_email="alice@x.test", user_role=LitellmUserRoles.PROXY_ADMIN.value)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("permissions", [{}, {"impersonate": False}, {"impersonate": "true"}])
async def test_key_without_the_permission_is_forbidden(permissions: dict[str, object]) -> None:
    result = await impersonate(_service_key(permissions), ImpersonationRequest("alice", None), _directory)
    assert isinstance(result, ImpersonationForbidden)
    assert "control-plane" in result.detail


@pytest.mark.asyncio
async def test_unknown_subject_is_not_found_not_created() -> None:
    result = await impersonate(_service_key({"impersonate": True}), ImpersonationRequest("ghost", None), _directory)
    assert result == ImpersonatedUserNotFound("ghost")


@pytest.mark.asyncio
async def test_permitted_key_becomes_the_subject_and_records_the_actor() -> None:
    result = await impersonate(_service_key({"impersonate": True}), ImpersonationRequest("alice", None), _directory)
    assert isinstance(result, UserAPIKeyAuth)
    assert (result.user_id, result.user_email, result.user_role) == (
        "alice",
        "alice@x.test",
        LitellmUserRoles.PROXY_ADMIN,
    )
    assert result.impersonated_by == "control-plane"
    assert result.api_key == "hashed"


@pytest.mark.asyncio
async def test_subject_does_not_inherit_the_service_keys_team_models_or_permissions() -> None:
    key = _service_key({"impersonate": True}).model_copy(
        update={"models": ["gpt-4o"], "max_budget": 1000.0, "metadata": {"tier": "platform"}}
    )
    result = await impersonate(key, ImpersonationRequest("alice", None), _directory)
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
    mismatch = await impersonate(
        _service_key({"impersonate": True}), ImpersonationRequest("alice", "bob@x.test"), _directory
    )
    assert mismatch == ImpersonatedUserNotFound("alice")
    same = await impersonate(
        _service_key({"impersonate": True}), ImpersonationRequest("alice", "ALICE@x.test"), _directory
    )
    assert isinstance(same, UserAPIKeyAuth)


@pytest.mark.asyncio
async def test_subject_budget_never_lands_on_the_shared_key_fields() -> None:
    budgeted = ALICE.model_copy(update={"max_budget": 5.0, "spend": 4.0})

    async def directory(user_id: str) -> LiteLLM_UserTable | None:
        return budgeted

    result = await impersonate(_service_key({"impersonate": True}), ImpersonationRequest("alice", None), directory)
    assert isinstance(result, UserAPIKeyAuth)
    assert (result.max_budget, result.spend) == (None, 0.0)


@pytest.mark.asyncio
async def test_the_service_keys_route_allowlist_stays_a_ceiling_on_the_subject() -> None:
    key = _service_key({"impersonate": True}).model_copy(update={"allowed_routes": ["/user/info", "/key/list"]})
    result = await impersonate(key, ImpersonationRequest("alice", None), _directory)
    assert isinstance(result, UserAPIKeyAuth)
    assert result.allowed_routes == ["/user/info", "/key/list"]
