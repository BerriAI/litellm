from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi.encoders import jsonable_encoder

import litellm
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_AccessGroupTable,
    LiteLLM_BudgetTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import can_key_call_model, team_max_budget_check
from litellm.proxy.auth.ui_session_models import load_ui_session_model_sources
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    team_membership_reservation_cache_key,
)
from litellm.proxy.utils import (
    PrismaClient,
    ProxyLogging,
    get_available_models_for_user,
)
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
    ModelGroupInfoResponse,
    ModelGroupTeam,
)


@pytest.fixture
def router():
    return litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/gpt-4.1-mini",
                    "api_key": "test-key",
                },
            }
            for name in ("personal", "research", "shared", "forbidden")
        ]
    )


@pytest.fixture
async def account(monkeypatch, router):
    cache = UserApiKeyCache()
    logging = ProxyLogging(cache)
    # Every row is in the real auth cache; the database must never be touched.
    prisma = cast(PrismaClient, object())
    user = LiteLLM_UserTable(
        user_id="researcher",
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        models=["personal"],
        teams=["research-team"],
        max_budget=10,
    )
    team = LiteLLM_TeamTableCachedObj(
        team_id="research-team",
        team_alias="Research",
        models=["research", "shared"],
        max_budget=20,
    )
    member = LiteLLM_TeamMembership(user_id=user.user_id, team_id=team.team_id, spend=0)
    auth = UserAPIKeyAuth(
        token="session-hash",
        user_id=user.user_id,
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        models=["personal"],
        max_budget=3,
    )
    await cache.async_set_cache(key=user.user_id, value=user, model_type=LiteLLM_UserTable)
    await cache.async_set_cache(key=f"team_id:{team.team_id}", value=team, model_type=LiteLLM_TeamTableCachedObj)
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user.user_id, team.team_id),
        value=member,
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging)
    return auth, user, team, member, prisma, cache, logging


async def sources(account, router):
    auth, _, _, _, prisma, cache, logging = account
    result = await load_ui_session_model_sources(auth, router, prisma, cache, logging)
    assert result is not None
    return result


async def test_discovery_includes_team_models_without_exposing_unrelated_models(account, router):
    auth, _, _, _, prisma, cache, logging = account
    models = await get_available_models_for_user(auth, router, {}, None, prisma, logging, user_api_key_cache=cache)
    access = await sources(account, router)
    assert {model: await access.available_teams(model) for model in models} == {
        "personal": (),
        "research": (ModelGroupTeam(team_id="research-team", team_alias="Research"),),
        "shared": (ModelGroupTeam(team_id="research-team", team_alias="Research"),),
    }


async def test_sole_team_preserves_identity_role_and_budgets_without_changing_cached_session(account, router):
    access = await sources(account, router)
    scoped = await access.for_request(("research",), None)
    assert (
        scoped.team_id,
        scoped.user_id,
        scoped.user_role,
        scoped.max_budget,
        scoped.team_max_budget,
    ) == (
        "research-team",
        "researcher",
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        3,
        20,
    )
    assert account[0].team_id == UI_SESSION_TOKEN_TEAM_ID
    assert account[0].models == ["personal"]


async def test_session_grants_resolve_before_the_auth_builder_attaches_the_user_role(account, router):
    auth, _, _, _, prisma, cache, logging = account
    access = await load_ui_session_model_sources(
        auth.model_copy(update={"user_role": None}), router, prisma, cache, logging
    )
    assert access is not None
    assert (await access.for_request(("research",), None)).team_id == "research-team"


async def test_direct_user_access_is_the_default_without_a_team_selection(account, router):
    access = await sources(account, router)
    assert await access.for_request(("personal",), None) == account[0]


async def test_session_model_access_fails_closed_without_a_database(account, router):
    auth, _, _, _, _, cache, logging = account
    with pytest.raises(ProxyException) as error:
        await load_ui_session_model_sources(auth, router, None, cache, logging)
    assert (error.value.code, error.value.type, error.value.param) == ("503", "no_db_connection", None)


async def test_deleted_user_cannot_reuse_model_grants_from_the_session(account, router, monkeypatch):
    auth, _, _, _, _, cache, logging = account
    prisma = PrismaClient(database_url="postgresql://localhost/session-tests", proxy_logging_obj=logging)
    missing_user = AsyncMock(return_value=None)
    monkeypatch.setattr("prisma.actions.LiteLLM_UserTableActions.find_unique", missing_user)
    deleted_user_auth = auth.model_copy(update={"user_id": "deleted-session-user"})

    with pytest.raises(ProxyException) as error:
        await load_ui_session_model_sources(deleted_user_auth, router, prisma, cache, logging)

    assert (error.value.code, error.value.type, error.value.param) == ("403", "auth_error", None)
    missing_user.assert_awaited_once_with(
        where={"user_id": "deleted-session-user"}, include={"organization_memberships": True}
    )


async def test_deleted_team_is_not_discoverable_or_selectable_with_stale_membership(account, router, monkeypatch):
    auth, _, team, _, _, cache, logging = account
    prisma = PrismaClient(database_url="postgresql://localhost/session-tests", proxy_logging_obj=logging)
    missing_team = AsyncMock(return_value=None)
    monkeypatch.setattr("prisma.actions.LiteLLM_TeamTableActions.find_unique", missing_team)
    await cache.async_delete_cache(f"team_id:{team.team_id}")

    access = await load_ui_session_model_sources(auth, router, prisma, cache, logging)

    assert access is not None
    assert await access.available_teams("personal") == ()
    assert await access.available_teams("research") is None
    with pytest.raises(ProxyException) as error:
        await access.for_request(("research",), team.team_id)
    assert (error.value.code, error.value.type, error.value.param) == (
        "403",
        "team_model_access_denied",
        "x-litellm-team-id",
    )
    missing_team.assert_awaited_once()


async def test_changed_live_user_role_stops_internal_session_grant_resolution(account, router):
    auth, user, _, _, prisma, cache, logging = account
    await cache.async_set_cache(
        key=user.user_id,
        value=user.model_copy(update={"user_role": LitellmUserRoles.PROXY_ADMIN}),
        model_type=LiteLLM_UserTable,
    )
    assert await load_ui_session_model_sources(auth, router, prisma, cache, logging) is None
    assert auth.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.parametrize("live_models", [["no-default-models"], ["shared"]])
async def test_removed_user_and_team_grants_deny_even_when_the_session_snapshot_allows(account, router, live_models):
    auth, user, _, _, prisma, cache, logging = account
    await cache.async_set_cache(
        key=user.user_id,
        value=user.model_copy(update={"models": live_models, "teams": []}),
        model_type=LiteLLM_UserTable,
    )
    stale_auth = auth.model_copy(update={"models": ["personal", "research"]})
    access = await load_ui_session_model_sources(stale_auth, router, prisma, cache, logging)
    assert access is not None
    for model in ("personal", "research"):
        assert await access.available_teams(model) is None
        with pytest.raises(ProxyException) as error:
            await access.for_request((model,), None)
        assert (error.value.code, error.value.type, error.value.param) == ("403", "key_model_access_denied", "model")


async def test_new_direct_grants_are_listed_and_usable_without_logging_in_again(account, router):
    auth, user, _, _, prisma, cache, logging = account
    await cache.async_set_cache(
        key=user.user_id,
        value=user.model_copy(update={"models": ["personal", "forbidden"]}),
        model_type=LiteLLM_UserTable,
    )
    models = await get_available_models_for_user(auth, router, {}, None, prisma, logging, user_api_key_cache=cache)
    access = await sources(account, router)
    scoped = await access.for_request(("forbidden",), None)
    assert set(models) == {"personal", "research", "shared", "forbidden"}
    assert await access.available_teams("forbidden") == ()
    assert await can_key_call_model("forbidden", None, scoped, router)
    assert (scoped.team_id, scoped.user_id, scoped.max_budget) == (UI_SESSION_TOKEN_TEAM_ID, "researcher", 3)
    assert auth.models == ["personal"]


async def test_explicit_team_is_not_silently_replaced_by_a_direct_user_grant(account, router):
    _, _, team, _, _, cache, _ = account
    await cache.async_set_cache(
        key=f"team_id:{team.team_id}",
        value=team.model_copy(update={"models": ["personal", "research"]}),
        model_type=LiteLLM_TeamTableCachedObj,
    )
    access = await sources(account, router)
    assert (await access.for_request(("personal",), "research-team")).team_id == "research-team"
    with pytest.raises(ProxyException) as error:
        await access.for_request(("personal",), "removed-team")
    assert error.value.code == "403"


async def test_team_access_group_grants_are_discoverable_and_usable(account, router):
    _, _, team, _, _, cache, _ = account
    group = LiteLLM_AccessGroupTable(
        access_group_id="research-group", access_group_name="Research models", access_model_names=["research"]
    )
    await cache.async_set_cache(key="access_group_id:research-group", value=group, model_type=LiteLLM_AccessGroupTable)
    await cache.async_set_cache(
        key=f"team_id:{team.team_id}",
        value=team.model_copy(update={"models": ["shared"], "access_group_ids": ["research-group"]}),
        model_type=LiteLLM_TeamTableCachedObj,
    )
    access = await sources(account, router)
    assert await access.available_teams("research") == (ModelGroupTeam(team_id="research-team", team_alias="Research"),)
    assert (await access.for_request(("research",), None)).team_id == "research-team"


async def test_multiple_granting_teams_require_explicit_choice(account, router):
    auth, user, _, _, _, cache, _ = account
    other = LiteLLM_TeamTableCachedObj(team_id="other-team", models=["research"])
    await cache.async_set_cache(
        key=user.user_id,
        value=user.model_copy(update={"teams": ["research-team", "other-team"]}),
        model_type=LiteLLM_UserTable,
    )
    await cache.async_set_cache(key="team_id:other-team", value=other, model_type=LiteLLM_TeamTableCachedObj)
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user.user_id, other.team_id),
        value=LiteLLM_TeamMembership(user_id=user.user_id, team_id=other.team_id),
    )
    access = await sources(account, router)
    with pytest.raises(ProxyException) as error:
        await access.for_request(("research",), None)
    assert (error.value.code, error.value.type, error.value.param) == (
        "409",
        "team_selection_required",
        "x-litellm-team-id",
    )
    assert (await access.for_request(("research",), "other-team")).team_id == "other-team"
    assert auth.team_id == UI_SESSION_TOKEN_TEAM_ID


@pytest.mark.parametrize("reason", ["blocked", "removed", "member_scope"])
async def test_unavailable_team_grants_are_not_listed_or_selectable(account, router, reason):
    _, user, team, member, _, cache, _ = account
    if reason == "blocked":
        await cache.async_set_cache(
            key=f"team_id:{team.team_id}",
            value=team.model_copy(update={"blocked": True}),
            model_type=LiteLLM_TeamTableCachedObj,
        )
    elif reason == "removed":
        await cache.async_set_cache(
            key=user.user_id,
            value=user.model_copy(update={"teams": []}),
            model_type=LiteLLM_UserTable,
        )
    else:
        await cache.async_set_cache(
            key=team_membership_reservation_cache_key(user.user_id, team.team_id),
            value=member.model_copy(update={"litellm_budget_table": LiteLLM_BudgetTable(allowed_models=["shared"])}),
        )
    access = await sources(account, router)
    assert await access.available_teams("research") is None
    with pytest.raises(ProxyException) as error:
        await access.for_request(("research",), "research-team")
    assert error.value.code == "403"


async def test_forbidden_model_and_fallback_keep_normal_key_denial(account, router):
    access = await sources(account, router)
    with pytest.raises(ProxyException) as error:
        await access.for_request(("research", "forbidden"), None)
    assert (error.value.code, error.value.type) == ("403", "key_model_access_denied")


async def test_selected_team_still_enforces_its_budget(account, router):
    access = await sources(account, router)
    scoped = await access.for_request(("research",), None)
    team = account[2].model_copy(update={"max_budget": 1, "spend": 2})
    with pytest.raises(litellm.BudgetExceededError):
        await team_max_budget_check(team, scoped, account[6])


@pytest.mark.parametrize(
    "auth",
    [
        UserAPIKeyAuth(
            user_id="researcher",
            team_id="research-team",
            user_role=LitellmUserRoles.INTERNAL_USER,
        ),
        UserAPIKeyAuth(user_id="researcher", user_role=LitellmUserRoles.INTERNAL_USER),
        UserAPIKeyAuth(
            user_id="admin",
            team_id=UI_SESSION_TOKEN_TEAM_ID,
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
    ],
)
async def test_other_credentials_are_never_widened(auth, router, account):
    assert await load_ui_session_model_sources(auth, router, account[4], account[5], account[6]) is None


def test_typed_response_preserves_existing_alias_and_arbitrary_metadata():
    group = ModelGroupInfoProxy(
        model_group="team-visible-alias",
        providers=["openai"],
        custom_tags=("research",),
        custom_metadata={"n": 5, "flag": True, "label": "Research"},
        supported_endpoints=("/v1/responses",),
    )
    assert ModelGroupInfoResponse(data=(group,)).model_dump(mode="json") == jsonable_encoder({"data": [group]})


async def test_team_public_model_preserves_capabilities_in_session_discovery(account, monkeypatch):
    auth, user, team, _, _, cache, _ = account
    router = litellm.Router(
        model_list=[
            {
                "model_name": "model_name_research-team_private",
                "litellm_params": {"model": "openai/gpt-4.1-mini", "api_key": "test-key"},
                "model_info": {"team_id": "research-team", "team_public_model_name": "team-chat", "mode": "chat"},
            }
        ]
    )
    await cache.async_set_cache(
        key=f"team_id:{team.team_id}",
        value=team.model_copy(update={"models": ["model_name_research-team_private"]}),
        model_type=LiteLLM_TeamTableCachedObj,
    )
    await cache.async_set_cache(
        key=user.user_id,
        value=user.model_copy(update={"models": ["no-default-models"]}),
        model_type=LiteLLM_UserTable,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.model_list)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    response = await proxy_server.model_group_info(user_api_key_dict=auth)
    assert [(group.model_group, group.providers, group.mode, group.available_teams) for group in response["data"]] == [
        (
            "model_name_research-team_private",
            ["openai"],
            "chat",
            (ModelGroupTeam(team_id="research-team", team_alias="Research"),),
        ),
    ]
