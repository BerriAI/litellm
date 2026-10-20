from collections.abc import Iterator, Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_AccessGroupTable,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ModelAccessDeniedProxyException,
    ProxyErrorTypes,
    UserAPIKeyAuth,
)
from litellm.proxy.auth import auth_checks
from litellm.proxy.auth.auth_checks import (
    can_key_call_model,
    can_key_call_resolved_model,
    can_user_call_model,
    common_checks,
)
from litellm.proxy.auth.model_access_group_grants import ungranted_access_group_models

_ROUTER: Final = litellm.Router(
    model_list=[
        {
            "model_name": "private-model",
            "litellm_params": {"model": "openai/private-only-upstream", "api_key": "sk-fake"},
            "model_info": {"id": "private-deployment-id", "access_groups": ["group-x"]},
        },
        {
            "model_name": "openai/private-ft",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"},
            "model_info": {"access_groups": ["group-x"]},
        },
        {"model_name": "public-model", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"}},
    ],
    model_group_alias={"alias-to-private": "private-model"},
    routing_groups=[
        {"group_name": "mixed-group", "models": ["private-model", "public-model"], "routing_strategy": "simple-shuffle"}
    ],
)


@pytest.fixture
def deny_by_default(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setitem(proxy_server.general_settings, "model_access_group_deny_by_default", True)
    yield


@pytest.fixture
def no_team_membership(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    async def _no_membership(**_: object) -> None:
        return None

    monkeypatch.setattr(auth_checks, "get_team_membership", _no_membership)
    yield


async def _common_checks(
    model: str,
    token: UserAPIKeyAuth,
    team: LiteLLM_TeamTable | None = None,
    user: LiteLLM_UserTable | None = None,
) -> bool:
    return await common_checks(
        request_body={"model": model},
        team_object=team,
        user_object=user,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings=proxy_server.general_settings,
        route="/chat/completions",
        llm_router=_ROUTER,
        proxy_logging_obj=MagicMock(),
        valid_token=token,
        request=MagicMock(spec=Request),
    )


def _team(models: Sequence[str], access_group_ids: Sequence[str] = ()) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(team_id="team-1", models=list(models), access_group_ids=list(access_group_ids))


_TEAM_KEY: Final = UserAPIKeyAuth(token="hashed-team-key", team_id="team-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "team_models, model, allowed",
    [
        ([], "private-model", False),
        (["*"], "private-model", False),
        (["all-proxy-models"], "private-model", False),
        (["openai/*"], "openai/private-ft", False),
        (["group-x"], "private-model", True),
        (["private-model"], "private-model", True),
        ([], "public-model", True),
    ],
)
async def test_team_needs_an_explicit_grant_for_access_group_models(
    deny_by_default: None, no_team_membership: None, team_models: list[str], model: str, allowed: bool
) -> None:
    if allowed:
        assert await _common_checks(model, _TEAM_KEY, team=_team(team_models)) is True
        return
    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        await _common_checks(model, _TEAM_KEY, team=_team(team_models))
    assert exc_info.value.type == ProxyErrorTypes.team_model_access_denied
    assert str(exc_info.value.code) == "403"


@pytest.mark.asyncio
@pytest.mark.parametrize("team_models, model", [([], "private-model"), (["openai/*"], "openai/private-ft")])
async def test_blanket_team_grant_is_unchanged_when_setting_is_off(
    monkeypatch: pytest.MonkeyPatch, no_team_membership: None, team_models: list[str], model: str
) -> None:
    monkeypatch.delitem(proxy_server.general_settings, "model_access_group_deny_by_default", raising=False)
    assert await _common_checks(model, _TEAM_KEY, team=_team(team_models)) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["private-deployment-id", "openai/private-only-upstream", "mixed-group"])
async def test_blanket_team_cannot_reach_an_access_group_deployment_by_another_name(
    deny_by_default: None, no_team_membership: None, model: str
) -> None:
    with pytest.raises(ModelAccessDeniedProxyException):
        await _common_checks(model, _TEAM_KEY, team=_team([]))


@pytest.mark.parametrize(
    "name, granted, ungranted",
    [
        ("private-deployment-id", ["private-model"], ()),
        ("private-deployment-id", ["private-deployment-id"], ()),
        ("openai/private-only-upstream", ["group-x"], ()),
        ("mixed-group", ["private-model"], ()),
        ("mixed-group", ["public-model"], ("mixed-group",)),
        ("alias-to-private", ["public-model"], ("alias-to-private",)),
        ("alias-to-private", ["group-x"], ()),
        ("mixed-group", ["*", "all-proxy-models", "openai/*"], ("mixed-group",)),
    ],
)
def test_another_name_for_a_deployment_is_granted_by_its_model_or_group(
    name: str, granted: list[str], ungranted: tuple[str, ...]
) -> None:
    assert ungranted_access_group_models(names=[name], granted_models=granted, llm_router=_ROUTER, team_id=None) == (
        ungranted
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("team_models, allowed", [([], False), (["group-x"], True)])
async def test_team_alias_cannot_reach_an_ungranted_access_group_model(
    deny_by_default: None, no_team_membership: None, team_models: list[str], allowed: bool
) -> None:
    token: Final = UserAPIKeyAuth(
        token="hashed-team-key", team_id="team-1", team_model_aliases={"nice-name": "private-model"}
    )
    if allowed:
        assert await _common_checks("nice-name", token, team=_team(team_models)) is True
        return
    with pytest.raises(ModelAccessDeniedProxyException):
        await _common_checks("nice-name", token, team=_team(team_models))


@pytest.mark.asyncio
@pytest.mark.parametrize("granted_names, allowed", [(["private-model"], True), (["public-model"], False)])
async def test_team_unified_access_group_counts_as_an_explicit_grant(
    deny_by_default: None, no_team_membership: None, granted_names: list[str], allowed: bool
) -> None:
    group: Final = LiteLLM_AccessGroupTable(
        access_group_id="ag-1", access_group_name="private", access_model_names=granted_names
    )
    with patch.object(auth_checks, "get_access_object", new_callable=AsyncMock, return_value=group):
        team: Final = _team([], access_group_ids=["ag-1"])
        if allowed:
            assert await _common_checks("private-model", _TEAM_KEY, team=team) is True
            return
        with pytest.raises(ModelAccessDeniedProxyException):
            await _common_checks("private-model", _TEAM_KEY, team=team)


@pytest.mark.asyncio
async def test_empty_key_list_on_a_team_key_defers_to_the_team_grant(deny_by_default: None) -> None:
    token: Final = UserAPIKeyAuth(token="hashed-team-key", team_id="team-1", models=[])
    assert await can_key_call_model(model="private-model", llm_model_list=None, valid_token=token, llm_router=_ROUTER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user, allowed",
    [
        (LiteLLM_UserTable(user_id="u1", models=[]), False),
        (LiteLLM_UserTable(user_id="u1", models=["group-x"]), True),
        (LiteLLM_UserTable(user_id="u1", models=[], user_role=LitellmUserRoles.PROXY_ADMIN.value), True),
    ],
)
async def test_personal_key_needs_an_explicit_user_grant(
    deny_by_default: None, user: LiteLLM_UserTable, allowed: bool
) -> None:
    token: Final = UserAPIKeyAuth(token="hashed-personal-key", user_id="u1")
    if allowed:
        assert await _common_checks("private-model", token, user=user) is True
        return
    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        await _common_checks("private-model", token, user=user)
    assert exc_info.value.type == ProxyErrorTypes.user_model_access_denied


@pytest.mark.asyncio
async def test_user_grant_is_skipped_when_owner_teams_decide(deny_by_default: None) -> None:
    assert await can_user_call_model(
        model="private-model",
        llm_router=_ROUTER,
        user_object=LiteLLM_UserTable(user_id="u1", models=[]),
        teams_grant_access_group_models=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_models, allowed", [([], False), (["private-model"], True)])
async def test_resolved_model_for_a_personal_key_needs_the_owner_grant(
    monkeypatch: pytest.MonkeyPatch, deny_by_default: None, owner_models: list[str], allowed: bool
) -> None:
    async def _get_user_object(**_: object) -> LiteLLM_UserTable:
        return LiteLLM_UserTable(user_id="u1", models=owner_models)

    monkeypatch.setattr(auth_checks, "get_user_object", _get_user_object)
    token: Final = UserAPIKeyAuth(token="hashed-personal-key", user_id="u1")
    if allowed:
        await can_key_call_resolved_model(
            model="private-model", llm_model_list=None, valid_token=token, llm_router=_ROUTER
        )
        return
    with pytest.raises(ModelAccessDeniedProxyException):
        await can_key_call_resolved_model(
            model="private-model", llm_model_list=None, valid_token=token, llm_router=_ROUTER
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token, allowed",
    [
        (UserAPIKeyAuth(token="hashed-service-key", models=[]), False),
        (UserAPIKeyAuth(token="hashed-service-key", models=["group-x"]), True),
        (UserAPIKeyAuth(token="hashed-service-key", models=[], user_role=LitellmUserRoles.PROXY_ADMIN), True),
    ],
)
async def test_key_with_no_team_or_user_owns_its_grant(
    deny_by_default: None, token: UserAPIKeyAuth, allowed: bool
) -> None:
    if allowed:
        assert await can_key_call_model(
            model="private-model", llm_model_list=None, valid_token=token, llm_router=_ROUTER
        )
        return
    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        await can_key_call_model(model="private-model", llm_model_list=None, valid_token=token, llm_router=_ROUTER)
    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token, allowed",
    [
        (UserAPIKeyAuth(token="hashed-service-key", models=["all-team-models"]), False),
        (UserAPIKeyAuth(token="hashed-personal-key", user_id="missing-user", models=[]), False),
        (UserAPIKeyAuth(token="hashed-personal-key", user_id="missing-user", models=["group-x"]), True),
    ],
)
async def test_teamless_key_without_a_loaded_owner_needs_its_own_grant(
    deny_by_default: None, token: UserAPIKeyAuth, allowed: bool
) -> None:
    if allowed:
        assert await _common_checks("private-model", token) is True
        return
    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        await _common_checks("private-model", token)
    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied


@pytest.mark.asyncio
async def test_resolved_model_for_an_all_team_models_key_with_no_team_needs_its_own_grant(
    deny_by_default: None,
) -> None:
    token: Final = UserAPIKeyAuth(token="hashed-service-key", models=["all-team-models"])
    with pytest.raises(ModelAccessDeniedProxyException):
        await can_key_call_resolved_model(
            model="private-model", llm_model_list=None, valid_token=token, llm_router=_ROUTER
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("key_group_models, allowed", [(["*"], False), (["private-model"], True)])
async def test_key_access_group_overrides_the_team_only_with_an_explicit_grant(
    monkeypatch: pytest.MonkeyPatch,
    deny_by_default: None,
    no_team_membership: None,
    key_group_models: list[str],
    allowed: bool,
) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    group: Final = LiteLLM_AccessGroupTable(
        access_group_id="key-ag",
        access_group_name="key-group",
        access_model_names=key_group_models,
        assigned_key_ids=["hashed-team-key"],
    )
    token: Final = UserAPIKeyAuth(token="hashed-team-key", team_id="team-1", access_group_ids=["key-ag"])
    with patch.object(auth_checks, "get_access_object", new_callable=AsyncMock, return_value=group):
        if allowed:
            assert await _common_checks("private-model", token, team=_team([])) is True
            return
        with pytest.raises(ModelAccessDeniedProxyException):
            await _common_checks("private-model", token, team=_team([]))


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_team_models, allowed", [(["group-x"], True), ([], False)])
async def test_personal_key_reaches_an_access_group_model_through_the_owner_team_grant(
    monkeypatch: pytest.MonkeyPatch,
    deny_by_default: None,
    no_team_membership: None,
    owner_team_models: list[str],
    allowed: bool,
) -> None:
    async def _get_team_object(**_: object) -> LiteLLM_TeamTableCachedObj:
        return LiteLLM_TeamTableCachedObj(team_id="team-1", models=owner_team_models)

    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", True)
    monkeypatch.setattr(auth_checks, "get_team_object", _get_team_object)
    token: Final = UserAPIKeyAuth(token="hashed-personal-key", user_id="u1")
    owner: Final = LiteLLM_UserTable(user_id="u1", models=[], teams=["team-1"])
    if allowed:
        assert await _common_checks("private-model", token, user=owner) is True
        return
    with pytest.raises(ModelAccessDeniedProxyException):
        await _common_checks("private-model", token, user=owner)
