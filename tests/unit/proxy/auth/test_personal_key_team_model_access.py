from collections.abc import Iterator, Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ModelAccessDeniedProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth import auth_checks
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model, common_checks
from litellm.proxy.utils import get_available_models_for_user

_TEAMS: Final[Mapping[str, LiteLLM_TeamTableCachedObj]] = {
    UI_TEAM_ID: LiteLLM_TeamTableCachedObj(team_id=UI_TEAM_ID, models=[]),
    "team-a": LiteLLM_TeamTableCachedObj(team_id="team-a", models=["gpt-a", "shared"]),
    "team-b": LiteLLM_TeamTableCachedObj(team_id="team-b", models=["gpt-b", "shared"]),
    "team-open": LiteLLM_TeamTableCachedObj(team_id="team-open", models=[]),
    "team-blocked": LiteLLM_TeamTableCachedObj(team_id="team-blocked", models=["gpt-a"], blocked=True),
    "team-group": LiteLLM_TeamTableCachedObj(
        team_id="team-group", models=["no-default-models"], access_group_ids=["team-models"]
    ),
    "team-exhausted": LiteLLM_TeamTableCachedObj(team_id="team-exhausted", models=["gpt-a"], spend=2, max_budget=1),
    "team-native-group": LiteLLM_TeamTableCachedObj(team_id="team-native-group", models=["team-access"]),
}

_ROUTER: Final = litellm.Router(
    model_list=[
        {
            "model_name": name,
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"},
            "model_info": {"access_groups": ["team-access"] if name == "gpt-a" else []},
        }
        for name in ("gpt-a", "gpt-b", "shared", "gpt-c")
    ]
)


@pytest.fixture
def teams(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    async def _get_team_object(team_id: str, **_: object) -> LiteLLM_TeamTableCachedObj:
        if team_id not in _TEAMS:
            raise Exception(f"team {team_id} lookup failed")
        return _TEAMS[team_id]

    async def _no_membership(**_: object) -> None:
        return None

    monkeypatch.setattr(auth_checks, "get_team_object", _get_team_object)
    monkeypatch.setattr(auth_checks, "get_team_membership", _no_membership)
    yield


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", True)


async def _common_checks(
    model: str | list[str],
    user: LiteLLM_UserTable | None,
    token: UserAPIKeyAuth,
    fallbacks: tuple[str, ...] = (),
) -> bool:
    request_body: Final = {"model": model} | ({"fallbacks": list(fallbacks)} if fallbacks else {})
    return await common_checks(
        request_body=request_body,
        team_object=None,
        user_object=user,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings={},
        route="/chat/completions",
        llm_router=_ROUTER,
        proxy_logging_obj=MagicMock(),
        valid_token=token,
        request=MagicMock(spec=Request),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, allowed",
    [
        ("gpt-a", True),
        ("gpt-b", True),
        ("shared", True),
        ("gpt-c", False),
    ],
)
async def test_personal_key_inherits_the_union_of_its_owner_team_models(
    monkeypatch: pytest.MonkeyPatch, teams: None, model: str, allowed: bool
) -> None:
    _enable(monkeypatch)
    user: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a", "team-b"])
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")

    if allowed:
        assert await _common_checks(model, user, token) is True
    else:
        with pytest.raises(ModelAccessDeniedProxyException):
            await _common_checks(model, user, token)


@pytest.mark.asyncio
async def test_personal_key_without_teams_gets_no_models(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    with pytest.raises(ModelAccessDeniedProxyException):
        await _common_checks(
            "gpt-a", LiteLLM_UserTable(user_id="u1", teams=[]), UserAPIKeyAuth(token="k1", user_id="u1")
        )


@pytest.mark.asyncio
async def test_personal_key_denied_when_owner_not_loaded(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    with pytest.raises(ModelAccessDeniedProxyException):
        await _common_checks("gpt-a", None, UserAPIKeyAuth(token="k1", user_id="u1"))


@pytest.mark.asyncio
async def test_proxy_admin_personal_key_is_not_capped(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    user: Final = LiteLLM_UserTable(user_id="admin", teams=[], user_role=LitellmUserRoles.PROXY_ADMIN.value)
    token: Final = UserAPIKeyAuth(token="k1", user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    assert await _common_checks("gpt-c", user, token) is True


@pytest.mark.asyncio
async def test_personal_key_unrestricted_when_flag_off(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", False)
    user: Final = LiteLLM_UserTable(user_id="u1", teams=[])
    assert await _common_checks("gpt-c", user, UserAPIKeyAuth(token="k1", user_id="u1")) is True


@pytest.mark.asyncio
async def test_personal_key_fallbacks_are_capped_by_owner_teams(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a"])
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")

    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        await _common_checks("gpt-a", owner, token, fallbacks=("gpt-c",))
    assert str(exc_info.value.code) == "403"
    assert await _common_checks("gpt-a", owner, token, fallbacks=("shared",)) is True


@pytest.mark.asyncio
async def test_personal_key_list_models_can_be_allowed_by_different_teams(
    monkeypatch: pytest.MonkeyPatch, teams: None
) -> None:
    _enable(monkeypatch)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a", "team-b"])
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")

    assert await _common_checks(["gpt-a", "gpt-b"], owner, token) is True


@pytest.mark.asyncio
async def test_personal_key_list_models_denies_the_unavailable_model(
    monkeypatch: pytest.MonkeyPatch, teams: None
) -> None:
    _enable(monkeypatch)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a", "team-b"])
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")

    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        await _common_checks(["gpt-a", "gpt-c"], owner, token)

    assert exc_info.value.code == "403"
    assert "gpt-c" in exc_info.value.message


@pytest.mark.asyncio
async def test_personal_key_fallbacks_are_unchanged_when_flag_off(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", False)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a"])
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")
    assert await _common_checks("gpt-a", owner, token, fallbacks=("gpt-c",)) is True


@pytest.mark.asyncio
async def test_personal_key_fallbacks_load_each_owner_team_once(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    get_team: Final = AsyncMock(wraps=auth_checks.get_team_object)
    monkeypatch.setattr(auth_checks, "get_team_object", get_team)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a", "team-b"])
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")

    assert await _common_checks("gpt-a", owner, token, fallbacks=("shared", "gpt-a")) is True
    assert sorted(call.kwargs["team_id"] for call in get_team.call_args_list) == ["team-a", "team-b"]


@pytest.mark.asyncio
async def test_team_key_is_not_capped_by_owner_teams(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    user: Final = LiteLLM_UserTable(user_id="u1", teams=[])
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1", team_id="team-open")
    assert await _common_checks("gpt-c", user, token, fallbacks=("gpt-c",)) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "teams_of_user, allowed",
    [
        (["team-missing", "team-a"], True),
        (["team-missing"], False),
        (["team-blocked"], False),
        (["team-blocked", "team-a"], True),
    ],
)
async def test_unloadable_or_blocked_team_grants_nothing(
    monkeypatch: pytest.MonkeyPatch, teams: None, teams_of_user: list[str], allowed: bool
) -> None:
    _enable(monkeypatch)
    user: Final = LiteLLM_UserTable(user_id="u1", teams=teams_of_user)
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")
    if allowed:
        assert await _common_checks("gpt-a", user, token) is True
    else:
        with pytest.raises(ModelAccessDeniedProxyException):
            await _common_checks("gpt-a", user, token)


@pytest.mark.asyncio
async def test_membership_change_applies_to_existing_key(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")
    assert await _common_checks("gpt-a", LiteLLM_UserTable(user_id="u1", teams=["team-a"]), token) is True
    with pytest.raises(ModelAccessDeniedProxyException):
        await _common_checks("gpt-a", LiteLLM_UserTable(user_id="u1", teams=["team-b"]), token)


@pytest.mark.asyncio
async def test_resolved_model_check_caps_personal_key(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a"])

    async def _get_user_object(**_: object) -> LiteLLM_UserTable:
        return owner

    monkeypatch.setattr(auth_checks, "get_user_object", _get_user_object)
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")

    await can_key_call_resolved_model(model="gpt-a", llm_model_list=None, valid_token=token, llm_router=_ROUTER)
    with pytest.raises(ModelAccessDeniedProxyException):
        await can_key_call_resolved_model(model="gpt-b", llm_model_list=None, valid_token=token, llm_router=_ROUTER)


@pytest.mark.asyncio
async def test_resolved_model_check_denies_when_personal_key_owner_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, teams: None
) -> None:
    _enable(monkeypatch)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    token: Final = UserAPIKeyAuth(token="k1", user_id="u-no-db")
    await proxy_server.user_api_key_cache.async_delete_cache("u-no-db")
    assert await proxy_server.user_api_key_cache.async_get_cache(key="u-no-db", model_type=LiteLLM_UserTable) is None

    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        await can_key_call_resolved_model(model="gpt-a", llm_model_list=None, valid_token=token, llm_router=_ROUTER)
    assert str(exc_info.value.code) == "403"


async def _list_models(monkeypatch: pytest.MonkeyPatch, owner: LiteLLM_UserTable, token: UserAPIKeyAuth) -> list[str]:
    async def _get_user_object(**_: object) -> LiteLLM_UserTable:
        return owner

    monkeypatch.setattr(auth_checks, "get_user_object", _get_user_object)
    return await get_available_models_for_user(
        user_api_key_dict=token,
        llm_router=_ROUTER,
        general_settings={},
        user_model=None,
        prisma_client=MagicMock(),
        proxy_logging_obj=MagicMock(),
        user_api_key_cache=MagicMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize(
    "own_models, owner_teams, expected",
    [
        (["no-default-models"], ["team-a"], {"gpt-a", "shared"}),
        (["gpt-c"], [], {"gpt-c"}),
        (["gpt-c"], ["team-a"], {"gpt-a", "shared", "gpt-c"}),
        ([], [], {"gpt-a", "gpt-b", "shared", "gpt-c"}),
        ([], ["team-a"], {"gpt-a", "gpt-b", "shared", "gpt-c"}),
        (["no-default-models"], [], set()),
        (["no-default-models"], ["team-blocked"], set()),
        (["no-default-models"], ["team-missing"], set()),
        (["no-default-models"], ["team-exhausted"], set()),
        pytest.param(["no-default-models"], ["team-native-group"], {"gpt-a"}, id="team-model-access-group"),
    ],
)
async def test_dashboard_lists_and_calls_own_and_team_models(
    monkeypatch: pytest.MonkeyPatch,
    teams: None,
    flag: bool,
    own_models: list[str],
    owner_teams: list[str],
    expected: set[str],
) -> None:
    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", flag)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    owner: Final = LiteLLM_UserTable(user_id="u1", models=own_models, teams=owner_teams)
    token: Final = UserAPIKeyAuth(
        token="dashboard-key",
        user_id="u1",
        team_id=UI_TEAM_ID,
        models=[],
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    assert set(await _list_models(monkeypatch, owner, token)) == expected
    for model in ("gpt-a", "gpt-b", "shared", "gpt-c"):
        if model in expected:
            await auth_checks.can_key_call_model(
                model=model, llm_model_list=None, valid_token=token, llm_router=_ROUTER
            )
            await can_key_call_resolved_model(model=model, llm_model_list=None, valid_token=token, llm_router=_ROUTER)
            assert await _common_checks(model, owner, token) is True
        else:
            with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
                await can_key_call_resolved_model(
                    model=model, llm_model_list=None, valid_token=token, llm_router=_ROUTER
                )
            assert str(exc_info.value.code) == "403"
            with pytest.raises(ModelAccessDeniedProxyException):
                await auth_checks.can_key_call_model(
                    model=model, llm_model_list=None, valid_token=token, llm_router=_ROUTER
                )
            with pytest.raises(ModelAccessDeniedProxyException):
                await _common_checks(model, owner, token)


@pytest.mark.asyncio
async def test_admin_dashboard_models_are_not_capped_by_teams(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    owner: Final = LiteLLM_UserTable(user_id="admin", teams=["team-a"], user_role=LitellmUserRoles.PROXY_ADMIN.value)
    token: Final = UserAPIKeyAuth(
        token="dashboard-key",
        user_id="admin",
        team_id=UI_TEAM_ID,
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    assert set(await _list_models(monkeypatch, owner, token)) == {"gpt-a", "gpt-b", "shared", "gpt-c"}
    await can_key_call_resolved_model(model="gpt-c", llm_model_list=None, valid_token=token, llm_router=_ROUTER)


@pytest.mark.parametrize("flag", [False, True])
def test_dashboard_does_not_inherit_mcp_or_agent_team_grants(monkeypatch: pytest.MonkeyPatch, flag: bool) -> None:
    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", flag)
    assert (
        auth_checks.personal_key_inherits_team_grants(
            UserAPIKeyAuth(token="dashboard-key", user_id="u1", team_id=UI_TEAM_ID)
        )
        is False
    )


@pytest.mark.asyncio
async def test_dashboard_lists_and_calls_team_access_group_models(
    monkeypatch: pytest.MonkeyPatch,
    teams: None,
) -> None:
    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", False)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(auth_checks, "get_models_from_access_groups", AsyncMock(return_value=["gpt-b"]))
    owner: Final = LiteLLM_UserTable(user_id="u1", models=["no-default-models"], teams=["team-group"])
    token: Final = UserAPIKeyAuth(
        token="dashboard-key",
        user_id="u1",
        team_id=UI_TEAM_ID,
        models=owner.models,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    assert await _list_models(monkeypatch, owner, token) == ["gpt-b"]
    await can_key_call_resolved_model(model="gpt-b", llm_model_list=None, valid_token=token, llm_router=_ROUTER)
    with pytest.raises(ModelAccessDeniedProxyException):
        await can_key_call_resolved_model(model="gpt-c", llm_model_list=None, valid_token=token, llm_router=_ROUTER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_models, expected",
    [
        ([], {"gpt-a", "gpt-b", "shared"}),
        (["gpt-a", "gpt-c"], {"gpt-a"}),
    ],
)
async def test_model_listing_matches_team_derived_access(
    monkeypatch: pytest.MonkeyPatch, teams: None, key_models: list[str], expected: set[str]
) -> None:
    _enable(monkeypatch)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a", "team-b"])
    listed: Final = await _list_models(monkeypatch, owner, UserAPIKeyAuth(token="k1", user_id="u1", models=key_models))
    assert set(listed) == expected


@pytest.mark.asyncio
async def test_model_listing_empty_without_teams(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=[])
    assert await _list_models(monkeypatch, owner, UserAPIKeyAuth(token="k1", user_id="u1")) == []


@pytest.mark.asyncio
async def test_model_listing_unchanged_when_flag_off(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", False)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=[])
    listed: Final = await _list_models(monkeypatch, owner, UserAPIKeyAuth(token="k1", user_id="u1"))
    assert set(listed) == {"gpt-a", "gpt-b", "shared", "gpt-c"}


@pytest.mark.asyncio
async def test_model_listing_loads_each_team_once(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    get_team: Final = AsyncMock(wraps=auth_checks.get_team_object)
    monkeypatch.setattr(auth_checks, "get_team_object", get_team)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a", "team-b", "team-a"])
    listed: Final = await _list_models(monkeypatch, owner, UserAPIKeyAuth(token="k1", user_id="u1"))
    assert set(listed) == {"gpt-a", "gpt-b", "shared"}
    assert sorted(call.kwargs["team_id"] for call in get_team.call_args_list) == ["team-a", "team-b"]


@pytest.mark.asyncio
async def test_team_check_error_grants_nothing_for_that_team(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    check_team: Final = auth_checks.can_team_access_model

    async def _failing_for_team_b(model: str, team_object: LiteLLM_TeamTableCachedObj, **_: object) -> bool:
        if team_object.team_id == "team-b":
            raise RuntimeError("access group lookup failed")
        return await check_team(model=model, team_object=team_object, llm_router=_ROUTER)

    monkeypatch.setattr(auth_checks, "can_team_access_model", _failing_for_team_b)
    owner: Final = LiteLLM_UserTable(user_id="u1", teams=["team-a", "team-b"])
    token: Final = UserAPIKeyAuth(token="k1", user_id="u1")
    assert await _common_checks("gpt-a", owner, token) is True
    with pytest.raises(ModelAccessDeniedProxyException):
        await _common_checks("gpt-b", owner, token)


async def _owner_lookup_fails(**_: object) -> LiteLLM_UserTable:
    raise RuntimeError("database unavailable")


async def _owner_missing(**_: object) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("get_owner", [_owner_lookup_fails, _owner_missing])
async def test_model_listing_empty_when_owner_not_loaded(
    monkeypatch: pytest.MonkeyPatch, teams: None, get_owner: object
) -> None:
    _enable(monkeypatch)
    monkeypatch.setattr(auth_checks, "get_user_object", get_owner)
    listed: Final = await get_available_models_for_user(
        user_api_key_dict=UserAPIKeyAuth(token="k1", user_id="u1"),
        llm_router=_ROUTER,
        general_settings={},
        user_model=None,
        prisma_client=MagicMock(),
        proxy_logging_obj=MagicMock(),
        user_api_key_cache=MagicMock(),
    )
    assert listed == []


@pytest.mark.asyncio
async def test_model_listing_empty_without_database(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    _enable(monkeypatch)
    listed: Final = await get_available_models_for_user(
        user_api_key_dict=UserAPIKeyAuth(token="k1", user_id="u1"),
        llm_router=_ROUTER,
        general_settings={},
        user_model=None,
        prisma_client=None,
        proxy_logging_obj=MagicMock(),
        user_api_key_cache=MagicMock(),
    )
    assert listed == []
