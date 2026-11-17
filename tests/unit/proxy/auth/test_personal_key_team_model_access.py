from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    ModelAccessDeniedProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth import auth_checks
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model, common_checks
from litellm.proxy.auth.user_api_key_auth import _authorize_authenticated_request
from litellm.proxy.auth.fallback_model_access import RouterFallbackAccessCheck
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup, move_guardrails_to_metadata
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    _check_dashboard_request_billing,
    _fallback_grant,
    global_then_key_alias_target,
)
from litellm.proxy.policy_engine import policy_registry
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.spend_tracking.budget_reservation import get_reserved_counter_keys, release_budget_reservation
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.router_api import RoutingContext
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import get_available_models_for_user

_TEAMS: Final[Mapping[str, LiteLLM_TeamTableCachedObj]] = {
    UI_TEAM_ID: LiteLLM_TeamTableCachedObj(team_id=UI_TEAM_ID, models=[]),
    "team-a": LiteLLM_TeamTableCachedObj(
        team_id="team-a", models=["gpt-a", "shared"], members_with_roles=[Member(user_id="u1", role="user")]
    ),
    "team-b": LiteLLM_TeamTableCachedObj(
        team_id="team-b", models=["gpt-b", "shared"], members_with_roles=[Member(user_id="u1", role="user")]
    ),
    "team-open": LiteLLM_TeamTableCachedObj(team_id="team-open", models=[]),
    "team-blocked": LiteLLM_TeamTableCachedObj(team_id="team-blocked", models=["gpt-a"], blocked=True),
    "team-group": LiteLLM_TeamTableCachedObj(
        team_id="team-group", models=["no-default-models"], access_group_ids=["team-models"]
    ),
    "team-exhausted": LiteLLM_TeamTableCachedObj(team_id="team-exhausted", models=["gpt-a"], spend=2, max_budget=1),
    "team-native-group": LiteLLM_TeamTableCachedObj(team_id="team-native-group", models=["team-access"]),
    "team-billing-a": LiteLLM_TeamTableCachedObj(
        team_id="team-billing-a", models=["shared", "gpt-b"], max_budget=10, spend=2
    ),
    "team-billing-b": LiteLLM_TeamTableCachedObj(team_id="team-billing-b", models=["shared"], max_budget=10, spend=3),
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


async def _list_models(
    monkeypatch: pytest.MonkeyPatch,
    owner: LiteLLM_UserTable | None,
    token: UserAPIKeyAuth,
    *,
    include_model_access_groups: bool = False,
    only_model_access_groups: bool = False,
    team_id: str | None = None,
    llm_router: litellm.Router = _ROUTER,
    return_wildcard_routes: bool = False,
) -> list[str]:
    async def _get_user_object(**_: object) -> LiteLLM_UserTable | None:
        return owner

    monkeypatch.setattr(auth_checks, "get_user_object", _get_user_object)
    return await get_available_models_for_user(
        user_api_key_dict=token,
        llm_router=llm_router,
        general_settings={},
        user_model=None,
        prisma_client=MagicMock(),
        proxy_logging_obj=MagicMock(),
        user_api_key_cache=MagicMock(),
        include_model_access_groups=include_model_access_groups,
        only_model_access_groups=only_model_access_groups,
        team_id=team_id,
        return_wildcard_routes=return_wildcard_routes,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("return_wildcard_routes", [False, True])
async def test_dashboard_listing_respects_wildcard_exclusion(
    monkeypatch: pytest.MonkeyPatch, teams: None, return_wildcard_routes: bool
) -> None:
    router: Final = litellm.Router(model_list=[
        {"model_name": name, "litellm_params": {"model": name, "api_key": "test-key"}}
        for name in ("openai/gpt-5.6", "openai/*")
    ])
    owner: Final = LiteLLM_UserTable(user_id="u1", models=[], teams=[])
    token: Final = UserAPIKeyAuth(
        token="dashboard-key", user_id="u1", team_id=UI_TEAM_ID, models=[],
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    listed: Final = await _list_models(
        monkeypatch, owner, token, llm_router=router, return_wildcard_routes=return_wildcard_routes
    )
    ordinary: Final = await _list_models(
        monkeypatch, owner, token.model_copy(update={"team_id": None, "user_role": LitellmUserRoles.PROXY_ADMIN}),
        llm_router=router, return_wildcard_routes=return_wildcard_routes,
    )
    assert set(listed) == set(ordinary)
    assert ("openai/*" in listed) is return_wildcard_routes


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
@pytest.mark.parametrize(
    "own_models, owner_teams, expected",
    [
        ([], [], {"team-access"}),
        (["gpt-c"], [], set()),
        (["no-default-models"], [], set()),
        (["no-default-models"], ["team-native-group"], {"team-access"}),
    ],
)
async def test_dashboard_group_only_listing_does_not_restore_model_names(
    monkeypatch: pytest.MonkeyPatch,
    teams: None,
    own_models: list[str],
    owner_teams: list[str],
    expected: set[str],
) -> None:
    owner: Final = LiteLLM_UserTable(user_id="u1", models=own_models, teams=owner_teams)
    token: Final = UserAPIKeyAuth(
        token="dashboard-key", user_id="u1", team_id=UI_TEAM_ID, models=[], user_role=LitellmUserRoles.INTERNAL_USER
    )
    assert (
        set(
            await _list_models(
                monkeypatch, owner, token, include_model_access_groups=True, only_model_access_groups=True
            )
        )
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("own_models", [[], ["gpt-c"], ["no-default-models"]])
@pytest.mark.parametrize("team_id, expected", [("team-a", {"gpt-a", "shared"}), ("team-b", {"gpt-b", "shared"})])
async def test_dashboard_team_listing_excludes_own_models_and_other_teams(
    monkeypatch: pytest.MonkeyPatch,
    teams: None,
    own_models: list[str],
    team_id: str,
    expected: set[str],
) -> None:
    owner: Final = LiteLLM_UserTable(user_id="u1", models=own_models, teams=["team-a", "team-b"])
    token: Final = UserAPIKeyAuth(
        token="dashboard-key", user_id="u1", team_id=UI_TEAM_ID, models=[], user_role=LitellmUserRoles.INTERNAL_USER
    )
    assert set(await _list_models(monkeypatch, owner, token, team_id=team_id)) == expected


@pytest.mark.asyncio
async def test_dashboard_validated_team_listing_does_not_require_owner_lookup(
    monkeypatch: pytest.MonkeyPatch, teams: None
) -> None:
    token: Final = UserAPIKeyAuth(
        token="dashboard-key", user_id="u1", team_id=UI_TEAM_ID, models=[], user_role=LitellmUserRoles.INTERNAL_USER
    )
    assert set(await _list_models(monkeypatch, None, token, team_id="team-a")) == {"gpt-a", "shared"}


@pytest.mark.asyncio
async def test_dashboard_team_listing_requires_membership(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    owner: Final = LiteLLM_UserTable(user_id="u1", models=[], teams=["team-a"])
    token: Final = UserAPIKeyAuth(
        token="dashboard-key", user_id="u1", team_id=UI_TEAM_ID, models=[], user_role=LitellmUserRoles.INTERNAL_USER
    )
    with pytest.raises(HTTPException) as exc_info:
        await _list_models(monkeypatch, owner, token, team_id="team-native-group")
    assert exc_info.value.status_code == 403


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


def _billing_database(monkeypatch: pytest.MonkeyPatch, owner: LiteLLM_UserTable) -> None:
    cache: Final = UserApiKeyCache()
    cache.set_cache(owner.user_id, owner, model_type=LiteLLM_UserTable)
    for team in _TEAMS.values():
        cache.set_cache(f"team_id:{team.team_id}", team, model_type=LiteLLM_TeamTableCachedObj)
    database: Final = MagicMock()
    database.db.litellm_teammembership.find_unique = AsyncMock(return_value=None)

    async def team_row(where: Mapping[str, object], **_: object) -> LiteLLM_TeamTableCachedObj | None:
        return _TEAMS.get(str(where["team_id"]))

    database.db.litellm_teamtable.find_unique = AsyncMock(side_effect=team_row)
    database.db.litellm_usertable.find_unique = AsyncMock(return_value=owner)
    database.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    database.get_data = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "llm_router", _ROUTER)
    monkeypatch.setattr(proxy_server, "llm_model_list", _ROUTER.model_list)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "spend_counter_store", SpendCounterStore(connect(None).spend_counters()))
    monkeypatch.setattr(proxy_server, "master_key", "test-master")
    monkeypatch.setattr(proxy_server, "user_custom_auth", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "own_models, denied",
    [(["no-default-models"], True), (["gpt-a"], True), (["gpt-a", "shared"], False), ([], False)],
)
async def test_dashboard_batches_require_own_model_grants(
    monkeypatch: pytest.MonkeyPatch, teams: None, own_models: list[str], denied: bool
) -> None:
    owner: Final = LiteLLM_UserTable(user_id="u1", models=own_models, teams=["team-a"])
    _billing_database(monkeypatch, owner)
    token: Final = UserAPIKeyAuth(
        api_key="dashboard-key", user_id="u1", team_id=UI_TEAM_ID, models=[],
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    request: Final = Request({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    try:
        if denied:
            with pytest.raises(HTTPException) as error:
                await _authorize_authenticated_request(
                    token, request, {"model": "gpt-a,shared"}, "/chat/completions", "dashboard-key",
                    force_virtual_key_checks=True,
                )
            assert error.value.status_code == 403
        else:
            billed: Final = await _authorize_authenticated_request(
                token, request, {"model": "gpt-a,shared"}, "/chat/completions", "dashboard-key",
                force_virtual_key_checks=True,
            )
            assert (billed or token).team_id == UI_TEAM_ID
    finally:
        if token.budget_reservation is not None:
            await release_budget_reservation(token.budget_reservation)


@pytest.mark.asyncio
async def test_dashboard_admitted_team_reservation_does_not_deny_same_team_fallback(
    monkeypatch: pytest.MonkeyPatch, teams: None
) -> None:
    window_limit: Final = 0.000001
    limited: Final = _TEAMS["team-billing-a"].model_copy(
        update={"budget_limits": [{"budget_duration": "30d", "max_budget": window_limit}]}
    )
    monkeypatch.setitem(_TEAMS, limited.team_id, limited)
    owner: Final = LiteLLM_UserTable(user_id="u1", models=["no-default-models"], teams=[limited.team_id])
    _billing_database(monkeypatch, owner)
    counter: Final = auth_checks.team_window_counter(limited.team_id, "30d")
    proxy_server.prisma_client.db.litellm_budgetwindowspend.find_unique = AsyncMock(return_value=None)
    proxy_server.prisma_client.db.litellm_spendlogs.group_by = AsyncMock(return_value=[])
    proxy_server.spend_counter_store.set_local(counter, 0.0)
    token: Final = UserAPIKeyAuth(
        api_key="dashboard-key", user_id="u1", team_id=UI_TEAM_ID, models=[],
        user_role=LitellmUserRoles.INTERNAL_USER, max_budget=20, spend=0,
    )
    request: Final = Request({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    billed: Final = await _authorize_authenticated_request(
        token, request, {"model": "shared"}, "/chat/completions", "dashboard-key", force_virtual_key_checks=True
    )
    assert billed is not None and billed.budget_reservation is not None
    reservation: Final = billed.budget_reservation
    assert reservation["reserved_cost"] == pytest.approx(window_limit)
    assert proxy_server.spend_counter_store.get_local(counter) == pytest.approx(window_limit)
    try:
        assert await RouterFallbackAccessCheck(is_enforced=lambda: True)(
            model="gpt-b", request_kwargs={"metadata": {"user_api_key_auth": billed}}, ctx=RoutingContext(_ROUTER)
        ) is True
        await _check_dashboard_request_billing(
            SimpleNamespace(data={"model": "gpt-b"}, dashboard_fallback_model="gpt-b"), billed, _ROUTER
        )
        assert billed.team_id == limited.team_id
        assert billed.budget_reservation is reservation
    finally:
        await release_budget_reservation(reservation)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_unavailable", [False, True])
async def test_dashboard_own_primary_alias_keeps_its_admitted_billing_identity(
    monkeypatch: pytest.MonkeyPatch, teams: None, owner_unavailable: bool
) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {"caller-alias": "shared"})
    owner: Final = LiteLLM_UserTable(user_id="u1", models=["caller-alias"], teams=["team-billing-a"])
    _billing_database(monkeypatch, owner)
    token: Final = UserAPIKeyAuth(
        api_key="dashboard-key", user_id="u1", team_id=UI_TEAM_ID, models=[],
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    request: Final = Request({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    billed: Final = await _authorize_authenticated_request(
        token, request, {"model": "caller-alias"}, "/chat/completions", "dashboard-key", force_virtual_key_checks=True
    )
    assert billed is not None
    if owner_unavailable:
        proxy_server.user_api_key_cache.delete_cache(owner.user_id)
        proxy_server.prisma_client.db.litellm_usertable.find_unique = AsyncMock(side_effect=RuntimeError("offline"))
    prepared: Final = ProxyBaseLLMRequestProcessing(
        data={"model": global_then_key_alias_target("caller-alias", billed)}
    )
    await _check_dashboard_request_billing(prepared, billed, _ROUTER)
    assert billed.team_id == UI_TEAM_ID
    proxy_server.user_api_key_cache.set_cache(owner.user_id, owner, model_type=LiteLLM_UserTable)
    assert await RouterFallbackAccessCheck(is_enforced=lambda: True)(
        model="shared", request_kwargs={"metadata": {"user_api_key_auth": billed}}, ctx=RoutingContext(_ROUTER)
    ) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "own_models, owner_teams, model, expected_team",
    [
        (["no-default-models"], ["team-billing-b", "team-billing-a"], "shared", "team-billing-a"),
        (["no-default-models"], ["team-native-group"], "gpt-a", "team-native-group"),
        (["no-default-models"], ["team-exhausted", "team-a"], "gpt-a", "team-a"),
        (["shared"], ["team-b", "team-a"], "shared", UI_TEAM_ID),
        ([], ["team-a"], "shared", UI_TEAM_ID),
    ],
)
async def test_dashboard_spend_row_and_budget_use_the_same_granting_team(
    monkeypatch: pytest.MonkeyPatch,
    teams: None,
    own_models: list[str],
    owner_teams: list[str],
    model: str,
    expected_team: str,
) -> None:
    owner: Final = LiteLLM_UserTable(user_id="u1", models=own_models, teams=owner_teams)
    _billing_database(monkeypatch, owner)
    token: Final = UserAPIKeyAuth(
        api_key="dashboard-key",
        user_id="u1",
        team_id=UI_TEAM_ID,
        models=[],
        user_role=LitellmUserRoles.INTERNAL_USER,
        max_budget=20,
        spend=0,
    )
    request: Final = Request({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    admitted: Final = await _authorize_authenticated_request(
        token, request, {"model": model}, "/chat/completions", "dashboard-key", force_virtual_key_checks=True
    )
    billed: Final = admitted or token
    metadata: Final = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(billed)
    timestamp: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row: Final = get_logging_payload(
        kwargs={"model": model, "response_cost": 0.25, "litellm_params": {"metadata": metadata}},
        response_obj={},
        start_time=timestamp,
        end_time=timestamp,
    )
    assert (billed.team_id, metadata["user_api_key_team_id"], row["team_id"]) == (expected_team,) * 3
    assert row["spend"] == 0.25
    assert billed.team_spend == _TEAMS[expected_team].spend
    assert billed.max_budget == token.max_budget
    assert token.team_id == UI_TEAM_ID
    assert billed.budget_reservation is request.state.budget_reservation
    assert billed.budget_reservation is not None
    if expected_team == "team-billing-a":
        assert any(expected_team in key for key in get_reserved_counter_keys(billed.budget_reservation))
        assert not any("team-billing-b" in key for key in get_reserved_counter_keys(billed.budget_reservation))
    await release_budget_reservation(billed.budget_reservation)


@pytest.mark.asyncio
async def test_dashboard_denied_fallback_keeps_its_existing_reservation(
    monkeypatch: pytest.MonkeyPatch, teams: None
) -> None:
    _billing_database(monkeypatch, LiteLLM_UserTable(user_id="u1", models=["no-default-models"], teams=["team-a"]))
    token: Final = UserAPIKeyAuth(
        api_key="dashboard-key",
        user_id="u1",
        team_id=UI_TEAM_ID,
        user_role=LitellmUserRoles.INTERNAL_USER,
        max_budget=20,
        spend=0,
    )
    request: Final = Request({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    billed: Final = await _authorize_authenticated_request(
        token, request, {"model": "gpt-a"}, "/chat/completions", "dashboard-key", force_virtual_key_checks=True
    )
    assert billed is not None
    reservation: Final = billed.budget_reservation
    metadata: Final = {"user_api_key_auth": billed, "user_api_key_team_id": billed.team_id}
    assert (
        await RouterFallbackAccessCheck(is_enforced=lambda: True)(
            model="gpt-c", request_kwargs={"metadata": metadata}, ctx=RoutingContext(_ROUTER)
        )
        is False
    )
    assert metadata["user_api_key_team_id"] == "team-a"
    assert request.state.budget_reservation is billed.budget_reservation is reservation
    assert reservation is not None and reservation["finalized"] is False
    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_dashboard_billing_denies_an_exhausted_only_team(monkeypatch: pytest.MonkeyPatch, teams: None) -> None:
    owner: Final = LiteLLM_UserTable(user_id="u1", models=["no-default-models"], teams=["team-exhausted"])
    _billing_database(monkeypatch, owner)
    token: Final = UserAPIKeyAuth(user_id="u1", team_id=UI_TEAM_ID, user_role=LitellmUserRoles.INTERNAL_USER)
    with pytest.raises(ModelAccessDeniedProxyException):
        await auth_checks.dashboard_model_billing_auth(token, "gpt-a", _ROUTER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "team_id,role", [(None, LitellmUserRoles.INTERNAL_USER), (UI_TEAM_ID, LitellmUserRoles.PROXY_ADMIN)]
)
async def test_dashboard_model_billing_keeps_admin_and_personal_keys_unchanged(
    monkeypatch: pytest.MonkeyPatch, teams: None, team_id: str | None, role: LitellmUserRoles
) -> None:
    _enable(monkeypatch)
    _billing_database(monkeypatch, LiteLLM_UserTable(user_id="u1", models=["no-default-models"], teams=["team-a"]))
    token: Final = UserAPIKeyAuth(user_id="u1", team_id=team_id, user_role=role)
    assert await auth_checks.dashboard_model_billing_auth(token, "gpt-a", _ROUTER) is token


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("enforced", [True, False])
@pytest.mark.parametrize(
    "own_models,initial,target,expected_team,allowed",
    [
        (["gpt-c"], "shared", "gpt-a", "team-billing-a", False),
        (["gpt-c"], "shared", "gpt-c", "team-billing-a", False),
        (["gpt-c"], "gpt-c", "shared", UI_TEAM_ID, False),
        (["gpt-c"], "shared", "gpt-b", "team-billing-a", True),
        (["gpt-c", "gpt-b"], "gpt-c", "gpt-b", UI_TEAM_ID, True),
    ],
)
async def test_dashboard_fallback_stays_with_the_original_billing_team(
    monkeypatch: pytest.MonkeyPatch,
    teams: None,
    metadata_field: str,
    enforced: bool,
    own_models: Sequence[str],
    initial: str,
    target: str,
    expected_team: str,
    allowed: bool,
) -> None:
    owner: Final = LiteLLM_UserTable(
        user_id="u1", models=list(own_models), teams=["team-billing-b", "team-native-group", "team-billing-a"]
    )
    _billing_database(monkeypatch, owner)
    monkeypatch.setattr(proxy_server, "general_settings", {"enforce_fallback_model_access": enforced})
    token: Final = UserAPIKeyAuth(
        api_key="dashboard-key",
        user_id="u1",
        team_id=UI_TEAM_ID,
        models=[],
        user_role=LitellmUserRoles.INTERNAL_USER,
        max_budget=20,
        spend=0,
    )
    request: Final = Request({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    billed: Final = await _authorize_authenticated_request(
        token, request, {"model": initial}, "/chat/completions", "dashboard-key", force_virtual_key_checks=True
    )
    assert billed is not None
    previous: Final = billed.budget_reservation
    assert previous is not None
    metadata: Final = {
        **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(billed),
        "user_api_key_auth": LiteLLMProxyRequestSetup.auth_for_request_metadata(billed),
        "user_api_key_budget_reservation": previous,
    }
    original_metadata: Final = dict(metadata)
    assert (
        await RouterFallbackAccessCheck(is_enforced=lambda: enforced)(
            model=target, request_kwargs={metadata_field: metadata}, ctx=RoutingContext(_ROUTER)
        )
        is allowed
    )
    assert (
        await _fallback_grant(target, billed, _ROUTER, proxy_server.proxy_logging_obj, initial) is not None
    ) is allowed
    assert billed.team_id == metadata["user_api_key_team_id"] == expected_team
    assert metadata == original_metadata
    assert previous["finalized"] is False
    assert metadata["user_api_key_budget_reservation"] is billed.budget_reservation is request.state.budget_reservation
    assert billed.budget_reservation is previous
    assert not any("team-billing-b" in key for key in get_reserved_counter_keys(billed.budget_reservation))
    await release_budget_reservation(billed.budget_reservation)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "team_config,expected_guardrails",
    [
        ({}, frozenset()),
        ({"guardrails": ["team-input"]}, frozenset({"team-input"})),
        (
            {"guardrails": ["team-input"], "policies": ["team-policy"]},
            frozenset({"team-input", "policy-input"}),
        ),
    ],
)
async def test_dashboard_primary_call_inherits_team_guardrails_and_policies_like_a_team_key(
    monkeypatch: pytest.MonkeyPatch,
    teams: None,
    team_config: Mapping[str, object],
    expected_guardrails: frozenset[str],
) -> None:
    _billing_database(
        monkeypatch, LiteLLM_UserTable(user_id="u1", models=["no-default-models"], teams=["team-billing-a"])
    )
    guarded_team: Final = _TEAMS["team-billing-a"].model_copy(update={"metadata": team_config})
    proxy_server.user_api_key_cache.set_cache(
        "team_id:team-billing-a", guarded_team, model_type=LiteLLM_TeamTableCachedObj
    )
    registry: Final = policy_registry.PolicyRegistry()
    registry.load_policies({"team-policy": {"guardrails": {"add": ["policy-input"]}}})
    monkeypatch.setattr(policy_registry, "_policy_registry", registry)
    monkeypatch.setattr(proxy_server, "premium_user", True)

    async def guarded_team_row(team_id: str, **_: object) -> LiteLLM_TeamTableCachedObj:
        return guarded_team if team_id == guarded_team.team_id else _TEAMS[team_id]

    monkeypatch.setattr(auth_checks, "get_team_object", guarded_team_row)
    token: Final = UserAPIKeyAuth(
        api_key="dashboard-key", user_id="u1", team_id=UI_TEAM_ID, user_role=LitellmUserRoles.INTERNAL_USER
    )
    request: Final = Request({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
    billed: Final = await _authorize_authenticated_request(
        token, request, {"model": "shared"}, "/chat/completions", "dashboard-key", force_virtual_key_checks=True
    )
    assert billed is not None
    team_key: Final = UserAPIKeyAuth(
        api_key="team-key", user_id="u1", team_id=guarded_team.team_id, team_metadata=guarded_team.metadata
    )
    dashboard_data: Final = {"model": "shared", "metadata": {}}
    team_data: Final = {"model": "shared", "metadata": {}}
    await move_guardrails_to_metadata(dashboard_data, "metadata", billed)
    await move_guardrails_to_metadata(team_data, "metadata", team_key)
    assert dashboard_data == team_data
    assert frozenset(dashboard_data["metadata"].get("guardrails", [])) == expected_guardrails
    await release_budget_reservation(billed.budget_reservation)
