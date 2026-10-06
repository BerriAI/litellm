from collections.abc import Mapping
from typing import Final, TypedDict
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    TeamObjectLoader,
    can_caller_call_search_tool,
    check_unregistered_search_fallback,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, object_permission_cache_key

_DENY_ON: Final = {"search_tool_deny_by_default": True}
_KEY: Final = ProxyErrorTypes.key_search_tool_access_denied
_TEAM: Final = ProxyErrorTypes.team_search_tool_access_denied
_USER: Final = ProxyErrorTypes.user_search_tool_access_denied
_LEGACY: Final = ProxyErrorTypes.key_model_access_denied
_TEAM_LOAD_FAILS: Final = "team-load-fails"
_NO_ROW: Final = "no-row"


class CallerFields(TypedDict, total=False):
    virtual_key: bool
    key_tools: list[str] | None | str
    team_id: str | None
    user_id: str | None
    user_role: LitellmUserRoles | None
    api_key: str


def _grant(search_tools: list[str] | None, object_permission_id: str = "op") -> LiteLLM_ObjectPermissionTable:
    return LiteLLM_ObjectPermissionTable(object_permission_id=object_permission_id, search_tools=search_tools)


def _caller(
    virtual_key: bool = True,
    key_tools: list[str] | None | str = _NO_ROW,
    team_id: str | None = None,
    user_id: str | None = "user-1",
    user_role: LitellmUserRoles | None = LitellmUserRoles.INTERNAL_USER,
    api_key: str = "sk-caller",
) -> UserAPIKeyAuth:
    token: Final = UserAPIKeyAuth(
        api_key=api_key,
        user_id=user_id,
        user_role=user_role,
        team_id=team_id,
        object_permission_id=None if key_tools == _NO_ROW else "op-key",
        object_permission=None if key_tools == _NO_ROW else _grant(key_tools, "op-key"),
    )
    token.via_virtual_key = virtual_key
    return token


def _team_loader(team_tools: list[str] | None | str) -> TeamObjectLoader:
    async def load() -> LiteLLM_TeamTable | None:
        if team_tools == _TEAM_LOAD_FAILS:
            raise ProxyException(message="team lookup failed", type="auth_error", param="team_id", code=404)
        if team_tools == _NO_ROW:
            return LiteLLM_TeamTable(team_id="team-1")
        return LiteLLM_TeamTable(
            team_id="team-1", object_permission_id="op-team", object_permission=_grant(team_tools, "op-team")
        )

    return load


async def _no_team() -> None:
    return None


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> UserApiKeyCache:
    user_api_key_cache: Final = UserApiKeyCache()
    monkeypatch.setattr(proxy_server, "user_api_key_cache", user_api_key_cache)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    return user_api_key_cache


def _cache_user(cache: UserApiKeyCache, user_tools: list[str] | None) -> None:
    cache.set_cache(key="user-1", value=LiteLLM_UserTable(user_id="user-1", object_permission_id="op-user"))
    cache.set_cache(key=object_permission_cache_key("op-user"), value=_grant(user_tools, "op-user"))


async def _denied_by(
    general_settings: Mapping[str, object], caller: UserAPIKeyAuth, load_team: TeamObjectLoader = _no_team
) -> ProxyErrorTypes | None:
    try:
        await can_caller_call_search_tool("search-a", caller, general_settings, load_team)
    except ProxyException as e:
        denial = e
    else:
        return None
    assert (denial.code, denial.param) == ("403", "search_tool_name")
    return ProxyErrorTypes(denial.type)


@pytest.mark.parametrize(
    "general_settings",
    [{}, {"search_tool_deny_by_default": False}],
    ids=["omitted", "false"],
)
@pytest.mark.parametrize(
    "caller, team_tools, expected",
    [
        pytest.param(CallerFields(), None, None, id="no grants anywhere"),
        pytest.param(CallerFields(key_tools=[]), None, None, id="empty key list"),
        pytest.param(CallerFields(team_id="team-1"), [], None, id="empty team list"),
        pytest.param(CallerFields(key_tools=["search-b"]), None, _LEGACY, id="key allowlist excludes"),
        pytest.param(CallerFields(team_id="team-1"), ["search-b"], _LEGACY, id="team allowlist excludes"),
        pytest.param(CallerFields(key_tools=["search-a"], team_id="team-1"), ["search-a"], None, id="both allow"),
    ],
)
@pytest.mark.asyncio
async def test_search_tool_access_is_unchanged_without_search_tool_deny_by_default(
    cache: UserApiKeyCache,
    general_settings: dict[str, bool],
    caller: CallerFields,
    team_tools: list[str] | None,
    expected: ProxyErrorTypes | None,
):
    _cache_user(cache, ["search-b"])
    load_team: Final = _no_team if team_tools is None else _team_loader(team_tools)
    assert await _denied_by(general_settings, _caller(**caller), load_team) == expected


@pytest.mark.parametrize(
    "caller, team_tools, user_tools, expected",
    [
        pytest.param(CallerFields(), None, ["search-a"], _KEY, id="standalone key, no permission row"),
        pytest.param(CallerFields(key_tools=None), None, None, _KEY, id="standalone key, search_tools null"),
        pytest.param(CallerFields(key_tools=[]), None, None, _KEY, id="standalone key, empty grant"),
        pytest.param(CallerFields(key_tools=["search-b"]), None, None, _KEY, id="standalone key, grants another tool"),
        pytest.param(CallerFields(key_tools=["search-a"]), None, [], None, id="standalone key, key grant is enough"),
        pytest.param(CallerFields(key_tools=[]), None, ["search-a"], _KEY, id="standalone key, user cannot stand in"),
        pytest.param(
            CallerFields(key_tools=["search-a"], team_id="team-1"), ["search-a"], None, None, id="team key, both"
        ),
        pytest.param(CallerFields(key_tools=[], team_id="team-1"), ["search-a"], None, _KEY, id="team key, empty key"),
        pytest.param(
            CallerFields(key_tools=["search-a"], team_id="team-1"), [], None, _TEAM, id="team key, empty team"
        ),
        pytest.param(
            CallerFields(key_tools=["search-a"], team_id="team-1"), None, None, _TEAM, id="team key, team null"
        ),
        pytest.param(
            CallerFields(key_tools=["search-a"], team_id="team-1"), _NO_ROW, None, _TEAM, id="team key, team has no row"
        ),
        pytest.param(
            CallerFields(key_tools=["search-a"], team_id="team-1"),
            _TEAM_LOAD_FAILS,
            None,
            _TEAM,
            id="team key, team fails to load",
        ),
        pytest.param(
            CallerFields(key_tools=["search-a"], team_id="team-1"),
            ["search-b"],
            ["search-a"],
            _TEAM,
            id="team key, user cannot stand in for team",
        ),
        pytest.param(
            CallerFields(virtual_key=False, team_id="team-1"), ["search-a"], [], None, id="keyless team member"
        ),
        pytest.param(
            CallerFields(virtual_key=False, team_id="team-1"), [], ["search-a"], _TEAM, id="keyless, empty team"
        ),
        pytest.param(CallerFields(virtual_key=False), None, ["search-a"], None, id="keyless user, user grants"),
        pytest.param(CallerFields(virtual_key=False), None, None, _USER, id="keyless user, search_tools null"),
        pytest.param(CallerFields(virtual_key=False), None, [], _USER, id="keyless user, empty grant"),
        pytest.param(
            CallerFields(virtual_key=False, user_id="user-2"), None, None, _USER, id="keyless user fails to load"
        ),
        pytest.param(CallerFields(virtual_key=False, user_id=None), None, None, _USER, id="keyless caller, no user"),
        pytest.param(
            CallerFields(user_role=LitellmUserRoles.PROXY_ADMIN),
            None,
            None,
            _KEY,
            id="proxy admin virtual key not exempt",
        ),
        pytest.param(CallerFields(api_key=LITELLM_PROXY_MASTER_KEY_ALIAS), None, None, None, id="master key exempt"),
        pytest.param(
            CallerFields(team_id=UI_TEAM_ID, virtual_key=False), None, None, None, id="dashboard session exempt"
        ),
    ],
)
@pytest.mark.asyncio
async def test_search_tool_deny_by_default_requires_every_owning_identity_to_grant(
    cache: UserApiKeyCache,
    caller: CallerFields,
    team_tools: list[str] | str | None,
    user_tools: list[str] | None,
    expected: ProxyErrorTypes | None,
):
    if user_tools is not None:
        _cache_user(cache, user_tools)
    elif caller.get("user_id", "user-1") == "user-1":
        _cache_user(cache, None)
    load_team: Final = _no_team if team_tools is None and "team_id" not in caller else _team_loader(team_tools)
    assert await _denied_by(_DENY_ON, _caller(**caller), load_team) == expected


@pytest.mark.parametrize("value", ["true", "enabled", 1], ids=["string-true", "string", "int"])
@pytest.mark.asyncio
async def test_non_boolean_search_tool_deny_by_default_enables_the_policy(cache: UserApiKeyCache, value: object):
    assert await _denied_by({"search_tool_deny_by_default": value}, _caller(key_tools=[])) == _KEY


@pytest.mark.asyncio
async def test_search_tool_deny_by_default_denies_when_no_database_is_connected(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    assert await _denied_by(_DENY_ON, _caller(key_tools=["search-a"])) == _KEY


@pytest.mark.asyncio
async def test_search_tool_deny_by_default_reads_the_user_grant_on_every_call(cache: UserApiKeyCache):
    caller: Final = _caller(virtual_key=False)
    _cache_user(cache, ["search-a"])
    assert await _denied_by(_DENY_ON, caller) is None
    cache.set_cache(key=object_permission_cache_key("op-user"), value=_grant([], "op-user"))
    assert await _denied_by(_DENY_ON, caller) == _USER


@pytest.mark.asyncio
async def test_search_tool_deny_by_default_does_not_load_the_user_when_off(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
):
    get_user_object: Final = AsyncMock()
    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_user_object", get_user_object)
    assert await _denied_by({}, _caller(virtual_key=False)) is None
    get_user_object.assert_not_awaited()


@pytest.mark.parametrize(
    "general_settings, caller, expected",
    [
        pytest.param({}, _caller(), None, id="flag off"),
        pytest.param(_DENY_ON, _caller(), _KEY, id="virtual key"),
        pytest.param(_DENY_ON, _caller(virtual_key=False, team_id="team-1"), _TEAM, id="keyless team member"),
        pytest.param(_DENY_ON, _caller(virtual_key=False), _USER, id="keyless user"),
        pytest.param(_DENY_ON, _caller(user_role=LitellmUserRoles.PROXY_ADMIN), _KEY, id="proxy admin key"),
        pytest.param(_DENY_ON, _caller(api_key=LITELLM_PROXY_MASTER_KEY_ALIAS), None, id="master key"),
    ],
)
def test_unregistered_search_fallback_follows_search_tool_deny_by_default(
    general_settings: Mapping[str, object], caller: UserAPIKeyAuth, expected: ProxyErrorTypes | None
):
    if expected is None:
        assert check_unregistered_search_fallback(caller, general_settings) is True
        return
    with pytest.raises(ProxyException) as exc_info:
        check_unregistered_search_fallback(caller, general_settings)
    assert (exc_info.value.type, exc_info.value.code) == (expected, "403")
