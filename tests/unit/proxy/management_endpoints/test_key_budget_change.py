from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.models.verification_token import LiteLLM_VerificationToken
from litellm.proxy._types import (
    LiteLLM_TeamTableCachedObj,
    LitellmUserRoles,
    Member,
    ProxyException,
    UpdateKeyRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.key_budget_change import (
    KeyBudgetAdminOnly,
    KeyBudgetLoosened,
    KeyBudgetTightened,
    KeyBudgetUnchanged,
    classify_key_budget_change,
    resolve_self_serve_budget_policy,
)
from litellm.proxy.management_endpoints.key_management_endpoints import _validate_update_key_data

_OWNER = "owner-1"
_TEAM = "team-1"
_WEEKLY = {"budget_duration": "7d", "max_budget": 50.0, "reset_at": "2026-10-08T00:00:00+00:00"}


def _key(**overrides):
    return LiteLLM_VerificationToken(
        **{"token": "hashed-owner-key", "user_id": _OWNER, "created_by": _OWNER, "metadata": {}, **overrides}
    )


def _window(duration, cap):
    return {"budget_duration": duration, "max_budget": cap}


@pytest.mark.parametrize(
    ("update", "stored", "expected"),
    [
        ({"key_alias": "renamed"}, {"max_budget": 10.0}, KeyBudgetUnchanged()),
        ({"max_budget": 10.0}, {"max_budget": 10.0}, KeyBudgetUnchanged()),
        ({"max_budget": 5.0}, {"max_budget": 10.0}, KeyBudgetTightened()),
        ({"max_budget": 5.0}, {}, KeyBudgetTightened()),
        ({"max_budget": 20.0}, {"max_budget": 10.0}, KeyBudgetLoosened()),
        ({"max_budget": None}, {"max_budget": 10.0}, KeyBudgetLoosened()),
        ({"budget_limits": [_window("24h", 5.0)]}, {}, KeyBudgetTightened()),
        ({"budget_limits": [_WEEKLY, _window("24h", 5.0)]}, {"budget_limits": [_WEEKLY]}, KeyBudgetTightened()),
        ({"budget_limits": [_window("7d", 50.0)]}, {"budget_limits": [_WEEKLY]}, KeyBudgetTightened()),
        ({"budget_limits": [_window("7d", 40.0)]}, {"budget_limits": [_WEEKLY]}, KeyBudgetTightened()),
        ({"budget_limits": [_window("7d", 60.0)]}, {"budget_limits": [_WEEKLY]}, KeyBudgetLoosened()),
        ({"budget_limits": [_window("24h", 5.0)]}, {"budget_limits": [_WEEKLY]}, KeyBudgetLoosened()),
        ({"budget_limits": []}, {"budget_limits": [_WEEKLY]}, KeyBudgetLoosened()),
        ({"budget_limits": None}, {"budget_limits": [_WEEKLY]}, KeyBudgetLoosened()),
        ({"budget_limits": [_window("1d", 5.0)]}, {"budget_limits": [_window("24h", 5.0)]}, KeyBudgetLoosened()),
        (
            {"budget_limits": [_window("7d", 40.0)]},
            {"budget_limits": [_WEEKLY, _window("7d", 30.0)]},
            KeyBudgetLoosened(),
        ),
        (
            {"budget_limits": [_window("24h", 5.0)]},
            {"budget_limits": [{"budget_duration": "24h"}]},
            KeyBudgetLoosened(),
        ),
        (
            {"max_budget": 5.0, "budget_limits": []},
            {"max_budget": 10.0, "budget_limits": [_WEEKLY]},
            KeyBudgetLoosened(),
        ),
        ({"spend": 0.0}, {}, KeyBudgetAdminOnly()),
        ({"soft_budget": 1.0}, {}, KeyBudgetAdminOnly()),
        ({"soft_budget": None, "max_budget": 5.0}, {"max_budget": 10.0}, KeyBudgetAdminOnly()),
    ],
)
def test_classify_key_budget_change(update, stored, expected):
    assert classify_key_budget_change(UpdateKeyRequest(key="sk-1", **update), _key(**stored)) == expected


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, "disabled"),
        ({"self_serve_budget_policy": "lower_only"}, "lower_only"),
        ({"self_serve_budget_policy": "ceiling"}, "ceiling"),
        ({"self_serve_budget_policy": "everything"}, "disabled"),
        ({"self_serve_budget_policy": True}, "disabled"),
    ],
)
def test_resolve_self_serve_budget_policy(settings, expected):
    assert resolve_self_serve_budget_policy(settings) == expected


def _team(member_permissions):
    return LiteLLM_TeamTableCachedObj(
        team_id=_TEAM,
        members_with_roles=[Member(user_id="team-admin", role="admin"), Member(user_id=_OWNER, role="user")],
        team_member_permissions=member_permissions,
    )


def _caller(**overrides):
    return UserAPIKeyAuth(**{"user_id": _OWNER, "user_role": LitellmUserRoles.INTERNAL_USER, **overrides})


async def _update(monkeypatch, *, policy, data, existing, team=None, caller=None):
    settings = {} if policy is None else {"self_serve_budget_policy": policy}
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", settings)
    get_team = AsyncMock(return_value=team)
    monkeypatch.setattr("litellm.proxy.management_endpoints.key_management_endpoints.get_team_object", get_team)
    monkeypatch.setattr("litellm.proxy.management_helpers.team_member_permission_checks.get_team_object", get_team)
    prisma = MagicMock()
    prisma.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=existing)
    await _validate_update_key_data(
        data=data,
        existing_key_row=existing,
        user_api_key_dict=caller or _caller(),
        llm_router=None,
        premium_user=False,
        prisma_client=prisma,
        user_api_key_cache=MagicMock(),
    )


_ADD_DAILY_WINDOW = UpdateKeyRequest(key="sk-1", budget_limits=[_WEEKLY, _window("24h", 5.0)])
_RAISE_WEEKLY_WINDOW = UpdateKeyRequest(key="sk-1", budget_limits=[_window("7d", 80.0)])
_TEAM_KEY_WITH_WEEKLY = {"team_id": _TEAM, "budget_limits": [_WEEKLY]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "data", "member_permissions"),
    [
        ("lower_only", _ADD_DAILY_WINDOW, ["/key/update"]),
        ("ceiling", _ADD_DAILY_WINDOW, ["/key/update"]),
        ("ceiling", _RAISE_WEEKLY_WINDOW, ["/key/update", "/key/generate"]),
    ],
)
async def test_owner_changes_team_key_budget_when_policy_allows(monkeypatch, policy, data, member_permissions):
    await _update(
        monkeypatch,
        policy=policy,
        data=data,
        existing=_key(**_TEAM_KEY_WITH_WEEKLY),
        team=_team(member_permissions),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "data", "member_permissions"),
    [
        (None, _ADD_DAILY_WINDOW, ["/key/update", "/key/generate"]),
        ("disabled", _ADD_DAILY_WINDOW, ["/key/update", "/key/generate"]),
        ("lower_only", _RAISE_WEEKLY_WINDOW, ["/key/update", "/key/generate"]),
        ("ceiling", _RAISE_WEEKLY_WINDOW, ["/key/update"]),
        ("ceiling", UpdateKeyRequest(key="sk-1", soft_budget=1.0), ["/key/update", "/key/generate"]),
        ("ceiling", UpdateKeyRequest(key="sk-1", spend=0.0), ["/key/update", "/key/generate"]),
    ],
)
async def test_owner_team_key_budget_change_needs_admin(monkeypatch, policy, data, member_permissions):
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy=policy,
            data=data,
            existing=_key(**_TEAM_KEY_WITH_WEEKLY),
            team=_team(member_permissions),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_key_created_by_someone_else_needs_admin_even_to_tighten(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=_ADD_DAILY_WINDOW,
            existing=_key(created_by="team-admin", **_TEAM_KEY_WITH_WEEKLY),
            team=_team(["/key/update", "/key/generate"]),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_team_member_without_update_permission_is_still_rejected(monkeypatch):
    with pytest.raises(ProxyException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=_ADD_DAILY_WINDOW,
            existing=_key(**_TEAM_KEY_WITH_WEEKLY),
            team=_team(["/key/generate"]),
        )
    assert str(exc.value.code) == "401"


@pytest.mark.asyncio
async def test_ceiling_lets_owner_raise_personal_key_budget(monkeypatch):
    await _update(
        monkeypatch,
        policy="ceiling",
        data=UpdateKeyRequest(key="sk-1", max_budget=30.0),
        existing=_key(max_budget=10.0),
    )


@pytest.mark.asyncio
async def test_ceiling_caps_owner_raise_at_callers_own_budget(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=UpdateKeyRequest(key="sk-1", max_budget=30.0),
            existing=_key(max_budget=10.0),
            caller=_caller(max_budget=20.0),
        )
    assert exc.value.status_code == 400
    assert "cannot exceed the caller's own max_budget (20.0)" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_ceiling_respects_personal_key_generation_role_restriction(monkeypatch):
    monkeypatch.setattr(
        "litellm.key_generation_settings",
        {"personal_key_generation": {"allowed_user_roles": [LitellmUserRoles.PROXY_ADMIN.value]}},
    )
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=UpdateKeyRequest(key="sk-1", max_budget=None),
            existing=_key(max_budget=10.0),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_lower_only_lets_owner_lower_personal_key_budget_without_generate_rights(monkeypatch):
    monkeypatch.setattr(
        "litellm.key_generation_settings",
        {"personal_key_generation": {"allowed_user_roles": [LitellmUserRoles.PROXY_ADMIN.value]}},
    )
    await _update(
        monkeypatch,
        policy="lower_only",
        data=UpdateKeyRequest(key="sk-1", max_budget=5.0),
        existing=_key(max_budget=10.0),
    )
