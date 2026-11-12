from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.models.verification_token import LiteLLM_VerificationToken
from litellm.proxy._types import (
    LiteLLM_TeamTableCachedObj,
    LitellmUserRoles,
    Member,
    ProxyException,
    UpdateKeyRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_endpoints.key_budget_change import (
    KeyBudgetAdminOnly,
    KeyBudgetChange,
    KeyBudgetLoosened,
    KeyBudgetTightened,
    KeyBudgetUnchanged,
    classify_key_budget_change,
    resolve_self_serve_budget_policy,
)
from litellm.proxy.management_endpoints.key_management_endpoints import _validate_update_key_data
from litellm.types.proxy.management_endpoints.ui_sso import LiteLLM_UpperboundKeyGenerateParams

_OWNER: Final = "owner-1"
_TEAM_ADMIN: Final = "team-admin"
_TEAM: Final = "team-1"
_WEEKLY: Final[Mapping[str, object]] = MappingProxyType(
    {"budget_duration": "7d", "max_budget": 50.0, "reset_at": "2026-10-08T00:00:00+00:00"}
)


def _key(
    *,
    user_id: str = _OWNER,
    created_by: str = _OWNER,
    team_id: str | None = None,
    max_budget: float | None = None,
    budget_limits: Sequence[Mapping[str, object]] | None = None,
    budget_id: str | None = None,
    budget_duration: str | None = None,
) -> LiteLLM_VerificationToken:
    return LiteLLM_VerificationToken(
        token="hashed-owner-key",
        user_id=user_id,
        created_by=created_by,
        team_id=team_id,
        max_budget=max_budget,
        budget_limits=None if budget_limits is None else [dict(window) for window in budget_limits],
        budget_id=budget_id,
        budget_duration=budget_duration,
        metadata={},
    )


def _window(duration: str, cap: float) -> Mapping[str, object]:
    return MappingProxyType({"budget_duration": duration, "max_budget": cap})


@pytest.mark.parametrize(
    ("update", "stored", "expected"),
    [
        (UpdateKeyRequest(key="sk-1", key_alias="renamed"), _key(max_budget=10.0), KeyBudgetUnchanged()),
        (UpdateKeyRequest(key="sk-1", max_budget=10.0), _key(max_budget=10.0), KeyBudgetUnchanged()),
        (UpdateKeyRequest(key="sk-1", max_budget=5.0), _key(max_budget=10.0), KeyBudgetTightened()),
        (UpdateKeyRequest(key="sk-1", max_budget=5.0), _key(), KeyBudgetTightened()),
        (UpdateKeyRequest(key="sk-1", max_budget=20.0), _key(max_budget=10.0), KeyBudgetLoosened()),
        (UpdateKeyRequest(key="sk-1", max_budget=None), _key(max_budget=10.0), KeyBudgetLoosened()),
        (UpdateKeyRequest(key="sk-1", budget_limits=[_window("24h", 5.0)]), _key(), KeyBudgetTightened()),
        (
            UpdateKeyRequest(key="sk-1", budget_limits=[_WEEKLY, _window("24h", 5.0)]),
            _key(budget_limits=[_WEEKLY]),
            KeyBudgetTightened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_limits=[_window("7d", 50.0)]),
            _key(budget_limits=[_WEEKLY]),
            KeyBudgetTightened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_limits=[_window("7d", 40.0)]),
            _key(budget_limits=[_WEEKLY]),
            KeyBudgetTightened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_limits=[_window("7d", 60.0)]),
            _key(budget_limits=[_WEEKLY]),
            KeyBudgetLoosened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_limits=[_window("24h", 5.0)]),
            _key(budget_limits=[_WEEKLY]),
            KeyBudgetLoosened(),
        ),
        (UpdateKeyRequest(key="sk-1", budget_limits=[]), _key(budget_limits=[_WEEKLY]), KeyBudgetLoosened()),
        (UpdateKeyRequest(key="sk-1", budget_limits=None), _key(budget_limits=[_WEEKLY]), KeyBudgetLoosened()),
        (
            UpdateKeyRequest(key="sk-1", budget_limits=[_window("1d", 5.0)]),
            _key(budget_limits=[_window("24h", 5.0)]),
            KeyBudgetLoosened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_limits=[_window("7d", 40.0)]),
            _key(budget_limits=[_WEEKLY, _window("7d", 30.0)]),
            KeyBudgetLoosened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_limits=[_window("7d", 40.0), _window("24h", 5.0)]),
            _key(budget_limits=[_window("7d", 30.0), _window("24h", 5.0), _window("7d", 50.0)]),
            KeyBudgetLoosened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_limits=[_window("24h", 5.0)]),
            _key(budget_limits=[{"budget_duration": "24h"}]),
            KeyBudgetLoosened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", max_budget=5.0, budget_limits=[]),
            _key(max_budget=10.0, budget_limits=[_WEEKLY]),
            KeyBudgetLoosened(),
        ),
        (UpdateKeyRequest(key="sk-1", spend=0.0), _key(), KeyBudgetAdminOnly()),
        (UpdateKeyRequest(key="sk-1", soft_budget=1.0), _key(), KeyBudgetAdminOnly()),
        (UpdateKeyRequest(key="sk-1", soft_budget=None, max_budget=5.0), _key(max_budget=10.0), KeyBudgetAdminOnly()),
        (
            UpdateKeyRequest(key="sk-1", budget_duration="1s"),
            _key(max_budget=10.0, budget_duration="30d"),
            KeyBudgetLoosened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_duration="60d"),
            _key(max_budget=10.0, budget_duration="30d"),
            KeyBudgetLoosened(),
        ),
        (UpdateKeyRequest(key="sk-1", budget_duration="1d"), _key(max_budget=10.0), KeyBudgetLoosened()),
        (
            UpdateKeyRequest(key="sk-1", budget_duration=None),
            _key(max_budget=10.0, budget_duration="30d"),
            KeyBudgetTightened(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_duration="30d"),
            _key(max_budget=10.0, budget_duration="30d"),
            KeyBudgetUnchanged(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_duration="30d"),
            _key(max_budget=10.0, budget_duration="Monthly"),
            KeyBudgetUnchanged(),
        ),
        (UpdateKeyRequest(key="sk-1", budget_duration="1s"), _key(budget_duration="30d"), KeyBudgetUnchanged()),
        (UpdateKeyRequest(key="sk-1", max_budget=5.0, budget_duration="1s"), _key(), KeyBudgetTightened()),
        (
            UpdateKeyRequest(key="sk-1", budget_duration="1s"),
            _key(budget_id="shared-tier-10", budget_duration="30d"),
            KeyBudgetAdminOnly(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_duration=None),
            _key(budget_id="shared-tier-10", budget_duration="30d"),
            KeyBudgetAdminOnly(),
        ),
        (
            UpdateKeyRequest(key="sk-1", budget_id="shared-tier-100"),
            _key(budget_id="shared-tier-10"),
            KeyBudgetAdminOnly(),
        ),
        (UpdateKeyRequest(key="sk-1", budget_id=None), _key(budget_id="shared-tier-10"), KeyBudgetAdminOnly()),
        (UpdateKeyRequest(key="sk-1", budget_id="shared-tier-10"), _key(), KeyBudgetAdminOnly()),
        (
            UpdateKeyRequest(key="sk-1", budget_id="shared-tier-10"),
            _key(budget_id="shared-tier-10"),
            KeyBudgetUnchanged(),
        ),
    ],
)
def test_classify_key_budget_change(
    update: UpdateKeyRequest, stored: LiteLLM_VerificationToken, expected: KeyBudgetChange
) -> None:
    assert classify_key_budget_change(update, stored) == expected


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
def test_resolve_self_serve_budget_policy(settings: Mapping[str, object], expected: str) -> None:
    assert resolve_self_serve_budget_policy(settings) == expected


def _team(member_permissions: Sequence[str]) -> LiteLLM_TeamTableCachedObj:
    return LiteLLM_TeamTableCachedObj(
        team_id=_TEAM,
        members_with_roles=[Member(user_id=_TEAM_ADMIN, role="admin"), Member(user_id=_OWNER, role="user")],
        team_member_permissions=list(member_permissions),
    )


def _caller(
    *,
    user_id: str = _OWNER,
    max_budget: float | None = None,
    team_id: str | None = None,
    user_max_budget: float | None = None,
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        max_budget=max_budget,
        team_id=team_id,
        user_max_budget=user_max_budget,
    )


async def _update(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy: str | None,
    data: UpdateKeyRequest,
    existing: LiteLLM_VerificationToken,
    team: LiteLLM_TeamTableCachedObj | None = None,
    caller: UserAPIKeyAuth | None = None,
) -> None:
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {} if policy is None else {"self_serve_budget_policy": policy},
    )
    prisma: Final = MagicMock()
    prisma.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=existing)
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    prisma.writer_db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    await _validate_update_key_data(
        data=data,
        existing_key_row=existing,
        user_api_key_dict=caller or _caller(),
        llm_router=None,
        premium_user=False,
        prisma_client=prisma,
        user_api_key_cache=UserApiKeyCache(),
    )


async def _denial(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy: str | None,
    data: UpdateKeyRequest,
    existing: LiteLLM_VerificationToken,
    team: LiteLLM_TeamTableCachedObj | None = None,
    caller: UserAPIKeyAuth | None = None,
) -> HTTPException | None:
    try:
        await _update(monkeypatch, policy=policy, data=data, existing=existing, team=team, caller=caller)
    except HTTPException as exc:
        return exc
    return None


_ADD_DAILY_WINDOW: Final = UpdateKeyRequest(key="sk-1", budget_limits=[_WEEKLY, _window("24h", 5.0)])
_RAISE_WEEKLY_WINDOW: Final = UpdateKeyRequest(key="sk-1", budget_limits=[_window("7d", 80.0)])


def _team_key_with_weekly(*, user_id: str = _OWNER, created_by: str = _OWNER) -> LiteLLM_VerificationToken:
    return _key(user_id=user_id, created_by=created_by, team_id=_TEAM, budget_limits=[_WEEKLY])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "data", "member_permissions"),
    [
        ("lower_only", _ADD_DAILY_WINDOW, ["/key/update"]),
        ("ceiling", _ADD_DAILY_WINDOW, ["/key/update"]),
        ("ceiling", _RAISE_WEEKLY_WINDOW, ["/key/update", "/key/generate"]),
    ],
)
async def test_owner_changes_team_key_budget_when_policy_allows(
    monkeypatch: pytest.MonkeyPatch, policy: str, data: UpdateKeyRequest, member_permissions: Sequence[str]
) -> None:
    assert (
        await _denial(
            monkeypatch,
            policy=policy,
            data=data,
            existing=_team_key_with_weekly(),
            team=_team(member_permissions),
        )
        is None
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
async def test_owner_team_key_budget_change_needs_admin(
    monkeypatch: pytest.MonkeyPatch, policy: str | None, data: UpdateKeyRequest, member_permissions: Sequence[str]
) -> None:
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy=policy,
            data=data,
            existing=_team_key_with_weekly(),
            team=_team(member_permissions),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_key_created_by_someone_else_needs_admin_even_to_tighten(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=_ADD_DAILY_WINDOW,
            existing=_team_key_with_weekly(created_by=_TEAM_ADMIN),
            team=_team(["/key/update", "/key/generate"]),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_team_member_without_update_permission_is_still_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ProxyException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=_ADD_DAILY_WINDOW,
            existing=_team_key_with_weekly(),
            team=_team(["/key/generate"]),
        )
    assert str(exc.value.code) == "401"


@pytest.mark.asyncio
async def test_ceiling_lets_owner_raise_personal_key_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        await _denial(
            monkeypatch,
            policy="ceiling",
            data=UpdateKeyRequest(key="sk-1", max_budget=30.0),
            existing=_key(max_budget=10.0),
        )
        is None
    )


@pytest.mark.asyncio
async def test_ceiling_caps_owner_raise_at_callers_own_budget(monkeypatch: pytest.MonkeyPatch) -> None:
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
async def test_ceiling_does_not_cap_a_team_admin_raising_their_own_team_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        await _denial(
            monkeypatch,
            policy="ceiling",
            data=UpdateKeyRequest(key="sk-1", max_budget=30.0),
            existing=_key(user_id=_TEAM_ADMIN, created_by=_TEAM_ADMIN, team_id=_TEAM, max_budget=10.0),
            team=_team(["/key/update"]),
            caller=_caller(user_id=_TEAM_ADMIN, max_budget=20.0),
        )
        is None
    )


@pytest.mark.asyncio
async def test_ceiling_caps_ui_session_owner_raise_at_user_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=UpdateKeyRequest(key="sk-1", max_budget=30.0),
            existing=_key(max_budget=10.0),
            caller=_caller(team_id=UI_SESSION_TOKEN_TEAM_ID, user_max_budget=20.0),
        )
    assert exc.value.status_code == 400
    assert "cannot exceed the caller's own max_budget (20.0)" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_ceiling_respects_personal_key_generation_role_restriction(monkeypatch: pytest.MonkeyPatch) -> None:
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
async def test_lower_only_lets_owner_lower_personal_key_budget_without_generate_rights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "litellm.key_generation_settings",
        {"personal_key_generation": {"allowed_user_roles": [LitellmUserRoles.PROXY_ADMIN.value]}},
    )
    assert (
        await _denial(
            monkeypatch,
            policy="lower_only",
            data=UpdateKeyRequest(key="sk-1", max_budget=5.0),
            existing=_key(max_budget=10.0),
        )
        is None
    )


@pytest.mark.asyncio
async def test_ceiling_respects_team_key_generation_role_restriction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "litellm.key_generation_settings",
        {"team_key_generation": {"allowed_team_member_roles": ["admin"]}},
    )
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=_RAISE_WEEKLY_WINDOW,
            existing=_team_key_with_weekly(),
            team=_team(["/key/update", "/key/generate"]),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["lower_only", "ceiling"])
async def test_owner_cannot_override_a_cap_inherited_from_a_linked_budget(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy=policy,
            data=UpdateKeyRequest(key="sk-1", max_budget=100.0),
            existing=_key(max_budget=None, budget_id="shared-tier-10"),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_ceiling_cannot_leave_a_key_uncapped_when_an_upperbound_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.upperbound_key_generate_params", LiteLLM_UpperboundKeyGenerateParams(max_budget=20.0))
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=UpdateKeyRequest(key="sk-1", max_budget=None),
            existing=_key(max_budget=10.0),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_ceiling_checks_the_retained_max_budget_when_only_windows_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy="ceiling",
            data=UpdateKeyRequest(key="sk-1", budget_limits=[]),
            existing=_key(max_budget=100.0, budget_limits=[_WEEKLY]),
            caller=_caller(max_budget=20.0),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [None, "disabled"])
async def test_owner_cannot_clear_max_budget_by_default(monkeypatch: pytest.MonkeyPatch, policy: str | None) -> None:
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy=policy,
            data=UpdateKeyRequest(key="sk-1", max_budget=None),
            existing=_key(max_budget=10.0),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "data", "existing"),
    [
        (None, UpdateKeyRequest(key="sk-1", budget_duration="1s"), _key(max_budget=10.0, budget_duration="30d")),
        (
            "lower_only",
            UpdateKeyRequest(key="sk-1", budget_duration="1s"),
            _key(max_budget=10.0, budget_duration="30d"),
        ),
        ("ceiling", UpdateKeyRequest(key="sk-1", budget_id="shared-tier-100"), _key(budget_id="shared-tier-10")),
        ("ceiling", UpdateKeyRequest(key="sk-1", budget_duration="1s"), _key(budget_id="shared-tier-10")),
    ],
)
async def test_owner_cannot_restart_or_relink_a_key_budget_without_admin(
    monkeypatch: pytest.MonkeyPatch, policy: str | None, data: UpdateKeyRequest, existing: LiteLLM_VerificationToken
) -> None:
    with pytest.raises(HTTPException) as exc:
        await _update(monkeypatch, policy=policy, data=data, existing=existing)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "data", "existing"),
    [
        (None, UpdateKeyRequest(key="sk-1", budget_duration="1d"), _key(budget_duration="30d")),
        (None, UpdateKeyRequest(key="sk-1", budget_duration="24h"), _key(max_budget=10.0, budget_duration="daily")),
        (
            "lower_only",
            UpdateKeyRequest(key="sk-1", budget_duration=None),
            _key(max_budget=10.0, budget_duration="30d"),
        ),
        ("ceiling", UpdateKeyRequest(key="sk-1", budget_duration="1s"), _key(max_budget=10.0, budget_duration="30d")),
    ],
)
async def test_owner_changes_budget_duration_when_it_cannot_restart_a_cap_or_policy_allows(
    monkeypatch: pytest.MonkeyPatch, policy: str | None, data: UpdateKeyRequest, existing: LiteLLM_VerificationToken
) -> None:
    assert await _denial(monkeypatch, policy=policy, data=data, existing=existing) is None


@pytest.mark.asyncio
async def test_team_member_cannot_restart_a_team_key_budget_created_by_someone_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(HTTPException) as exc:
        await _update(
            monkeypatch,
            policy=None,
            data=UpdateKeyRequest(key="sk-1", budget_duration="1s"),
            existing=_key(created_by=_TEAM_ADMIN, team_id=_TEAM, max_budget=10.0, budget_duration="30d"),
            team=_team(["/key/update"]),
        )
    assert exc.value.status_code == 403
