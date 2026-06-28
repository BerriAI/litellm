"""
Single policy entrypoint for authorizing a key mutation.

Background: prior to this module, six write paths each enforced subsets of the
key-mutation policy at their own call sites, with different gating logic for
`permissions`, budget admin authority, and the delegation ceiling. The drift
produced real bypasses (non-admin self-grant on `permissions`, NaN smuggled
into `budget_limits`, CLI session token treating its own absent budget as
unlimited delegation authority, bulk paths skipping checks the single path
enforced).

This module collapses the policy into one helper, `authorize_key_mutation`,
that:

  1. Validates numeric hygiene on every submitted budget value (NaN / inf
     / None / wrong-type are rejected with 400 before any comparison).
  2. Diffs the incoming request against the existing key (None on create)
     to know which fields actually changed.
  3. Enforces three guards by changed field:
       a. `permissions` is proxy-admin-only. On create, only a non-empty
          dict trips the gate; on update-like paths, any explicit presence
          in `model_fields_set` trips it (so `{}` and `null` cannot clear
          an admin-set capability).
       b. On update-like paths, any budget change (max_budget / spend /
          budget_limits) requires key-admin authority over the target key.
          Personal-key-owner and team-member-grant fast paths apply to
          non-budget changes only.
       c. On every write path, the delegation ceiling caps the submitted
          budget against the caller's own authority. Proxy admins and the
          UI team-admin session are exempt from the ceiling. A CLI session
          token with no team and an explicit budget is hard-rejected
          rather than treated as unlimited.

The six handlers each call this once, in the position where they already
have `existing_key_row` (None on create) and `team_table` resolved.
"""

import math
from typing import Any, List, Optional, Union

from fastapi import HTTPException

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.models.team import BudgetLimitEntry
from litellm.proxy._types import (
    GenerateKeyRequest,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_VerificationToken,
    LitellmUserRoles,
    PermissionsDict,
    RegenerateKeyRequest,
    UpdateKeyRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.utils import PrismaClient

KeyWriteRequest = Union[GenerateKeyRequest, UpdateKeyRequest, RegenerateKeyRequest]


def _is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value


def _is_ui_team_admin_session(
    user_api_key_dict: UserAPIKeyAuth,
    requested_team_id: Optional[str],
) -> bool:
    """The UI signs the team admin into a session token with a fixed
    sentinel team_id; when they then act on a real team key they keep
    their normal delegation authority."""
    return user_api_key_dict.team_id == UI_SESSION_TOKEN_TEAM_ID and requested_team_id is not None


def _resolve_delegation_ceiling(
    user_api_key_dict: UserAPIKeyAuth,
    team_table: Optional[LiteLLM_TeamTableCachedObj],
) -> Optional[float]:
    """Caller's own max_budget when set; for CLI session tokens acting on
    a team key, fall back to the team's max_budget. `None` means "no
    ceiling configured" (legitimate when an admin granted unlimited
    delegation); only the session-token-no-team case rejects on `None`,
    handled separately."""
    if user_api_key_dict.max_budget is not None:
        return user_api_key_dict.max_budget
    if user_api_key_dict.is_session_token and team_table is not None:
        return team_table.max_budget
    return None


def _reject_non_finite(value: Any, field: str) -> None:
    """Reject NaN, inf, None, or non-numeric. `NaN > x` is False so a
    NaN budget bypasses the ceiling check and disables downstream
    enforcement (`spend > NaN` is always False). Numeric hygiene is
    not role-dependent — admins are also rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"{field} must be a finite number; got {value!r}",
            },
        )
    if not math.isfinite(value):
        raise HTTPException(
            status_code=400,
            detail={"error": f"{field} must be a finite number; got {value}"},
        )


def _normalize_window_for_compare(window: Any) -> Any:
    """Reduce a budget_limits window to a comparable shape. The wire
    form is a `BudgetLimitEntry` (Pydantic), the DB form is a plain
    dict carrying `reset_at`. Drop `reset_at` and any None-valued
    keys so a UI prefill that round-trips unchanged compares equal."""
    if isinstance(window, BudgetLimitEntry):
        window = window.model_dump(mode="json", exclude_none=True)
    if isinstance(window, dict):
        return {k: v for k, v in window.items() if v is not None and k != "reset_at"}
    return window


def _budget_limits_equal(
    submitted: Optional[List[BudgetLimitEntry]],
    existing: Any,
) -> bool:
    """A True result lets the ceiling check skip the field; used only
    to suppress noise on unchanged resubmits, never to bypass a
    real change."""
    if submitted is None and existing is None:
        return True
    if submitted is None or existing is None:
        return False
    if not isinstance(existing, list):
        return False
    sub = [_normalize_window_for_compare(w) for w in submitted]
    ex = [_normalize_window_for_compare(w) for w in existing]
    return sub == ex


def _check_permissions_field(
    data: KeyWriteRequest,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    is_create_path: bool,
) -> None:
    """`permissions` grants ambient capabilities (e.g. `get_spend_routes`
    exposes `/global/spend/*`, `enable_llm_guard_check` toggles a
    callback). Proxy-admin-only on every write path.

    Create-path semantic: the empty-dict default on `GenerateKeyRequest`
    is the legitimate non-admin shape, so only a non-empty dict trips
    the gate. Update-path semantic: an explicit `{}` or `null` from a
    non-admin owner would clear an admin-set capability such as
    `enable_llm_guard_check`, so any explicit presence in
    `model_fields_set` trips the gate."""
    if _is_proxy_admin(user_api_key_dict):
        return
    permissions: Optional[PermissionsDict] = getattr(data, "permissions", None)
    if is_create_path:
        if not permissions:
            return
    else:
        if "permissions" not in data.model_fields_set:
            return
    raise HTTPException(
        status_code=403,
        detail={"error": "Only proxy admins can write `permissions` on a key."},
    )


async def _check_budget_admin_authority(
    data: KeyWriteRequest,
    existing_key_row: LiteLLM_VerificationToken,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: Any,
    route_label: str,
) -> None:
    """Update-like paths only. Any budget-field change requires admin
    authority over the key (proxy admin / team admin of key's team /
    org admin of that team's org). Personal-key-owner and team-member
    grant fast paths apply only to non-budget changes — the DB spend
    counter lags the live cross-pod counter, so letting an unchanged
    spend through the non-admin path would let a key owner silently
    overwrite the live counter below real usage."""
    if _is_proxy_admin(user_api_key_dict):
        return
    if prisma_client is None:
        return
    is_budget_change = (
        (data.max_budget is not None and data.max_budget != existing_key_row.max_budget)
        or getattr(data, "spend", None) is not None
        or "budget_limits" in data.model_fields_set
    )
    caller_is_creator = (
        user_api_key_dict.user_id is not None
        and getattr(existing_key_row, "created_by", None) == user_api_key_dict.user_id
        and getattr(existing_key_row, "user_id", None) == user_api_key_dict.user_id
    )
    key_is_team_key = getattr(existing_key_row, "team_id", None) is not None
    can_skip = (caller_is_creator or key_is_team_key) and not is_budget_change
    if can_skip:
        return
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _check_key_admin_access,
    )

    await _check_key_admin_access(
        user_api_key_dict=user_api_key_dict,
        hashed_token=existing_key_row.token,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        route=f"{route_label} (max_budget/spend)" if is_budget_change else route_label,
    )


def _check_delegation_ceiling(
    data: KeyWriteRequest,
    existing_key_row: Optional[LiteLLM_VerificationToken],
    user_api_key_dict: UserAPIKeyAuth,
    team_table: Optional[LiteLLM_TeamTableCachedObj],
) -> None:
    """A caller may not delegate budget beyond their own authority. On
    update-like paths, only enforce the ceiling on budget values that
    actually changed against the existing key (so a UI prefill that
    resubmits the existing $1M cap on a $100-ceiling team admin while
    editing an unrelated field passes).

    Numeric hygiene runs before any role check — NaN is rejected for
    everyone, including proxy admins, because a NaN at rest disables
    enforcement forever."""
    if data.max_budget is not None:
        _reject_non_finite(data.max_budget, "max_budget")
    if data.budget_limits:
        for window in data.budget_limits:
            _reject_non_finite(window.max_budget, "budget_limits entry max_budget")

    if _is_proxy_admin(user_api_key_dict):
        return
    if _is_ui_team_admin_session(user_api_key_dict, data.team_id):
        return

    if (
        user_api_key_dict.is_session_token
        and team_table is None
        and (data.max_budget is not None or "budget_limits" in data.model_fields_set)
    ):
        raise HTTPException(
            status_code=400,
            detail={"error": ("A CLI session token cannot delegate budget for a personal key; specify `team_id`.")},
        )

    delegation_ceiling = _resolve_delegation_ceiling(user_api_key_dict, team_table)
    if delegation_ceiling is None:
        return

    max_budget_changed = data.max_budget is not None and (
        existing_key_row is None or data.max_budget != existing_key_row.max_budget
    )
    if max_budget_changed and data.max_budget > delegation_ceiling:
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    f"max_budget ({data.max_budget}) cannot exceed the caller's own max_budget ({delegation_ceiling})."
                )
            },
        )

    if not data.budget_limits:
        return
    existing_windows = getattr(existing_key_row, "budget_limits", None) if existing_key_row is not None else None
    if existing_key_row is not None and _budget_limits_equal(data.budget_limits, existing_windows):
        return
    over = next(
        (w for w in data.budget_limits if w.max_budget > delegation_ceiling),
        None,
    )
    if over is not None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    f"budget_limits entry max_budget ({over.max_budget}) "
                    f"cannot exceed the caller's own max_budget "
                    f"({delegation_ceiling})."
                )
            },
        )


async def authorize_key_mutation(
    *,
    data: KeyWriteRequest,
    existing_key_row: Optional[LiteLLM_VerificationToken],
    user_api_key_dict: UserAPIKeyAuth,
    team_table: Optional[LiteLLM_TeamTableCachedObj],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: Any,
    route_label: str,
) -> None:
    """Single policy entrypoint for any key-write handler.

    `existing_key_row` is None on create paths (`/key/generate`,
    `/key/service-account/generate`) and populated on update-like paths
    (`/key/update`, `/key/regenerate`, `/key/bulk_update`,
    `/team/key/bulk_update`). `is_create_path` is derived from that.

    Order matters: permissions admin gate runs first because it is the
    cheapest check and short-circuits non-admin self-grant. Budget admin
    gate runs second so update-like paths reject non-admin budget
    changes before reaching the ceiling. Delegation ceiling runs last
    and is the only check that constrains admin callers (proxy admin is
    exempt, but team admin / org admin / session token are bounded by
    their own authority)."""
    is_create_path = existing_key_row is None

    _check_permissions_field(
        data=data,
        user_api_key_dict=user_api_key_dict,
        is_create_path=is_create_path,
    )

    if not is_create_path:
        await _check_budget_admin_authority(
            data=data,
            existing_key_row=existing_key_row,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            route_label=route_label,
        )

    _check_delegation_ceiling(
        data=data,
        existing_key_row=existing_key_row,
        user_api_key_dict=user_api_key_dict,
        team_table=team_table,
    )
