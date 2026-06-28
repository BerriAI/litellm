"""
Edge-case matrix for the centralized key-mutation policy helper.

Each test exercises one rule from the seven-rule policy:

  Rule 1: permissions is admin-only; create rejects only non-empty,
          update/regenerate/bulk reject any explicit presence in
          model_fields_set.
  Rule 2: budget changes on update-like paths require key-admin
          authority; personal-key and team-member fast paths apply
          to non-budget changes only.
  Rule 3: numeric hygiene + delegation ceiling on every write path;
          NaN/inf/None/wrong-type rejected before comparison.
  Rule 4: ceiling derivation from caller.max_budget, falling back to
          team budget for session tokens in a team context; CLI
          session token + no team + budget = hard reject.
  Rule 5: delegation_ceiling=None means no ceiling configured (admin
          granted unlimited delegation); proxy admins and the UI team
          admin session are exempt from the ceiling.
  Rule 6: on update-like paths, the ceiling is enforced only on
          budget values that actually changed against the existing
          key row; unchanged resubmits pass.
  Rule 7: no mutable defaults; permissions defaults to None internally
          and is normalized at the json.dumps site.
"""

import math
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import (
    GenerateKeyRequest,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_VerificationToken,
    LitellmUserRoles,
    RegenerateKeyRequest,
    UpdateKeyRequest,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_helpers.key_mutation_authz import authorize_key_mutation


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")


def _internal_user(max_budget: float = 100.0) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="alice",
        max_budget=max_budget,
    )


def _internal_user_unlimited() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="alice")


def _cli_session_token(team_id: str | None = None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="alice",
        team_id=team_id,
        is_session_token=True,
    )


def _ui_team_admin_session() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="alice",
        team_id=UI_SESSION_TOKEN_TEAM_ID,
    )


def _existing_key(**overrides) -> LiteLLM_VerificationToken:
    fields = dict(
        token="hashed-key",
        user_id="alice",
        created_by="alice",
        max_budget=100.0,
        spend=0.0,
    )
    fields.update(overrides)
    return LiteLLM_VerificationToken(**fields)


# ─────────────────────────────────────────────────────────────────────
# Rule 1 — permissions admin-only
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_non_admin_empty_permissions_allowed():
    """The empty {} default on GenerateRequestBase.permissions is the
    legitimate non-admin shape; rule 1 only rejects non-empty values on
    the create path."""
    await authorize_key_mutation(
        data=GenerateKeyRequest(),
        existing_key_row=None,
        user_api_key_dict=_internal_user(),
        team_table=None,
        prisma_client=None,
        user_api_key_cache=None,
        route_label="/key/generate",
    )


@pytest.mark.asyncio
async def test_create_non_admin_non_empty_permissions_rejected():
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=GenerateKeyRequest(permissions={"get_spend_routes": True}),
            existing_key_row=None,
            user_api_key_dict=_internal_user(),
            team_table=None,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/generate",
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_create_admin_permissions_allowed():
    await authorize_key_mutation(
        data=GenerateKeyRequest(permissions={"get_spend_routes": True}),
        existing_key_row=None,
        user_api_key_dict=_admin(),
        team_table=None,
        prisma_client=None,
        user_api_key_cache=None,
        route_label="/key/generate",
    )


@pytest.mark.asyncio
async def test_update_non_admin_explicit_empty_permissions_rejected():
    """An explicit `permissions={}` from a non-admin owner clears an
    admin-set capability such as enable_llm_guard_check, so it must trip
    the gate even though `{}` is falsy."""
    data = UpdateKeyRequest(key="sk-alice", permissions={})
    assert "permissions" in data.model_fields_set
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=data,
            existing_key_row=_existing_key(),
            user_api_key_dict=_internal_user(),
            team_table=None,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/update",
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_update_non_admin_omitted_permissions_allowed():
    """Omitting `permissions` is not the same as submitting it; the gate
    keys off model_fields_set, so an unrelated update passes."""
    data = UpdateKeyRequest(key="sk-alice", metadata={"team": "data"})
    assert "permissions" not in data.model_fields_set
    await authorize_key_mutation(
        data=data,
        existing_key_row=_existing_key(),
        user_api_key_dict=_internal_user(),
        team_table=None,
        prisma_client=None,
        user_api_key_cache=None,
        route_label="/key/update",
    )


# ─────────────────────────────────────────────────────────────────────
# Rule 2 — budget admin gate on update-like paths
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_non_admin_owner_budget_change_rejected():
    """Personal-key ownership lets the caller update non-budget fields,
    but budget changes still require admin authority."""
    data = UpdateKeyRequest(key="sk-alice", max_budget=50.0)
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
        side_effect=HTTPException(status_code=403, detail={"error": "not authorized"}),
    ):
        with pytest.raises(HTTPException) as exc:
            await authorize_key_mutation(
                data=data,
                existing_key_row=_existing_key(max_budget=100.0),
                user_api_key_dict=_internal_user(),
                team_table=None,
                prisma_client=MagicMock(),
                user_api_key_cache=None,
                route_label="/key/update",
            )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_update_non_admin_owner_non_budget_change_allowed():
    """Personal-key bypass applies to non-budget changes; the helper
    must not raise."""
    data = UpdateKeyRequest(key="sk-alice", metadata={"team": "data"})
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
    ) as check:
        await authorize_key_mutation(
            data=data,
            existing_key_row=_existing_key(),
            user_api_key_dict=_internal_user(),
            team_table=None,
            prisma_client=MagicMock(),
            user_api_key_cache=None,
            route_label="/key/update",
        )
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_team_member_non_budget_change_allowed_on_team_key():
    """Team-member fast path on team keys for non-budget changes:
    can_team_member_execute_key_management_endpoint has already
    validated team membership upstream, so the budget admin gate
    must not require admin authority for a non-budget edit here."""
    data = UpdateKeyRequest(key="sk-team", metadata={"app": "billing"})
    existing = _existing_key(user_id="someone-else", created_by="someone-else", team_id="team-x")
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
    ) as check:
        await authorize_key_mutation(
            data=data,
            existing_key_row=existing,
            user_api_key_dict=_internal_user(),
            team_table=None,
            prisma_client=MagicMock(),
            user_api_key_cache=None,
            route_label="/key/update",
        )
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_team_member_budget_change_requires_admin():
    """Team-member fast path does NOT extend to budget changes; the
    admin check must run and reject if the team member is not also a
    team admin / org admin / proxy admin."""
    data = UpdateKeyRequest(key="sk-team", budget_limits=[])
    existing = _existing_key(user_id="someone-else", created_by="someone-else", team_id="team-x")
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
        side_effect=HTTPException(status_code=403, detail={"error": "not authorized"}),
    ):
        with pytest.raises(HTTPException) as exc:
            await authorize_key_mutation(
                data=data,
                existing_key_row=existing,
                user_api_key_dict=_internal_user(),
                team_table=None,
                prisma_client=MagicMock(),
                user_api_key_cache=None,
                route_label="/key/update",
            )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_update_explicit_spend_requires_admin():
    """spend is gated on presence alone — the DB spend lags the live
    counter, so re-submitting an unchanged value through the non-admin
    path would silently weaken enforcement."""
    data = UpdateKeyRequest(key="sk-alice", spend=0.0)
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
        side_effect=HTTPException(status_code=403, detail={"error": "not authorized"}),
    ):
        with pytest.raises(HTTPException) as exc:
            await authorize_key_mutation(
                data=data,
                existing_key_row=_existing_key(),
                user_api_key_dict=_internal_user(),
                team_table=None,
                prisma_client=MagicMock(),
                user_api_key_cache=None,
                route_label="/key/update",
            )
    assert exc.value.status_code == 403


# ─────────────────────────────────────────────────────────────────────
# Rule 3 — numeric hygiene + delegation ceiling on every write path
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
async def test_create_non_finite_max_budget_rejected_even_for_admin(value):
    """NaN/inf at rest disables downstream enforcement (`spend > NaN`
    is always False), so hygiene is not role-dependent — even admin
    callers must be rejected before a non-finite value reaches the DB."""
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=GenerateKeyRequest(max_budget=value),
            existing_key_row=None,
            user_api_key_dict=_admin(),
            team_table=None,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/generate",
        )
    assert exc.value.status_code == 400
    assert "finite" in str(exc.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_create_non_finite_budget_limits_window_rejected_for_admin(value):
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=GenerateKeyRequest(budget_limits=[{"budget_duration": "1d", "max_budget": value}]),
            existing_key_row=None,
            user_api_key_dict=_admin(),
            team_table=None,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/generate",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_create_non_admin_over_ceiling_max_budget_rejected():
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=GenerateKeyRequest(max_budget=1_000_000.0),
            existing_key_row=None,
            user_api_key_dict=_internal_user(max_budget=10.0),
            team_table=None,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/generate",
        )
    assert exc.value.status_code == 400
    assert "exceed" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_create_non_admin_over_ceiling_budget_window_rejected():
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=GenerateKeyRequest(budget_limits=[{"budget_duration": "1d", "max_budget": 1_000_000.0}]),
            existing_key_row=None,
            user_api_key_dict=_internal_user(max_budget=10.0),
            team_table=None,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/generate",
        )
    assert exc.value.status_code == 400
    assert "exceed" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_create_non_admin_at_ceiling_max_budget_allowed():
    """Boundary: a value exactly equal to the caller's ceiling is
    permitted. Pins the strict `>` semantics."""
    await authorize_key_mutation(
        data=GenerateKeyRequest(max_budget=10.0),
        existing_key_row=None,
        user_api_key_dict=_internal_user(max_budget=10.0),
        team_table=None,
        prisma_client=None,
        user_api_key_cache=None,
        route_label="/key/generate",
    )


# ─────────────────────────────────────────────────────────────────────
# Rule 4 — ceiling derivation + CLI session token hard-reject
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cli_session_token_personal_key_max_budget_rejected():
    """CLI session token with no team submitting any budget = hard
    reject. The token's own max_budget=None must not be read as
    unlimited delegation authority for a personal key."""
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=GenerateKeyRequest(max_budget=5.0),
            existing_key_row=None,
            user_api_key_dict=_cli_session_token(),
            team_table=None,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/generate",
        )
    assert exc.value.status_code == 400
    assert "session token" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_cli_session_token_personal_key_budget_limits_rejected():
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=GenerateKeyRequest(budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}]),
            existing_key_row=None,
            user_api_key_dict=_cli_session_token(),
            team_table=None,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/generate",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_cli_session_token_team_key_uses_team_ceiling():
    """Session token + team table delegates against the team's budget,
    not the caller's None max_budget."""
    team = LiteLLM_TeamTableCachedObj(team_id="team-1", max_budget=50.0)
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=GenerateKeyRequest(max_budget=1000.0, team_id="team-1"),
            existing_key_row=None,
            user_api_key_dict=_cli_session_token(team_id="team-1"),
            team_table=team,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/generate",
        )
    assert exc.value.status_code == 400
    assert "exceed" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_cli_session_token_team_key_within_team_budget_allowed():
    team = LiteLLM_TeamTableCachedObj(team_id="team-1", max_budget=50.0)
    await authorize_key_mutation(
        data=GenerateKeyRequest(max_budget=25.0, team_id="team-1"),
        existing_key_row=None,
        user_api_key_dict=_cli_session_token(team_id="team-1"),
        team_table=team,
        prisma_client=None,
        user_api_key_cache=None,
        route_label="/key/generate",
    )


# ─────────────────────────────────────────────────────────────────────
# Rule 5 — None ceiling carve-outs
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_admin_unlimited_caller_can_set_any_budget():
    """A non-admin caller whose admin granted them no ceiling
    (`max_budget=None`, not a session token) can mint a key with any
    budget — admin chose not to bound them."""
    await authorize_key_mutation(
        data=GenerateKeyRequest(max_budget=1_000_000.0),
        existing_key_row=None,
        user_api_key_dict=_internal_user_unlimited(),
        team_table=None,
        prisma_client=None,
        user_api_key_cache=None,
        route_label="/key/generate",
    )


@pytest.mark.asyncio
async def test_admin_can_exceed_own_ceiling():
    """Proxy admins are exempt from the delegation ceiling."""
    await authorize_key_mutation(
        data=GenerateKeyRequest(
            max_budget=1_000_000.0,
            budget_limits=[{"budget_duration": "1d", "max_budget": 1_000_000.0}],
        ),
        existing_key_row=None,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            user_id="admin",
            max_budget=10.0,
        ),
        team_table=None,
        prisma_client=None,
        user_api_key_cache=None,
        route_label="/key/generate",
    )


@pytest.mark.asyncio
async def test_ui_team_admin_session_exempt_from_ceiling():
    """UI signs the team admin into a session token with the sentinel
    team_id; when they then act on a real team key they keep their
    normal delegation authority and the ceiling is not enforced."""
    await authorize_key_mutation(
        data=GenerateKeyRequest(max_budget=1_000_000.0, team_id="team-real"),
        existing_key_row=None,
        user_api_key_dict=_ui_team_admin_session(),
        team_table=None,
        prisma_client=None,
        user_api_key_cache=None,
        route_label="/key/generate",
    )


# ─────────────────────────────────────────────────────────────────────
# Rule 6 — diff-based change detection on update-like paths
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_unchanged_over_ceiling_max_budget_allowed():
    """A team admin with max_budget=$100 whose UI prefills the
    existing $1M cap while updating an unrelated field must not be
    blocked by the delegation ceiling — the value didn't change."""
    existing = _existing_key(
        user_id="someone-else",
        created_by="someone-else",
        team_id="team-x",
        max_budget=1_000_000.0,
    )
    data = UpdateKeyRequest(key="sk-team", max_budget=1_000_000.0)
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
    ):
        await authorize_key_mutation(
            data=data,
            existing_key_row=existing,
            user_api_key_dict=_internal_user(max_budget=100.0),
            team_table=None,
            prisma_client=MagicMock(),
            user_api_key_cache=None,
            route_label="/key/update",
        )


@pytest.mark.asyncio
async def test_update_changed_over_ceiling_max_budget_rejected():
    """When max_budget IS being raised, the ceiling fires."""
    existing = _existing_key(
        user_id="someone-else",
        created_by="someone-else",
        team_id="team-x",
        max_budget=100.0,
    )
    data = UpdateKeyRequest(key="sk-team", max_budget=1_000_000.0)
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
    ):
        with pytest.raises(HTTPException) as exc:
            await authorize_key_mutation(
                data=data,
                existing_key_row=existing,
                user_api_key_dict=_internal_user(max_budget=100.0),
                team_table=None,
                prisma_client=MagicMock(),
                user_api_key_cache=None,
                route_label="/key/update",
            )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_unchanged_over_ceiling_budget_limits_allowed():
    """Same diff semantics for the windows: round-trip resubmit
    passes even when the windows are above the caller's ceiling."""
    existing_windows = [{"budget_duration": "1d", "max_budget": 1_000_000.0}]
    existing = _existing_key(
        user_id="someone-else",
        created_by="someone-else",
        team_id="team-x",
        budget_limits=existing_windows,
    )
    data = UpdateKeyRequest(
        key="sk-team",
        budget_limits=[{"budget_duration": "1d", "max_budget": 1_000_000.0}],
    )
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
    ):
        await authorize_key_mutation(
            data=data,
            existing_key_row=existing,
            user_api_key_dict=_internal_user(max_budget=100.0),
            team_table=None,
            prisma_client=MagicMock(),
            user_api_key_cache=None,
            route_label="/key/update",
        )


@pytest.mark.asyncio
async def test_update_changed_over_ceiling_budget_limits_rejected():
    existing = _existing_key(
        user_id="someone-else",
        created_by="someone-else",
        team_id="team-x",
        budget_limits=[{"budget_duration": "1d", "max_budget": 100.0}],
    )
    data = UpdateKeyRequest(
        key="sk-team",
        budget_limits=[{"budget_duration": "1d", "max_budget": 1_000_000.0}],
    )
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
    ):
        with pytest.raises(HTTPException) as exc:
            await authorize_key_mutation(
                data=data,
                existing_key_row=existing,
                user_api_key_dict=_internal_user(max_budget=100.0),
                team_table=None,
                prisma_client=MagicMock(),
                user_api_key_cache=None,
                route_label="/key/update",
            )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_nan_window_rejected_even_unchanged():
    """A NaN window must always be rejected at the hygiene step, even
    if it matches what's already on the existing key — the existing
    value is the bug we're trying to keep out of the DB."""
    existing = _existing_key(
        user_id="someone-else",
        created_by="someone-else",
        team_id="team-x",
        budget_limits=[{"budget_duration": "1d", "max_budget": float("nan")}],
    )
    data = UpdateKeyRequest(
        key="sk-team",
        budget_limits=[{"budget_duration": "1d", "max_budget": float("nan")}],
    )
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_key_admin_access",
        new_callable=AsyncMock,
    ):
        with pytest.raises(HTTPException) as exc:
            await authorize_key_mutation(
                data=data,
                existing_key_row=existing,
                user_api_key_dict=_admin(),
                team_table=None,
                prisma_client=MagicMock(),
                user_api_key_cache=None,
                route_label="/key/update",
            )
    assert exc.value.status_code == 400
    assert "finite" in str(exc.value.detail)


# ─────────────────────────────────────────────────────────────────────
# Rule 7 — mutable defaults normalized
# ─────────────────────────────────────────────────────────────────────


def test_generate_key_helper_fn_has_no_mutable_permissions_default():
    """generate_key_helper_fn used to default `permissions={}` — a
    mutable default shared across calls. The signature must use None
    so the function normalizes internally and there is no shared dict."""
    import inspect

    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_helper_fn,
    )

    sig = inspect.signature(generate_key_helper_fn)
    assert sig.parameters["permissions"].default is None


def test_permissions_param_is_typed_with_permissionsdict():
    """The `permissions` parameter on every internal helper that holds the
    payload is typed with `PermissionsDict`, not a bare `dict`. Names the
    keys the proxy actually reads (`get_spend_routes`,
    `enable_llm_guard_check`) so basedpyright catches typos at the
    call sites; `total=False` keeps user-defined guardrail keys valid."""
    import inspect
    from typing import Optional

    from litellm.proxy._types import PermissionsDict
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_helper_fn,
    )
    from litellm.proxy.management_helpers.key_mutation_authz import (
        _check_permissions_field,
    )
    from litellm.repositories.verification_token_repository import (
        VerificationTokenRepository,
    )

    expected = Optional[PermissionsDict]
    assert inspect.signature(generate_key_helper_fn).parameters["permissions"].annotation == expected
    assert inspect.signature(VerificationTokenRepository.create_token).parameters["permissions"].annotation == expected
    # The helper's gate function inspects but does not declare a
    # permissions param; just verify the type is importable next to it.
    assert _check_permissions_field is not None
    assert "get_spend_routes" in PermissionsDict.__annotations__
    assert "enable_llm_guard_check" in PermissionsDict.__annotations__


# ─────────────────────────────────────────────────────────────────────
# Composition / regenerate
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_regenerate_non_admin_permissions_rejected():
    """Regenerate inherits the same admin-only permissions gate. The
    create-path semantic (truthiness) applies because regenerate's body
    populates a fresh key from a base."""
    with pytest.raises(HTTPException) as exc:
        await authorize_key_mutation(
            data=RegenerateKeyRequest(key="sk-alice", permissions={"get_spend_routes": True}),
            existing_key_row=_existing_key(),
            user_api_key_dict=_internal_user(),
            team_table=None,
            prisma_client=None,
            user_api_key_cache=None,
            route_label="/key/regenerate",
        )
    assert exc.value.status_code == 403
