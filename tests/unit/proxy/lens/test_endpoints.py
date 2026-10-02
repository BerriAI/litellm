from collections.abc import Callable
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.lens.endpoints import list_agents, user_scope


@pytest.mark.parametrize("role", (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
@pytest.mark.asyncio
async def test_agent_discovery_without_trace_storage_is_empty(role: LitellmUserRoles) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role)
    assert await list_agents(auth, None) == ()


@pytest.mark.asyncio
async def test_agent_discovery_without_trace_storage_still_requires_admin_access() -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    with pytest.raises(HTTPException) as error:
        await list_agents(auth, None)
    assert error.value.status_code == 403


@pytest.mark.parametrize(
    "role",
    (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.TEAM),
)
def test_non_admin_cannot_start_analysis_spending(role: LitellmUserRoles) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role, team_id="team", token="hashed-test-key")
    with pytest.raises(HTTPException) as error:
        user_scope(auth, write=True)
    assert error.value.status_code == 403


def test_admin_can_configure_lens_and_viewer_can_only_read() -> None:
    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    viewer: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    assert user_scope(admin, write=True).all_teams
    assert user_scope(viewer).all_teams


@pytest.mark.parametrize("identity", ("not-an-execution", "W10=", "WyJvdGhlciIsICIiLCAiaWQiXQ=="))
def test_invalid_explicit_execution_ids_are_rejected(identity: str) -> None:
    from litellm.proxy.lens.endpoints import validate_selection
    from tests.unit.proxy.lens.test_state import lens

    settings: Final = lens().settings.model_copy(update={"execution_ids": (identity,)})
    with pytest.raises(HTTPException) as error:
        validate_selection(settings)
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_incompatible_worker_is_rejected_before_claiming_work() -> None:
    from litellm.proxy.lens.endpoints import claim
    from tests.unit.proxy.lens.test_state import worker

    with pytest.raises(HTTPException) as error:
        await claim(worker(), protocol_version=1)
    assert error.value.status_code == 409
    assert "Upgrade" in error.value.detail


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM, None))
def test_regular_keys_cannot_read_lens_results(role: LitellmUserRoles | None) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role, team_id="team", token="hashed-test-key")
    with pytest.raises(HTTPException) as error:
        user_scope(auth)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_updating_an_unknown_finding_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime, timezone

    from litellm.proxy.lens import endpoints
    from litellm.proxy.lens.models import Finding, FindingUpdate, Lens
    from tests.unit.proxy.lens.test_state import finding, lens

    now: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)
    draft: Final = finding("run")
    known: Final = Finding(
        title=draft.title,
        description=draft.description,
        check_id=draft.check_id,
        evidence=draft.evidence,
        id="known",
        first_seen=now,
        last_seen=now,
        revision=1,
    )
    stored: Final = lens().model_copy(update={"findings": (known,)})

    class Repository:
        async def get(self, lens_id: str) -> Lens | None:
            return stored if lens_id == stored.id else None

        async def update(self, lens_id: str, transform: Callable[[Lens], Lens]) -> Lens | None:
            return transform(stored)

    monkeypatch.setattr(endpoints, "repository", Repository)
    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    with pytest.raises(HTTPException) as error:
        await endpoints.update_finding("lens", "missing", FindingUpdate(status="dismissed"), admin)
    assert error.value.status_code == 404

    updated: Final = await endpoints.update_finding("lens", "known", FindingUpdate(status="dismissed"), admin)
    assert updated.findings[0].status == "dismissed"
