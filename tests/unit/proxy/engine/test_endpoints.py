from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.engine import endpoints
from litellm.proxy.engine.endpoints import Preview, user_scope
from litellm.proxy.engine.models import Check, Engine, EngineSettings, Scope
from litellm.proxy.spend_tracking import spend_management_endpoints
from litellm.proxy.spend_tracking.log_visibility import LogVisibility

NOW: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)


class EngineRepositoryStub:
    def __init__(self, engine: Engine | None) -> None:
        self.engine = engine

    async def get(self, engine_id: str) -> Engine | None:
        return self.engine if self.engine is not None and self.engine.id == engine_id else None


def _engine(engine_scope: Scope) -> Engine:
    return Engine(
        id="lens-1",
        scope=engine_scope,
        settings=EngineSettings(
            name="Research",
            model="analysis",
            checks=(Check(id="retries", instruction="Find unrecovered retries"),),
        ),
        created_at=NOW,
        next_run_at=NOW,
        budget_month="2026-01",
    )


@pytest.mark.parametrize(
    "role",
    (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.TEAM),
)
def test_non_admin_cannot_start_analysis_spending(role: LitellmUserRoles) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role, team_id="team", token="hashed-test-key")
    with pytest.raises(HTTPException) as error:
        user_scope(auth)
    assert (error.value.status_code, error.value.detail) == (
        403,
        "Only proxy admins can configure or run Lens",
    )


def test_only_proxy_admin_gets_write_scope() -> None:
    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    assert user_scope(admin).all_teams


@pytest.mark.asyncio
async def test_team_engine_read_is_hidden_without_spend_log_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    engine: Final = _engine(Scope(team_id="team-a"))
    monkeypatch.setattr(endpoints, "repository", lambda: EngineRepositoryStub(engine))

    with pytest.raises(HTTPException) as error:
        await endpoints.get_visible_engine("lens-1", LogVisibility(user_id="reader"))

    assert (error.value.status_code, error.value.detail) == (404, "Lens not found")


@pytest.mark.asyncio
async def test_evidence_content_is_hidden_without_spend_log_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    engine: Final = _engine(Scope(team_id="team-a"))
    monkeypatch.setattr(endpoints, "repository", lambda: EngineRepositoryStub(engine))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(
        spend_management_endpoints,
        "_get_permitted_team_ids_for_spend_logs",
        AsyncMock(return_value=[]),
    )
    with pytest.raises(HTTPException) as error:
        await endpoints.evidence_content(
            "lens-1",
            "invalid-execution",
            UserAPIKeyAuth(token="key-a", user_id="reader"),
        )

    assert (error.value.status_code, error.value.detail) == (404, "Lens not found")


@pytest.mark.asyncio
async def test_key_scoped_engine_is_visible_to_matching_key_only_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    engine: Final = _engine(Scope(api_key_hash="key-a"))
    monkeypatch.setattr(endpoints, "repository", lambda: EngineRepositoryStub(engine))

    visible: Final = await endpoints.get_visible_engine("lens-1", LogVisibility(api_key_hash="key-a"))

    assert visible is engine


@pytest.mark.asyncio
async def test_all_team_engine_is_visible_only_to_admin_viewers(monkeypatch: pytest.MonkeyPatch) -> None:
    engine: Final = _engine(Scope(all_teams=True))
    monkeypatch.setattr(endpoints, "repository", lambda: EngineRepositoryStub(engine))

    with pytest.raises(HTTPException) as error:
        await endpoints.get_visible_engine("lens-1", LogVisibility(user_id="reader"))
    assert (error.value.status_code, error.value.detail) == (404, "Lens not found")
    assert await endpoints.get_visible_engine("lens-1", LogVisibility(all_teams=True)) is engine


@pytest.mark.asyncio
async def test_preview_sample_is_admin_only(monkeypatch: pytest.MonkeyPatch) -> None:
    sample: Final = AsyncMock()
    monkeypatch.setattr(endpoints, "source_reader", lambda: MagicMock(sample=sample))

    with pytest.raises(HTTPException) as error:
        await endpoints.preview_sample(
            Preview(
                settings=EngineSettings(
                    name="Research",
                    model="analysis",
                    checks=(Check(id="retries", instruction="Find unrecovered retries"),),
                )
            ),
            UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, token="key-a"),
        )

    assert (error.value.status_code, error.value.detail) == (403, "Only proxy admins can configure or run Lens")
    sample.assert_not_awaited()
