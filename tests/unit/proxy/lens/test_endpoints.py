from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.lens import endpoints
from litellm.proxy.lens.endpoints import Preview, claim, user_scope, validate_selection
from litellm.proxy.lens.models import Check, Job, Lens, LensSettings, Scope
from litellm.proxy.lens.sources import SourceReader, execution_id
from litellm.proxy.spend_tracking import spend_management_endpoints
from litellm.proxy.spend_tracking.log_visibility import LogVisibility
from tests.unit.proxy.lens.test_state import lens as sample_lens
from tests.unit.proxy.lens.test_state import worker

NOW: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)


class LensRepositoryStub:
    def __init__(self, lens: Lens | None, job: Job | None = None) -> None:
        self.lens = lens
        self.run = job

    async def get(self, lens_id: str) -> Lens | None:
        return self.lens if self.lens is not None and self.lens.id == lens_id else None

    async def jobs(self, lens_id: str, offset: int = 0) -> tuple[Job, ...]:
        return (self.run,) if self.run is not None else ()

    async def job(self, lens_id: str, job_id: str) -> Job | None:
        return self.run if self.run is not None and self.run.id == job_id else None


class EvidenceStorageStub:
    async def lens_sample(self, parameters: Mapping[str, object]) -> object:
        raise AssertionError("Evidence reads do not sample")

    async def lens_content(self, parameters: Mapping[str, object]) -> object:
        assert parameters["team"] == "team-a"
        assert parameters["record_team"] == "team-a"
        assert parameters["trace_ref"] == "trace-ref"
        return (
            {
                "span_id": "span-a",
                "parent_span_id": "",
                "name": "completion",
                "kind": "llm",
                "content": "verified output",
                "truncated": 0,
            },
        )

    async def lens_evidence(self, parameters: Mapping[str, object]) -> object:
        raise AssertionError("Evidence reads do not verify evidence")


def _lens(lens_scope: Scope) -> Lens:
    return Lens(
        id="lens-1",
        scope=lens_scope,
        settings=LensSettings(
            name="Research",
            model="analysis",
            checks=(Check(id="retries", instruction="Find unrecovered retries"),),
        ),
        created_at=NOW,
        next_run_at=NOW,
        budget_month="2026-01",
    )


def _job(lens: Lens) -> Job:
    return Job(
        id="run-1",
        created_at=NOW,
        start=NOW,
        end=NOW,
        settings=lens.settings,
        revision=lens.revision,
    )


def _team_member() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.TEAM,
        team_id="team-a",
        token="key-a",
        user_id="reader",
    )


def _set_permitted_teams(monkeypatch: pytest.MonkeyPatch, teams: tuple[str, ...]) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(
        spend_management_endpoints,
        "_get_permitted_team_ids_for_spend_logs",
        AsyncMock(return_value=teams),
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
async def test_read_lens_requires_permitted_team_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    lens: Final = _lens(Scope(team_id="team-a"))
    auth: Final = _team_member()
    monkeypatch.setattr(endpoints, "repository", lambda: LensRepositoryStub(lens))
    _set_permitted_teams(monkeypatch, ())

    with pytest.raises(HTTPException) as error:
        await endpoints.read_lens("lens-1", auth)
    assert (error.value.status_code, error.value.detail) == (404, "Lens not found")

    _set_permitted_teams(monkeypatch, ("team-a",))
    assert await endpoints.read_lens("lens-1", auth) is lens


@pytest.mark.asyncio
async def test_list_runs_requires_permitted_team_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    lens: Final = _lens(Scope(team_id="team-a"))
    job: Final = _job(lens)
    auth: Final = _team_member()
    monkeypatch.setattr(endpoints, "repository", lambda: LensRepositoryStub(lens, job))
    _set_permitted_teams(monkeypatch, ())

    with pytest.raises(HTTPException) as error:
        await endpoints.list_runs("lens-1", auth, offset=0)
    assert (error.value.status_code, error.value.detail) == (404, "Lens not found")

    _set_permitted_teams(monkeypatch, ("team-a",))
    runs: Final = await endpoints.list_runs("lens-1", auth, offset=0)
    assert tuple(run.id for run in runs) == ("run-1",)


@pytest.mark.asyncio
async def test_read_run_requires_permitted_team_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    lens: Final = _lens(Scope(team_id="team-a"))
    job: Final = _job(lens)
    auth: Final = _team_member()
    monkeypatch.setattr(endpoints, "repository", lambda: LensRepositoryStub(lens, job))
    _set_permitted_teams(monkeypatch, ())

    with pytest.raises(HTTPException) as error:
        await endpoints.read_run("lens-1", "run-1", auth)
    assert (error.value.status_code, error.value.detail) == (404, "Lens not found")

    _set_permitted_teams(monkeypatch, ("team-a",))
    assert await endpoints.read_run("lens-1", "run-1", auth) is job


@pytest.mark.asyncio
async def test_evidence_content_requires_permitted_team_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    lens: Final = _lens(Scope(team_id="team-a"))
    auth: Final = _team_member()
    identity: Final = execution_id("traces", "team-a", "trace-a", "trace-ref")
    reader: Final = SourceReader(EvidenceStorageStub())
    monkeypatch.setattr(endpoints, "repository", lambda: LensRepositoryStub(lens))
    _set_permitted_teams(monkeypatch, ())

    with pytest.raises(HTTPException) as error:
        await endpoints.evidence_content("lens-1", identity, auth, reader, offset=0)
    assert (error.value.status_code, error.value.detail) == (404, "Lens not found")

    _set_permitted_teams(monkeypatch, ("team-a",))
    content: Final = await endpoints.evidence_content("lens-1", identity, auth, reader, offset=0)
    assert content.execution.trace_ref == "trace-ref"
    assert content.parts[0].content == "verified output"


@pytest.mark.asyncio
async def test_key_scoped_lens_is_visible_to_matching_key_only_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    lens: Final = _lens(Scope(api_key_hash="key-a"))
    monkeypatch.setattr(endpoints, "repository", lambda: LensRepositoryStub(lens))

    visible: Final = await endpoints.get_visible_lens("lens-1", LogVisibility(api_key_hash="key-a"))

    assert visible is lens


@pytest.mark.asyncio
async def test_all_team_lens_is_visible_only_to_admin_viewers(monkeypatch: pytest.MonkeyPatch) -> None:
    lens: Final = _lens(Scope(all_teams=True))
    monkeypatch.setattr(endpoints, "repository", lambda: LensRepositoryStub(lens))

    with pytest.raises(HTTPException) as error:
        await endpoints.get_visible_lens("lens-1", LogVisibility(user_id="reader"))
    assert (error.value.status_code, error.value.detail) == (404, "Lens not found")
    assert await endpoints.get_visible_lens("lens-1", LogVisibility(all_teams=True)) is lens


@pytest.mark.asyncio
async def test_preview_sample_is_admin_only() -> None:
    with pytest.raises(HTTPException) as error:
        await endpoints.preview_sample(
            Preview(
                settings=LensSettings(
                    name="Research",
                    model="analysis",
                    checks=(Check(id="retries", instruction="Find unrecovered retries"),),
                )
            ),
            UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, token="key-a"),
        )

    assert (error.value.status_code, error.value.detail) == (403, "Only proxy admins can configure or run Lens")


@pytest.mark.parametrize("identity", ("not-an-execution", "W10=", "WyJvdGhlciIsICIiLCAiaWQiXQ=="))
def test_invalid_explicit_execution_ids_are_rejected(identity: str) -> None:
    settings: Final = sample_lens().settings.model_copy(update={"execution_ids": (identity,)})
    with pytest.raises(HTTPException) as error:
        validate_selection(settings)
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_incompatible_worker_is_rejected_before_claiming_work() -> None:
    with pytest.raises(HTTPException) as error:
        await claim(worker(), protocol_version=1)
    assert error.value.status_code == 409
    assert "Upgrade" in error.value.detail


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM, None))
def test_regular_keys_cannot_configure_lens(role: LitellmUserRoles | None) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role, team_id="team", token="hashed-test-key")
    with pytest.raises(HTTPException) as error:
        user_scope(auth)
    assert error.value.status_code == 403
