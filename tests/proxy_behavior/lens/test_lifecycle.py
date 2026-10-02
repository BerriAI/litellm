import asyncio
import hashlib
import os
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Final
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import TypeAdapter

from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.lens import endpoints
from litellm.proxy.lens.models import (
    Check,
    Coverage,
    Lens,
    LensSettings,
    ModelRequest,
    Progress,
    Result,
    RunRequest,
    Scope,
    Worker,
)
from litellm.proxy.lens.repository import Database, LensRepository, Row
from litellm.proxy.lens.state import can_access
from litellm.proxy.utils import PrismaClient, ProxyLogging


@pytest_asyncio.fixture(loop_scope="function")
async def lens_database() -> AsyncIterator[PrismaClient]:
    original_db: Final = proxy_server.prisma_client
    original_router: Final = proxy_server.llm_router
    original_settings: Final = proxy_server.general_settings
    proxy_server.general_settings = {
        **original_settings,
        "allowed_ips": ["127.0.0.1"],
        "use_x_forwarded_for": True,
        "mcp_trusted_proxy_ranges": ["192.0.2.100/32"],
        "mcp_xff_num_trusted_hops": 1,
    }
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    await client.connect()
    proxy_server.prisma_client = client
    proxy_server.llm_router = Router(
        model_list=[
            {
                "model_name": "lens-test-analysis",
                "litellm_params": {
                    "model": "openai/lens-test-analysis",
                    "api_key": "test-only",
                    "mock_response": '{"observations":[]}',
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000002,
                },
            },
            {
                "model_name": "lens-team-route",
                "model_info": {"team_id": "lens-test-team-a", "team_public_model_name": "private/*"},
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "test-only",
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000002,
                },
            },
            {"model_name": "unpriced/*", "litellm_params": {"model": "openai/*", "api_key": "test-only"}},
        ]
    )
    try:
        yield client
    finally:
        proxy_server.general_settings = original_settings
        proxy_server.prisma_client = original_db
        proxy_server.llm_router = original_router
        await client.disconnect()


class _ObservedDatabase:
    def __init__(self, db: Database) -> None:
        self.db: Final = db
        self.page_sizes: tuple[int, ...] = ()

    async def query_raw(self, query: str, *args: object) -> object:
        rows: Final = TypeAdapter(tuple[Row, ...]).validate_python(await self.db.query_raw(query, *args))
        self.page_sizes = (*self.page_sizes, len(rows))
        return rows

    async def execute_raw(self, query: str, *args: object) -> int:
        return await self.db.execute_raw(query, *args)


@pytest.mark.parametrize("kind", ("all", "team", "key"))
@pytest.mark.asyncio
async def test_eligible_workers_filter_before_bounded_pages(lens_database: PrismaClient, kind: str) -> None:
    prefix: Final = str(uuid4())
    now: Final = datetime.now(timezone.utc)
    scopes: Final = {
        "all": Scope(all_teams=True),
        "team": Scope(team_id=prefix),
        "key": Scope(api_key_hash=prefix),
    }
    workers: Final = (
        *(Worker(id=f"{prefix}-{i:03}", name=prefix, scope=scopes["all"], last_seen=now) for i in range(65)),
        Worker(id=f"{prefix}-team", name=prefix, scope=scopes["team"], last_seen=now),
        Worker(id=f"{prefix}-key", name=prefix, scope=scopes["key"], last_seen=now),
        Worker(id=f"{prefix}-foreign", name=prefix, scope=Scope(team_id="other"), last_seen=now),
        Worker(id=f"{prefix}-other-key", name=prefix, scope=Scope(api_key_hash="other"), last_seen=now),
        Worker(id=f"{prefix}-revoked", name=prefix, scope=scopes["all"], last_seen=now, revoked=True),
    )
    repo: Final = endpoints.repository()
    try:
        for worker in workers:
            await repo.save_worker(worker, hashlib.sha256(worker.id.encode()).hexdigest())
        observed: Final = _ObservedDatabase(repo.db)
        eligible: Final = [worker async for worker in LensRepository(observed).eligible_workers(scopes[kind])]
        expected: Final = tuple(w for w in workers if not w.revoked and can_access(w.scope, scopes[kind]))
        assert tuple(w.id for w in eligible) == tuple(sorted(w.id for w in expected))
        assert observed.page_sizes == (50, len(expected) - 50)
    finally:
        await lens_database.db.execute_raw("DELETE FROM \"LiteLLM_LensWorker\" WHERE data->>'name'=$1", prefix)


@pytest.mark.parametrize("enabled", (True, False))
@pytest.mark.asyncio
async def test_unpriced_saved_model_allows_edits_but_not_new_runs(lens_database: PrismaClient, enabled: bool) -> None:
    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    now: Final = datetime.now(timezone.utc)
    original: Final = Lens(
        id=str(uuid4()),
        scope=Scope(all_teams=True),
        created_at=now,
        next_run_at=now,
        budget_month=now.strftime("%Y-%m"),
        settings=LensSettings(
            name="Saved investigation", model="unpriced/lens-saved-model", context="Answer questions", enabled=enabled
        ),
    )
    await endpoints.repository().create(original)
    try:
        settings: Final = original.settings.model_copy(update={"context": "Use cited sources", "enabled": False})
        edited: Final = await endpoints.update_lens(original.id, settings, admin)
        assert edited.settings == settings
        assert edited.revision == original.revision + 1
        assert (await endpoints.read_lens(original.id, admin)).settings == settings
        for operation in (
            endpoints.run_lens(original.id, RunRequest(), admin),
            endpoints.update_lens(original.id, settings.model_copy(update={"enabled": True}), admin),
            endpoints.update_lens(original.id, settings.model_copy(update={"model": "unpriced/other-model"}), admin),
        ):
            with pytest.raises(HTTPException) as error:
                await operation
            assert error.value.status_code == 400
            assert "Pricing is not configured" in error.value.detail
        with pytest.raises(HTTPException) as invalid_selection:
            await endpoints.update_lens(original.id, settings.model_copy(update={"execution_ids": ("invalid",)}), admin)
        assert invalid_selection.value.status_code == 422
        assert (await endpoints.read_lens(original.id, admin)).settings == settings
    finally:
        await lens_database.db.execute_raw('DELETE FROM "LiteLLM_Lens" WHERE id=$1', original.id)


@pytest.mark.asyncio
async def test_team_route_requires_a_worker_with_matching_model_access(lens_database: PrismaClient) -> None:
    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, team_id="lens-test-team-a")
    name: Final = f"Team route regression {uuid4()}"
    settings: Final = LensSettings(name=name, model="private/analysis", context="Answer questions", enabled=False)
    lens: Final = await endpoints.create_lens(settings, admin)
    key_a: Final = hashlib.sha256(uuid4().bytes).hexdigest()
    key_b: Final = hashlib.sha256(uuid4().bytes).hexdigest()
    await lens_database.db.litellm_verificationtoken.create(
        data={"token": key_a, "team_id": "lens-test-team-a", "models": ["private/*"]}
    )
    await lens_database.db.litellm_verificationtoken.create(
        data={"token": key_b, "team_id": "lens-test-team-b", "models": ["private/*"]}
    )
    try:
        wrong_team: Final = await endpoints.register_worker(endpoints.WorkerName(analysis_key_id=key_b), admin)
        assert await endpoints.claim_candidate(lens, wrong_team.worker, datetime.now(timezone.utc)) is None
        for operation in (
            endpoints.create_lens(settings, admin),
            endpoints.run_lens(lens.id, RunRequest(), admin),
        ):
            with pytest.raises(HTTPException) as error:
                await operation
            assert error.value.status_code == 400
            assert "worker" in error.value.detail
        edited: Final = await endpoints.update_lens(
            lens.id, settings.model_copy(update={"context": "Use sources"}), admin
        )
        assert edited.settings.context == "Use sources"
        right_team: Final = await endpoints.register_worker(endpoints.WorkerName(analysis_key_id=key_a), admin)
        await endpoints.validate_workers(settings, lens.scope)
        claim: Final = await endpoints.claim_candidate(lens, right_team.worker, datetime.now(timezone.utc))
        assert claim is not None and claim.job.worker_id == right_team.worker.id
    finally:
        await lens_database.db.execute_raw(
            """DELETE FROM "LiteLLM_LensRun" WHERE lens_id IN
            (SELECT id FROM "LiteLLM_Lens" WHERE data->'settings'->>'name'=$1)""",
            name,
        )
        await lens_database.db.execute_raw("DELETE FROM \"LiteLLM_Lens\" WHERE data->'settings'->>'name'=$1", name)
        await lens_database.db.execute_raw(
            "DELETE FROM \"LiteLLM_LensWorker\" WHERE data->>'analysis_key_id' IN ($1, $2)", key_a, key_b
        )
        await lens_database.db.execute_raw(
            'DELETE FROM "LiteLLM_VerificationToken" WHERE token IN ($1, $2)', key_a, key_b
        )


@pytest.mark.asyncio
async def test_scan_lifecycle_persists_results_and_revokes_worker(lens_database: PrismaClient) -> None:
    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    settings: Final = LensSettings(
        name="Lifecycle regression",
        model="lens-test-analysis",
        enabled=False,
        checks=(Check(id="retries", instruction="Find unrecovered retries"),),
    )
    lens: Final = await endpoints.create_lens(settings, admin)
    key_id: Final = hashlib.sha256(uuid4().bytes).hexdigest()
    await lens_database.db.litellm_verificationtoken.create(data={"token": key_id, "models": ["lens-test-analysis"]})
    registration: Final = await endpoints.register_worker(
        endpoints.WorkerName(name="Test analyzer", analysis_key_id=key_id), admin
    )
    credentials: Final = HTTPAuthorizationCredentials(scheme="Bearer", credentials=registration.token)
    worker: Final = await endpoints.worker_auth(credentials)
    try:
        assert lens.jobs[0].status == "queued"
        stored_worker: Final = await endpoints.repository().worker(
            hashlib.sha256(registration.token.encode()).hexdigest()
        )
        assert stored_worker is not None and stored_worker.id == worker.id
        assert worker.id == registration.worker.id
        listing: Final = await endpoints.list_lenses(admin, storage=None)
        assert lens.id in tuple(e.id for e in listing.lenses)
        assert worker.id in tuple(w.id for w in listing.workers)
        claims: Final = await asyncio.gather(
            *(endpoints.claim_candidate(lens, worker, datetime.now(timezone.utc)) for _ in range(8))
        )
        winners: Final = tuple(claim for claim in claims if claim is not None)
        assert len(winners) == 1
        claimed: Final = winners[0]
        assert claimed.job.worker_id == worker.id
        assert (
            await endpoints.claim_candidate(
                await endpoints.get_lens(lens.id, worker.scope), worker, datetime.now(timezone.utc)
            )
            is None
        )
        assert await endpoints.progress(
            lens.id, claimed.job.id, Progress(stage="Reviewing", coverage=Coverage(screened=2)), worker
        )
        assert await endpoints.heartbeat(lens.id, claimed.job.id, worker)
        response: Final = await endpoints.model(
            lens.id,
            claimed.job.id,
            ModelRequest(prompt="Return an empty observations list", purpose="extract"),
            worker,
            Request(
                {
                    "type": "http",
                    "scheme": "http",
                    "path": "/lens/worker/model",
                    "headers": [],
                    "client": ("127.0.0.1", 1234),
                }
            ),
        )
        assert '"observations"' in response.content
        with pytest.raises(HTTPException) as denied_ip:
            await endpoints.model(
                lens.id,
                claimed.job.id,
                ModelRequest(prompt="Must not run", purpose="extract"),
                worker,
                Request(
                    {
                        "type": "http",
                        "scheme": "http",
                        "path": "/lens/worker/model",
                        "headers": [(b"x-forwarded-for", b"127.0.0.1")],
                        "client": ("192.0.2.1", 1234),
                    }
                ),
            )
        assert denied_ip.value.status_code == 403
        forwarded: Final = await endpoints.model(
            lens.id,
            claimed.job.id,
            ModelRequest(prompt="Return an empty observations list", purpose="extract"),
            worker,
            Request(
                {
                    "type": "http",
                    "scheme": "http",
                    "path": "/lens/worker/model",
                    "headers": [(b"x-forwarded-for", b"127.0.0.1")],
                    "client": ("192.0.2.100", 1234),
                }
            ),
        )
        assert '"observations"' in forwarded.content
        with pytest.raises(HTTPException) as spoofed_chain:
            await endpoints.model(
                lens.id,
                claimed.job.id,
                ModelRequest(prompt="Must not run", purpose="extract"),
                worker,
                Request(
                    {
                        "type": "http",
                        "scheme": "http",
                        "path": "/lens/worker/model",
                        "headers": [(b"x-forwarded-for", b"127.0.0.1, 192.0.2.1")],
                        "client": ("192.0.2.100", 1234),
                    }
                ),
            )
        assert spoofed_chain.value.status_code == 403
        charged: Final = await endpoints.get_lens(lens.id, worker.scope)
        assert charged.spent == pytest.approx(response.cost + forwarded.cost)
        assert charged.jobs[0].cost == pytest.approx(response.cost + forwarded.cost)
        legacy: Final = worker.model_copy(update={"analysis_key_id": None})
        await endpoints.repository().save_worker(legacy)
        authenticated_legacy: Final = await endpoints.worker_auth(credentials)
        assert authenticated_legacy.analysis_key_id is None
        with pytest.raises(HTTPException) as needs_billing:
            await endpoints.claim(authenticated_legacy, protocol_version=2)
        assert needs_billing.value.status_code == 409
        assert await endpoints.heartbeat(lens.id, claimed.job.id, authenticated_legacy)
        finished: Final = await endpoints.result(
            lens.id, claimed.job.id, Result(coverage=Coverage(screened=2)), authenticated_legacy, storage=None
        )
        assert finished.jobs[0].status == "completed"
        assert finished.jobs[0].coverage.screened == 2
        assert finished.last_scan_at == claimed.job.end
        assert finished.next_run_at > finished.jobs[0].finished_at
        assert (
            await endpoints.result(lens.id, claimed.job.id, Result(coverage=Coverage()), worker, storage=None)
            == finished
        )
        with pytest.raises(HTTPException) as stale:
            await endpoints.heartbeat(lens.id, claimed.job.id, worker)
        assert stale.value.status_code == 409
        edited: Final = await endpoints.update_lens(lens.id, settings.model_copy(update={"interval_minutes": 7}), admin)
        assert edited.revision == lens.revision + 1
        with pytest.raises(HTTPException) as unavailable_worker:
            await endpoints.run_lens(lens.id, RunRequest(lookback_hours=3), admin)
        assert unavailable_worker.value.status_code == 400
        await endpoints.set_worker_billing(worker.id, endpoints.WorkerBilling(analysis_key_id=key_id), admin)
        rerun: Final = await endpoints.run_lens(lens.id, RunRequest(lookback_hours=3), admin)
        assert rerun.jobs[0].settings.interval_minutes == 7
        assert rerun.jobs[0].created_at - rerun.jobs[0].start == timedelta(hours=3)
        history: Final = await endpoints.list_runs(lens.id, admin, offset=0)
        assert {job.id for job in history} == {claimed.job.id, rerun.jobs[0].id}
        archived: Final = await endpoints.read_run(lens.id, claimed.job.id, admin)
        assert archived == finished.jobs[0]
        assert archived.settings.interval_minutes == 15
        assert archived.findings == ()
        with pytest.raises(HTTPException) as foreign_history:
            await endpoints.read_run(lens.id, claimed.job.id, UserAPIKeyAuth(team_id="other"))
        assert foreign_history.value.status_code == 403
        cancelled: Final = await endpoints.cancel_lens(lens.id, admin)
        assert cancelled.jobs[0].status == "cancelled"
        assert await endpoints.cancel_lens(lens.id, admin) == cancelled
        assert await endpoints.revoke_worker(worker.id, admin)
        assert await endpoints.repository().set_worker_billing(worker.id, key_id) is None
        with pytest.raises(HTTPException) as revoked_billing:
            await endpoints.set_worker_billing(worker.id, endpoints.WorkerBilling(analysis_key_id=key_id), admin)
        assert revoked_billing.value.status_code == 409
        with pytest.raises(HTTPException) as revoked:
            await endpoints.worker_auth(credentials)
        assert revoked.value.status_code == 401
        with pytest.raises(HTTPException) as foreign:
            await endpoints.get_lens(lens.id, endpoints.Scope(team_id="other"))
        assert foreign.value.status_code == 404
    finally:
        await lens_database.db.execute_raw('DELETE FROM "LiteLLM_LensRun" WHERE lens_id=$1', lens.id)
        await lens_database.db.execute_raw('DELETE FROM "LiteLLM_Lens" WHERE id=$1', lens.id)
        await lens_database.db.execute_raw('DELETE FROM "LiteLLM_LensWorker" WHERE id=$1', worker.id)
        await lens_database.db.execute_raw('DELETE FROM "LiteLLM_VerificationToken" WHERE token=$1', key_id)
