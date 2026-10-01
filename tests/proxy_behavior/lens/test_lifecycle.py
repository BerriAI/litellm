import hashlib
import os
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.engine import endpoints
from litellm.proxy.engine.models import Check, Coverage, EngineSettings, ModelRequest, Progress, Result, RunRequest
from litellm.proxy.utils import PrismaClient, ProxyLogging


@pytest_asyncio.fixture(loop_scope="function")
async def lens_database() -> AsyncIterator[PrismaClient]:
    original_db: Final = proxy_server.prisma_client
    original_router: Final = proxy_server.llm_router
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
            }
        ]
    )
    try:
        yield client
    finally:
        proxy_server.prisma_client = original_db
        proxy_server.llm_router = original_router
        await client.disconnect()


@pytest.mark.asyncio
async def test_scan_lifecycle_persists_results_and_revokes_worker(lens_database: PrismaClient) -> None:
    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    settings: Final = EngineSettings(
        name="Lifecycle regression",
        model="lens-test-analysis",
        enabled=False,
        checks=(Check(id="retries", instruction="Find unrecovered retries"),),
    )
    engine: Final = await endpoints.create_engine(settings, admin)
    registration: Final = await endpoints.register_worker(endpoints.WorkerName(name="Test analyzer"), admin)
    credentials: Final = HTTPAuthorizationCredentials(scheme="Bearer", credentials=registration.token)
    worker: Final = await endpoints.worker_auth(credentials)
    try:
        assert engine.jobs[0].status == "queued"
        stored_worker: Final = await endpoints.repository().worker(
            hashlib.sha256(registration.token.encode()).hexdigest()
        )
        assert stored_worker is not None and stored_worker.id == worker.id
        assert worker.id == registration.worker.id
        listing: Final = await endpoints.list_engines(admin)
        assert engine.id in tuple(e.id for e in listing.engines)
        assert worker.id in tuple(w.id for w in listing.workers)
        claimed: Final = await endpoints.claim_candidate(engine, worker, datetime.now(timezone.utc))
        assert claimed is not None
        assert claimed.job.worker_id == worker.id
        assert (
            await endpoints.claim_candidate(
                await endpoints.get_engine(engine.id, worker.scope), worker, datetime.now(timezone.utc)
            )
            is None
        )
        assert await endpoints.progress(
            engine.id, claimed.job.id, Progress(stage="Reviewing", coverage=Coverage(screened=2)), worker
        )
        assert await endpoints.heartbeat(engine.id, claimed.job.id, worker)
        response: Final = await endpoints.model(
            engine.id,
            claimed.job.id,
            ModelRequest(prompt="Return an empty observations list", purpose="extract"),
            worker,
        )
        assert '"observations"' in response.content
        charged: Final = await endpoints.get_engine(engine.id, worker.scope)
        assert charged.spent == pytest.approx(response.cost)
        assert charged.jobs[0].cost == pytest.approx(response.cost)
        finished: Final = await endpoints.result(
            engine.id, claimed.job.id, Result(coverage=Coverage(screened=2)), worker
        )
        assert finished.jobs[0].status == "completed"
        assert finished.jobs[0].coverage.screened == 2
        assert finished.last_scan_at == claimed.job.end
        assert finished.next_run_at > finished.jobs[0].finished_at
        assert await endpoints.result(engine.id, claimed.job.id, Result(coverage=Coverage()), worker) == finished
        with pytest.raises(HTTPException) as stale:
            await endpoints.heartbeat(engine.id, claimed.job.id, worker)
        assert stale.value.status_code == 409
        edited: Final = await endpoints.update_engine(
            engine.id, settings.model_copy(update={"interval_minutes": 7}), admin
        )
        assert edited.revision == engine.revision + 1
        rerun: Final = await endpoints.run_engine(engine.id, RunRequest(lookback_hours=3), admin)
        assert rerun.jobs[0].settings.interval_minutes == 7
        assert rerun.jobs[0].created_at - rerun.jobs[0].start == timedelta(hours=3)
        cancelled: Final = await endpoints.cancel_engine(engine.id, admin)
        assert cancelled.jobs[0].status == "cancelled"
        assert await endpoints.cancel_engine(engine.id, admin) == cancelled
        assert await endpoints.revoke_worker(worker.id, admin)
        with pytest.raises(HTTPException) as revoked:
            await endpoints.worker_auth(credentials)
        assert revoked.value.status_code == 401
        with pytest.raises(HTTPException) as foreign:
            await endpoints.get_engine(engine.id, endpoints.user_scope(UserAPIKeyAuth(team_id="other")))
        assert foreign.value.status_code == 404
    finally:
        await lens_database.db.execute_raw('DELETE FROM "LiteLLM_Engine" WHERE id=$1', engine.id)
        await lens_database.db.execute_raw('DELETE FROM "LiteLLM_EngineWorker" WHERE id=$1', worker.id)
