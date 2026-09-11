import asyncio

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.debug_utils import router as debug_router


async def _park_for_test() -> None:
    await asyncio.sleep(30)


def _test_app(user_role: LitellmUserRoles) -> FastAPI:
    app = FastAPI()
    app.include_router(debug_router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=user_role)
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


@pytest.mark.asyncio
async def test_task_stacks_require_proxy_admin() -> None:
    app = _test_app(LitellmUserRoles.INTERNAL_USER)
    async with _client(app) as client:
        response = await client.get("/debug/asyncio-tasks/stacks")

    assert response.status_code == 403
    assert response.json()["detail"] == "Only proxy admins can read asyncio task stacks"


@pytest.mark.asyncio
async def test_task_stacks_include_parked_task() -> None:
    app = _test_app(LitellmUserRoles.PROXY_ADMIN)
    parked_task = asyncio.create_task(_park_for_test(), name="parked-test-task")
    try:
        async with _client(app) as client:
            response = await client.get("/debug/asyncio-tasks/stacks")
    finally:
        parked_task.cancel()
        await asyncio.gather(parked_task, return_exceptions=True)

    assert response.status_code == 200
    body = response.json()
    parked_group = next(group for group in body["groups"] if "_park_for_test" in group["coroutine"])
    assert any(frame["function"] == "_park_for_test" for frame in parked_group["stack"])
    assert any(frame["file"].endswith("test_debug_utils.py") for frame in parked_group["stack"])
    assert body["total_active_tasks"] >= 1


@pytest.mark.asyncio
async def test_task_stacks_respect_max_frames() -> None:
    app = _test_app(LitellmUserRoles.PROXY_ADMIN)
    parked_task = asyncio.create_task(_park_for_test(), name="parked-test-task")
    try:
        async with _client(app) as client:
            response = await client.get("/debug/asyncio-tasks/stacks?max_frames=1")
    finally:
        parked_task.cancel()
        await asyncio.gather(parked_task, return_exceptions=True)

    assert response.status_code == 200
    assert all(len(group["stack"]) <= 1 for group in response.json()["groups"])
