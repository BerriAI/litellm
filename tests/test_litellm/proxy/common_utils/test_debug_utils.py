import asyncio

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.debug_utils import router as debug_router


async def _park_for_test() -> None:
    await asyncio.sleep(30)


async def _park_inner(event: asyncio.Event) -> None:
    await event.wait()


async def _park_outer(event: asyncio.Event) -> None:
    await _park_inner(event)


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
async def test_task_stacks_allow_proxy_admin_view_only() -> None:
    app = _test_app(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    async with _client(app) as client:
        response = await client.get("/debug/asyncio-tasks/stacks")

    assert response.status_code == 200
    assert "groups" in response.json()


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
async def test_task_stacks_include_nested_await_frames() -> None:
    app = _test_app(LitellmUserRoles.PROXY_ADMIN)
    event = asyncio.Event()
    parked_task = asyncio.create_task(_park_outer(event), name="nested-parked-test-task")
    await asyncio.sleep(0)
    try:
        async with _client(app) as client:
            response = await client.get("/debug/asyncio-tasks/stacks")
    finally:
        parked_task.cancel()
        await asyncio.gather(parked_task, return_exceptions=True)

    assert response.status_code == 200
    nested_group = next(
        group for group in response.json()["groups"] if group["task_names"] == ["nested-parked-test-task"]
    )
    frames = nested_group["stack"]
    functions = [frame["function"] for frame in frames]
    assert functions.index("_park_outer") < functions.index("_park_inner")
    assert any(frame["file"].endswith("asyncio/locks.py") or frame["function"] == "wait" for frame in frames)


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
    groups = response.json()["groups"]
    parked_group = next(group for group in groups if "_park_for_test" in group["coroutine"])
    assert len(parked_group["stack"]) == 1
    assert parked_group["stack"][0]["function"] == "_park_for_test"
    assert all(len(group["stack"]) <= 1 for group in groups)
