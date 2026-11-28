import asyncio
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm.proxy.in_flight_registry as registry
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.in_flight_endpoints import router


async def _sleep_forever() -> None:
    await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_register_snapshot_unregister_round_trip():
    task: Final = asyncio.create_task(_sleep_forever())
    handle: Final = registry.register("r1", task, model="gpt-4o", user_id="u1", key_alias="prod", is_streaming=True)
    assert handle is not None
    try:
        snap: Final = registry.snapshot()
        assert [item["request_id"] for item in snap] == ["r1"]
        assert snap[0]["handle"] == handle
        assert snap[0]["model"] == "gpt-4o"
        assert snap[0]["key_alias"] == "prod"
        assert snap[0]["is_streaming"] is True
        assert snap[0]["state"] == "running"
    finally:
        registry.unregister_if_owned(handle, task)
        task.cancel()
    assert registry.snapshot() == ()


@pytest.mark.asyncio
async def test_register_ignores_missing_id_or_task():
    assert registry.register("", asyncio.current_task() or asyncio.create_task(_sleep_forever()), model="m") is None
    assert registry.register("r2", None, model="m") is None
    assert registry.snapshot() == ()


@pytest.mark.asyncio
async def test_cancel_by_id_cancels_only_that_request():
    first: Final = asyncio.create_task(_sleep_forever())
    second: Final = asyncio.create_task(_sleep_forever())
    h1: Final = registry.register("c1", first, model="m")
    h2: Final = registry.register("c2", second, model="m")
    try:
        cancelled: Final = registry.cancel((h1,))
        assert cancelled == (h1,)
        assert first.cancelled() or first.cancelling()
        assert not second.cancelled()
        assert [item["state"] for item in registry.snapshot()] == ["cancelling", "running"]
    finally:
        registry.unregister_if_owned(h1, first)
        registry.unregister_if_owned(h2, second)
        for task in (first, second):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_cancel_all_skips_finished_and_second_cancel_is_noop():
    running: Final = asyncio.create_task(_sleep_forever())
    done: Final = asyncio.create_task(_sleep_forever())
    done.cancel()
    try:
        await done
    except asyncio.CancelledError:
        pass
    ha: Final = registry.register("a1", running, model="m")
    hb: Final = registry.register("a2", done, model="m")
    try:
        assert registry.cancel() == (ha,)
        assert registry.cancel() == ()
        assert registry.cancel(("unknown",)) == ()
    finally:
        registry.unregister_if_owned(ha, running)
        registry.unregister_if_owned(hb, done)
        running.cancel()
        try:
            await running
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_duplicate_client_ids_get_distinct_handles_and_survive_each_other_unregistration():
    long_task: Final = asyncio.create_task(_sleep_forever())
    short_task: Final = asyncio.create_task(_sleep_forever())
    h_long: Final = registry.register("same-id", long_task, model="long-model")
    h_short: Final = registry.register("same-id", short_task, model="short-model")
    assert h_long is not None and h_short is not None and h_long != h_short
    try:
        registry.unregister_if_owned(h_short, short_task)
        snap: Final = registry.snapshot()
        assert [item["handle"] for item in snap] == [h_long]
        assert snap[0]["model"] == "long-model"
        assert registry.cancel(("all",)) == ()
        assert registry.cancel((h_long,)) == (h_long,)
        assert long_task.cancelled() or long_task.cancelling()
        registry.unregister_if_owned(h_long, short_task)
        assert [item["handle"] for item in registry.snapshot()] == [h_long]
    finally:
        for task in (long_task, short_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        registry.unregister_if_owned(h_long, long_task)
        registry.unregister_if_owned(h_short, short_task)
    assert registry.snapshot() == ()


def _app_for_role(role: LitellmUserRoles) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="k", user_role=role)
    return app


def test_list_requires_admin_role():
    client = TestClient(_app_for_role(LitellmUserRoles.INTERNAL_USER))
    assert client.get("/in_flight/requests").status_code == 403
    admin = TestClient(_app_for_role(LitellmUserRoles.PROXY_ADMIN))
    response: Final = admin.get("/in_flight/requests")
    assert response.status_code == 200
    assert response.json()["requests"] == []


def test_cancel_requires_admin_and_selector():
    client = TestClient(_app_for_role(LitellmUserRoles.PROXY_ADMIN))
    assert client.post("/in_flight/cancel", json={}).status_code == 400
    viewer = TestClient(_app_for_role(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
    assert viewer.post("/in_flight/cancel", json={"all": True}).status_code == 403
    assert client.post("/in_flight/cancel", json={"all": True}).json() == {"cancelled": []}
