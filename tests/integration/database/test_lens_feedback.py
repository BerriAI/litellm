"""Real ClickHouse feedback CRUD; trace storage is a controlled boundary fixture."""

import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.authorization import OwnedRows
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.lens.feedback import (
    FeedbackCreate,
    FeedbackRepository,
    FeedbackTarget,
    feedback_repository,
    router,
)
from litellm.proxy.lens.feedback_store import FeedbackStore
from litellm.proxy.tracing_endpoints import TraceAccessContext, provide_trace_access
from litellm.rust_bridge.trace.storage import TraceStorageConfig
from litellm.tracing import TraceReceiver


@pytest_asyncio.fixture(loop_scope="function")
async def feedback_db() -> AsyncIterator[FeedbackRepository]:
    url = os.environ["CLICKHOUSE_URL"]
    schema = "feedback_" + uuid4().hex
    async with httpx.AsyncClient() as client:
        response = await client.post(url, content=f"CREATE DATABASE {schema}")
        response.raise_for_status()
    store = FeedbackStore(TraceStorageConfig(url=url, database=schema))
    await store.ensure_schema()
    try:
        yield FeedbackRepository(store)
    finally:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, content=f"DROP DATABASE {schema}")
            response.raise_for_status()


@pytest.mark.asyncio
async def test_concurrent_retries_count_once_and_different_authors_remain_separate(
    feedback_db: FeedbackRepository,
) -> None:
    target = FeedbackTarget(trace_id="response", trace_ref="tenant-a", span_id="span")
    body = FeedbackCreate(**target.model_dump(), value=0)
    empty = await feedback_db.summary(target, "alice")
    assert (empty.count, empty.average, empty.mine) == (0, None, None)
    assert empty.distribution == {str(i): 0 for i in range(11)}
    saved = await asyncio.gather(*(feedback_db.save(body, target, "alice") for _ in range(8)))
    assert len({item.id for item in saved}) == 1
    assert (await feedback_db.summary(target, "alice")).count == 1
    await feedback_db.save(body.model_copy(update={"value": 10}), target, "bob")
    summary = await feedback_db.summary(target, "alice")
    assert (summary.count, summary.average, summary.mine.value) == (2, 5.0, 0)
    assert summary.distribution == {str(i): int(i in (0, 10)) for i in range(11)}
    other = target.model_copy(update={"trace_ref": "tenant-b"})
    await feedback_db.save(body.model_copy(update={"value": 9}), other, "alice")
    assert (await feedback_db.summary(target, "alice")).average == 5.0
    assert (await feedback_db.summary(other, "alice")).average == 9.0
    page = await feedback_db.page(target, 1, 0)
    second = await feedback_db.page(target, 1, page.next_offset)
    assert len(page.data) == len(second.data) == 1
    assert page.data[0].id != second.data[0].id
    assert second.next_offset is None


class TraceStorageFixture:
    """Emulates the external trace store's scoped result, not feedback persistence."""

    async def get_trace(self, trace_id, scope, trace_ref, cursor, page_size):
        if scope["user_id"] == "outsider" or trace_id != "response" or trace_ref not in ("", "tenant-a"):
            return None
        return {"summary": {"trace_id": trace_id, "trace_ref": "tenant-a"}}

    async def get_span(self, trace_id, span_id, scope, trace_ref):
        return {"span_id": span_id} if span_id == "span" and trace_ref == "tenant-a" else None


@pytest.mark.asyncio
async def test_http_create_edit_delete_and_authorization(feedback_db: FeedbackRepository) -> None:
    app = FastAPI()
    app.include_router(router)
    auth = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)
    receiver = TraceReceiver(storage=TraceStorageFixture())
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    app.dependency_overrides[feedback_repository] = lambda: feedback_db
    app.dependency_overrides[provide_trace_access] = lambda: TraceAccessContext(receiver, OwnedRows(auth.user_id), None)
    target = {"trace_id": "response", "span_id": "span"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for value in (-1, 11, True, "8", 2.5):
            invalid = await client.post("/v1/feedback", json={**target, "value": value})
            assert invalid.status_code == 422, invalid.text
        missing_span = await client.post("/v1/feedback", json={**target, "span_id": "foreign", "value": 8})
        assert missing_span.status_code == 404, missing_span.text
        missing_trace = await client.post("/v1/feedback", json={**target, "trace_id": "not-ingested", "value": 8})
        assert missing_trace.status_code == 404, missing_trace.text
        created = await client.post("/v1/feedback", json={**target, "value": 0, "comment": "Not useful"})
        assert created.status_code == 200, created.text
        stored = created.json()
        assert stored["value"] == 0 and stored["trace_ref"] == "tenant-a"
        assert "author_id" not in stored
        identity = stored["id"]
        retry = await client.post("/v1/feedback", json={**target, "value": 0})
        assert retry.json()["id"] == identity
        edited = await client.patch(f"/v1/feedback/{identity}", json={"value": 10, "comment": "Now useful"})
        assert edited.status_code == 200, edited.text
        assert edited.json()["value"] == 10
        assert (await client.get(f"/v1/feedback/{identity}")).json() == edited.json()
        summary = (await client.get("/v1/feedback/summary", params=target)).json()
        assert (summary["count"], summary["average"], summary["mine"]["value"]) == (1, 10.0, 10)
        listed = (await client.get("/v1/feedback", params=target)).json()
        assert listed == {"data": [edited.json()], "next_offset": None}

        auth = UserAPIKeyAuth(user_id="bob", user_role=LitellmUserRoles.INTERNAL_USER)
        forbidden = await client.patch(f"/v1/feedback/{identity}", json={"value": 2})
        assert forbidden.status_code == 403, forbidden.text
        assert (await client.delete(f"/v1/feedback/{identity}")).status_code == 403
        assert (await client.post("/v1/feedback", json={**target, "value": 0})).status_code == 200
        summary = (await client.get("/v1/feedback/summary", params=target)).json()
        assert (summary["count"], summary["average"], summary["mine"]["value"]) == (2, 5.0, 0)

        auth = UserAPIKeyAuth(user_id="outsider", user_role=LitellmUserRoles.INTERNAL_USER)
        for path in ("/v1/feedback", "/v1/feedback/summary", f"/v1/feedback/{identity}"):
            assert (await client.get(path, params=target)).status_code == 404
        assert (await client.post("/v1/feedback", json={**target, "value": 8})).status_code == 404
        assert (await client.patch(f"/v1/feedback/{identity}", json={"value": 8})).status_code == 404
        assert (await client.delete(f"/v1/feedback/{identity}")).status_code == 404

        auth = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        assert (await client.get("/v1/feedback/summary", params=target)).json()["can_rate"] is False
        assert (await client.post("/v1/feedback", json={**target, "value": 8})).status_code == 403
        assert (await client.patch(f"/v1/feedback/{identity}", json={"value": 8})).status_code == 403
        assert (await client.delete(f"/v1/feedback/{identity}")).status_code == 403
        auth = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)
        assert (await client.delete(f"/v1/feedback/{identity}")).status_code == 204
        assert (await client.get(f"/v1/feedback/{identity}")).status_code == 404
        summary = (await client.get("/v1/feedback/summary", params=target)).json()
        assert (summary["count"], summary["average"], summary["mine"]) == (1, 0.0, None)
