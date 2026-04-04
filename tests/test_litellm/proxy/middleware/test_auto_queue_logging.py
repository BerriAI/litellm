import json
import os
import sys

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)
for _name in list(sys.modules):
    if _name == "litellm" or _name.startswith("litellm."):
        sys.modules.pop(_name, None)

from litellm.proxy.middleware.auto_queue_logging import (
    AUTOQ_METADATA_KEY,
    append_autoq_event,
    finalize_autoq_summary,
)
from litellm.proxy.middleware.auto_queue_middleware import (
    AutoQueueMiddleware,
    AutoQueueRedis,
)


def test_should_append_bounded_ordered_autoq_events():
    request_data = {}

    for idx in range(5):
        append_autoq_event(
            request_data,
            event=f"event-{idx}",
            payload={"idx": idx},
            at_ms=idx,
            max_events=3,
        )

    autoq_metadata = request_data[AUTOQ_METADATA_KEY]
    assert [event["event"] for event in autoq_metadata["events"]] == [
        "event-2",
        "event-3",
        "event-4",
    ]
    assert [event["at_ms"] for event in autoq_metadata["events"]] == [2, 3, 4]


def test_should_merge_autoq_summary_without_dropping_existing_fields():
    request_data = {}

    finalize_autoq_summary(
        request_data,
        {
            "request_id": "req-1",
            "model": "gpt-4",
        },
    )
    finalize_autoq_summary(
        request_data,
        {
            "decision": "queued",
            "queued": True,
        },
    )

    assert request_data[AUTOQ_METADATA_KEY]["summary"] == {
        "request_id": "req-1",
        "model": "gpt-4",
        "decision": "queued",
        "queued": True,
    }


@pytest.mark.asyncio
async def test_should_capture_autoq_metadata_for_logging_without_forwarding_it():
    captured_body = {}
    captured_autoq_metadata = None
    redis = fakeredis.aioredis.FakeRedis()
    aqr = AutoQueueRedis(
        redis=redis,
        default_max_concurrent=2,
        ceiling=10,
        scale_up_threshold=3,
        scale_down_step=1,
    )

    async def handler(request):
        nonlocal captured_autoq_metadata
        captured_body.update(json.loads(await request.body()))
        captured_autoq_metadata = getattr(request.state, "autoq_metadata", None)
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/chat/completions", handler, methods=["POST"])])
    app = AutoQueueMiddleware(app, aqr=aqr, enabled=True)
    client = AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    )
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
    finally:
        await client.aclose()
        await redis.aclose()

    assert response.status_code == 200
    assert AUTOQ_METADATA_KEY not in captured_body

    assert isinstance(captured_autoq_metadata, dict)
    autoq_metadata = captured_autoq_metadata
    assert [event["event"] for event in autoq_metadata["events"]] == [
        "received",
        "decision",
        "claim_acquired",
        "forwarded",
    ]
    assert autoq_metadata["summary"]["model"] == "gpt-4"
    assert autoq_metadata["summary"]["decision"] == "admit_now"
    assert autoq_metadata["summary"]["queued"] is False
    assert autoq_metadata["summary"]["request_id"].startswith("gpt-4-")
