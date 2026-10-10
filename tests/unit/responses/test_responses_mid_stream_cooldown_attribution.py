"""
Regression for #45645: a mid-stream 429 on a streaming /v1/responses call must be
recorded against the deployment that streamed it, not against the fallback
deployment the router picks next.

All attempts of one request share one Logging object. The stream iterator hands
the sync failure handler to a thread pool without waiting, and the fallback
attempt calls update_from_kwargs(model_info=<fallback>) on that same object
right away. When the thread runs late, Router.deployment_callback_on_failure
reads the fallback's model_info.id and cools down a deployment that never
returned a 429.

A slow sync failure callback registered ahead of the router's makes "the thread
runs late" deterministic, as in the issue's repro.
"""

import json
import threading
import time
from datetime import datetime

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.router_utils.cooldown_handlers import async_get_cooldown_deployments

PRIMARY_BASE = "http://fake-45645/primary/v1"
BACKUP_BASE = "http://fake-45645/backup/v1"


def _request_logging_obj(call_id: str) -> Logging:
    """One Logging object shared by every attempt of a request, as the proxy passes it."""
    return Logging(
        model="primary",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        call_type="aresponses",
        litellm_call_id=call_id,
        start_time=datetime.now(),
        function_id=call_id,
    )


async def _drain_stream(router: litellm.Router, call_id: str) -> None:
    stream = await router.aresponses(
        model="primary", input="hello", stream=True, litellm_logging_obj=_request_logging_obj(call_id)
    )
    async for _event in stream:
        pass


def _primary_rate_limited_stream() -> bytes:
    resp = {
        "id": "resp_fake",
        "object": "response",
        "created_at": 0,
        "status": "in_progress",
        "model": "primary",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    events = [
        {"type": "response.created", "sequence_number": 0, "response": resp},
        {"type": "response.in_progress", "sequence_number": 1, "response": resp},
        {
            "type": "error",
            "sequence_number": 2,
            "error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "rate limit", "param": None},
        },
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


@pytest.mark.asyncio
async def test_mid_stream_429_does_not_cool_down_the_fallback_deployment(monkeypatch):
    seen_ids: list[str | None] = []
    callbacks_done = threading.Semaphore(0)

    def slow_failure_logger(kwargs, completion_response, start_time, end_time):
        time.sleep(0.15)  # stands in for other sync failure callbacks / host load
        seen_ids.append(((kwargs.get("litellm_params") or {}).get("model_info") or {}).get("id"))
        callbacks_done.release()

    monkeypatch.setattr(litellm, "failure_callback", [slow_failure_logger])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)  # so respx sees the calls

    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/primary", "api_base": PRIMARY_BASE, "api_key": "x"},
                "model_info": {"id": "primary-1"},
            },
            {
                "model_name": "backup",
                "litellm_params": {"model": "openai/backup", "api_base": BACKUP_BASE, "api_key": "a"},
                "model_info": {"id": "backup-a"},
            },
            {
                "model_name": "backup",
                "litellm_params": {"model": "openai/backup", "api_base": BACKUP_BASE, "api_key": "b"},
                "model_info": {"id": "backup-b"},
            },
        ],
        fallbacks=[{"primary": ["backup"]}],
        num_retries=0,
        cooldown_time=60,
    )

    with respx.mock(assert_all_called=False) as mock:
        primary_route = mock.post(f"{PRIMARY_BASE}/responses").mock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_primary_rate_limited_stream()
            )
        )
        backup_route = mock.post(f"{BACKUP_BASE}/responses").mock(
            return_value=httpx.Response(
                400,
                json={"error": {"message": "bad request", "type": "invalid_request_error", "param": None, "code": "x"}},
            )
        )

        for i in range(2):
            with pytest.raises(litellm.RateLimitError):  # primary's 429, surfaced after the backup's 400
                await _drain_stream(router, call_id=f"req-{i}")

    assert primary_route.call_count == 2 and backup_route.call_count == 2, (
        f"expected each request to stream from primary then fall back to backup; "
        f"primary calls={primary_route.call_count}, backup calls={backup_route.call_count}"
    )

    # Each request logs the primary's 429 once; wait for those sync handlers to finish.
    for _ in range(2):
        assert callbacks_done.acquire(timeout=5), f"sync failure handler never ran; saw {seen_ids}"

    assert not callbacks_done.acquire(timeout=0.5), f"more sync failure logs than requests: {seen_ids}"

    cooldown_ids = await async_get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None)
    assert not {"backup-a", "backup-b"} & set(cooldown_ids), (
        f"fallback deployments cooled down for the primary's mid-stream 429; "
        f"cooldown set={cooldown_ids}, failure callback saw model_info.id={seen_ids}"
    )
    # The backup's own 400 stays deduplicated on the shared Logging object, as before this fix.
    assert seen_ids == ["primary-1", "primary-1"], f"failure callbacks saw model_info.id={seen_ids}"
