from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
from litellm.proxy.health_check_utils.recent_traffic import RecentTrafficHealth


def _model(ident: str = "served", mode: str = "chat", **probe: object) -> dict[str, object]:
    return {
        "model_name": "shared group",
        "model_info": {"id": ident, "mode": mode, **probe},
        "litellm_params": {"model": "openai/test-model"},
    }


def _event(success: bool = True, **logged: object) -> dict[str, object]:
    return {
        "response_headers": {"content-type": "application/json"},
        "current_attempt_api_call_start_time": datetime.fromtimestamp(99, tz=timezone.utc),
        "standard_logging_object": {
            "model_id": "served",
            "call_type": "acompletion",
            "status": "success" if success else "failure",
            **logged,
        },
    }


@pytest.mark.asyncio
async def test_only_a_newer_success_of_the_same_deployment_and_api_suppresses_its_probe() -> None:
    clock: Final = Mock(return_value=100.0)
    observer: Final = RecentTrafficHealth(10, lambda: None, clock)
    candidates: Final = [_model(), _model("sibling")]
    older: Final = datetime.fromtimestamp(98, tz=timezone.utc)
    first: Final = datetime.fromtimestamp(99, tz=timezone.utc)
    now: Final = datetime.fromtimestamp(100, tz=timezone.utc)
    await observer.async_log_success_event(_event(), {}, older, first)
    snapshot: Final = await observer.snapshot(candidates)
    assert snapshot.successes == {"served": 99.0}
    assert not (await observer.snapshot([_model(mode="responses")])).successes
    assert not (await observer.snapshot([_model(health_check_model="special-probe")])).successes
    assert not (await observer.snapshot([_model(health_check_params={"temperature": 0.5})])).successes
    await observer.async_log_failure_event(_event(False), None, first, now)
    await observer.async_log_success_event(_event(), {}, older, first)
    failed: Final = await observer.snapshot(candidates)
    assert not failed.successes
    assert not failed.accepts(
        [{"model_id": "served", "health_check_source": "recent_traffic", "last_success_at": 99.0}], 99
    )
    clock.return_value = 101.0
    await observer.async_log_success_event(_event(), {}, now, datetime.fromtimestamp(101, tz=timezone.utc))
    assert (await observer.snapshot(candidates)).successes == {"served": 101.0}
    clock.return_value = 111.0
    assert not (await observer.snapshot(candidates)).successes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("details", "logged", "response"),
    [
        ({"cache_hit": True}, {}, {}),
        ({}, {"cache_hit": True}, {}),
        ({}, {"request_tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME]}, {}),
        ({}, {"metadata": {"user_api_key_team_id": LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME}}, {}),
        ({}, {"metadata": {"internal_call_origin": "shadow_eval"}}, {}),
        ({"litellm_params": {"mock_response": "ok"}}, {}, {}),
        ({"current_attempt_api_call_start_time": None}, {}, {}),
        ({"response_headers": None}, {}, {}),
        ({"httpx_response": httpx.Response(503)}, {}, {}),
        ({"stream": True}, {}, {}),
        ({}, {}, {"status": "in_progress"}),
        ({}, {"call_type": "aretrieve_responses"}, {}),
        ({}, {"model_id": None}, {}),
    ],
)
async def test_non_qualifying_events_leave_active_probes_enabled(
    details: Mapping[str, object], logged: Mapping[str, object], response: Mapping[str, object]
) -> None:
    observer: Final = RecentTrafficHealth(10, lambda: None, lambda: 100.0)
    now: Final = datetime.fromtimestamp(100, tz=timezone.utc)
    await observer.async_log_success_event({**_event(**logged), **details}, response, now, now)
    assert not (await observer.snapshot([_model()])).successes


@pytest.mark.asyncio
async def test_worker_waiting_for_the_probe_lock_sees_traffic_that_arrived_while_it_waited(
    router_redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import respx
    from litellm.proxy.health_check_utils import shared_health_check_manager as shared

    observer: Final = RecentTrafficHealth(10, lambda: None, lambda: 100.0)
    holder: Final = shared.SharedHealthCheckManager(redis_cache=router_redis[0], traffic_aware=True)
    waiter: Final = shared.SharedHealthCheckManager(redis_cache=router_redis[0], traffic_aware=True)
    assert await holder.acquire_health_check_lock()
    finished: Final = datetime.fromtimestamp(100, tz=timezone.utc)

    async def finish_request_while_waiting(_seconds: float) -> None:
        await observer.async_log_success_event(_event(), {}, finished, finished)
        await holder.release_health_check_lock()

    monkeypatch.setattr(shared, "asyncio", SimpleNamespace(sleep=finish_request_while_waiting))
    model: Final = {
        "model_info": {"id": "served", "mode": "chat"},
        "litellm_params": {"model": "openai/test-model", "api_base": "https://served.example/v1", "api_key": "test"},
    }
    with respx.mock(assert_all_called=False) as upstream:
        probe: Final = upstream.post("https://served.example/v1/chat/completions").mock(
            return_value=httpx.Response(500)
        )
        healthy, unhealthy, _ = await waiter.perform_shared_health_check([model], recent_traffic=observer)
    assert probe.call_count == 0
    assert not unhealthy
    assert [(ep["model_id"], ep["health_check_source"], ep["last_success_at"]) for ep in healthy] == [
        ("served", "recent_traffic", 100.0)
    ]
