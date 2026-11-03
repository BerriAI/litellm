import json
import os
import uuid
from itertools import chain
from typing import Final

import litellm
import pytest
from integration._support.wire import Reply, Request, wire_server
from litellm import Router
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from prometheus_client import REGISTRY

CHAT_RESPONSE: Final = json.dumps(
    {
        "id": "chatcmpl_redis_service_metrics",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "metrics"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
).encode()
LABELS: Final = {"redis": "redis"}


def _reply(request: Request) -> Reply:
    return Reply(body=CHAT_RESPONSE)


def _redis_metrics() -> tuple[float, float, float]:
    failed_metrics: Final = tuple(
        metric for metric in REGISTRY.collect() if metric.name == "litellm_redis_failed_requests"
    )
    samples: Final = chain.from_iterable(metric.samples for metric in failed_metrics)
    failed: Final = sum(sample.value for sample in samples if sample.name.endswith("_total"))
    return (
        REGISTRY.get_sample_value("litellm_redis_total_requests_total", LABELS) or 0.0,
        REGISTRY.get_sample_value("litellm_redis_latency_count", LABELS) or 0.0,
        failed,
    )


@pytest.mark.asyncio
async def test_router_redis_traffic_is_counted_in_the_prometheus_service_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "service_callback", ["prometheus_system"])
    with wire_server(_reply) as wire:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "redis-metrics",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": f"{wire.url}/v1",
                        "api_key": "synthetic-redis-metrics-key",
                        "tpm": tpm,
                    },
                }
                for tpm in (100, 1000)
            ],
            routing_strategy="usage-based-routing-v2",
            redis_host=os.environ["REDIS_HOST"],
            redis_port=int(os.environ["REDIS_PORT"]),
        )
        before: Final = _redis_metrics()
        responses: Final = [
            await router.acompletion(
                model="redis-metrics", messages=[{"role": "user", "content": f"metrics {uuid.uuid4().hex}"}]
            )
            for _ in range(2)
        ]
        await GLOBAL_LOGGING_WORKER.flush()
        after: Final = _redis_metrics()
        assert [response.usage.total_tokens for response in responses] == [7, 7]
        assert len(wire.drain()) == 2
    total_delta, latency_delta, failed_delta = (now - then for now, then in zip(after, before, strict=True))
    assert total_delta > 0, (before, after)
    assert latency_delta > 0, (before, after)
    assert failed_delta == 0, (before, after)
