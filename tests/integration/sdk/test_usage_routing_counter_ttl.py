import json
import os
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.wire import Reply, Request, Wire, wire_server
from litellm import Router
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from redis import Redis

COUNTER_TTL_SECONDS: Final = 60
CHAT_RESPONSE: Final = json.dumps(
    {
        "id": "chatcmpl_usage_counter_ttl",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ttl"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
    }
).encode()


def _reply(request: Request) -> Reply:
    assert request.target == "/v1/chat/completions", request.target
    return Reply(body=CHAT_RESPONSE)


@pytest.fixture
def redis_client() -> Iterator[Redis]:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as client:
        yield client


def _router(wire: Wire, deployment_id: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "usage-ttl",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{wire.url}/v1",
                    "api_key": "synthetic-usage-ttl-key",
                    "tpm": 1440,
                },
                "model_info": {"id": deployment_id},
            }
        ],
        routing_strategy="usage-based-routing-v2",
        redis_host=os.environ["REDIS_HOST"],
        redis_port=int(os.environ["REDIS_PORT"]),
    )


def _counter_ttls(redis_client: Redis, deployment_id: str) -> dict[str, int]:
    keys: Final = tuple(redis_client.scan_iter(match=f"{deployment_id}:*"))
    return {key: int(redis_client.ttl(key)) for key in keys}


def _expiring(ttls: dict[str, int]) -> bool:
    kinds: Final = {key.split(":")[-2] for key in ttls}
    return "tpm" in kinds and all(0 < ttl <= COUNTER_TTL_SECONDS for ttl in ttls.values())


@pytest.mark.asyncio
async def test_async_usage_counters_land_in_redis_with_a_one_minute_expiry(redis_client: Redis) -> None:
    deployment_id: Final = f"usage-ttl-{uuid.uuid4().hex}"
    with wire_server(_reply) as wire:
        router: Final = _router(wire, deployment_id)
        response: Final = await router.acompletion(
            model="usage-ttl", messages=[{"role": "user", "content": f"async {uuid.uuid4().hex}"}]
        )
        assert response.usage.total_tokens == 11
        await GLOBAL_LOGGING_WORKER.flush()
        ttls: Final = eventually(
            lambda: _counter_ttls(redis_client, deployment_id), _expiring, seconds=15, return_last_on_timeout=True
        )
        assert _expiring(ttls), ttls
        assert len(wire.drain()) == 1


def test_sync_usage_counters_land_in_redis_with_a_one_minute_expiry(redis_client: Redis) -> None:
    deployment_id: Final = f"usage-ttl-{uuid.uuid4().hex}"
    with wire_server(_reply) as wire:
        router: Final = _router(wire, deployment_id)
        response: Final = router.completion(
            model="usage-ttl", messages=[{"role": "user", "content": f"sync {uuid.uuid4().hex}"}]
        )
        assert response.usage.total_tokens == 11
        ttls: Final = eventually(
            lambda: _counter_ttls(redis_client, deployment_id), _expiring, seconds=15, return_last_on_timeout=True
        )
        assert _expiring(ttls), ttls
        assert len(wire.drain()) == 1
