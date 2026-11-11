import json
import os
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import AlertType
from litellm.types.integrations.slack_alerting import SlackAlertingCacheKeys
from redis import Redis

REPORT_SENT_KEY: Final = SlackAlertingCacheKeys.report_sent_key.value
FAILED_REQUESTS: Final = 3
API_BASE: Final = "http://daily-report-upstream.invalid/v1"


@pytest.fixture
def redis_client() -> Iterator[Redis]:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as client:
        client.delete(REPORT_SENT_KEY)
        yield client
        client.delete(REPORT_SENT_KEY)


def _accept(request: Request) -> Reply:
    return Reply(body=b"ok", content_type="text/plain")


def _pod(webhook: Wire) -> SlackAlerting:
    redis_cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    return SlackAlerting(
        internal_usage_cache=DualCache(redis_cache=redis_cache),
        alerting=["slack"],
        alert_types=[AlertType.daily_reports],
        alerting_args={"daily_report_frequency": 0},
        default_webhook_url=webhook.url,
    )


def _router(deployment_id: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "daily-report",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": API_BASE,
                    "api_key": "synthetic-daily-report-key",
                },
                "model_info": {"id": deployment_id},
            }
        ]
    )


@pytest.mark.asyncio
async def test_the_report_timestamp_one_pod_stores_in_redis_drives_the_next_pods_daily_report(
    redis_client: Redis,
) -> None:
    deployment_id: Final = f"daily-report-{uuid.uuid4().hex}"
    failed_key: Final = f"{deployment_id}:{SlackAlertingCacheKeys.failed_requests_key.value}"
    redis_client.set(failed_key, json.dumps(FAILED_REQUESTS), ex=300)
    router: Final = _router(deployment_id)
    with wire_server(_accept) as webhook:
        first_pod: Final = _pod(webhook)
        assert await first_pod._run_scheduler_helper(llm_router=router) is False
        stored: Final = redis_client.get(REPORT_SENT_KEY)
        assert stored is not None
        first_sent: Final = json.loads(stored)
        assert isinstance(first_sent, float), stored
        await first_pod.flush_queue()
        assert webhook.drain() == ()

        second_pod: Final = _pod(webhook)
        assert await second_pod._run_scheduler_helper(llm_router=router) is True
        await second_pod.flush_queue()
        delivered: Final = webhook.drain()
        assert len(delivered) == 1
        text: Final = json.loads(delivered[0].body)["text"]
        assert f"Failed Requests: `{FAILED_REQUESTS}`" in text, text
        assert API_BASE in text, text
        assert json.loads(redis_client.get(failed_key) or "null") == 0
        assert float(json.loads(redis_client.get(REPORT_SENT_KEY) or "null")) >= first_sent
        redis_client.delete(failed_key)
