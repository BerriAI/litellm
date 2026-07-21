"""Live e2e: RPM enforcement on the Redis-backed limiter path customers run.

Requires REDIS_HOST reachable from this process. A key with rpm_limit=1 must
serve the first chat and 429 the second.
"""

from __future__ import annotations

import os
import socket

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager
from models import KeyGenerateBody, LiteLLMParamsBody
from quota_client import QuotaClient

pytestmark = pytest.mark.e2e

BACKEND = "anthropic/claude-haiku-4-5-20251001"


def _require_redis_reachable() -> None:
    (host,) = require_env("REDIS_HOST")
    port = int((os.environ.get("REDIS_PORT") or "6379").strip() or "6379")
    try:
        with socket.create_connection((host, port), timeout=3):
            return
    except OSError as exc:
        raise AssertionError(
            f"REDIS_HOST={host!r} port={port} is not reachable ({exc}). "
            "Redis-backed rate limiting e2e needs a live Redis the proxy shares."
        ) from exc


class TestRedisBackedRateLimit:
    @pytest.mark.covers(
        "quota_management.ratelimit.redis_backed.blocks_over_limit",
        exercised_on=["chat_completions"],
    )
    def test_rpm_limit_one_blocks_second_call(
        self, client: QuotaClient, resources: ResourceManager
    ) -> None:
        _require_redis_reachable()
        model = f"e2e-redis-rpm-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))

        key = client.proxy.generate_key(
            KeyGenerateBody(
                models=[model],
                rpm_limit=1,
                key_alias=f"e2e-redis-rpm-{unique_marker()}",
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        info = client.proxy.key_info(key)
        assert info.rpm_limit == 1, f"key must echo rpm_limit=1: {info}"

        first = client.chat(key, model, f"ping {unique_marker()}")
        require_successful_call(first)

        second = client.chat(key, model, f"pong {unique_marker()}")
        assert second.status_code == 429, (
            f"second call over rpm_limit=1 must be 429, got {second.status_code}: "
            f"{second.body[:300]}"
        )
        assert "rate" in second.body.lower() or "limit" in second.body.lower(), (
            f"429 body should name the rate limit: {second.body[:300]}"
        )
