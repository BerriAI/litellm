"""Live e2e: Redis-backed rate limit path stays responsive (LIT-3523 shape).

With Redis up, burst past rpm_limit=1, then a fresh key must still complete a
chat in well under REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT.
"""

from __future__ import annotations

import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager
from models import KeyGenerateBody, LiteLLMParamsBody
from quota_client import QuotaClient

pytestmark = pytest.mark.e2e

BACKEND = "anthropic/claude-haiku-4-5-20251001"
RECOVERY_TIMEOUT = float(
    os.environ.get("REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60") or "60"
)


def _require_redis() -> None:
    (host,) = require_env("REDIS_HOST")
    port = int((os.environ.get("REDIS_PORT") or "6379").strip() or "6379")
    try:
        with socket.create_connection((host, port), timeout=3):
            return
    except OSError as exc:
        raise AssertionError(
            f"REDIS_HOST={host!r}:{port} unreachable ({exc}); "
            "LIT-3523 e2e needs Redis the proxy shares."
        ) from exc


class TestRedisCircuitBreakerPath:
    @pytest.mark.covers(
        "reliability.circuit_breaker.redis.trips_then_recovers",
        exercised_on=["chat_completions"],
    )
    def test_burst_rate_limit_does_not_freeze_fresh_key(
        self, client: QuotaClient, resources: ResourceManager
    ) -> None:
        _require_redis()
        model = f"e2e-cb-model-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))

        hot_key = client.proxy.generate_key(
            KeyGenerateBody(
                models=[model],
                rpm_limit=1,
                key_alias=f"e2e-cb-hot-{unique_marker()}",
            )
        )
        resources.defer(lambda: client.proxy.delete_key(hot_key))
        cool_key = client.proxy.generate_key(
            KeyGenerateBody(models=[model], key_alias=f"e2e-cb-cool-{unique_marker()}")
        )
        resources.defer(lambda: client.proxy.delete_key(cool_key))

        def _hit() -> int:
            return client.chat(hot_key, model, f"burst {unique_marker()}").status_code

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_hit) for _ in range(12)]
            codes = tuple(f.result() for f in as_completed(futures))
        assert any(code == 429 for code in codes), (
            f"expected some 429 under rpm_limit=1 burst, got {codes}"
        )

        started = time.monotonic()
        cool = client.chat(cool_key, model, f"fresh {unique_marker()}")
        elapsed = time.monotonic() - started
        require_successful_call(cool)
        assert elapsed < RECOVERY_TIMEOUT * 0.5, (
            f"fresh key chat took {elapsed:.1f}s after redis rate-limit burst; "
            f"customers treat hangs near recovery_timeout={RECOVERY_TIMEOUT}s as "
            "LIT-3523 circuit-breaker pain"
        )
