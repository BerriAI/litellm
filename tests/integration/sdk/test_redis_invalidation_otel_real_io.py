import asyncio
import os
from collections.abc import Iterator
from typing import Final
from uuid import uuid4

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext

import litellm
from litellm.caching.redis_cache import RedisCache
from litellm.integrations.otel import OpenTelemetryV2Config
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import is_otel_v2_enabled
from litellm.integrations.otel.plumbing import providers
from litellm.integrations.otel.runtime import background_job_span
from litellm.proxy import proxy_server
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore
from litellm.redis.stores.cache_invalidation import PythonCaches, connect_invalidation
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.rust_bridge._native import NativeBudgets
from litellm.types.integrations.custom_logger_v2 import BudgetPlan, BudgetPlanCounter


@pytest.fixture
def exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[InMemorySpanExporter]:
    with monkeypatch.context() as settings:
        settings.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()
        memory: Final = InMemorySpanExporter()
        config: Final = OpenTelemetryV2Config(exporter="in_memory")
        provider: Final = providers.build_tracer_provider(config, exporter=memory)
        logger: Final = OpenTelemetryV2(config=config, tracer_provider=provider)
        settings.setattr(litellm, "service_callback", [logger])
        settings.setattr(proxy_server, "open_telemetry_logger", logger)
        try:
            yield memory
        finally:
            is_otel_v2_enabled.cache_clear()


async def _redis_reports(exporter: InMemorySpanExporter, count: int) -> list[tuple[str, SpanContext | None]]:
    async with asyncio.timeout(10):
        while True:
            reports = [
                (str(span.attributes["call_type"]), span.parent)
                for span in exporter.get_finished_spans()
                if span.attributes is not None and span.attributes.get("service") == "redis"
            ]
            if len(reports) >= count:
                return reports
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_native_cache_invalidation_preserves_the_job_on_both_publish_paths(
    exporter: InMemorySpanExporter,
) -> None:
    redis: Final = connect(RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])))
    channel: Final = connect_invalidation(
        redis,
        None,
        AuthCacheStore(redis.auth_cache(AuthCacheSettings(200, 60))),
        SpendCounterStore(redis.spend_counters()),
        PythonCaches(),
    )
    assert channel is not None

    with background_job_span("invalidate_auth_cache"):
        parent: Final = trace.get_current_span().get_span_context()
        assert channel.publish("team_id:a")
        await channel.broadcast_evictions(("team_id:a", "team_id:b"))
        calls = await _redis_reports(exporter, 2)
    assert sorted(calls, key=lambda call: call[0]) == [("PUBLISH", parent), ("pipeline", parent)]


@pytest.mark.asyncio
async def test_native_budget_reservation_keeps_its_redis_operations_under_the_job(
    exporter: InMemorySpanExporter,
) -> None:
    redis: Final = connect(RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])))
    counter: Final = f"spend:key:{uuid4().hex}"
    plan: Final = BudgetPlan(
        counters=(
            BudgetPlanCounter(
                counter_key=counter, max_budget=10.0, fallback_spend=0.0, entity_type="key", entity_id="k"
            ),
        ),
        reservation_cost=1.0,
        input_cost=0.0,
        status_code=429,
    )

    with background_job_span("reserve_budget"):
        parent: Final = trace.get_current_span().get_span_context()
        admission: Final = await NativeBudgets(redis.spend_counters()).reserve(
            plan, ({"counter_key": counter, "spend": 0.0},)
        )
        calls = await _redis_reports(exporter, 1)

    assert isinstance(admission, dict) and admission["admission"] == "held"
    assert all(call_parent == parent for _, call_parent in calls)
