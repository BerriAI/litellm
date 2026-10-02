import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm._service_logger import ServiceTypes
from litellm.proxy.db.db_lookup_gate import bounded_db_lookup
from litellm.proxy.db.log_db_metrics import log_db_metrics
from litellm.proxy.db.prisma_client import _PrismaDrainTracker, _TrackedPrismaEngine


def _tracked_engine() -> _TrackedPrismaEngine:
    raw_engine: Final = SimpleNamespace(query=AsyncMock(return_value={"data": {}}))
    return _TrackedPrismaEngine(raw_engine, _PrismaDrainTracker())


@pytest.fixture
def success_hook() -> Iterator[AsyncMock]:
    hook: Final = AsyncMock()
    with patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        MagicMock(service_logging_obj=MagicMock(async_service_success_hook=hook)),
    ):
        yield hook


async def _db_call_types(hook: AsyncMock) -> tuple[str, ...]:
    await asyncio.sleep(0)
    return tuple(call.kwargs["call_type"] for call in hook.await_args_list if call.kwargs["service"] == ServiceTypes.DB)


@pytest.mark.asyncio
async def test_a_decorated_call_that_never_queries_the_engine_emits_no_db_event(success_hook: AsyncMock) -> None:
    @log_db_metrics
    async def cache_hit(**kwargs: object) -> str:
        return "cached"

    assert await cache_hit(parent_otel_span="span") == "cached"
    assert await _db_call_types(success_hook) == ()


@pytest.mark.asyncio
async def test_a_decorated_call_that_queries_the_engine_emits_one_db_event_named_after_it(
    success_hook: AsyncMock,
) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def read_user_row(**kwargs: object) -> object:
        return await engine.query("{}", tx_id=None)

    await read_user_row(parent_otel_span="span", table_name="LiteLLM_UserTable")

    assert await _db_call_types(success_hook) == ("read_user_row",)
    event: Final = success_hook.await_args_list[0].kwargs
    assert (event["parent_otel_span"], event["event_metadata"]) == ("span", {"table_name": "LiteLLM_UserTable"})


@pytest.mark.asyncio
async def test_a_query_behind_the_bounded_lookup_task_still_counts_for_the_enclosing_call(
    success_hook: AsyncMock,
) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def read_through_gate(**kwargs: object) -> object:
        return await bounded_db_lookup(engine.query("{}", tx_id=None), name="user")

    await read_through_gate()

    assert await _db_call_types(success_hook) == ("read_through_gate",)


@pytest.mark.asyncio
async def test_a_query_inside_a_nested_decorated_call_counts_for_both_callers(success_hook: AsyncMock) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def get_data(**kwargs: object) -> object:
        return await engine.query("{}", tx_id=None)

    @log_db_metrics
    async def get_key_object(**kwargs: object) -> object:
        return await get_data()

    await get_key_object()

    assert sorted(await _db_call_types(success_hook)) == ["get_data", "get_key_object"]


@pytest.mark.asyncio
async def test_a_cache_hit_after_a_sibling_db_read_emits_no_db_event(success_hook: AsyncMock) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def read_row(**kwargs: object) -> object:
        return await engine.query("{}", tx_id=None)

    @log_db_metrics
    async def cache_hit(**kwargs: object) -> str:
        return "cached"

    await read_row()
    await cache_hit()

    assert await _db_call_types(success_hook) == ("read_row",)
