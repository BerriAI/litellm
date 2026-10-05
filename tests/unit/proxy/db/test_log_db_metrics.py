import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.errors import PrismaError

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


@pytest.fixture
def failure_hook() -> Iterator[AsyncMock]:
    hook: Final = AsyncMock()
    with patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        MagicMock(service_logging_obj=MagicMock(async_service_failure_hook=hook)),
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
async def test_one_query_inside_a_nested_decorated_call_emits_only_the_inner_event(success_hook: AsyncMock) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def get_data(**kwargs: object) -> object:
        return await engine.query("{}", tx_id=None)

    @log_db_metrics
    async def get_key_object(**kwargs: object) -> object:
        return await get_data()

    await get_key_object()

    assert await _db_call_types(success_hook) == ("get_data",)


@pytest.mark.asyncio
async def test_an_outer_call_that_also_queries_outside_the_inner_call_emits_its_own_event(
    success_hook: AsyncMock,
) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def get_object_permission(**kwargs: object) -> object:
        return await engine.query("{}", tx_id=None)

    @log_db_metrics
    async def get_key_object(**kwargs: object) -> object:
        await engine.query("{}", tx_id=None)
        return await get_object_permission()

    await get_key_object()

    assert await _db_call_types(success_hook) == ("get_object_permission", "get_key_object")


@pytest.mark.asyncio
async def test_a_query_inside_an_inner_call_that_fails_without_a_db_error_is_reported_by_the_outer_call(
    success_hook: AsyncMock,
) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def read_row(**kwargs: object) -> object:
        await engine.query("{}", tx_id=None)
        raise ValueError("row did not validate")

    @log_db_metrics
    async def get_key_object(**kwargs: object) -> str:
        try:
            await read_row()
        except ValueError:
            return "fallback"
        return "row"

    assert await get_key_object() == "fallback"
    assert await _db_call_types(success_hook) == ("get_key_object",)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lookup", "table_name"),
    [
        ({"token": "sk-hashed"}, "key"),
        ({"tokens": ["sk-hashed"]}, "key"),
        ({"user_id": "u-1"}, "user"),
        ({"team_id": "t-1"}, "team"),
        ({"token": "sk-hashed", "user_id": "u-1"}, "key"),
        ({"table_name": "spend", "token": "sk-hashed"}, "spend"),
    ],
)
async def test_a_crud_method_called_without_table_name_reports_the_table_its_lookup_key_selects(
    success_hook: AsyncMock, lookup: dict[str, object], table_name: str
) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def get_data(*, table_name: str | None = None, **kwargs: object) -> object:
        return await engine.query("{}", tx_id=None)

    await get_data(**lookup)

    await asyncio.sleep(0)
    assert success_hook.await_args_list[0].kwargs["event_metadata"] == {"table_name": table_name}


@pytest.mark.asyncio
async def test_a_helper_without_a_table_name_parameter_gets_no_inferred_table(success_hook: AsyncMock) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def get_team_member_default_budget(*, team_id: str, user_id: str) -> object:
        return await engine.query("{}", tx_id=None)

    await get_team_member_default_budget(team_id="t-1", user_id="u-1")

    await asyncio.sleep(0)
    assert success_hook.await_args_list[0].kwargs["event_metadata"] is None


_FIND_UNIQUE_KEY_PAYLOAD: Final = (
    '{"query": "query { result: findUniqueLiteLLM_VerificationToken(where: {token: \\"h\\"}) { token } }"}'
)
_RAW_SELECT_PAYLOAD: Final = '{"query": "mutation { result: queryRaw(query: \\"SELECT 1 FROM \\\\\\"LiteLLM_UserTable\\\\\\"\\", parameters: \\"[]\\") }"}'


@pytest.mark.asyncio
async def test_an_undecorated_prisma_query_emits_one_db_event_named_from_the_engine_payload(
    success_hook: AsyncMock,
) -> None:
    engine: Final = _tracked_engine()

    await engine.query(_RAW_SELECT_PAYLOAD, tx_id=None)
    await engine.query(_FIND_UNIQUE_KEY_PAYLOAD, tx_id=None)

    assert await _db_call_types(success_hook) == ("query_raw", "find_unique")
    raw, model = (call.kwargs["event_metadata"] for call in success_hook.await_args_list)
    assert raw == {"table_name": "LiteLLM_UserTable", "db_operation": "select"}
    assert model == {"table_name": "LiteLLM_VerificationToken", "db_operation": "select"}


@pytest.mark.asyncio
async def test_a_decorated_call_owns_its_query_so_the_engine_fallback_stays_silent(success_hook: AsyncMock) -> None:
    engine: Final = _tracked_engine()

    @log_db_metrics
    async def read_key_row(**kwargs: object) -> object:
        return await engine.query(_FIND_UNIQUE_KEY_PAYLOAD, tx_id=None)

    await read_key_row(parent_otel_span="span", token="h")

    assert await _db_call_types(success_hook) == ("read_key_row",)


@pytest.mark.asyncio
async def test_a_task_spawned_by_a_decorated_call_that_queries_after_it_returned_emits_its_own_event(
    success_hook: AsyncMock,
) -> None:
    engine: Final = _tracked_engine()
    released: Final = asyncio.Event()

    async def write_after_the_caller_returned() -> object:
        await released.wait()
        return await engine.query(_FIND_UNIQUE_KEY_PAYLOAD, tx_id=None)

    @log_db_metrics
    async def read_key_row(**kwargs: object) -> asyncio.Task[object]:
        await engine.query(_FIND_UNIQUE_KEY_PAYLOAD, tx_id=None)
        return asyncio.create_task(write_after_the_caller_returned())

    background: Final = await read_key_row(token="h")
    released.set()
    await background

    assert await _db_call_types(success_hook) == ("read_key_row", "find_unique")


@pytest.mark.asyncio
async def test_a_raising_failure_hook_never_replaces_the_prisma_error(failure_hook: AsyncMock) -> None:
    failure_hook.side_effect = RuntimeError("exporter down")

    @log_db_metrics
    async def insert_data(**kwargs: object) -> None:
        raise PrismaError("connection reset")

    with pytest.raises(PrismaError, match="connection reset"):
        await insert_data(table_name="key")

    assert failure_hook.await_count == 1
    assert failure_hook.await_args_list[0].kwargs["call_type"] == "insert_data"
