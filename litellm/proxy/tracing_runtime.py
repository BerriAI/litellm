from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Final

from fastapi import HTTPException, Request
from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.clickhouse.clickhouse_spend_logger import ClickHouseSpendLogger
from litellm.rust_bridge.traces import ClickHouseStorage
from litellm.tracing import TraceReceiver

_RECEIVER_ADAPTER: Final[TypeAdapter[TraceReceiver | None]] = TypeAdapter(
    TraceReceiver | None, config=ConfigDict(arbitrary_types_allowed=True)
)
_UNAVAILABLE_DETAIL: Final = "Agent tracing is not enabled. Set `tracing:` in general_settings and CLICKHOUSE_URL."


def require_receiver(tracing: TraceReceiver | None) -> TraceReceiver:
    if tracing is None:
        raise HTTPException(status_code=501, detail=_UNAVAILABLE_DETAIL)
    return tracing


async def provide_receiver(request: Request) -> TraceReceiver | None:
    return _RECEIVER_ADAPTER.validate_python(getattr(request.state, "tracing_receiver", None))


async def provide_storage(request: Request) -> ClickHouseStorage | None:
    tracing: Final = await provide_receiver(request)
    return tracing.store.storage if tracing is not None else None


async def _start_receiver(factory: Callable[[], TraceReceiver]) -> TraceReceiver | None:
    try:
        tracing: Final = factory()
        await tracing.start()
        return tracing
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        verbose_proxy_logger.warning("Agent tracing unavailable: %s", error)
        return None


@asynccontextmanager
async def manage_tracing(
    enabled: bool, receiver_factory: Callable[[], TraceReceiver] = TraceReceiver.from_env
) -> AsyncGenerator[TraceReceiver | None, None]:
    tracing: Final = await _start_receiver(receiver_factory) if enabled else None
    if tracing is None:
        yield tracing
        return

    spend_logger: Final = ClickHouseSpendLogger(storage=tracing.store.storage)
    manager: Final = litellm.logging_callback_manager
    manager.add_litellm_callback(spend_logger)
    manager.add_litellm_success_callback(spend_logger)
    manager.add_litellm_failure_callback(spend_logger)
    manager.add_litellm_async_success_callback(spend_logger)
    manager.add_litellm_async_failure_callback(spend_logger)
    verbose_proxy_logger.info("Agent tracing enabled (store=clickhouse)")
    try:
        yield tracing
    finally:
        manager.remove_callback_from_all_lists(spend_logger)
        await spend_logger.aclose()
