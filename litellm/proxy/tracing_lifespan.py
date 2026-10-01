from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Final, TypedDict

from fastapi import HTTPException, Request
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import ReadOnly

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.clickhouse.clickhouse_spend_logger import ClickHouseSpendLogger
from litellm.tracing import TraceReceiver


class TracingState(TypedDict):
    tracing_receiver: ReadOnly[TraceReceiver | None]


_RECEIVER_ADAPTER: Final[TypeAdapter[TraceReceiver | None]] = TypeAdapter(
    TraceReceiver | None, config=ConfigDict(arbitrary_types_allowed=True)
)


def require_receiver(tracing: TraceReceiver | None) -> TraceReceiver:
    if tracing is None:
        raise HTTPException(
            status_code=501,
            detail="Agent tracing is not enabled. Set `tracing:` in general_settings and CLICKHOUSE_URL.",
        )
    return tracing


async def provide_receiver(request: Request) -> TraceReceiver | None:
    return _RECEIVER_ADAPTER.validate_python(getattr(request.state, "tracing_receiver", None))


async def _start_receiver(factory: Callable[[], TraceReceiver]) -> TraceReceiver | None:
    try:
        tracing: Final = factory()
        await tracing.start()
        return tracing
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        verbose_proxy_logger.warning("Agent tracing unavailable: %s", error)
        return None


@asynccontextmanager
async def tracing_lifespan(
    enabled: bool, receiver_factory: Callable[[], TraceReceiver] = TraceReceiver.from_env
) -> AsyncGenerator[TracingState, None]:
    tracing: Final = await _start_receiver(receiver_factory) if enabled else None
    state: Final[TracingState] = {"tracing_receiver": tracing}
    if tracing is None:
        yield state
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
        yield state
    finally:
        manager.remove_callback_from_all_lists(spend_logger)
        await spend_logger.aclose()
