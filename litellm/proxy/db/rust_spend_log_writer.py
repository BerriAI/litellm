import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import SPEND_LOG_QUEUE_MAX_BYTES, SPEND_LOG_WRITE_BATCH_MAX_BYTES, SPEND_LOG_WRITE_BATCH_MAX_ROWS
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.rust_bridge._native import NativeDatabase, NativeDatabaseFailure, RustSpendLogWriter

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging

_ROW: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _Installed:
    client: object
    writer: RustSpendLogWriter


_installed: Final[list[_Installed]] = []  # mutable-ok: registry rewritten in place by install and uninstall


def rust_spend_log_writer(prisma_client: object) -> RustSpendLogWriter | None:
    return next((installed.writer for installed in _installed if installed.client is prisma_client), None)


def rust_spend_log_queue_size(prisma_client: object) -> int:
    writer: Final = rust_spend_log_writer(prisma_client)
    return 0 if writer is None else len(writer)


def spend_log_writer_runs_in_rust(general_settings: Mapping[str, object]) -> bool:
    return general_settings.get("use_spend_logs_ledger", True) is not False


def _json_default(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _encoded(row: Mapping[str, object]) -> str:
    try:
        return json.dumps(row, default=_json_default)
    except (TypeError, ValueError):
        return safe_dumps(row)


def enqueue_in_rust(
    writer: RustSpendLogWriter, logs: Sequence[Mapping[str, object]], *, at_head: bool, max_bytes: int
) -> None:
    writer.enqueue([_encoded(row) for row in logs], at_head, max_bytes)


def dequeue_from_rust(
    writer: RustSpendLogWriter, limit: int
) -> list[dict[str, object]]:  # mutable-ok: dequeue_spend_logs hands callers the rows the Python queue would
    return [_ROW.validate_json(text) for text in writer.dequeue(limit)]


def _parks_spend_logs(proxy_logging_obj: "ProxyLogging") -> bool:
    buffer: Final = proxy_logging_obj.db_spend_update_writer.redis_update_buffer
    return buffer.redis_cache is not None and RedisUpdateBuffer.should_commit_spend_updates_to_redis()


async def _park_failed_pass(
    writer: RustSpendLogWriter, prisma_client: "PrismaClient", proxy_logging_obj: "ProxyLogging", rows: int
) -> None:
    from litellm.proxy.utils import requeue_spend_logs

    if rows > 0 and _parks_spend_logs(proxy_logging_obj):
        await requeue_spend_logs(prisma_client, proxy_logging_obj, dequeue_from_rust(writer, rows))


def spend_log_retention() -> tuple[tuple[str, ...] | None, tuple[str, ...] | None] | None:
    from litellm.proxy.spend_tracking.spend_tracking_utils import configured_spend_logs_metadata_fields

    fields: Final = configured_spend_logs_metadata_fields()
    return None if fields is None else (fields.include, fields.exclude)


async def flush_in_rust(
    writer: RustSpendLogWriter, prisma_client: "PrismaClient", proxy_logging_obj: "ProxyLogging", limit: int
) -> None:
    """Write one pass from the Rust queue, park a failed pass in Redis when the buffer is on, and run the
    guardrail usage rollup on the written logs."""
    from litellm.proxy.guardrails.usage_tracking import (
        process_spend_logs_guardrail_usage,
        retry_pending_guardrail_usage,
    )
    from litellm.proxy.utils import raise_failed_update_spend_exception

    taken: Final = min(limit, len(writer))
    start_time: Final = time.time()
    try:
        written: Final = await writer.flush(limit, spend_log_retention())
    except NativeDatabaseFailure as e:
        await _park_failed_pass(writer, prisma_client, proxy_logging_obj, taken)
        raise_failed_update_spend_exception(e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj)
    try:
        if written:
            await process_spend_logs_guardrail_usage(prisma_client=prisma_client, logs_to_process=written)
        elif taken:
            await retry_pending_guardrail_usage(prisma_client)
    except Exception as guardrail_tracking_err:  # noqa: BLE001  # the rollup never fails the spend log job
        verbose_proxy_logger.warning(
            "Spend tracking - guardrail usage tracking failed (non-fatal): %s", guardrail_tracking_err
        )


def _request_flush_from_any_thread(loop: asyncio.AbstractEventLoop, prisma_client: "PrismaClient") -> None:
    from litellm.proxy.utils import request_spend_log_flush

    if not loop.is_closed():
        loop.call_soon_threadsafe(request_spend_log_flush, prisma_client)


async def install_rust_spend_log_writer(
    prisma_client: "PrismaClient",
    database: NativeDatabase,
    general_settings: Mapping[str, object],
) -> RustSpendLogWriter | None:
    if not spend_log_writer_runs_in_rust(general_settings):
        return None
    writer: Final = RustSpendLogWriter(
        database,
        queue_max_bytes=SPEND_LOG_QUEUE_MAX_BYTES,
        write_batch_max_bytes=SPEND_LOG_WRITE_BATCH_MAX_BYTES,
        write_batch_max_rows=SPEND_LOG_WRITE_BATCH_MAX_ROWS,
        on_flush_request=partial(_request_flush_from_any_thread, asyncio.get_running_loop(), prisma_client),
    )
    from litellm.proxy.utils import dequeue_spend_logs

    await uninstall_rust_spend_log_writer(prisma_client)
    pending: Final = await dequeue_spend_logs(prisma_client, sys.maxsize)
    _installed.append(_Installed(client=prisma_client, writer=writer))
    if pending:
        enqueue_in_rust(writer, pending, at_head=True, max_bytes=SPEND_LOG_QUEUE_MAX_BYTES)
    return writer


async def uninstall_rust_spend_log_writer(prisma_client: "PrismaClient") -> None:
    """Forget the client's writer, moving any rows still queued in Rust back to the Python queue."""
    from litellm.proxy.utils import enqueue_spend_logs

    writer: Final = rust_spend_log_writer(prisma_client)
    if writer is None:
        return
    await writer.wait_idle()
    remaining: Final = dequeue_from_rust(writer, sys.maxsize)
    _installed[:] = [installed for installed in _installed if installed.client is not prisma_client]
    if remaining:
        await enqueue_spend_logs(prisma_client, remaining, at_head=True)
