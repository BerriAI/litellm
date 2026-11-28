import asyncio
import atexit
import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Final, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.redact_messages import (
    redacted_standard_logging_payload,
    should_redact_message_logging,  # pyright: ignore[reportUnknownVariableType]  # legacy redaction helper has an untyped dict parameter
)

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_COLUMNS: Final = (
    "event_id",
    "request_id",
    "status",
    "model",
    "start_time",
    "end_time",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "response_cost",
    "payload",
)
_INTEGER: Final[TypeAdapter[int | None]] = TypeAdapter(int | None)
_FLOAT: Final[TypeAdapter[float | None]] = TypeAdapter(float | None)
Row: TypeAlias = tuple[str | datetime | int | float | None, ...]


def _without_sensitive_fields(payload: Mapping[str, object]) -> Mapping[str, object]:
    details: Final = payload.get("error_information")
    return {
        **{
            key: value
            for key, value in payload.items()
            if key not in ("error_str", "metadata", "model_parameters", "response")
        },
        **(
            {
                "error_information": {
                    key: value
                    for key, value in _MAPPING.validate_python(details).items()
                    if key not in ("error_message", "traceback")
                }
            }
            if details is not None
            else {}
        ),
    }


class PostgresOlapConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dsn: SecretStr = Field(repr=False)
    schema_name: str = Field(default="public", min_length=1)
    table_name: str = Field(default="litellm_logs", min_length=1)
    batch_size: int = Field(default=100, ge=1, le=1000)
    flush_interval: float = Field(default=1.0, gt=0)
    max_queue_bytes: int = Field(default=16 * 1024 * 1024, ge=1)
    max_event_bytes: int = Field(default=1024 * 1024, ge=1)
    connect_timeout: int = Field(default=5, ge=1)
    statement_timeout_ms: int = Field(default=10000, ge=1)
    retries: int = Field(default=2, ge=0, le=5)
    shutdown_timeout: float = Field(default=10.0, gt=0)
    turn_off_message_logging: bool = True


def write_batch(config: PostgresOlapConfig, rows: Sequence[Row]) -> None:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict

    if not rows:
        return
    placeholders: Final = sql.SQL("({})").format(sql.SQL(", ").join(sql.Placeholder() for _ in _COLUMNS))
    values: Final = sql.SQL(", ").join(placeholders for _ in rows)
    statement: Final = sql.SQL("INSERT INTO {}.{} ({}) VALUES {}").format(
        sql.Identifier(config.schema_name),
        sql.Identifier(config.table_name),
        sql.SQL(", ").join(map(sql.Identifier, _COLUMNS)),
        values,
    )
    parameters: Final = tuple(value for row in rows for value in row)  # comprehension-ok: flatten SQL rows
    dsn: Final = config.dsn.get_secret_value()
    sslmode: Final = conninfo_to_dict(dsn).get("sslmode") or os.environ.get("PGSSLMODE") or "verify-full"
    with (
        psycopg.connect(
            dsn,
            sslmode=sslmode,
            autocommit=True,
            connect_timeout=config.connect_timeout,
            cursor_factory=psycopg.ClientCursor,
        ) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(sql.SQL("SET statement_timeout = {}").format(sql.Literal(config.statement_timeout_ms)))
        cursor.execute(statement, parameters)


class PostgresOlapLogger(CustomLogger):
    def __init__(
        self,
        config: PostgresOlapConfig | None = None,
        writer: Callable[[PostgresOlapConfig, Sequence[Row]], None] = write_batch,
    ) -> None:
        self.config: Final = config or PostgresOlapConfig(dsn=SecretStr(os.environ["POSTGRES_OLAP_DSN"]))
        super().__init__(turn_off_message_logging=self.config.turn_off_message_logging)
        self._writer: Final = writer
        self._condition: Final = threading.Condition()
        self._queue: Final[deque[tuple[Row, int]]] = deque()  # mutable-ok: condition-protected producer/consumer queue
        self._bytes = 0
        self._accepted = 0
        self._completed = 0
        self._delivered = 0
        self._dropped = 0
        self._closing = False
        self._flush_requested = False
        self._thread: Final = threading.Thread(target=self._run, name="litellm-postgres-olap", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    @property
    def statistics(self) -> Mapping[str, int]:
        with self._condition:
            return {
                "accepted": self._accepted,
                "delivered": self._delivered,
                "dropped": self._dropped,
                "pending": self._accepted - self._completed,
            }

    def _log(self, kwargs: Mapping[str, object], start_time: datetime, end_time: datetime) -> None:
        if kwargs.get("standard_logging_object") is None:
            return
        try:
            details: Final = self.redact_standard_logging_payload_from_model_call_details(dict(kwargs))  # pyright: ignore[reportUnknownMemberType]  # legacy base accepts an untyped dict; the result is validated below
            payload: Final = _MAPPING.validate_python(details.get("standard_logging_object"))
            redacted: Final = (
                _without_sensitive_fields(redacted_standard_logging_payload(payload))
                if self.turn_off_message_logging or should_redact_message_logging(dict(kwargs))
                else payload
            )
            serialized: Final = json.dumps(redacted, ensure_ascii=False, allow_nan=False, default=str)
            row: Final[Row] = (
                str(uuid4()),
                str(redacted.get("id", "")),
                str(redacted.get("status", "")),
                str(redacted.get("model", "")),
                start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc),
                end_time if end_time.tzinfo else end_time.replace(tzinfo=timezone.utc),
                _INTEGER.validate_python(redacted.get("prompt_tokens")),
                _INTEGER.validate_python(redacted.get("completion_tokens")),
                _INTEGER.validate_python(redacted.get("total_tokens")),
                _FLOAT.validate_python(redacted.get("response_cost")),
                serialized,
            )
            size: Final = (
                len(serialized.encode("utf-8")) + sum(len(str(value).encode("utf-8")) for value in row[:4]) + 128
            )
            with self._condition:
                if (
                    self._closing
                    or size > self.config.max_event_bytes
                    or self._bytes + size > self.config.max_queue_bytes
                ):
                    self._dropped += 1
                    return
                self._queue.append((row, size))
                self._bytes += size
                self._accepted += 1
                self._condition.notify_all()
        except Exception:  # noqa: BLE001  # logging must not alter model responses or disclose payloads
            with self._condition:
                self._dropped += 1

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._log(kwargs, start_time, end_time)

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._log(kwargs, start_time, end_time)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await asyncio.to_thread(self._log, kwargs, start_time, end_time)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await asyncio.to_thread(self._log, kwargs, start_time, end_time)

    def _send(self, rows: Sequence[Row]) -> bool:
        for attempt in range(self.config.retries + 1):
            try:
                self._writer(self.config, rows)
            except Exception as error:  # noqa: BLE001  # driver messages can contain credentials or rejected log contents
                verbose_logger.warning("Postgres OLAP batch failed (%s)", type(error).__name__)
                if attempt < self.config.retries:
                    time.sleep(min(float(1 << attempt), self.config.flush_interval))
            else:
                return True
        return False

    def _run(self) -> None:
        while self._flush_batch():
            pass

    def _flush_batch(self) -> bool:
        with self._condition:
            self._condition.wait_for(
                lambda: self._closing or self._flush_requested or len(self._queue) >= self.config.batch_size,
                timeout=self.config.flush_interval,
            )
            batch: Final = tuple(self._queue.popleft() for _ in range(min(len(self._queue), self.config.batch_size)))
            self._flush_requested = self._flush_requested and bool(self._queue)
            if not batch:
                return not self._closing
        delivered: Final = self._send(tuple(item[0] for item in batch))
        with self._condition:
            self._bytes -= sum(item[1] for item in batch)
            self._completed += len(batch)
            if delivered:
                self._delivered += len(batch)
            else:
                self._dropped += len(batch)
            self._condition.notify_all()
        return True

    def flush(self, timeout: float | None = None) -> bool:
        with self._condition:
            target: Final = self._accepted
            self._flush_requested = True
            self._condition.notify_all()
            return self._condition.wait_for(
                lambda: self._completed >= target,
                self.config.shutdown_timeout if timeout is None else max(0.0, timeout),
            )

    async def flush_queue(self) -> None:
        await asyncio.to_thread(self.flush)

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(self.config.shutdown_timeout)
        if self._thread.is_alive():
            verbose_logger.warning("Postgres OLAP shutdown timed out with pending events")
        else:
            atexit.unregister(self.close)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    async def async_close_proxy(self) -> None:
        await self.aclose()
