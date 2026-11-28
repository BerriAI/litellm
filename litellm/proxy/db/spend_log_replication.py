from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from time import monotonic
from typing import Annotated, Final, TypeAlias
from urllib.parse import parse_qsl, unquote, urlsplit
from uuid import uuid4

import psycopg
from psycopg import Connection
from psycopg.conninfo import make_conninfo
from psycopg.rows import TupleRow
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.fleet_budget_spend import replicated_budget_entity_id
from litellm.repositories.spend_log_insert import replicated_spend_log_insert

STREAM_NAME: Final = "litellm-spend-logs"
_BATCH_SIZE: Final = 100
_MAX_RECORD_BYTES: Final = 256 * 1024
_MAX_DISPLAY_FIELD_BYTES: Final = 2048
_CLAIM_IDLE_MS: Final = 30_000
_TRIM_INTERVAL_SECONDS: Final = 60
_RETENTION_MS: Final = 7 * 24 * 60 * 60 * 1000
StreamEntry: TypeAlias = tuple[str, dict[str, str]]
_STREAM_READ: Final = TypeAdapter(tuple[tuple[str, tuple[StreamEntry, ...]], ...])
_STREAM_CLAIM: Final = TypeAdapter(tuple[str, tuple[StreamEntry, ...], tuple[str, ...]])
_OUTBOX_BATCH: Final = TypeAdapter(tuple[tuple[int, str], ...])
_REQUEST_ID_KEY_SETTING: Final = "LITELLM_SPEND_LOG_REQUEST_ID_KEY"
_TRIM_ACKNOWLEDGED: Final = """
local function less_number(a, b)
    return #a < #b or (#a == #b and a < b)
end
local function earlier(a, b)
    local am, as = string.match(a, '^(%d+)-(%d+)$')
    local bm, bs = string.match(b, '^(%d+)-(%d+)$')
    if am == bm then return less_number(as, bs) end
    return less_number(am, bm)
end
local key = KEYS[1]
local groups = redis.call('XINFO', 'GROUPS', key)
if #groups == 0 then return 0 end
local now = redis.call('TIME')
local cutoff = string.format('%.0f-0', math.max(0, now[1] * 1000 + math.floor(now[2] / 1000) - ARGV[1]))
for _, info in ipairs(groups) do
    local group = {}
    for i = 1, #info, 2 do group[info[i]] = info[i + 1] end
    local delivered = group['last-delivered-id']
    if earlier(delivered, cutoff) then cutoff = delivered end
    local pending = redis.call('XPENDING', key, group['name'])
    if pending[1] > 0 and earlier(pending[2], cutoff) then cutoff = pending[2] end
end
return redis.call('XTRIM', key, 'MINID', '~', cutoff)
"""
_PRISMA_QUERY_OPTIONS: Final = frozenset(
    {
        "connection_limit",
        "schema",
        "pool_timeout",
        "socket_timeout",
        "pgbouncer",
        "max_idle_connection_lifetime",
        "max_connection_lifetime",
        "statement_cache_size",
    }
)


class SpendLogMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    request_id: Annotated[str, Field(min_length=1)]
    call_type: str
    api_key: Annotated[str, Field(pattern=r"^(?:[a-fA-F0-9]{64})?$")] = ""
    spend: float = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    startTime: str
    endTime: str
    request_duration_ms: int | None = None
    completionStartTime: str | None = None
    model: str = ""
    model_id: str | None = None
    model_group: str | None = None
    custom_llm_provider: str | None = None
    user: str | None = None
    end_user: str | None = None
    team_id: str | None = None
    organization_id: str | None = None
    status: str | None = None
    created_at: str
    updated_at: str


class SpendLogEvent(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    region: Annotated[str, Field(min_length=1)]
    row: SpendLogMetadata


@contextmanager
def connect_regional_db(database_url: str) -> Generator[Connection[TupleRow]]:
    base, query_separator, query = database_url.partition("?")
    uri_query: Final = query if base.startswith(("postgresql://", "postgres://")) else ""
    schema: Final = dict(parse_qsl(uri_query, keep_blank_values=True)).get("schema")
    libpq_query: Final = "&".join(
        field for field in uri_query.split("&") if unquote(field.partition("=")[0]) not in _PRISMA_QUERY_OPTIONS
    )
    libpq_url: Final = base + (query_separator + libpq_query if libpq_query else "") if uri_query else database_url
    dsn: Final = make_conninfo(libpq_url, connect_timeout=5)
    with psycopg.connect(dsn, autocommit=True, prepare_threshold=None) as db:
        if schema is not None:
            db.execute("SELECT set_config('search_path', quote_ident(%s), false)", (schema,))
        yield db


def enable_capture(db: Connection[TupleRow], region: str, request_id_key: str) -> None:
    try:
        key: Final = bytes.fromhex(request_id_key)
    except ValueError:
        raise ValueError(f"{_REQUEST_ID_KEY_SETTING} must be a 32-byte hex key") from None
    if len(key) != 32:
        raise ValueError(f"{_REQUEST_ID_KEY_SETTING} must be a 32-byte hex key")
    if db.info.server_version < 110000 and db.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'pgcrypto')"
    ).fetchone() != (True,):
        raise ValueError("Spend log replication on PostgreSQL 10 requires the pgcrypto extension")
    block: Final = key.ljust(64, b"\0")
    hmac_pads: Final = bytes(byte ^ 0x36 for byte in block) + bytes(byte ^ 0x5C for byte in block)
    with db.transaction():
        migration_ready: Final = db.execute(
            """SELECT to_regclass('"LiteLLM_SpendLogReporting"') IS NOT NULL AND COUNT(*) = 4
                FROM pg_catalog.pg_trigger
                WHERE tgenabled IN ('O', 'A') AND (
                    (tgrelid = to_regclass('"LiteLLM_SpendLogs"') AND tgname IN (
                        'litellm_capture_spend_log_insert',
                        'litellm_capture_spend_log_update',
                        'litellm_capture_spend_log_delete'
                    )) OR
                    (tgrelid = to_regclass('"LiteLLM_ReplicatedSpendLogs"')
                        AND tgname = 'litellm_track_replicated_budget_spend')
                )"""
        ).fetchone()
        if migration_ready != (True,):
            raise ValueError("Apply the LiteLLM database migrations before starting spend log replication")
        settings_query: Final = (
            'SELECT region = %s, request_id_hmac_pads = %s FROM "LiteLLM_SpendLogReplication" WHERE id = 1'
        )
        current: Final = db.execute(settings_query, (region, hmac_pads)).fetchone()
        if current is None:
            # Writes that began before the first activation must finish before retention can prepare a partition.
            db.execute("SET LOCAL lock_timeout = '10s'")
            db.execute('LOCK TABLE ONLY "LiteLLM_SpendLogs" IN SHARE ROW EXCLUSIVE MODE')
            db.execute(
                'INSERT INTO "LiteLLM_SpendLogReplication" (id, region, request_id_hmac_pads) '
                "VALUES (1, %s, %s) ON CONFLICT (id) DO NOTHING",
                (region, hmac_pads),
            )
        configured: Final = db.execute(settings_query, (region, hmac_pads)).fetchone() if current is None else current
        if configured != (True, True):
            raise ValueError(
                "The regional database is already configured with a different replication region or request ID key"
            )


def _stream_event(payload: str, region: str) -> str:
    row: Final = SpendLogMetadata.model_validate_json(payload)
    bounded_row: Final = row.model_copy(
        update={
            "team_id": replicated_budget_entity_id(row.team_id),
            "user": replicated_budget_entity_id(row.user),
            "end_user": replicated_budget_entity_id(row.end_user),
            "model": _bounded_display_field(row.model) or "",
            "model_id": _bounded_display_field(row.model_id),
            "model_group": _bounded_display_field(row.model_group),
            "custom_llm_provider": _bounded_display_field(row.custom_llm_provider),
            "organization_id": _bounded_display_field(row.organization_id),
            "status": _bounded_display_field(row.status),
        }
    )
    return SpendLogEvent(region=region, row=bounded_row).model_dump_json()


def _bounded_display_field(value: str | None) -> str | None:
    return value if value is not None and len(value.encode()) <= _MAX_DISPLAY_FIELD_BYTES else None


async def publish_batch(db: Connection[TupleRow], redis: Redis[str], region: str) -> bool:
    failure: RedisError | None = None  # rebind-ok: preserve retryable rows while removing oversized events
    with db.transaction():
        rows: Final = _OUTBOX_BATCH.validate_python(
            db.execute(
                'SELECT id, "row"::text FROM "LiteLLM_SpendLogOutbox" ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED',
                (_BATCH_SIZE,),
            ).fetchall()
        )
        if not rows:
            return False

        records: Final = tuple((record_id, _stream_event(payload, region)) for record_id, payload in rows)
        oversized: Final = [record_id for record_id, value in records if len(value.encode()) > _MAX_RECORD_BYTES]
        if oversized:
            db.execute('DELETE FROM "LiteLLM_SpendLogOutbox" WHERE id = ANY(%s)', (oversized,))
        queued: Final = tuple((record_id, value) for record_id, value in records if record_id not in oversized)
        if queued:
            try:
                async with redis.pipeline(transaction=False) as pipe:
                    for _, value in queued:
                        pipe.xadd(STREAM_NAME, {"event": value})  # pyright: ignore[reportUnknownMemberType]  # redis stream stubs
                    await pipe.execute()  # pyright: ignore[reportUnknownMemberType]  # untyped pipeline result
            except RedisError as error:
                failure = error  # rebind-ok: raise after the oversized deletions commit
            else:
                db.execute('DELETE FROM "LiteLLM_SpendLogOutbox" WHERE id = ANY(%s)', ([row[0] for row in queued],))
    if oversized:
        verbose_proxy_logger.warning("Removed %s oversized spend-log stream events from the outbox", len(oversized))
    if failure is not None:
        raise failure
    return True


def import_batch(db: Connection[TupleRow], payloads: Sequence[str], region: str) -> None:
    with db.transaction():
        for payload in payloads:
            event = SpendLogEvent.model_validate_json(payload)
            if event.region == region:
                continue
            query, params = replicated_spend_log_insert(
                event.row.model_dump(mode="json", exclude_unset=True), region=event.region
            )
            db.execute(query, (params,))


async def consume_batch(
    db: Connection[TupleRow], redis: Redis[str], entries: Sequence[StreamEntry], region: str
) -> None:
    if entries:
        import_batch(db, tuple(fields["event"] for _, fields in entries), region)
        await redis.xack(  # pyright: ignore[reportUnknownMemberType]  # redis stream stubs
            STREAM_NAME, region, *(entry_id for entry_id, _ in entries)
        )


async def ensure_group(redis: Redis[str], region: str) -> None:
    try:
        await redis.xgroup_create(  # pyright: ignore[reportUnknownMemberType]  # redis stream stubs
            STREAM_NAME, region, id="0-0", mkstream=True
        )
    except ResponseError as error:
        if not str(error).startswith("BUSYGROUP "):
            raise


async def claim_batch(
    redis: Redis[str], region: str, consumer: str, cursor: str
) -> tuple[str, tuple[StreamEntry, ...]]:
    result: Final[object] = await redis.xautoclaim(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # redis stream stubs
        STREAM_NAME, region, consumer, _CLAIM_IDLE_MS, cursor, count=_BATCH_SIZE
    )
    next_cursor, entries, deleted = _STREAM_CLAIM.validate_python(result)
    if deleted:
        verbose_proxy_logger.error("Redis lost %s pending spend-log entries for a regional group", len(deleted))
    return next_cursor, entries


async def read_batch(redis: Redis[str], region: str, consumer: str) -> tuple[StreamEntry, ...]:
    result: Final[object] = await redis.xreadgroup(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # redis stream stubs
        region, consumer, {STREAM_NAME: ">"}, count=_BATCH_SIZE
    )
    streams: Final = _STREAM_READ.validate_python(result)
    return streams[0][1] if streams else ()


async def trim_stream(redis: Redis[str]) -> None:
    await redis.eval(  # pyright: ignore[reportUnknownMemberType]  # redis stubs do not type eval
        _TRIM_ACKNOWLEDGED, 1, STREAM_NAME, _RETENTION_MS
    )


def replication_redis(url: str) -> Redis[str]:
    parsed: Final = urlsplit(url)
    if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
        raise ValueError("Spend log replication requires a Redis URL")
    if parsed.scheme == "rediss":
        tls_error: Final = "Spend log replication TLS requires certificate and hostname verification"
        for name, value in parse_qsl(parsed.query):
            if name == "ssl_cert_reqs" and value != "required":
                raise ValueError(tls_error)
            if name == "ssl_check_hostname" and value.lower() not in {"true", "1", "yes"}:
                raise ValueError(tls_error)
        return Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            ssl_cert_reqs="required",
            ssl_check_hostname=True,
        )
    return Redis.from_url(url, decode_responses=True, socket_connect_timeout=5, socket_timeout=5)


async def _replicate_once(
    db: Connection[TupleRow], redis: Redis[str], region: str, consumer: str, cursor: str
) -> tuple[str, bool]:
    next_cursor, pending = await claim_batch(redis, region, consumer, cursor)
    await consume_batch(db, redis, pending, region)
    entries: Final = await read_batch(redis, region, consumer)
    await consume_batch(db, redis, entries, region)
    published: Final = await publish_batch(db, redis, region)
    return next_cursor, bool(pending or entries or published)


async def run_replication(db: Connection[TupleRow], settings: Mapping[str, str], stop: asyncio.Event) -> None:
    region: Final = settings["LITELLM_SPEND_LOG_REGION"].strip()
    if not region:
        raise ValueError("Spend log replication requires a region")
    consumer: Final = uuid4().hex
    cursor = "0-0"  # rebind-ok: XAUTOCLAIM scans the pending list in batches
    next_trim = 0.0  # rebind-ok: bound trim frequency even when the stream is busy
    group_ready = False  # rebind-ok: ensure the group again after a Redis error
    async with replication_redis(settings["LITELLM_SPEND_LOG_REDIS_URL"]) as redis:
        enable_capture(db, region, settings.get(_REQUEST_ID_KEY_SETTING, ""))
        verbose_proxy_logger.info("Redis spend-log replication started")
        while not stop.is_set():
            try:
                if not group_ready:
                    await ensure_group(redis, region)
                    group_ready = True
                cursor, active = await _replicate_once(db, redis, region, consumer, cursor)
                if monotonic() >= next_trim:
                    await trim_stream(redis)
                    next_trim = monotonic() + _TRIM_INTERVAL_SECONDS
                if not active:
                    await asyncio.sleep(0.1)
            except RedisError:
                group_ready = False
                cursor = "0-0"
                verbose_proxy_logger.warning("Redis spend-log replication failed. The worker will retry")
                await asyncio.sleep(1)


async def _run() -> None:
    stop: Final = asyncio.Event()
    loop: Final = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    with connect_regional_db(os.environ["DATABASE_URL"]) as db:
        await run_replication(db, os.environ, stop)


def main() -> None:
    try:
        asyncio.run(_run())
    except (KeyError, ValueError, psycopg.Error, RedisError) as error:
        raise SystemExit(f"Redis spend-log replication stopped ({type(error).__name__})") from None


if __name__ == "__main__":
    main()
