from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.rust_bridge.trace.generated.types import AllQueryScope, Trace
from litellm.rust_bridge.trace.storage import ClickHouseStorage, Tenant, span_rows
from litellm.tracing.config import trace_storage_config
from litellm.tracing.types import SpendLogRecord

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma.types import LiteLLM_SpendLogsCreateWithoutRelationsInput

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
TRACE_FIXTURES: Final = REPO_ROOT / "litellm-rust/crates/traces/tests/fixtures"
SPEND_FIXTURES: Final = REPO_ROOT / "litellm-rust/crates/traces-clickhouse/tests/fixtures"
JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
SPEND_ROWS: Final = TypeAdapter(tuple[SpendLogRecord, ...])
TRACE: Final = TypeAdapter(Trace)
NANOSECOND_FIELDS: Final = frozenset({"startTimeUnixNano", "endTimeUnixNano", "timeUnixNano"})
TRACE_ID_FIELDS: Final = frozenset({"traceId", "trace_id", "session_id"})
TRACE_ID_ATTRIBUTES: Final = frozenset({"session.id"})
SPAN_ID_FIELDS: Final = frozenset({"spanId", "parentSpanId", "span_id"})
COPY_WINDOW_MS: Final = 24 * 60 * 60 * 1000
LONG_SESSION_SOURCE: Final = "openai_agents_swarm"
LONG_SESSION_REPEATS: Final = (50, 400, 4000)


class TenantIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    team_id: str
    api_key: str
    user: str


class FixtureCapture(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    trace_id: str
    spend_linked: bool
    spend_complete: bool = True


@dataclass(frozen=True, slots=True)
class FixtureReplay:
    name: str
    export: JsonValue
    offset_ms: int
    namespace: str


def spend_fixtures(directory: Path = SPEND_FIXTURES) -> tuple[tuple[str, tuple[SpendLogRecord, ...]], ...]:
    return tuple(
        (
            path.stem.removesuffix("_spend_logs"),
            SPEND_ROWS.validate_python(
                tuple(
                    {"litellm_call_id": "", **JSON_OBJECT.validate_json(line)} for line in path.read_text().splitlines()
                )
            ),
        )
        for path in sorted(directory.glob("*_spend_logs.jsonl"))
    )


def managed_response(value: str) -> str | None:
    if not value.startswith("resp_"):
        return None
    try:
        decoded: Final = base64.b64decode(value[5:], validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    return decoded if "response_id:" in decoded else None


def response_ids(rows: tuple[SpendLogRecord, ...]) -> Iterator[str]:
    for row in rows:
        yield row["request_id"]
        yield row["response_id"]
        if (decoded := managed_response(row["response_id"])) is not None:
            if (upstream := re.search(r"response_id:([^;]+)", decoded)) is not None:
                yield upstream.group(1)


def response_pattern(rows: tuple[SpendLogRecord, ...]) -> re.Pattern[str]:
    identities: Final = sorted(
        frozenset(
            identity for identity in chain(response_ids(rows), (row["litellm_call_id"] for row in rows)) if identity
        ),
        key=len,
        reverse=True,
    )
    return re.compile("|".join(re.escape(identity) for identity in identities) or r"(?!)")


def rebased_response(value: str, namespace: str, pattern: re.Pattern[str]) -> str:
    decoded: Final = managed_response(value)
    if decoded is None:
        return f"seed-{namespace}-{value}"
    payload: Final = pattern.sub(lambda match: f"seed-{namespace}-{match.group()}", decoded)
    return "resp_" + base64.b64encode(payload.encode()).decode()


@cache
def fixture_exports(directory: Path) -> tuple[tuple[str, JsonValue], ...]:
    return tuple((path.stem, JSON.validate_json(path.read_bytes())) for path in sorted(directory.glob("*.json")))


def fixture_replays(
    directory: Path, now_ms: int, namespace: str, response_pattern: re.Pattern[str]
) -> tuple[FixtureReplay, ...]:
    exports: Final = fixture_exports(directory)
    query_latest: Final = max(
        (max(timestamps(export)) for name, export in exports if name.startswith("query_")), default=0
    )

    def replay(name: str, export: JsonValue) -> FixtureReplay:
        group: Final = "query" if name.startswith("query_") else name
        latest_ns: Final = query_latest if group == "query" else max(timestamps(export))
        offset_ms: Final = now_ms - latest_ns // 1_000_000 - 1000
        capture_namespace: Final = f"{namespace}-{group}"
        return FixtureReplay(
            name=name,
            export=rebase(export, offset_ms * 1_000_000, capture_namespace, response_pattern),
            offset_ms=offset_ms,
            namespace=capture_namespace,
        )

    return tuple(replay(name, export) for name, export in exports)


def timestamps(value: JsonValue) -> Iterator[int]:
    if isinstance(value, list):
        for item in value:
            yield from timestamps(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in NANOSECOND_FIELDS and isinstance(item, (str, int)) and int(item) > 0:
                yield int(item)
            else:
                yield from timestamps(item)


def seed_id(value: str, namespace: str, length: int) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()[:length] if value else ""


def rebase(
    value: JsonValue, offset_ns: int, namespace: str, response_pattern: re.Pattern[str], field: str = ""
) -> JsonValue:
    if field in NANOSECOND_FIELDS and isinstance(value, (str, int)):
        return str(int(value) + offset_ns) if int(value) else value
    if isinstance(value, str):
        if field == "metadata":
            return json.dumps(rebase(JSON.validate_json(value), offset_ns, namespace, response_pattern))
        if field == "bytesValue":
            return base64.b64encode(
                re.sub(
                    response_pattern.pattern.encode(),
                    lambda match: rebased_response(match.group().decode(), namespace, response_pattern).encode(),
                    base64.b64decode(value),
                )
            ).decode()
        if field in TRACE_ID_FIELDS:
            return seed_id(value, namespace, 32)
        if field in SPAN_ID_FIELDS:
            return seed_id(value, namespace, 16)
        return response_pattern.sub(lambda match: rebased_response(match.group(), namespace, response_pattern), value)
    if isinstance(value, list):
        return [rebase(item, offset_ns, namespace, response_pattern) for item in value]
    if isinstance(value, dict):
        attribute_key: Final = value.get("key")
        attribute_value: Final = value.get("value")
        session_id: Final = (
            attribute_value.get("stringValue") if isinstance(attribute_value, dict) else None
        )
        if (
            isinstance(attribute_key, str)
            and attribute_key in TRACE_ID_ATTRIBUTES
            and isinstance(attribute_value, dict)
            and isinstance(session_id, str)
        ):
            return {
                **{
                    key: rebase(item, offset_ns, namespace, response_pattern, key)
                    for key, item in value.items()
                    if key != "value"
                },
                "value": {
                    **{
                        key: rebase(item, offset_ns, namespace, response_pattern, key)
                        for key, item in attribute_value.items()
                        if key != "stringValue"
                    },
                    "stringValue": seed_id(session_id, namespace, 32),
                },
            }
        return {key: rebase(item, offset_ns, namespace, response_pattern, key) for key, item in value.items()}
    return value


def rebase_spend(
    rows: tuple[SpendLogRecord, ...], offset_ms: int, namespace: str, response_pattern: re.Pattern[str]
) -> tuple[SpendLogRecord, ...]:
    return SPEND_ROWS.validate_python(
        tuple(
            {
                **JSON_OBJECT.validate_python(rebase(JSON.validate_python(row), 0, namespace, response_pattern)),
                "start_time": row["start_time"] + offset_ms,
                "end_time": row["end_time"] + offset_ms,
                "completion_start_time": (
                    row["completion_start_time"] + offset_ms if row["completion_start_time"] is not None else None
                ),
            }
            for row in rows
        )
    )


def postgres_row(row: SpendLogRecord) -> LiteLLM_SpendLogsCreateWithoutRelationsInput:
    from prisma import Json
    from prisma.types import LiteLLM_SpendLogsCreateWithoutRelationsInput

    return LiteLLM_SpendLogsCreateWithoutRelationsInput(
        request_id=row["request_id"],
        call_type=row["call_type"],
        api_key=row["api_key"],
        user=row["user"],
        team_id=row["team_id"],
        spend=row["spend"],
        model=row["model"],
        model_group=row["model_group"],
        custom_llm_provider=row["custom_llm_provider"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        total_tokens=row["total_tokens"],
        startTime=datetime.fromtimestamp(row["start_time"] / 1000, tz=timezone.utc),
        endTime=datetime.fromtimestamp(row["end_time"] / 1000, tz=timezone.utc),
        request_duration_ms=row["end_time"] - row["start_time"],
        session_id=row["session_id"],
        status=row["status"],
        cache_hit=str(row["cache_hit"]),
        request_tags=Json(list(row["request_tags"])),
        metadata=Json(JSON.validate_json(row["metadata"])),
        messages=Json(JSON.validate_json(row["messages"])),
        response=Json(JSON.validate_json(row["response"])),
        proxy_server_request=Json(None),
    )


class SeedOptions(BaseModel):
    model_config = ConfigDict(frozen=True)
    profile: Literal["default", "large"]
    copies: int | None
    timeout_seconds: float = 120


def seed_arguments(argv: Sequence[str] | None = None) -> SeedOptions:
    parser: Final = argparse.ArgumentParser(description="Replay tracing fixtures into a running local Lens stack")
    parser.add_argument("--profile", choices=("default", "large"), default="default")
    parser.add_argument(
        "--copies",
        type=int,
        default=os.environ.get("LENS_DEV_SEED_COPIES"),
        help="Override fixture copies (default: 1, large: 2000; env: LENS_DEV_SEED_COPIES)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=os.environ.get("LENS_DEV_SEED_TIMEOUT_SECONDS", "120"))
    arguments: Final = SeedOptions.model_validate(vars(parser.parse_args(argv)))
    if arguments.copies is not None and arguments.copies < 1:
        parser.error("--copies must be positive")
    if not math.isfinite(arguments.timeout_seconds) or arguments.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    return arguments


async def seed_copy(
    client: httpx.AsyncClient,
    storage: ClickHouseStorage,
    database: Prisma,
    replays: tuple[FixtureReplay, ...],
    fixtures: tuple[tuple[str, tuple[SpendLogRecord, ...]], ...],
    pattern: re.Pattern[str],
) -> tuple[tuple[str, tuple[SpendLogRecord, ...]], ...]:
    by_name: Final = MappingProxyType(dict(fixtures))
    paired: Final = tuple(
        (
            replay.name,
            rebase_spend(by_name[replay.name], replay.offset_ms, replay.namespace, pattern),
        )
        for replay in replays
        if replay.name in by_name
    )
    tenant: Final = await ingest_replays(
        client, storage, replays, fixture_capture(*next((name, rows[0]) for name, rows in paired)).trace_id
    )
    stamped: Final = tuple((name, tuple(stamp(row, tenant) for row in rows)) for name, rows in paired)
    stamped_spends: Final = tuple(chain.from_iterable(rows for _, rows in stamped))
    await storage.insert_rows("spend_logs", stamped_spends)
    await database.litellm_spendlogs.create_many(data=[postgres_row(row) for row in stamped_spends])
    return stamped


def stamp(row: SpendLogRecord, tenant: TenantIdentity) -> SpendLogRecord:
    return {**row, "team_id": tenant.team_id, "api_key": tenant.api_key, "user": tenant.user}


async def verify(
    client: httpx.AsyncClient, captures: tuple[tuple[str, tuple[SpendLogRecord, ...]], ...], trace_salt: str
) -> None:
    verified: Final = tuple(
        await asyncio.gather(*(verify_capture(client, name, rows, trace_salt) for name, rows in captures))
    )
    sys.stdout.write(
        json.dumps({"spend_rows": sum(len(rows) for _, rows in captures), "captures": verified}, indent=2) + "\n"
    )
    if not all(capture["verified"] for capture in verified):
        raise RuntimeError("Seed spend verification failed")


async def ingest_replays(
    client: httpx.AsyncClient, storage: ClickHouseStorage, replays: tuple[FixtureReplay, ...], trace_id: str
) -> TenantIdentity:
    for replay in replays:
        (
            await client.post(
                "/v1/traces", content=json.dumps(replay.export), headers={"Content-Type": "application/json"}
            )
        ).raise_for_status()
    identity: Final = await storage.query_sql(
        "SELECT DISTINCT TeamId AS team_id, ApiKeyHash AS api_key, UserId AS user "
        f"FROM otel_traces WHERE TraceId = '{trace_id}'",
        AllQueryScope(kind="all"),
        os.environ["LITELLM_MASTER_KEY"],
    )
    return TenantIdentity.model_validate(identity.data[0])


def bulk_span_rows(replays: tuple[FixtureReplay, ...], tenant: Tenant) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        chain.from_iterable(
            span_rows(json.dumps(replay.export).encode(), "application/json", tenant) for replay in replays
        )
    )


@dataclass(frozen=True, slots=True)
class Copies:
    """Server-side copies of seeded traces and their spend.

    Each copy `n` hashes trace ids with `session` (or `n` when empty), keeps root span ids, hashes the
    other span ids with `n`, rewrites seeded call ids from `source` to `{target}{n}-` and moves `n * step_ms`
    earlier. A `session` folds every copy into one trace under a single root that spans all of them.
    """

    trace_ids: tuple[str, ...]
    request_ids: tuple[str, ...]
    numbers: range
    step_ms: int
    source: str
    target: str
    session: str = ""


def copied_trace_id(trace_id: str, salt: str) -> str:
    return hashlib.sha256(f"{trace_id}:{salt}".encode()).hexdigest()[:32]


def clickhouse_call_id(column: str) -> str:
    target: Final = "concat({target:String}, toString(c.n), '-')"
    return (
        f"if(startsWith({column}, 'resp_'), concat('resp_', base64Encode(replaceAll("
        f"tryBase64Decode(substring({column}, 6)), {{source:String}}, {target}))), "
        f"replaceAll({column}, {{source:String}}, {target}))"
    )


def clickhouse_hash(column: str, salt: str, length: int) -> str:
    return f"if({column} = '', '', substring(lower(hex(SHA256(concat({column}, ':', {salt})))), 1, {length}))"


def clickhouse_copy_sql(database: str) -> tuple[str, str]:
    trace_salt: Final = "if({session:String} = '', toString(c.n), {session:String})"
    roots: Final = (
        f"(SELECT SpanId FROM {database}.otel_traces "
        "WHERE TraceId IN {trace_ids:Array(String)} AND ParentSpanId = '')"
    )
    folded_root: Final = "{session:String} != '' AND t.ParentSpanId = ''"
    shift: Final = f"toIntervalMillisecond(if({folded_root}, {{last:UInt64}}, c.n) * {{step_ms:UInt64}})"
    numbers: Final = "CROSS JOIN (SELECT number AS n FROM numbers({first:UInt64}, {count:UInt64})) AS c"
    spans: Final = f"""INSERT INTO {database}.otel_traces
SELECT t.* REPLACE (
    t.Timestamp - {shift} AS Timestamp,
    {clickhouse_hash("t.TraceId", trace_salt, 32)} AS TraceId,
    if(t.ParentSpanId = '', t.SpanId, {clickhouse_hash("t.SpanId", "toString(c.n)", 16)}) AS SpanId,
    if(t.ParentSpanId IN {roots}, t.ParentSpanId, {clickhouse_hash("t.ParentSpanId", "toString(c.n)", 16)})
        AS ParentSpanId,
    t.Duration + if({folded_root}, {{last:UInt64}} * {{step_ms:UInt64}} * 1000000, 0) AS Duration,
    arrayMap(at -> at - {shift}, t.`Events.Timestamp`) AS `Events.Timestamp`,
    mapApply((name, value) -> (name, {clickhouse_call_id("value")}), t.SpanAttributes) AS SpanAttributes,
    {clickhouse_call_id("t.LiteLLMRequestId")} AS LiteLLMRequestId,
    arrayMap(key -> concat(extract(key, '^[^:]*:'), {clickhouse_call_id("replaceRegexpOne(key, '^[^:]*:', '')")}),
        t.CallKeys) AS CallKeys
)
FROM {database}.otel_traces AS t {numbers}
WHERE t.TraceId IN {{trace_ids:Array(String)}}
    AND ({{session:String}} = '' OR t.ParentSpanId != '' OR c.n = {{first:UInt64}})"""
    spend: Final = f"""INSERT INTO {database}.spend_logs
SELECT s.* REPLACE (
    {clickhouse_call_id("s.request_id")} AS request_id,
    {clickhouse_call_id("s.response_id")} AS response_id,
    {clickhouse_call_id("s.litellm_call_id")} AS litellm_call_id,
    {clickhouse_hash("s.trace_id", trace_salt, 32)} AS trace_id,
    {clickhouse_hash("s.session_id", trace_salt, 32)} AS session_id,
    if(s.span_id IN {roots}, s.span_id, {clickhouse_hash("s.span_id", "toString(c.n)", 16)}) AS span_id,
    s.start_time - toIntervalMillisecond(c.n * {{step_ms:UInt64}}) AS start_time,
    s.end_time - toIntervalMillisecond(c.n * {{step_ms:UInt64}}) AS end_time,
    s.completion_start_time - toIntervalMillisecond(c.n * {{step_ms:UInt64}}) AS completion_start_time
)
FROM {database}.spend_logs AS s {numbers}
WHERE s.request_id IN {{request_ids:Array(String)}}"""
    return spans, spend


def clickhouse_array(values: tuple[str, ...]) -> str:
    return "[" + ",".join("'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'" for value in values) + "]"


async def copy_clickhouse(client: httpx.AsyncClient, database: str, copies: Copies) -> None:
    parameters: Final = {
        "param_trace_ids": clickhouse_array(copies.trace_ids),
        "param_request_ids": clickhouse_array(copies.request_ids),
        "param_first": str(copies.numbers.start),
        "param_count": str(len(copies.numbers)),
        "param_last": str(copies.numbers.stop - 1),
        "param_step_ms": str(copies.step_ms),
        "param_source": copies.source,
        "param_target": copies.target,
        "param_session": copies.session,
    }
    for sql in clickhouse_copy_sql(database):
        (await client.post("/", params=parameters, content=sql)).raise_for_status()


POSTGRES_COPY_TARGET: Final = "($2 || c.n || '-')"
POSTGRES_COPY_SHIFT: Final = "make_interval(secs => c.n * $6::bigint / 1000.0)"
POSTGRES_COPY_SQL: Final = f"""INSERT INTO "LiteLLM_SpendLogs"
SELECT (jsonb_populate_record(s, jsonb_build_object(
    'request_id', CASE WHEN left(s.request_id, 5) = 'resp_'
        THEN 'resp_' || translate(encode(convert_to(replace(convert_from(decode(substr(s.request_id, 6), 'base64'),
            'UTF8'), $1, {POSTGRES_COPY_TARGET}), 'UTF8'), 'base64'), E'\\n', '')
        ELSE replace(s.request_id, $1, {POSTGRES_COPY_TARGET}) END,
    'session_id', CASE WHEN coalesce(s.session_id, '') = '' THEN s.session_id
        ELSE substr(encode(sha256(convert_to(
            s.session_id || ':' || CASE WHEN $3 = '' THEN c.n::text ELSE $3 END, 'UTF8')), 'hex'), 1, 32) END,
    'startTime', s."startTime" - {POSTGRES_COPY_SHIFT},
    'endTime', s."endTime" - {POSTGRES_COPY_SHIFT},
    'completionStartTime', s."completionStartTime" - {POSTGRES_COPY_SHIFT}
))).*
FROM "LiteLLM_SpendLogs" AS s CROSS JOIN generate_series($4::int, $5::int) AS c(n)
WHERE s.request_id = ANY(string_to_array($7, E'\\n'))"""


async def copy_postgres(database: Prisma, copies: Copies) -> None:
    await database.execute_raw(
        POSTGRES_COPY_SQL,
        copies.source,
        copies.target,
        copies.session,
        copies.numbers.start,
        copies.numbers.stop - 1,
        copies.step_ms,
        "\n".join(copies.request_ids),
    )


def long_sessions(
    replays: tuple[FixtureReplay, ...],
    captures: tuple[tuple[str, tuple[SpendLogRecord, ...]], ...],
    source: str,
    target: str,
    repeats: tuple[int, ...] = LONG_SESSION_REPEATS,
) -> tuple[Copies, ...]:
    replay: Final = next(replay for replay in replays if replay.name == LONG_SESSION_SOURCE)
    rows: Final = dict(captures)[LONG_SESSION_SOURCE]
    span_ns: Final = tuple(timestamps(replay.export))
    return tuple(
        Copies(
            trace_ids=(fixture_capture(LONG_SESSION_SOURCE, rows[0]).trace_id,),
            request_ids=tuple(row["request_id"] for row in rows),
            numbers=range(count),
            step_ms=(max(span_ns) - min(span_ns)) // 1_000_000 + 1000,
            source=source,
            target=f"{target}s{count}x",
            session=f"session{count}",
        )
        for count in repeats
    )


async def seed(profile: str = "default", copies: int | None = None, timeout_seconds: float = 120) -> int:
    from prisma import Prisma

    fixtures: Final = spend_fixtures()
    pattern: Final = response_pattern(tuple(chain.from_iterable(rows for _, rows in fixtures)))
    count: Final = copies if copies is not None else (2000 if profile == "large" else 1)
    namespace: Final = uuid4().hex
    now_ms: Final = time.time_ns() // 1_000_000
    config: Final = trace_storage_config({})
    storage: Final = ClickHouseStorage(config)
    replays: Final = fixture_replays(TRACE_FIXTURES, now_ms, namespace + "-0", pattern)
    source: Final = f"seed-{namespace}-0-"
    target: Final = f"seed-{namespace}-"
    async with (
        httpx.AsyncClient(
            base_url=os.environ.get("PROXY_BASE_URL", "http://127.0.0.1:4002"),
            headers={"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"},
            timeout=timeout_seconds,
        ) as client,
        httpx.AsyncClient(base_url=config.url, params={"database": config.database}, timeout=600) as clickhouse,
        Prisma(http={"timeout": httpx.Timeout(600)}) as database,
    ):
        captures: Final = await seed_copy(client, storage, database, replays, fixtures, pattern)
        await verify(client, captures, "")
        repeated: Final = Copies(
            trace_ids=tuple(
                sorted(frozenset(str(span["TraceId"]) for span in bulk_span_rows(replays, Tenant("", ""))))
            ),
            request_ids=tuple(row["request_id"] for _, rows in captures for row in rows),
            numbers=range(1, count),
            step_ms=COPY_WINDOW_MS // count,
            source=source,
            target=target,
        )
        sessions: Final = long_sessions(replays, captures, source, target) if profile == "large" else ()
        for plan in (repeated, *sessions) if count > 1 else sessions:
            await copy_clickhouse(clickhouse, config.database, plan)
            await copy_postgres(database, plan)
        if count > 1:
            await verify(client, captures, str(count - 1))
        for plan in sessions:
            sys.stdout.write(
                f"Long session: {len(plan.numbers)} repeats, "
                f"trace_id={copied_trace_id(plan.trace_ids[0], plan.session)}\n"
            )
    sys.stdout.write(f"Seed complete: profile={profile}, copies={count}, namespace={namespace}\n")
    return 0


def fixture_capture(name: str, row: SpendLogRecord) -> FixtureCapture:
    metadata: Final = JSON_OBJECT.validate_json(row["metadata"])
    capture: Final = metadata.get("fixture_capture")
    return (
        FixtureCapture.model_validate(capture)
        if capture is not None
        else FixtureCapture(name=name, trace_id=row["trace_id"], spend_linked=True)
    )


async def verify_capture(
    client: httpx.AsyncClient, name: str, rows: tuple[SpendLogRecord, ...], trace_salt: str = ""
) -> Mapping[str, JsonValue]:
    capture: Final = fixture_capture(name, rows[0])
    trace_id: Final = copied_trace_id(capture.trace_id, trace_salt) if trace_salt else capture.trace_id
    detail: Final = await client.get(f"/v1/traces/{trace_id}")
    detail.raise_for_status()
    trace: Final = TRACE.validate_json(detail.content)
    expected: Final = sum(row["spend"] or 0 for row in rows)
    actual: Final = trace["summary"]["spend"]
    return {
        "fixture": name,
        "trace_id": trace_id,
        "spend_rows": len(rows),
        "recorded_spend": expected,
        "trace_spend": actual,
        "verified": math.isclose(actual, expected)
        if actual is not None
        else not (capture.spend_linked and capture.spend_complete),
    }


if __name__ == "__main__":
    arguments: Final = seed_arguments()
    raise SystemExit(asyncio.run(seed(arguments.profile, arguments.copies, arguments.timeout_seconds)))
