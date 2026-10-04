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
SPAN_ID_FIELDS: Final = frozenset({"spanId", "parentSpanId", "span_id"})


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
    batch_copies: int = 4
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
    parser.add_argument(
        "--batch-copies",
        type=int,
        default=os.environ.get("LENS_DEV_SEED_BATCH_COPIES", "4"),
        help="Copies per bulk insert (default: 4; env: LENS_DEV_SEED_BATCH_COPIES)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=os.environ.get("LENS_DEV_SEED_TIMEOUT_SECONDS", "120"))
    arguments: Final = SeedOptions.model_validate(vars(parser.parse_args(argv)))
    if arguments.copies is not None and arguments.copies < 1:
        parser.error("--copies must be positive")
    if arguments.batch_copies < 1:
        parser.error("--batch-copies must be positive")
    if not math.isfinite(arguments.timeout_seconds) or arguments.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    return arguments


async def seed_batch(
    client: httpx.AsyncClient,
    storage: ClickHouseStorage,
    database: Prisma,
    replays: tuple[FixtureReplay, ...],
    fixtures: tuple[tuple[str, tuple[SpendLogRecord, ...]], ...],
    pattern: re.Pattern[str],
    tenant: TenantIdentity | None,
    verify: bool,
) -> TenantIdentity:
    by_name: Final = MappingProxyType(dict(fixtures))
    paired: Final = tuple(
        (
            replay.name,
            rebase_spend(by_name[replay.name], replay.offset_ms, replay.namespace, pattern),
        )
        for replay in replays
        if replay.name in by_name
    )
    rebased_spends: Final = tuple(chain.from_iterable(rows for _, rows in paired))
    resolved_tenant: Final = await ingest_replays(
        client, storage, replays, fixture_capture(*next((name, rows[0]) for name, rows in paired)).trace_id, tenant
    )
    stamped_spends: Final[tuple[SpendLogRecord, ...]] = tuple(
        {**row, "team_id": resolved_tenant.team_id, "api_key": resolved_tenant.api_key, "user": resolved_tenant.user}
        for row in rebased_spends
    )
    await storage.insert_rows("spend_logs", stamped_spends)
    await database.litellm_spendlogs.create_many(data=[postgres_row(row) for row in stamped_spends])
    if verify:
        verified: Final = tuple(await asyncio.gather(*(verify_capture(client, name, rows) for name, rows in paired)))
        sys.stdout.write(json.dumps({"spend_rows": len(stamped_spends), "captures": verified}, indent=2) + "\n")
        if not all(capture["verified"] for capture in verified):
            raise RuntimeError("Seed spend verification failed")
    return resolved_tenant


async def ingest_replays(
    client: httpx.AsyncClient,
    storage: ClickHouseStorage,
    replays: tuple[FixtureReplay, ...],
    trace_id: str,
    tenant: TenantIdentity | None,
) -> TenantIdentity:
    if tenant is not None:
        await storage.insert_rows(
            "otel_traces", bulk_span_rows(replays, Tenant(tenant.team_id, tenant.api_key, user_id=tenant.user))
        )
        return tenant
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


def replay_batches(
    count: int, now_ms: int, namespace: str, pattern: re.Pattern[str], batch_copies: int = 4
) -> Iterator[tuple[int, tuple[FixtureReplay, ...]]]:
    for start, stop in ((start, min(start + batch_copies, count)) for start in range(1, count, batch_copies)):
        yield (
            stop,
            tuple(
                chain.from_iterable(
                    fixture_replays(TRACE_FIXTURES, now_ms - index * 1000, f"{namespace}-{index}", pattern)
                    for index in range(start, stop)
                )
            ),
        )


async def seed(
    profile: str = "default", copies: int | None = None, batch_copies: int = 4, timeout_seconds: float = 120
) -> int:
    from prisma import Prisma

    fixtures: Final = spend_fixtures()
    pattern: Final = response_pattern(tuple(chain.from_iterable(rows for _, rows in fixtures)))
    count: Final = copies if copies is not None else (2000 if profile == "large" else 1)
    namespace: Final = uuid4().hex
    now_ms: Final = time.time_ns() // 1_000_000
    storage: Final = ClickHouseStorage(trace_storage_config({}))
    async with (
        httpx.AsyncClient(
            base_url=os.environ.get("PROXY_BASE_URL", "http://127.0.0.1:4002"),
            headers={"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"},
            timeout=timeout_seconds,
        ) as client,
        Prisma() as database,
    ):
        tenant: Final = await seed_batch(
            client,
            storage,
            database,
            fixture_replays(TRACE_FIXTURES, now_ms, namespace + "-0", pattern),
            fixtures,
            pattern,
            None,
            True,
        )
        for stop, replays in replay_batches(count, now_ms, namespace, pattern, batch_copies):
            await seed_batch(client, storage, database, replays, fixtures, pattern, tenant, stop == count)
            sys.stdout.write(f"Seeded {stop}/{count} fixture copies\n")
            sys.stdout.flush()
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
    client: httpx.AsyncClient, name: str, rows: tuple[SpendLogRecord, ...]
) -> Mapping[str, JsonValue]:
    capture: Final = fixture_capture(name, rows[0])
    detail: Final = await client.get(f"/v1/traces/{capture.trace_id}")
    detail.raise_for_status()
    trace: Final = TRACE.validate_json(detail.content)
    expected: Final = sum(row["spend"] or 0 for row in rows)
    actual: Final = trace["summary"]["spend"]
    return {
        "fixture": name,
        "trace_id": capture.trace_id,
        "spend_rows": len(rows),
        "recorded_spend": expected,
        "trace_spend": actual,
        "verified": math.isclose(actual, expected)
        if actual is not None
        else not (capture.spend_linked and capture.spend_complete),
    }


if __name__ == "__main__":
    arguments: Final = seed_arguments()
    raise SystemExit(
        asyncio.run(seed(arguments.profile, arguments.copies, arguments.batch_copies, arguments.timeout_seconds))
    )
