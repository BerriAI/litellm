import asyncio
import base64
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
from prisma import Json, Prisma
from prisma.types import LiteLLM_SpendLogsCreateWithoutRelationsInput
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.rust_bridge.trace.generated.types import AllQueryScope, Trace
from litellm.rust_bridge.trace.storage import ClickHouseStorage
from litellm.tracing.config import trace_storage_config
from litellm.tracing.types import SpendLogRecord

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
TRACE_FIXTURES: Final = REPO_ROOT / "litellm-rust/crates/traces/tests/fixtures"
SPEND_FIXTURE: Final = (
    REPO_ROOT / "litellm-rust/crates/traces-clickhouse/tests/fixtures/deeplite_swarm_spend_logs.jsonl"
)
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
        if field == "bytesValue":
            return base64.b64encode(
                re.sub(
                    response_pattern.pattern.encode(),
                    lambda match: f"seed-{namespace}-".encode() + match.group(),
                    base64.b64decode(value),
                )
            ).decode()
        if field in TRACE_ID_FIELDS:
            return seed_id(value, namespace, 32)
        if field in SPAN_ID_FIELDS:
            return seed_id(value, namespace, 16)
        return response_pattern.sub(lambda match: f"seed-{namespace}-{match.group()}", value)
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


async def seed() -> int:
    exports: Final = tuple(
        JSON.validate_json((TRACE_FIXTURES / name).read_bytes())
        for name in ("deeplite_swarm.json", "deeplite_auth_error.json")
    )
    spends: Final = SPEND_ROWS.validate_python(
        tuple(json.loads(line) for line in SPEND_FIXTURE.read_text().splitlines())
    )
    namespace: Final = uuid4().hex
    response_pattern: Final = re.compile("|".join(re.escape(row["response_id"]) for row in spends))
    latest_ns: Final = max(max(timestamps(export)) for export in exports)
    offset_ms: Final = time.time_ns() // 1_000_000 - latest_ns // 1_000_000 - 1000
    rebased_exports: Final = tuple(
        rebase(export, offset_ms * 1_000_000, namespace, response_pattern) for export in exports
    )
    rebased_spends: Final = rebase_spend(spends, offset_ms, namespace, response_pattern)
    master_key: Final = os.environ["LITELLM_MASTER_KEY"]
    proxy_url: Final = os.environ.get("PROXY_BASE_URL", "http://127.0.0.1:4002")
    async with httpx.AsyncClient(
        base_url=proxy_url, headers={"Authorization": f"Bearer {master_key}"}, timeout=60
    ) as client:
        for export in rebased_exports:
            (
                await client.post(
                    "/v1/traces", content=json.dumps(export), headers={"Content-Type": "application/json"}
                )
            ).raise_for_status()
        storage: Final = ClickHouseStorage(trace_storage_config({}))
        trace_id: Final = rebased_spends[0]["trace_id"]
        identity: Final = await storage.query_sql(
            "SELECT DISTINCT TeamId AS team_id, ApiKeyHash AS api_key, UserId AS user "
            f"FROM otel_traces WHERE TraceId = '{trace_id}'",
            AllQueryScope(kind="all"),
            master_key,
        )
        tenant: Final = TenantIdentity.model_validate(identity.data[0])
        stamped_spends: Final[tuple[SpendLogRecord, ...]] = tuple(
            {**row, "team_id": tenant.team_id, "api_key": tenant.api_key, "user": tenant.user} for row in rebased_spends
        )
        await storage.insert_rows("spend_logs", stamped_spends)
        async with Prisma() as database:
            await database.litellm_spendlogs.create_many(data=[postgres_row(row) for row in stamped_spends])
        detail: Final = await client.get(f"/v1/traces/{trace_id}")
        detail.raise_for_status()
        trace: Final = TRACE.validate_json(detail.content)
        expected_spend: Final = sum(row["spend"] for row in rebased_spends)
        sys.stdout.write(
            json.dumps(
                {
                    "fixture": "deeplite_swarm",
                    "spend_rows": len(stamped_spends),
                    "synthetic_spend": True,
                    "summary": trace["summary"],
                },
                indent=2,
            )
            + "\n"
        )
        return (
            0
            if trace["summary"]["spend"] is not None and math.isclose(trace["summary"]["spend"], expected_spend)
            else 1
        )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(seed()))
