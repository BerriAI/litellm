from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from itertools import chain
from typing import TYPE_CHECKING, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from scripts.seed_tracing_fixtures import JSON_OBJECT, spend_fixtures

if TYPE_CHECKING:
    from prisma.types import LiteLLM_SpendLogsCreateWithoutRelationsInput

REQUEST_ID_PREFIX: Final = "seed-logs-"
SESSION_ID_PREFIX: Final = "seed-logs-session-"
WINDOW_HOURS: Final = 23
RNG_SEED: Final = 20261004
LARGE_COPIES: Final = 3000
PROFILES: Final = ("default", "large")
Profile = Literal["default", "large"]
JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)

WORDS: Final = (
    "trace", "span", "token", "request", "response", "latency", "router", "fallback", "cache", "budget",
    "guardrail", "stream", "deployment", "proxy", "callback", "cursor", "schema", "payload", "retry", "quota",
)


class SeededLog(BaseModel):
    """One synthetic spend-log row before it is shaped for Postgres."""

    model_config = ConfigDict(frozen=True)
    request_id: str
    label: str
    call_type: str
    model: str
    provider: str
    status: Literal["success", "failure"]
    session_id: str | None
    offset_minutes: int
    duration_ms: int
    prompt_tokens: int
    completion_tokens: int
    spend: float
    messages: JsonValue
    response: JsonValue
    proxy_server_request: JsonValue
    error_information: dict[str, JsonValue] | None = None


def prose(rng: random.Random, chars: int) -> str:
    words: Final[list[str]] = []
    length = 0
    while length < chars:
        word: Final = rng.choice(WORDS)
        words.append(word)
        length += len(word) + 1  # rebind-ok: accumulates generated text length
    return " ".join(words)[:chars]


def tool_definition(index: int) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {
            "name": f"seed_tool_{index}",
            "description": f"Synthetic tool number {index} used only by the request-log seeder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["query"],
            },
        },
    }


def tool_call(index: int, rng: random.Random) -> dict[str, JsonValue]:
    return {
        "id": f"call_seed_{index}",
        "type": "function",
        "function": {"name": f"seed_tool_{index}", "arguments": json.dumps({"query": prose(rng, 40), "limit": index})},
    }


def chat_turns(rng: random.Random, turns: int, chars_per_turn: int) -> list[JsonValue]:
    def turn(index: int) -> Iterator[JsonValue]:
        yield {"role": "user", "content": prose(rng, chars_per_turn)}
        if index % 3 == 0:
            yield {"role": "assistant", "content": None, "tool_calls": [tool_call(index % 7, rng)]}
            yield {"role": "tool", "tool_call_id": f"call_seed_{index % 7}", "content": prose(rng, chars_per_turn * 4)}
        else:
            yield {"role": "assistant", "content": prose(rng, chars_per_turn)}

    return list(chain.from_iterable(turn(index) for index in range(turns)))


def chat_response(content: str, tool_calls: list[JsonValue] | None, prompt_tokens: int, completion_tokens: int) -> JsonValue:
    message: dict[str, JsonValue] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-seed",
        "object": "chat.completion",
        "model": "gpt-5.5",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chat_log(
    rng: random.Random,
    label: str,
    *,
    messages: list[JsonValue],
    response_chars: int,
    tools: int = 0,
    called_tools: int = 0,
    offset_minutes: int,
    session_id: str | None = None,
) -> SeededLog:
    prompt_tokens: Final = len(json.dumps(messages)) // 4
    completion_tokens: Final = max(response_chars // 4, 1)
    tool_calls: Final = [tool_call(index, rng) for index in range(called_tools)] or None
    request: dict[str, JsonValue] = {"model": "gpt-5.5", "messages": messages, "stream": False}
    if tools:
        request["tools"] = [tool_definition(index) for index in range(tools)]
    return SeededLog(
        request_id=f"{REQUEST_ID_PREFIX}{label}",
        label=label,
        call_type="acompletion",
        model="gpt-5.5",
        provider="openai",
        status="success",
        session_id=session_id,
        offset_minutes=offset_minutes,
        duration_ms=1500 + completion_tokens // 10,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        spend=prompt_tokens * 0.000002 + completion_tokens * 0.000008,
        messages=messages,
        response=chat_response(prose(rng, response_chars), tool_calls, prompt_tokens, completion_tokens),
        proxy_server_request=request,
    )


def anthropic_log(rng: random.Random, offset_minutes: int) -> SeededLog:
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": [{"type": "text", "text": prose(rng, 2000)}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": prose(rng, 500)},
                {"type": "tool_use", "id": "toolu_seed_1", "name": "seed_tool_1", "input": {"query": "spend"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_seed_1", "content": prose(rng, 20_000)}]},
    ]
    tools: Final[list[JsonValue]] = [
        {"name": f"seed_tool_{index}", "description": "Synthetic Anthropic tool", "input_schema": {"type": "object"}}
        for index in range(3)
    ]
    response: Final[JsonValue] = {
        "id": "msg_seed",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {"type": "text", "text": prose(rng, 50_000)},
            {"type": "tool_use", "id": "toolu_seed_2", "name": "seed_tool_2", "input": {"query": "latency", "limit": 5}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 6000, "output_tokens": 12_500},
    }
    return SeededLog(
        request_id=f"{REQUEST_ID_PREFIX}anthropic-tool-use",
        label="anthropic-tool-use",
        call_type="anthropic_messages",
        model="claude-opus-5-5",
        provider="anthropic",
        status="success",
        session_id=None,
        offset_minutes=offset_minutes,
        duration_ms=9000,
        prompt_tokens=6000,
        completion_tokens=12_500,
        spend=6000 * 0.000015 + 12_500 * 0.000075,
        messages=messages,
        response=response,
        proxy_server_request={"model": "claude-opus-5-5", "max_tokens": 16_000, "messages": messages, "tools": tools},
    )


def failure_log(rng: random.Random, offset_minutes: int) -> SeededLog:
    messages: Final[list[JsonValue]] = [{"role": "user", "content": prose(rng, 300_000)}]
    return SeededLog(
        request_id=f"{REQUEST_ID_PREFIX}context-window-failure",
        label="context-window-failure",
        call_type="acompletion",
        model="gpt-5.5",
        provider="openai",
        status="failure",
        session_id=None,
        offset_minutes=offset_minutes,
        duration_ms=800,
        prompt_tokens=75_000,
        completion_tokens=0,
        spend=0.0,
        messages=messages,
        response={},
        proxy_server_request={"model": "gpt-5.5", "messages": messages},
        error_information={
            "error_code": "400",
            "error_class": "ContextWindowExceededError",
            "llm_provider": "openai",
            "error_message": "This model's maximum context length is 128000 tokens. Your messages resulted in 75000 tokens plus 300000 characters of synthetic prose.",
            "traceback": "Traceback (most recent call last):\n" + "\n".join(f"  File seed_{index}.py, line {index}" for index in range(40)),
        },
    )


def seeded_logs(rng: random.Random) -> tuple[SeededLog, ...]:
    """The size ladder: one axis per thing that can make the log drawer slow."""
    session: Final = f"{SESSION_ID_PREFIX}agent-run"
    single: Final = [{"role": "user", "content": "Summarise the seeded request logs in one paragraph."}]
    return (
        chat_log(rng, "baseline-small", messages=single, response_chars=400, offset_minutes=5),
        chat_log(rng, "response-100kb", messages=single, response_chars=100_000, offset_minutes=20),
        chat_log(rng, "response-1mb", messages=single, response_chars=1_000_000, offset_minutes=35),
        chat_log(rng, "response-5mb", messages=single, response_chars=5_000_000, offset_minutes=50),
        chat_log(rng, "turns-200", messages=chat_turns(rng, 200, 500), response_chars=2000, offset_minutes=70),
        chat_log(rng, "turns-1000", messages=chat_turns(rng, 1000, 500), response_chars=2000, offset_minutes=90),
        chat_log(rng, "system-prompt-200kb", messages=[{"role": "system", "content": prose(rng, 200_000)}, *single], response_chars=1500, offset_minutes=110),
        chat_log(rng, "tools-50", messages=single, response_chars=800, tools=50, called_tools=6, offset_minutes=130),
        anthropic_log(rng, offset_minutes=150),
        failure_log(rng, offset_minutes=170),
        *(
            chat_log(
                rng,
                f"session-call-{index:02d}",
                messages=chat_turns(rng, index + 1, 400),
                response_chars=3000,
                tools=4,
                called_tools=index % 3,
                offset_minutes=200 + index,
                session_id=session,
            )
            for index in range(30)
        ),
    )


def spread_offsets(logs: tuple[SeededLog, ...]) -> tuple[SeededLog, ...]:
    """Fit every row into the page's default 24h window, newest first."""
    last: Final = max(log.offset_minutes for log in logs)
    scale: Final = min(1.0, WINDOW_HOURS * 60 / max(last, 1))
    return tuple(log.model_copy(update={"offset_minutes": int(log.offset_minutes * scale)}) for log in logs)


def metadata(log: SeededLog, template: dict[str, JsonValue]) -> dict[str, JsonValue]:
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": log.prompt_tokens,
        "completion_tokens": log.completion_tokens,
        "total_tokens": log.prompt_tokens + log.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": log.prompt_tokens},
    }
    seeded: dict[str, JsonValue] = {
        **template,
        "status": log.status,
        "model_group": log.model,
        "deployment": f"{log.provider}/{log.model}",
        "deployment_model_name": f"{log.provider}/{log.model}",
        "user_api_key_team_alias": "seed-logs",
        "usage_object": usage,
        "additional_usage_values": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, **usage},
        "cost_breakdown": {
            "input_cost": log.prompt_tokens * 0.000002,
            "output_cost": log.completion_tokens * 0.000008,
            "total_cost": log.spend,
        },
        "litellm_overhead_time_ms": 12.5,
        "attempted_retries": 0,
        "max_retries": 2,
        "hidden_params": {"litellm_overhead_time_ms": 12.5, "response_cost": log.spend},
        "fixture_capture": None,
        "seed_label": log.label,
    }
    if log.error_information is not None:
        seeded["error_information"] = log.error_information
    return seeded


def postgres_row(log: SeededLog, template: dict[str, JsonValue], now: datetime) -> LiteLLM_SpendLogsCreateWithoutRelationsInput:
    from prisma import Json
    from prisma.types import LiteLLM_SpendLogsCreateWithoutRelationsInput

    end: Final = now - timedelta(minutes=log.offset_minutes)
    start: Final = end - timedelta(milliseconds=log.duration_ms)
    return LiteLLM_SpendLogsCreateWithoutRelationsInput(
        request_id=log.request_id,
        litellm_call_id=log.request_id,
        call_type=log.call_type,
        api_key=str(template.get("user_api_key", "seed-logs-key")),
        user="seed-logs-user",
        team_id="seed-logs-team",
        spend=log.spend,
        model=log.model,
        model_id=f"seed-logs-{log.model}",
        model_group=log.model,
        custom_llm_provider=log.provider,
        api_base=f"https://api.{log.provider}.example",
        prompt_tokens=log.prompt_tokens,
        completion_tokens=log.completion_tokens,
        total_tokens=log.prompt_tokens + log.completion_tokens,
        startTime=start,
        endTime=end,
        completionStartTime=start + timedelta(milliseconds=min(400, log.duration_ms // 2)),
        request_duration_ms=log.duration_ms,
        session_id=log.session_id,
        status=log.status,
        cache_hit="False",
        request_tags=Json(["seed-logs", log.label.split("-")[0]]),
        metadata=Json(metadata(log, template)),
        messages=Json(log.messages),
        response=Json(log.response),
        proxy_server_request=Json(log.proxy_server_request),
    )


COPY_SQL: Final = """INSERT INTO "LiteLLM_SpendLogs"
SELECT (jsonb_populate_record(s, jsonb_build_object(
    'request_id', s.request_id || '-copy-' || c.n,
    'litellm_call_id', s.request_id || '-copy-' || c.n,
    'session_id', NULL,
    'startTime', s."startTime" - make_interval(secs => c.n * $3::bigint / 1000.0),
    'endTime', s."endTime" - make_interval(secs => c.n * $3::bigint / 1000.0),
    'completionStartTime', s."completionStartTime" - make_interval(secs => c.n * $3::bigint / 1000.0)
))).*
FROM "LiteLLM_SpendLogs" AS s CROSS JOIN generate_series(1, $2::int) AS c(n)
WHERE s.request_id = $1"""


class SeedOptions(BaseModel):
    model_config = ConfigDict(frozen=True)
    profile: Profile
    timeout_seconds: float = 120


def seed_arguments(argv: Sequence[str] | None = None) -> SeedOptions:
    parser: Final = argparse.ArgumentParser(description="Insert synthetic request logs of controlled sizes into a local proxy DB")
    parser.add_argument("--profile", choices=PROFILES, default="default")
    parser.add_argument("--timeout-seconds", type=float, default=os.environ.get("LENS_DEV_SEED_TIMEOUT_SECONDS", "120"))
    arguments: Final = SeedOptions.model_validate(vars(parser.parse_args(argv)))
    if not math.isfinite(arguments.timeout_seconds) or arguments.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    return arguments


def metadata_template() -> dict[str, JsonValue]:
    """A real captured row's metadata, so the drawer sees the keys the gateway writes."""
    _, rows = spend_fixtures()[0]
    return JSON_OBJECT.validate_json(rows[0]["metadata"])


async def verify(client: httpx.AsyncClient, logs: tuple[SeededLog, ...], ui_base: str) -> None:
    async def fetch(log: SeededLog) -> dict[str, JsonValue]:
        detail: Final = await client.get(f"/spend/logs/ui/{log.request_id}")
        detail.raise_for_status()
        payload: Final = JSON.validate_json(detail.content)
        return {
            "label": log.label,
            "bytes": len(detail.content),
            "found": isinstance(payload, dict) and bool(payload),
            "url": f"{ui_base}/ui/?page=logs&log_id={log.request_id}"
            + (f"&session_id={log.session_id}" if log.session_id else ""),
        }

    results: Final = tuple(await asyncio.gather(*(fetch(log) for log in logs)))
    sys.stdout.write(json.dumps(list(results), indent=2) + "\n")
    if not all(result["found"] for result in results):
        raise RuntimeError("Seeded request logs did not round-trip through /spend/logs/ui/{request_id}")


async def seed(profile: Profile = "default", timeout_seconds: float = 120) -> int:
    from prisma import Prisma

    logs: Final = spread_offsets(seeded_logs(random.Random(RNG_SEED)))
    template: Final = metadata_template()
    now: Final = datetime.now(timezone.utc)
    proxy_url: Final = os.environ.get("PROXY_BASE_URL", "http://127.0.0.1:4000")
    ui_base: Final = os.environ.get("LENS_DEV_UI_URL", proxy_url)
    async with (
        httpx.AsyncClient(
            base_url=proxy_url,
            headers={"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"},
            timeout=timeout_seconds,
        ) as client,
        Prisma(http={"timeout": httpx.Timeout(600)}) as database,
    ):
        await database.litellm_spendlogs.delete_many(where={"request_id": {"startswith": REQUEST_ID_PREFIX}})
        await database.litellm_spendlogs.create_many(data=[postgres_row(log, template, now) for log in logs])
        if profile == "large":
            step_ms: Final = WINDOW_HOURS * 60 * 60 * 1000 // LARGE_COPIES
            await database.execute_raw(COPY_SQL, f"{REQUEST_ID_PREFIX}baseline-small", LARGE_COPIES, step_ms)
        await verify(client, tuple(log for log in logs if not log.session_id or log.label.endswith("-00")), ui_base)
    total: Final = len(logs) + (LARGE_COPIES if profile == "large" else 0)
    sys.stdout.write(f"Request log seed complete: profile={profile}, rows={total}, prefix={REQUEST_ID_PREFIX}\n")
    return 0


if __name__ == "__main__":
    arguments: Final = seed_arguments()
    raise SystemExit(asyncio.run(seed(arguments.profile, arguments.timeout_seconds)))
