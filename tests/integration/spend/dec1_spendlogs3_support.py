"""Shared building blocks for the `test_dec1_spendlogs3_*` contract files (Spend Logs V2, round 3).

Round 3 pins what the Rust `callbacks-spend-log` port (plan of 2026-11-16) and the rollup move that must land
before its PR 4 have to keep, beyond rounds 1 and 2: the rollups that ride on the spend writer today
(`LiteLLM_DailyRequestErrors`, `LiteLLM_SpendLogToolIndex`, `LiteLLM_DailyToolSpend`, `LiteLLM_DailyModelUsage`
task types), the row of every non chat inference route (speech, transcription, moderation, rerank, image
generation), the keying of the daily tables projected by the ledger worker, and request and response body storage.

It builds on `dec1_spend_support` (scripted provider priced 0.001 / 0.002 per token, so a chat call costs 0.02)
and `dec1_spendlogs_support` (owned proxy with settings), and adds: a chat provider that answers with tool calls, a
provider for the media and rerank routes, a provider with a long answer, and readers for the rollup tables.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.spend.dec1_spend_support import (
    CHAT_COMPLETION_TOKENS,
    CHAT_PROMPT_TOKENS,
    LANDING_SECONDS,
    chat_model,
    hashed,
    respond,
)

SL3: Final = "dec1-sl3"
TRANSCRIPT: Final = "dec1 sl3 synthetic transcript"
AUDIO_BYTES: Final = b"RIFF\x24\x00\x00\x00WAVEfmt dec1 sl3 synthetic audio"
SPEECH_BYTES: Final = b"ID3dec1-sl3-synthetic-mp3-bytes"
LONG_ANSWER_LENGTH: Final = 3000
"""Length of the answer `long_answer_upstream` sends: `x` * 1500 followed by `y` * 1500."""


def utc_today() -> str:
    """The UTC day a row written now is bucketed under (`startTime.split("T")[0]` on a UTC timestamp)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def fresh_tool_name() -> str:
    """A tool name no other test uses, so the global `(date, tool_name)` rollup row belongs to one test."""
    return f"dec1_sl3_tool_{uuid.uuid4().hex[:12]}"


def _chat_body(content: str | None, tool_calls: list[dict[str, JsonValue]] | None) -> bytes:
    message: Final[dict[str, JsonValue]] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return json.dumps(
        {
            "id": f"chatcmpl-dec1sl3-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {
                "prompt_tokens": CHAT_PROMPT_TOKENS,
                "completion_tokens": CHAT_COMPLETION_TOKENS,
                "total_tokens": CHAT_PROMPT_TOKENS + CHAT_COMPLETION_TOKENS,
            },
        }
    ).encode()


@contextmanager
def tool_call_upstream(*names: str) -> Iterator[Wire]:
    """A chat provider whose non streamed answer invokes the named tools, in order, as OpenAI `tool_calls`
    (each with arguments `{}`), with the usual 10 + 5 tokens. Other routes answer as `respond` does."""
    calls: Final[list[dict[str, JsonValue]]] = [
        {
            "id": f"call_{index}_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }
        for index, name in enumerate(names)
    ]

    def answer(request: Request) -> Reply:
        path: Final = request.target.split("?", 1)[0]
        if request.method == "POST" and path.endswith("/chat/completions"):
            body: Final = JSON_OBJECT.validate_json(request.body) if request.body else {}
            if body.get("stream") is not True:
                return Reply(body=_chat_body(None, calls))
        return respond(request)

    with wire_server(answer) as wire:
        yield wire


@contextmanager
def long_answer_upstream() -> Iterator[Wire]:
    """A chat provider whose answer text is `x` * 1500 + `y` * 1500 (3000 characters), with 10 + 5 tokens."""

    def answer(request: Request) -> Reply:
        path: Final = request.target.split("?", 1)[0]
        if request.method == "POST" and path.endswith("/chat/completions"):
            return Reply(body=_chat_body("x" * 1500 + "y" * 1500, None))
        return respond(request)

    with wire_server(answer) as wire:
        yield wire


def media_reply(request: Request) -> Reply:
    """Answers the OpenAI speech, transcription, moderation and image routes and the Cohere v2 rerank route;
    everything else is answered by `respond`."""
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    path: Final = request.target.split("?", 1)[0]
    if path.endswith("/audio/speech"):
        return Reply(body=SPEECH_BYTES, content_type="audio/mpeg")
    if path.endswith("/audio/transcriptions"):
        return Reply(body=json.dumps({"text": TRANSCRIPT}).encode())
    if path.endswith("/moderations"):
        return Reply(
            body=json.dumps(
                {
                    "id": f"modr-dec1sl3-{uuid.uuid4().hex}",
                    "model": "omni-moderation-latest",
                    "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
                }
            ).encode()
        )
    if path.endswith("/images/generations"):
        return Reply(
            body=json.dumps({"created": 1, "data": [{"url": "https://example.invalid/dec1-sl3.png"}]}).encode()
        )
    if path.endswith("/rerank"):
        return Reply(
            body=json.dumps(
                {
                    "id": f"dec1-sl3-rerank-{uuid.uuid4().hex}",
                    "results": [{"index": 0, "relevance_score": 0.9}],
                    "meta": {"billed_units": {"search_units": 1}},
                }
            ).encode()
        )
    return respond(request)


@contextmanager
def media_upstream() -> Iterator[Wire]:
    with wire_server(media_reply) as wire:
        yield wire


@dataclass(frozen=True, slots=True)
class Route:
    """One non chat inference route, the deployment that serves it from `media_upstream`, and what its row and
    daily rows are expected to carry (`ROUTE_ENDPOINT_MAPPING`, litellm/proxy/route_llm_request.py:156)."""

    name: str
    call_type: str
    endpoint: str
    upstream_model: str
    mode: str
    provider_id_prefix: str | None
    """Prefix of the provider's response id when the row is filed under it, None when the row falls back to the
    litellm call id (`get_spend_logs_id`, litellm/proxy/spend_tracking/spend_tracking_utils.py:408)."""

    def model(self, scenario: Scenario, wire: Wire) -> str:
        if self.name == "rerank":
            return chat_model(scenario, wire, model=self.upstream_model, api_base=f"{wire.url}/v2/rerank")
        return chat_model(scenario, wire, model=self.upstream_model, model_info={"mode": self.mode})

    def send(self, gateway: Gateway, model: str, key: str) -> httpx.Response:
        if self.name == "transcription":
            return gateway.request_multipart(
                "/v1/audio/transcriptions", {"model": model}, {"file": ("dec1.wav", AUDIO_BYTES, "audio/wav")}, key=key
            )
        body: Final[dict[str, JsonValue]] = {"model": model}
        if self.name == "speech":
            body.update(input=f"dec1 sl3 {uuid.uuid4().hex}", voice="alloy")
            return gateway.request("POST", "/v1/audio/speech", body, key=key)
        if self.name == "moderation":
            body.update(input=f"dec1 sl3 {uuid.uuid4().hex}")
            return gateway.request("POST", "/v1/moderations", body, key=key)
        if self.name == "image":
            body.update(prompt=f"dec1 sl3 {uuid.uuid4().hex}", n=1)
            return gateway.request("POST", "/v1/images/generations", body, key=key)
        body.update(query=f"dec1 sl3 {uuid.uuid4().hex}", documents=["only document"])
        return gateway.request("POST", "/v1/rerank", body, key=key)


SPEECH: Final = Route("speech", "aspeech", "/audio/speech", "openai/tts-1", "audio_speech", None)
TRANSCRIPTION: Final = Route(
    "transcription", "atranscription", "/audio/transcriptions", "openai/whisper-1", "audio_transcription", None
)
MODERATION: Final = Route(
    "moderation", "amoderation", "/moderations", "openai/omni-moderation-latest", "moderation", "modr-"
)
IMAGE: Final = Route("image", "aimage_generation", "/image/generations", "openai/dall-e-3", "image_generation", None)
RERANK: Final = Route("rerank", "arerank", "/rerank", "cohere/rerank-v3.5", "rerank", "dec1-sl3-rerank-")
ROUTES: Final = (SPEECH, TRANSCRIPTION, MODERATION, IMAGE, RERANK)


def wait_rows(
    gateway: Gateway, query: str, parameters: tuple[str, ...], count: int, *, seconds: float = LANDING_SECONDS
) -> list[dict[str, JsonValue]]:
    """Rows of a fixed query once exactly `count` exist."""
    return eventually(lambda: gateway.rows(query, parameters), lambda found: len(found) == count, seconds=seconds)


def stays_empty(gateway: Gateway, query: str, parameters: tuple[str, ...], *, seconds: float = 20) -> None:
    """Wait a bounded time for a row that must never appear; fail as soon as one does."""
    found: Final = eventually(
        lambda: gateway.rows(query, parameters),
        lambda values: len(values) > 0,
        seconds=seconds,
        return_last_on_timeout=True,
    )
    assert found == [], found


REQUEST_ERRORS_QUERY: Final = (
    "SELECT date, api_key, team_id, user_id, model_group, status_code, failed_requests::int AS failed_requests "
    'FROM "LiteLLM_DailyRequestErrors" WHERE api_key = %s ORDER BY status_code'
)


def request_error_rows(gateway: Gateway, key: str, count: int) -> list[dict[str, JsonValue]]:
    """The key's `LiteLLM_DailyRequestErrors` rows (by its sha256 hash), once exactly `count` exist."""
    return wait_rows(gateway, REQUEST_ERRORS_QUERY, (hashed(key),), count)


def request_errors_reach(gateway: Gateway, key: str, failed: int) -> list[dict[str, JsonValue]]:
    """The key's request error rows once their `failed_requests` add up to at least `failed`."""
    return eventually(
        lambda: gateway.rows(REQUEST_ERRORS_QUERY, (hashed(key),)),
        lambda found: sum(int(str(row["failed_requests"])) for row in found) >= failed,
        seconds=LANDING_SECONDS,
    )


TOOL_INDEX_QUERY: Final = (
    'SELECT request_id, tool_name, start_time::text AS start_time FROM "LiteLLM_SpendLogToolIndex" '
    "WHERE request_id = %s ORDER BY tool_name"
)
TOOL_DAY_QUERY: Final = (
    "SELECT date, tool_name, spend::float8 AS spend, total_tokens::int AS total_tokens, "
    'request_count::int AS request_count FROM "LiteLLM_DailyToolSpend" WHERE tool_name = %s'
)


def tool_index_rows(gateway: Gateway, request_id: str, count: int) -> list[dict[str, JsonValue]]:
    return wait_rows(gateway, TOOL_INDEX_QUERY, (request_id,), count)


def tool_day_row(gateway: Gateway, tool_name: str, *, requests: int = 1) -> dict[str, JsonValue]:
    """The one `LiteLLM_DailyToolSpend` row of a tool, once its `request_count` reaches `requests`."""
    rows: Final = eventually(
        lambda: gateway.rows(TOOL_DAY_QUERY, (tool_name,)),
        lambda found: len(found) == 1 and int(str(found[0]["request_count"])) >= requests,
        seconds=LANDING_SECONDS,
    )
    return rows[0]


MODEL_USAGE_QUERY: Final = (
    "SELECT date, model_group, model, custom_llm_provider, task_type, spend::float8 AS spend, "
    "prompt_tokens::int AS prompt_tokens, completion_tokens::int AS completion_tokens, "
    "request_count::int AS request_count, successful_requests::int AS successful_requests, "
    "failed_requests::int AS failed_requests, timed_requests::int AS timed_requests, "
    'total_response_time_ms::int AS total_response_time_ms FROM "LiteLLM_DailyModelUsage" '
    "WHERE model_group = %s ORDER BY task_type, model"
)


def model_usage_rows(gateway: Gateway, model_group: str, count: int) -> list[dict[str, JsonValue]]:
    return wait_rows(gateway, MODEL_USAGE_QUERY, (model_group,), count)


def model_usage_reaches(gateway: Gateway, model_group: str, requests: int) -> list[dict[str, JsonValue]]:
    """The group's `LiteLLM_DailyModelUsage` rows once their `request_count` adds up to at least `requests`."""
    return eventually(
        lambda: gateway.rows(MODEL_USAGE_QUERY, (model_group,)),
        lambda found: sum(int(str(row["request_count"])) for row in found) >= requests,
        seconds=LANDING_SECONDS,
    )


def daily_table_rows(gateway: Gateway, table: str, column: str, value: str, count: int) -> list[dict[str, JsonValue]]:
    """Rows of one `LiteLLM_Daily*Spend` table whose `column` equals `value`, as JSON objects, once exactly
    `count` exist."""
    assert table.startswith("LiteLLM_Daily") and table.isidentifier() and column.isidentifier(), (table, column)
    found: Final = wait_rows(
        gateway,
        f'SELECT to_jsonb(d) AS row FROM "{table}" d WHERE "{column}" = %s ORDER BY endpoint, model',
        (value,),
        count,
    )
    return [dict(JSON_OBJECT.validate_python(row["row"])) for row in found]


def stored_json(value: JsonValue) -> JsonValue:
    """A `Json` column as a value, whether the driver handed back text or a decoded object."""
    return json.loads(value) if isinstance(value, str) else value


def counters(row: Mapping[str, JsonValue], *names: str) -> tuple[int, ...]:
    return tuple(int(str(row[name])) for name in names)


__all__ = (
    "AUDIO_BYTES",
    "IMAGE",
    "LONG_ANSWER_LENGTH",
    "MODERATION",
    "RERANK",
    "ROUTES",
    "SL3",
    "SPEECH",
    "TRANSCRIPT",
    "TRANSCRIPTION",
    "Route",
    "counters",
    "daily_table_rows",
    "fresh_tool_name",
    "long_answer_upstream",
    "media_upstream",
    "model_usage_reaches",
    "model_usage_rows",
    "request_error_rows",
    "request_errors_reach",
    "stays_empty",
    "stored_json",
    "tool_call_upstream",
    "tool_day_row",
    "tool_index_rows",
    "utc_today",
    "wait_rows",
)
