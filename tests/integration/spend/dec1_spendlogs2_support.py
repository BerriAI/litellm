"""Shared building blocks for the `test_dec1_spendlogs2_*` contract files (Spend Logs V2, round 2).

Round 2 pins what the Rust `callbacks-spend-log` port (plan of 2026-11-16, PRs 0 to 5) must keep for every route
that writes a `LiteLLM_SpendLogs` row, for requests the proxy refuses before any provider call, for the ledger
delivery semantics between the row writer and the Python ledger worker, for the redaction settings matrix and for
the legacy read routes. It builds on `dec1_spend_support` (scripted provider, prices 0.001 / 0.002 per token, so a
chat call costs 0.02) and `dec1_spendlogs_support` (owned proxy with settings, row readers), and only adds what
those two lack: failure row readers, a provider that repeats one response id, guardrail and rate limit set up,
the per-provider `/v1/messages` rigs and small readers for the legacy routes.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.spend.dec1_spend_support import (
    LANDING_SECONDS,
    REPLY_TEXT,
    ROW_COLUMNS,
    ask,
    hashed,
    respond,
)

CALL_ID_HEADER: Final = "x-litellm-call-id"
FAILURE_COLUMNS: Final = (
    "request_id, call_type, api_key, spend, total_tokens, prompt_tokens, completion_tokens, model, model_group, "
    'custom_llm_provider, status, metadata, team_id, "user", litellm_call_id, request_tags'
)


def failure_rows(gateway: Gateway, key: str, count: int) -> list[dict[str, JsonValue]]:
    """The key's `status = 'failure'` rows, oldest first, once exactly `count` exist."""
    return eventually(
        lambda: gateway.rows(
            f'SELECT {FAILURE_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND status = %s '
            'ORDER BY "startTime", request_id',
            (hashed(key), "failure"),
        ),
        lambda found: len(found) == count,
        seconds=LANDING_SECONDS,
    )


def rows_with_call_id(gateway: Gateway, call_id: str, count: int) -> list[dict[str, JsonValue]]:
    """Rows whose `litellm_call_id` is `call_id`, once exactly `count` exist."""
    return eventually(
        lambda: gateway.rows(
            f'SELECT {ROW_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s ORDER BY "startTime"',
            (call_id,),
        ),
        lambda found: len(found) == count,
        seconds=LANDING_SECONDS,
    )


def no_rows_with_call_id(gateway: Gateway, call_id: str, *, seconds: float = 15) -> None:
    """Wait a bounded time for a row that must never land; fail as soon as one does."""
    found: Final = eventually(
        lambda: gateway.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s', (call_id,)),
        lambda values: len(values) > 0,
        seconds=seconds,
        return_last_on_timeout=True,
    )
    assert found == [], found


def error_information(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    metadata: Final = row["metadata"]
    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else metadata
    return object_value(object_value(parsed)["error_information"])


def metadata_dict(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    metadata: Final = row["metadata"]
    return object_value(json.loads(metadata) if isinstance(metadata, str) else metadata)


def json_column(value: JsonValue) -> JsonValue:
    return json.loads(value) if isinstance(value, str) else value


def call_id_of(response: httpx.Response) -> str:
    return response.headers[CALL_ID_HEADER]


def fresh_call_id() -> str:
    return f"dec1-sl2-{uuid.uuid4().hex}"


@contextmanager
def fixed_id_upstream(response_id: str) -> Iterator[Wire]:
    """A chat provider that answers every call with the same response `id` (and the usual 10 + 5 tokens), the
    way a misbehaving or replaying provider would."""

    def answer(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        return Reply(
            body=json.dumps(
                {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": REPLY_TEXT}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(answer) as wire:
        yield wire


def first_refusal(send: Callable[[], httpx.Response], status: int, attempts: int = 8) -> httpx.Response:
    """Send until one answer has `status`; the limiter counts asynchronously, so the first over-limit call is not
    always the second one. Fails if no attempt is refused."""
    for _ in range(attempts):
        response = send()
        if response.status_code == status:
            return response
        assert response.status_code == 200, response.text
    raise AssertionError(f"no {status} in {attempts} attempts")


def custom_code_guardrail(gateway: Gateway, scenario: Scenario, code: str, mode: str = "pre_call") -> str:
    """Create a `custom_code` guardrail with `POST /guardrails` (names given to requests must exist, addendum
    lesson) and delete it at teardown. Returns its name."""
    name: Final = f"dec1-sl2-guard-{uuid.uuid4().hex[:10]}"
    created: Final = gateway.post(
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": name,
                "litellm_params": {"guardrail": "custom_code", "mode": mode, "default_on": False, "custom_code": code},
            }
        },
    )
    identity: Final = string_value(created["guardrail_id"])
    scenario.cleanups.callback(gateway.request, "DELETE", f"/guardrails/{identity}")
    return name


BLOCKING_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n    return block('dec1 spendlogs2 blocked input')\n"
)


def guarded_ask(gateway: Gateway, model: str, key: str, guardrail: str) -> httpx.Response:
    """A chat call naming `guardrail`; polls because a guardrail created on one worker reaches the others on the
    next config sync."""
    return eventually(
        lambda: ask(gateway, model, key, metadata={"guardrails": [guardrail]}),
        lambda response: response.status_code != 200,
        seconds=30,
    )


def utc_day(offset: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset)).strftime("%Y-%m-%d")


def get_json(gateway: Gateway, path: str, params: Mapping[str, str] | None = None) -> JsonValue:
    response: Final = gateway.request("GET", path, params=dict(params or {}))
    assert response.status_code == 200, f"GET {path}: {response.status_code} {response.text}"
    parsed: Final[JsonValue] = json.loads(response.content)
    return parsed


def wait_for_key_spend(gateway: Gateway, key: str, amount: float) -> float:
    rows: Final = eventually(
        lambda: gateway.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),)),
        lambda found: bool(found) and float(str(found[0]["spend"])) >= amount - 1e-9,
        seconds=LANDING_SECONDS,
    )
    return float(str(rows[0]["spend"]))


def daily_user_counters(gateway: Gateway, key: str) -> dict[str, JsonValue] | None:
    """Sum of the key's `LiteLLM_DailyUserSpend` counters across rows, or None if none exist yet."""
    rows: Final = gateway.rows(
        "SELECT COALESCE(SUM(spend), 0)::float8 AS spend, COALESCE(SUM(api_requests), 0)::int AS api_requests, "
        "COALESCE(SUM(successful_requests), 0)::int AS successful_requests, "
        "COALESCE(SUM(failed_requests), 0)::int AS failed_requests, COUNT(*)::int AS row_count "
        'FROM "LiteLLM_DailyUserSpend" WHERE api_key = %s',
        (hashed(key),),
    )
    return None if not rows or rows[0]["row_count"] == 0 else rows[0]


def upstream_url() -> str:
    return os.environ["INTEGRATION_UPSTREAM_URL"].rstrip("/")


__all__ = (
    "BLOCKING_CODE",
    "CALL_ID_HEADER",
    "call_id_of",
    "custom_code_guardrail",
    "daily_user_counters",
    "error_information",
    "failure_rows",
    "first_refusal",
    "fixed_id_upstream",
    "fresh_call_id",
    "get_json",
    "guarded_ask",
    "json_column",
    "metadata_dict",
    "no_rows_with_call_id",
    "respond",
    "rows_with_call_id",
    "upstream_url",
    "utc_day",
    "wait_for_key_spend",
)
