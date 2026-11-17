"""Shared rig for the December 1 cost accuracy contracts (`test_dec1_costacc_*`).

These files pin the exact dollar amount the proxy bills for a scripted provider usage block on the big
providers, priced from the cost map the accounting group really loads: the integration runner sets
`LITELLM_LOCAL_MODEL_COST_MAP=True` (`tests/integration/run.py:128`), so the proxy reads the packaged
`litellm/model_prices_and_context_window_backup.json`, which is byte for byte the same file as the repo
root `model_prices_and_context_window.json` that every docstring cites by key and line.

Every expected value is a literal worked out by hand from those rows. Each test checks the two places a
user sees the cost: the `x-litellm-response-cost` header on the answer and the `spend` column of the one
`LiteLLM_SpendLogs` row of the test's own model group (each deployment gets a fresh random group name, so
the row is unambiguous). The row lands asynchronously and is polled, never slept for.

Provider rigs are the ones the provider contract suites already use, so the wire shapes agree:
`providers/dec1_bedrock_support.py` (Converse and InvokeModel), `providers/dec1_vertex_support.py`
(Vertex Gemini and Claude, with an owned OAuth token endpoint), `providers/dec1_gemini_support.py`
(AI Studio) and `providers/dec1_azure_support.py` (Azure OpenAI and Azure AI Foundry). Anthropic and
OpenAI get a small rig here. The shared proxy caches responses, so every prompt carries a fresh marker.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

LANDING_SECONDS: Final = 90
"""How long a SpendLogs row may take to land; the batch writer flushes on an interval."""
RELATIVE: Final = 1e-6
"""Float tolerance: the product multiplies in a different order than the hand-worked literal."""
ANTHROPIC_KEY: Final = "synthetic-dec1-costacc-anthropic"
OPENAI_KEY: Final = "synthetic-dec1-costacc-openai"
SPEND_SQL: Final = (
    "SELECT request_id, spend, prompt_tokens, completion_tokens, model, model_group, custom_llm_provider, "
    'status, call_type FROM "LiteLLM_SpendLogs" WHERE model_group=%s'
)


def marker() -> str:
    """A unique prompt suffix so the shared proxy's response cache never answers a test."""
    return f"dec1-costacc-{uuid.uuid4().hex}"


def _answering_probes(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        return respond(request)

    return answer


def json_reply(body: Mapping[str, JsonValue], *, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(body=json.dumps(body).encode(), headers=dict(headers or {}))


def always(reply: Reply) -> Callable[[Request], Reply]:
    def respond(_: Request) -> Reply:
        return reply

    return respond


@dataclass(frozen=True, slots=True)
class Rig:
    """One owned provider peer plus the model group name that routes to it."""

    gateway: Gateway
    scenario: Scenario
    wire: Wire
    model: str

    def chat(self, **fields: JsonValue) -> httpx.Response:
        body: Final[dict[str, JsonValue]] = {
            "model": self.model,
            "messages": [{"role": "user", "content": f"price this {marker()}"}],
            **fields,
        }
        return self.gateway.request("POST", "/v1/chat/completions", body)

    def messages(self, **fields: JsonValue) -> httpx.Response:
        body: Final[dict[str, JsonValue]] = {
            "model": self.model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": f"price this {marker()}"}],
            **fields,
        }
        return self.gateway.request("POST", "/v1/messages", body, headers={"anthropic-version": "2023-06-01"})

    def posts(self) -> tuple[Request, ...]:
        return tuple(request for request in self.wire.drain() if request.method == "POST")


@contextmanager
def deployment(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    model: str,
    base_suffix: str = "",
    model_info: Mapping[str, JsonValue] | None = None,
    **litellm_params: JsonValue,
) -> Iterator[Rig]:
    """Register `model` with `api_base` at a new owned peer (plus `base_suffix`) and yield the rig."""
    with wire_server(_answering_probes(respond)) as wire, gateway.scenario() as scenario:
        name: Final = scenario.model(
            model=model, api_base=wire.url + base_suffix, model_info=model_info, **litellm_params
        )
        yield Rig(gateway, scenario, wire, name)


# ---------------------------------------------------------------- Anthropic wire


def anthropic_usage(
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read: int | None = None,
    cache_write: int | None = None,
    write_5m: int | None = None,
    write_1h: int | None = None,
    **extra: JsonValue,
) -> dict[str, JsonValue]:
    """An Anthropic `usage` block. `write_5m` / `write_1h` add the `cache_creation` TTL breakdown;
    leaving both None sends `cache_creation_input_tokens` with no breakdown at all."""
    usage: Final[dict[str, JsonValue]] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write is not None:
        usage["cache_creation_input_tokens"] = cache_write
    breakdown: Final[dict[str, JsonValue]] = {}
    if write_5m is not None:
        breakdown["ephemeral_5m_input_tokens"] = write_5m
    if write_1h is not None:
        breakdown["ephemeral_1h_input_tokens"] = write_1h
    if breakdown:
        usage["cache_creation"] = breakdown
    return {**usage, **extra}


def anthropic_message(
    usage: Mapping[str, JsonValue], *, model: str, content: Sequence[JsonValue] | None = None
) -> dict[str, JsonValue]:
    return {
        "id": f"msg_costacc_{uuid.uuid4().hex[:20]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": list(content) if content is not None else [{"type": "text", "text": "priced"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": dict(usage),
    }


def anthropic_reply(usage: Mapping[str, JsonValue], *, model: str, **fields: JsonValue) -> Reply:
    return json_reply(anthropic_message(usage, model=model, **fields))  # pyright: ignore[reportArgumentType]  # test-local passthrough


def anthropic_sse(usage: Mapping[str, JsonValue], *, model: str) -> Reply:
    """A streamed Anthropic message: `message_start` carries the input side, `message_delta` the output count."""
    start_usage: Final = {key: value for key, value in usage.items() if key != "output_tokens"}
    events: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": f"msg_costacc_{uuid.uuid4().hex[:20]}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {**start_usage, "output_tokens": 1},
                },
            },
        ),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "priced"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events),
    )


@contextmanager
def anthropic(
    gateway: Gateway, reply: Reply, *, backend: str, **litellm_params: JsonValue
) -> Iterator[Rig]:
    """`anthropic/<backend>` at an owned peer that answers every POST with `reply` (the Anthropic config
    appends `/v1/messages` to the peer root)."""
    with deployment(
        gateway, always(reply), model=f"anthropic/{backend}", api_key=ANTHROPIC_KEY, **litellm_params
    ) as rig:
        yield rig


# ---------------------------------------------------------------- OpenAI wire


def openai_usage(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached: int | None = None,
    reasoning: int | None = None,
    audio_in: int | None = None,
    audio_out: int | None = None,
) -> dict[str, JsonValue]:
    """An OpenAI chat `usage` block; `prompt_tokens` already includes cached and audio input tokens."""
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    prompt_details: Final[dict[str, JsonValue]] = {}
    if cached is not None:
        prompt_details["cached_tokens"] = cached
    if audio_in is not None:
        prompt_details["audio_tokens"] = audio_in
    if prompt_details:
        usage["prompt_tokens_details"] = prompt_details
    completion_details: Final[dict[str, JsonValue]] = {}
    if reasoning is not None:
        completion_details["reasoning_tokens"] = reasoning
    if audio_out is not None:
        completion_details["audio_tokens"] = audio_out
    if completion_details:
        usage["completion_tokens_details"] = completion_details
    return usage


def openai_chat(usage: Mapping[str, JsonValue], *, model: str, service_tier: str | None = None) -> dict[str, JsonValue]:
    body: Final[dict[str, JsonValue]] = {
        "id": f"chatcmpl-costacc-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion",
        "created": 1763377200,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "priced"}, "finish_reason": "stop"}
        ],
        "usage": dict(usage),
    }
    if service_tier is not None:
        body["service_tier"] = service_tier
    return body


def openai_reply(usage: Mapping[str, JsonValue], *, model: str, service_tier: str | None = None) -> Reply:
    return json_reply(openai_chat(usage, model=model, service_tier=service_tier))


@contextmanager
def openai(gateway: Gateway, reply: Reply, *, backend: str, **litellm_params: JsonValue) -> Iterator[Rig]:
    """`openai/<backend>` at an owned peer (`api_base` ends in `/v1`) answering every POST with `reply`."""
    with deployment(
        gateway, always(reply), model=f"openai/{backend}", base_suffix="/v1", api_key=OPENAI_KEY, **litellm_params
    ) as rig:
        yield rig


# ---------------------------------------------------------------- what was billed


def header_cost(response: httpx.Response) -> float:
    """The `x-litellm-response-cost` header of a served answer, as a float."""
    assert response.status_code == 200, response.text
    value: Final = response.headers.get("x-litellm-response-cost")
    assert value is not None, dict(response.headers)
    return float(value)


def spend_row(gateway: Gateway, model_group: str, *, seconds: float = LANDING_SECONDS) -> dict[str, JsonValue]:
    """The one SpendLogs row of `model_group`. Polls, since rows land asynchronously."""
    rows: Final = eventually(
        lambda: read_rows(SPEND_SQL, (model_group,), database_url=gateway.database_url),
        lambda values: len(values) >= 1,
        seconds=seconds,
    )
    assert len(rows) == 1, rows
    return rows[0]


def row_spend(gateway: Gateway, model_group: str) -> float:
    return float(str(spend_row(gateway, model_group)["spend"]))


def assert_billed(gateway: Gateway, model_group: str, response: httpx.Response, expected: float) -> None:
    """The answer is 200, its cost header is `expected`, and the group's one spend row stores `expected`."""
    assert header_cost(response) == pytest.approx(expected, rel=RELATIVE), response.headers
    assert row_spend(gateway, model_group) == pytest.approx(expected, rel=RELATIVE)


def assert_row_billed(gateway: Gateway, model_group: str, expected: float) -> None:
    """Only the spend row: used for streamed answers, whose headers are sent before the usage is known."""
    assert row_spend(gateway, model_group) == pytest.approx(expected, rel=RELATIVE)
