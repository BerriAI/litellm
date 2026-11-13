from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import operator
import threading
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import openai
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.chat import ChatCompletionChunk
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(4 * graceful_stop_seconds() + 120)

SETTING: Final = "enable_mid_stream_fallback_continuation"
PARTIAL: Final = ("Alpha beta", " gamma")
JOINED_PARTIAL: Final = "".join(PARTIAL)
RELAY_TEXT: Final = " delta"
BACKUP_TEXT: Final = " omega."
PLAIN_TEXT: Final = "plain answer"
TOOL_PREAMBLE: Final = "Let me check"
BACKUP_USAGE: Final = (40, 6)
MAX_TOKENS: Final = 200
PRIMARY_PRICES: Final = (0.001, 0.002)
BACKUP_PRICES: Final = (0.003, 0.005)
SHARED_BACKEND: Final = "openai/gpt-4o-mini"
SEPARATE_BACKEND: Final = "openai/gpt-4o"
WEATHER_TOOL: Final[Mapping[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
TOOL_CALL_DELTA: Final[Mapping[str, JsonValue]] = {
    "index": 0,
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": ""},
}
NOTHING_SEEN: Final[tuple[ChatCompletionChunk, ...]] = ()
ALLOW_EVERYTHING: Final = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()\n"
EVENT_BATCH: Final = TypeAdapter(list[dict[str, JsonValue]])


@functools.cache
def cut_gate(marker: str, hop: str, attempt: int) -> threading.Event:
    return threading.Event()


@functools.cache
def attempts(marker: str, hop: str) -> Iterator[int]:
    return itertools.count()


def frame(identity: str, delta: Mapping[str, JsonValue], finish: str | None = None) -> bytes:
    body: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def complete_stream(identity: str, text: str, usage: tuple[int, int]) -> Reply:
    usage_frame: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
    }
    return Reply(
        content_type="text/event-stream",
        chunks=(
            frame(identity, {"role": "assistant", "content": text}),
            frame(identity, {}, finish="stop"),
            b"data: " + json.dumps(usage_frame).encode() + b"\n\n",
            b"data: [DONE]\n\n",
        ),
    )


def cut_after_caller_sees(marker: str, hop: str, *frames: bytes) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(b"".join(frames), b"data: never sent\n\n"),
        abort_after=1,
        gate_after_first=cut_gate(marker, hop, next(attempts(marker, hop))),
    )


def first_user_text(request: Request) -> str:
    messages: Final = JSON_OBJECT.validate_json(request.body)["messages"]
    assert isinstance(messages, list), messages
    first: Final = object_value(messages[0])["content"]
    assert isinstance(first, str), first
    return first


def marker_of(request: Request) -> str:
    return first_user_text(request).split()[-1]


def is_chat(request: Request) -> bool:
    return request.target.endswith("/chat/completions")


def model_list() -> Reply:
    return Reply(body=b'{"object": "list", "data": []}')


def primary(request: Request) -> Reply:
    if not is_chat(request):
        return model_list()
    scenario: Final = first_user_text(request).split()[0]
    marker: Final = marker_of(request)
    identity: Final = f"chatcmpl-{marker}-primary"
    if scenario == "reject":
        return Reply(status=500, body=b'{"error": {"message": "primary is down", "type": "server_error"}}')
    if scenario == "tool-cut":
        return cut_after_caller_sees(
            marker,
            "primary",
            frame(identity, {"role": "assistant", "content": TOOL_PREAMBLE}),
            frame(identity, {"tool_calls": [dict(TOOL_CALL_DELTA)]}),
        )
    return cut_after_caller_sees(
        marker,
        "primary",
        frame(identity, {"role": "assistant", "content": PARTIAL[0]}),
        frame(identity, {"content": PARTIAL[1]}),
    )


def relay(request: Request) -> Reply:
    if not is_chat(request):
        return model_list()
    marker: Final = marker_of(request)
    return cut_after_caller_sees(
        marker, "relay", frame(f"chatcmpl-{marker}-relay", {"role": "assistant", "content": RELAY_TEXT})
    )


def backup(request: Request) -> Reply:
    if not is_chat(request):
        return model_list()
    return complete_stream(f"chatcmpl-{marker_of(request)}-backup", BACKUP_TEXT, BACKUP_USAGE)


def no_prefill(request: Request) -> Reply:
    if not is_chat(request):
        return model_list()
    return complete_stream(f"chatcmpl-{marker_of(request)}-plain", PLAIN_TEXT, (9, 2))


def accept_events(request: Request) -> Reply:
    return Reply()


def registered_guardrail(name: str, mode: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "custom_code",
            "mode": mode,
            "default_on": False,
            "custom_code": ALLOW_EVERYTHING,
        },
    }


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    database_url: str
    primary: Wire
    relay: Wire
    backup: Wire
    no_prefill: Wire
    events: Wire


def joined_text(chunks: Iterable[ChatCompletionChunk]) -> str:
    return "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)


def has_tool_call(chunks: Iterable[ChatCompletionChunk]) -> bool:
    return any(choice.delta.tool_calls for chunk in chunks for choice in chunk.choices)


@dataclass(frozen=True, slots=True)
class Release:
    gate: threading.Event
    ready: Callable[[tuple[ChatCompletionChunk, ...]], bool]


def after_text(marker: str, hop: str, text: str, attempt: int = 0) -> Release:
    return Release(cut_gate(marker, hop, attempt), lambda seen: joined_text(seen) == text)


@dataclass(frozen=True, slots=True)
class Streamed:
    chunks: tuple[ChatCompletionChunk, ...]
    error: openai.APIError | None

    @property
    def text(self) -> str:
        return joined_text(self.chunks)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(chunk.id for chunk in self.chunks)

    @property
    def role_deltas(self) -> int:
        return sum(1 for chunk in self.chunks for choice in chunk.choices if choice.delta.role is not None)

    @property
    def finish_reasons(self) -> tuple[str, ...]:
        return tuple(choice.finish_reason for chunk in self.chunks for choice in chunk.choices if choice.finish_reason)

    @property
    def usage(self) -> tuple[int, int]:
        reported: Final = tuple(chunk.usage for chunk in self.chunks if chunk.usage is not None)
        assert reported, self.chunks
        return (reported[-1].prompt_tokens, reported[-1].completion_tokens)


def deployment(
    name: str,
    wire: Wire,
    prices: tuple[float, float],
    *,
    backend: str = SHARED_BACKEND,
    prefill: bool | None = None,
) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": backend,
            "api_base": wire.url + "/v1",
            "api_key": "synthetic-continuation-key",
            "input_cost_per_token": prices[0],
            "output_cost_per_token": prices[1],
        },
        **({"model_info": {"supports_assistant_prefill": prefill}} if prefill is not None else {}),
    }


def write_config(
    directory: Path,
    model_entries: Sequence[Mapping[str, JsonValue]],
    fallbacks: Sequence[Mapping[str, JsonValue]],
    router_settings: Mapping[str, JsonValue],
    guardrails: Sequence[Mapping[str, JsonValue]],
) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "model_list": [dict(entry) for entry in model_entries],
        "litellm_settings": {
            **object_value(base["litellm_settings"]),
            "callbacks": ["generic_api"],
            "DEFAULT_FLUSH_INTERVAL_SECONDS": 1,
        },
        "guardrails": [dict(entry) for entry in guardrails],
        "router_settings": {
            **object_value(base["router_settings"]),
            "num_retries": 0,
            "fallbacks": [dict(entry) for entry in fallbacks],
            **router_settings,
        },
    }
    path: Final = directory / f"continuation-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def booted_rig(
    directory: Path,
    fallbacks: Sequence[Mapping[str, JsonValue]],
    router_settings: Mapping[str, JsonValue],
    workers: int,
    guardrails: Sequence[Mapping[str, JsonValue]] = (),
) -> Generator[Rig]:
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        database_url: Final = stack.enter_context(scratch_database())
        wires: Final = tuple(
            stack.enter_context(wire_server(respond)) for respond in (primary, relay, backup, no_prefill, accept_events)
        )
        primary_wire, relay_wire, backup_wire, no_prefill_wire, events_wire = wires
        config: Final = write_config(
            directory,
            (
                *(
                    deployment(name, primary_wire, PRIMARY_PRICES)
                    for name in ("primary", "primary-skip", "primary-none", "primary-relay", "primary-shared")
                ),
                deployment("relay", relay_wire, PRIMARY_PRICES, prefill=True),
                deployment("backup", backup_wire, BACKUP_PRICES, prefill=True),
                deployment("no-prefill", no_prefill_wire, PRIMARY_PRICES, backend=SEPARATE_BACKEND),
                deployment("no-prefill-shared", no_prefill_wire, PRIMARY_PRICES),
            ),
            fallbacks,
            router_settings,
            guardrails,
        )
        gateway: Final = stack.enter_context(
            owned_proxy(
                shared,
                directory,
                {"DATABASE_URL": database_url, "GENERIC_LOGGER_ENDPOINT": events_wire.url},
                config=config,
                remove_environment=("DATABASE_URL_READ_REPLICA",),
                workers=workers,
            )
        )
        yield Rig(gateway, database_url, primary_wire, relay_wire, backup_wire, no_prefill_wire, events_wire)


@pytest.fixture(scope="module")
def configured(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    fallbacks: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"primary": ["backup"]},
        {"primary-skip": ["no-prefill", "backup"]},
        {"primary-none": ["no-prefill"]},
        {"primary-relay": ["relay"]},
        {"relay": ["backup"]},
        {"primary-shared": ["no-prefill-shared"]},
    )
    with booted_rig(
        tmp_path_factory.mktemp("continuation-configured"),
        fallbacks,
        {SETTING: True},
        2,
        (registered_guardrail("input-check", "pre_call"),),
    ) as rig:
        yield rig


@pytest.fixture(scope="module")
def output_guarded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with booted_rig(
        tmp_path_factory.mktemp("continuation-output-guarded"),
        ({"primary": ["backup"]},),
        {SETTING: True},
        2,
        (registered_guardrail("output-check", "post_call"),),
    ) as rig:
        yield rig


@pytest.fixture(scope="module")
def unset(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with booted_rig(
        tmp_path_factory.mktemp("continuation-unset"), ({"primary": ["no-prefill", "backup"]},), {}, 1
    ) as rig:
        yield rig


def new_key(rig: Rig) -> str:
    key: Final = rig.gateway.post("/key/generate", {})["key"]
    assert isinstance(key, str), key
    return key


def appended(seen: tuple[ChatCompletionChunk, ...], chunk: ChatCompletionChunk) -> tuple[ChatCompletionChunk, ...]:
    return (*seen, chunk)


def observed(
    response: Iterable[ChatCompletionChunk], releases: Sequence[Release]
) -> Iterator[ChatCompletionChunk | openai.APIError]:
    try:
        for seen in itertools.accumulate(response, appended, initial=NOTHING_SEEN):
            for release in releases:
                if release.ready(seen):
                    release.gate.set()
            yield from seen[-1:]
    except openai.APIError as error:
        yield error


def stream(
    rig: Rig,
    key: str,
    model: str,
    prompt: str,
    releases: Sequence[Release] = (),
    extra: Mapping[str, JsonValue] | None = None,
) -> Streamed:
    base_url: Final = f"{rig.gateway.client.base_url}".rstrip("/") + "/v1"
    try:
        with openai.OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=60) as client:
            response: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=MAX_TOKENS,
                extra_body=dict(extra or {}),
            )
            outcome: Final = tuple(observed(response, releases))
    finally:
        for release in releases:
            release.gate.set()
    errors: Final = tuple(item for item in outcome if isinstance(item, openai.APIError))
    return Streamed(
        tuple(item for item in outcome if isinstance(item, ChatCompletionChunk)), errors[0] if errors else None
    )


def chat_bodies(wire: Wire, marker: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(request.body)
        for request in wire.drain()
        if is_chat(request) and marker_of(request) == marker
    )


def spend_rows(rig: Rig, key: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, status, spend, prompt_tokens, completion_tokens, metadata FROM "LiteLLM_SpendLogs" '
            "WHERE api_key=%s",
            (hashlib.sha256(key.encode()).hexdigest(),),
            database_url=rig.database_url,
        ),
        lambda rows: len(rows) >= 1,
        seconds=70,
    )


def amount(value: JsonValue) -> float:
    assert isinstance(value, (int, float)), value
    return float(value)


def key_spend(rig: Rig, key: str) -> float:
    return amount(object_value(rig.gateway.get("/key/info", {"key": key})["info"])["spend"])


def close(observed_value: float, expected: float) -> bool:
    return math.isclose(observed_value, expected, rel_tol=1e-9, abs_tol=1e-12)


def cost(tokens: tuple[int, int], prices: tuple[float, float]) -> float:
    return tokens[0] * prices[0] + tokens[1] * prices[1]


def prefill_turn(text: str) -> dict[str, JsonValue]:
    return {"role": "assistant", "content": text, "prefix": True}


@dataclass(frozen=True, slots=True)
class Billed:
    rows: list[dict[str, JsonValue]]
    partial: tuple[int, int]


def billed_once(rig: Rig, key: str, received: Streamed) -> Billed:
    rows: Final = spend_rows(rig, key)
    assert [row["status"] for row in rows] == ["success"], rows
    row_tokens: Final = (rows[0]["prompt_tokens"], rows[0]["completion_tokens"])
    assert isinstance(row_tokens[0], int) and isinstance(row_tokens[1], int), rows
    assert received.usage == row_tokens, (received.usage, row_tokens)
    partial: Final = (row_tokens[0] - BACKUP_USAGE[0], row_tokens[1] - BACKUP_USAGE[1])
    assert partial[0] > 0 and partial[1] > 0, (row_tokens, BACKUP_USAGE)
    return Billed(rows, partial)


def assert_spend_prices_every_token(rig: Rig, key: str, billed: Billed) -> None:
    partial_cost: Final = cost(billed.partial, PRIMARY_PRICES)
    breakdown: Final = object_value(object_value(billed.rows[0]["metadata"])["cost_breakdown"])
    spend: Final = amount(billed.rows[0]["spend"])
    reported_partial_cost: Final = amount(object_value(breakdown["additional_costs"])["mid_stream_partial_cost"])
    assert close(spend, cost(BACKUP_USAGE, BACKUP_PRICES) + partial_cost) and close(
        reported_partial_cost, partial_cost
    ), f"partial={billed.partial} spend={spend} breakdown={json.dumps(breakdown)}"
    eventually(lambda: key_spend(rig, key), lambda value: close(value, spend), seconds=70)
    assert spend_rows(rig, key) == billed.rows


def test_text_cut_continues_on_the_prefill_fallback_and_bills_both_calls_once(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    key: Final = new_key(configured)
    prompt: Final = f"text-cut {marker}"
    received: Final = stream(configured, key, "primary", prompt, (after_text(marker, "primary", JOINED_PARTIAL),))
    assert received.error is None, received.error
    assert received.text == JOINED_PARTIAL + BACKUP_TEXT, received.chunks
    assert received.ids == {f"chatcmpl-{marker}-primary"}, received.ids
    assert received.role_deltas == 1, received.chunks
    assert received.finish_reasons == ("stop",), received.chunks
    assert len(chat_bodies(configured.primary, marker)) == 1
    continued: Final = chat_bodies(configured.backup, marker)
    assert len(continued) == 1, continued
    assert continued[0]["messages"] == [{"role": "user", "content": prompt}, prefill_turn(JOINED_PARTIAL)]
    assert "_mid_stream_continuation" not in continued[0], continued[0]
    billed: Final = billed_once(configured, key, received)
    assert continued[0]["max_tokens"] == MAX_TOKENS - billed.partial[1], (continued[0], billed.partial)
    assert_spend_prices_every_token(configured, key, billed)


def test_spend_row_of_a_continued_stream_carries_the_id_the_caller_received(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    key: Final = new_key(configured)
    received: Final = stream(
        configured, key, "primary", f"text-cut {marker}", (after_text(marker, "primary", JOINED_PARTIAL),)
    )
    assert received.error is None, received.error
    assert [row["request_id"] for row in spend_rows(configured, key)] == [received.chunks[0].id]


def test_tool_call_before_the_cut_keeps_the_original_error(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    received: Final = stream(
        configured,
        new_key(configured),
        "primary",
        f"tool-cut {marker}",
        (Release(cut_gate(marker, "primary", 0), has_tool_call),),
        {"tools": [dict(WEATHER_TOOL)]},
    )
    assert received.error is not None, received.chunks
    assert (received.text, has_tool_call(received.chunks)) == (TOOL_PREAMBLE, True), received.chunks
    assert len(chat_bodies(configured.primary, marker)) == 1
    assert chat_bodies(configured.backup, marker) == ()


def test_continuation_skips_a_fallback_without_prefill_support(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    prompt: Final = f"text-cut {marker}"
    received: Final = stream(
        configured, new_key(configured), "primary-skip", prompt, (after_text(marker, "primary", JOINED_PARTIAL),)
    )
    assert received.error is None, received.error
    assert received.text == JOINED_PARTIAL + BACKUP_TEXT, received.chunks
    assert chat_bodies(configured.no_prefill, marker) == ()
    assert [body["messages"] for body in chat_bodies(configured.backup, marker)] == [
        [{"role": "user", "content": prompt}, prefill_turn(JOINED_PARTIAL)]
    ]


def test_no_prefill_capable_fallback_keeps_the_original_error(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    received: Final = stream(
        configured,
        new_key(configured),
        "primary-none",
        f"text-cut {marker}",
        (after_text(marker, "primary", JOINED_PARTIAL),),
    )
    assert received.error is not None, received.chunks
    assert "No deployments available" not in str(received.error), received.error
    assert received.text == JOINED_PARTIAL, received.chunks
    assert len(chat_bodies(configured.primary, marker)) == 1
    assert chat_bodies(configured.no_prefill, marker) == ()


def test_undeclared_fallback_sharing_a_backend_with_a_prefill_deployment_keeps_the_original_error(
    configured: Rig,
) -> None:
    marker: Final = uuid.uuid4().hex
    received: Final = stream(
        configured,
        new_key(configured),
        "primary-shared",
        f"text-cut {marker}",
        (after_text(marker, "primary", JOINED_PARTIAL),),
    )
    assert (received.error is not None, received.text) == (True, JOINED_PARTIAL), received.chunks
    assert chat_bodies(configured.no_prefill, marker) == ()


def test_failure_before_content_still_falls_back_to_a_fallback_without_prefill_support(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    prompt: Final = f"reject {marker}"
    received: Final = stream(configured, new_key(configured), "primary-none", prompt)
    assert received.error is None, received.error
    assert received.text == PLAIN_TEXT, received.chunks
    assert [body["messages"] for body in chat_bodies(configured.no_prefill, marker)] == [
        [{"role": "user", "content": prompt}]
    ]


def stream_with_two_cuts(rig: Rig, key: str, marker: str) -> Streamed:
    return stream(
        rig,
        key,
        "primary-relay",
        f"text-cut {marker}",
        (after_text(marker, "primary", JOINED_PARTIAL), after_text(marker, "relay", JOINED_PARTIAL + RELAY_TEXT)),
    )


def test_second_cut_continues_with_both_partials_in_one_prefill(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    key: Final = new_key(configured)
    received: Final = stream_with_two_cuts(configured, key, marker)
    assert received.error is None, received.error
    assert received.text == JOINED_PARTIAL + RELAY_TEXT + BACKUP_TEXT, received.chunks
    assert received.role_deltas == 1, received.chunks
    user_turn: Final = {"role": "user", "content": f"text-cut {marker}"}
    relayed: Final = chat_bodies(configured.relay, marker)
    continued: Final = chat_bodies(configured.backup, marker)
    assert [body["messages"] for body in relayed] == [[user_turn, prefill_turn(JOINED_PARTIAL)]]
    assert [body["messages"] for body in continued] == [[user_turn, prefill_turn(JOINED_PARTIAL + RELAY_TEXT)]]
    partial: Final = billed_once(configured, key, received).partial
    relay_ceiling: Final = relayed[0]["max_tokens"]
    assert isinstance(relay_ceiling, int), relayed[0]
    assert 0 < MAX_TOKENS - relay_ceiling < partial[1], (relayed[0], partial)
    assert continued[0]["max_tokens"] == MAX_TOKENS - partial[1], (continued[0], partial)


def test_second_cut_bills_every_call_on_one_row(configured: Rig) -> None:
    key: Final = new_key(configured)
    received: Final = stream_with_two_cuts(configured, key, uuid.uuid4().hex)
    assert received.error is None, received.error
    assert_spend_prices_every_token(configured, key, billed_once(configured, key, received))


def test_repeating_a_continued_request_never_returns_only_the_continuation(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    key: Final = new_key(configured)
    prompt: Final = f"text-cut {marker}"
    answers: Final = tuple(
        stream(configured, key, "primary", prompt, (after_text(marker, "primary", JOINED_PARTIAL, attempt),))
        for attempt in range(2)
    )
    assert tuple((answer.error, answer.text) for answer in answers) == ((None, JOINED_PARTIAL + BACKUP_TEXT),) * 2


def test_without_the_setting_a_text_cut_keeps_the_original_error(unset: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    received: Final = stream(
        unset, new_key(unset), "primary", f"text-cut {marker}", (after_text(marker, "primary", JOINED_PARTIAL),)
    )
    assert received.error is not None, received.chunks
    assert received.text == JOINED_PARTIAL, received.chunks
    assert len(chat_bodies(unset.primary, marker)) == 1
    assert chat_bodies(unset.no_prefill, marker) == ()
    assert chat_bodies(unset.backup, marker) == ()


def router_setting(rig: Rig) -> JsonValue:
    return object_value(rig.gateway.get("/router/settings")["current_values"]).get(SETTING)


def test_turning_the_setting_on_at_runtime_continues_on_a_prefill_fallback(unset: Rig) -> None:
    rig: Final = unset
    rig.gateway.post("/config/update", {"router_settings": {SETTING: True}})
    try:
        assert router_setting(rig) is True
        marker: Final = uuid.uuid4().hex
        prompt: Final = f"text-cut {marker}"
        received: Final = stream(rig, new_key(rig), "primary", prompt, (after_text(marker, "primary", JOINED_PARTIAL),))
        assert received.error is None, received.error
        assert received.text == JOINED_PARTIAL + BACKUP_TEXT, received.chunks
        assert chat_bodies(rig.no_prefill, marker) == ()
        assert [body["messages"] for body in chat_bodies(rig.backup, marker)] == [
            [{"role": "user", "content": prompt}, prefill_turn(JOINED_PARTIAL)]
        ]
    finally:
        rig.gateway.post("/config/update", {"router_settings": {SETTING: False}})


def test_a_registered_output_guardrail_keeps_the_original_error(output_guarded: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    received: Final = stream(
        output_guarded,
        new_key(output_guarded),
        "primary",
        f"text-cut {marker}",
        (after_text(marker, "primary", JOINED_PARTIAL),),
    )
    assert (received.error is not None, received.text) == (True, JOINED_PARTIAL), received.chunks
    assert len(chat_bodies(output_guarded.primary, marker)) == 1
    assert chat_bodies(output_guarded.backup, marker) == ()


def test_a_request_fallback_bringing_its_own_messages_keeps_the_original_error(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    received: Final = stream(
        configured,
        new_key(configured),
        "primary",
        f"text-cut {marker}",
        (after_text(marker, "primary", JOINED_PARTIAL),),
        {"fallbacks": [{"model": "backup", "messages": [{"role": "user", "content": f"start over {marker}"}]}]},
    )
    assert (received.error is not None, received.text) == (True, JOINED_PARTIAL), received.chunks
    assert len(chat_bodies(configured.primary, marker)) == 1
    assert chat_bodies(configured.backup, marker) == ()


def carries_tag(event: Mapping[str, JsonValue], marker: str) -> bool:
    tags: Final = event.get("request_tags")
    return isinstance(tags, list) and marker in tags


def tagged_events(batches: Iterable[Request], marker: str) -> tuple[dict[str, JsonValue], ...]:
    events: Final = itertools.chain.from_iterable(EVENT_BATCH.validate_json(batch.body) for batch in batches)
    return tuple(event for event in events if carries_tag(event, marker))


def logged_events(sink: Wire, marker: str) -> Iterator[tuple[dict[str, JsonValue], ...]]:
    return itertools.accumulate((tagged_events(sink.drain(), marker) for _ in itertools.count()), operator.add)


def test_the_cut_call_logs_its_failure_under_its_own_deployment_and_price(configured: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    received: Final = stream(
        configured,
        new_key(configured),
        "primary",
        f"text-cut {marker}",
        (after_text(marker, "primary", JOINED_PARTIAL),),
        {"metadata": {"tags": [marker]}},
    )
    assert received.error is None, received.error
    polls: Final = logged_events(configured.events, marker)
    events: Final = eventually(lambda: next(polls), lambda logged: len(logged) >= 2, seconds=30)
    assert sorted(str(event["status"]) for event in events) == ["failure", "success"], events
    failure: Final = next(event for event in events if event["status"] == "failure")
    assert failure["model_group"] == "primary", failure
    assert str(failure["api_base"]).startswith(configured.primary.url), failure
    tokens: Final = (failure["prompt_tokens"], failure["completion_tokens"])
    assert isinstance(tokens[0], int) and isinstance(tokens[1], int) and tokens[0] > 0 and tokens[1] > 0, failure
    assert close(amount(failure["response_cost"]), cost((tokens[0], tokens[1]), PRIMARY_PRICES)), failure
