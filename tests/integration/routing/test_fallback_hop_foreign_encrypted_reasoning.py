"""Fallback hops to a deployment that cannot decrypt the previous deployment's encrypted reasoning.

A model group lists an order-1 and an order-2 deployment on distinct encryption boundaries (a
distinct ``api_base`` each, both played by the scripted upstream). The integration proxy keeps
``disable_cooldowns: true`` and ``num_retries: 0``, so every request tries order 1 first and a
failure there hops to order 2 through the router's order-based fallback. The hop must strip the
``encrypted_content`` order 2 cannot decrypt from Responses ``input`` items (summary kept) and the
bridge-tagged thinking blocks from ``messages``, the encrypted-content affinity pin must yield to
the hop's ``_target_order``, and a client cannot forge hop state (``fallback_depth``,
``_target_order``, ``attempted_targets``) through the request body while ``max_fallbacks`` stays
client-settable. Every request carries a unique marker so the response cache never serves it, and
a hop is proven by the upstream's own record: the order-1 attempt with the full history, then the
order-2 request. Every call goes through a lane: one keep-alive connection pinned to a single
worker, used only once ``/model/info`` on that same connection lists every deployment the call
needs, because the peer worker learns a ``/model/new`` row through the config-sync resync one to
sixteen seconds later, and a resync that lands between a group's two rows leaves that worker
serving the group with one deployment until the next resync.
"""

import asyncio
import json
import re
import signal
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request, wire_server
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, SseResponse, StoredResponse
from pydantic import JsonValue

ORDER_ONE_BLOB: Final = "gAAAAA-minted-by-order-1"
ORDER_TWO_BLOB: Final = "gAAAAA-minted-by-order-2"
SUMMARY_TEXT: Final = "multiply 17 by 23"
TAGGED_SIGNATURE: Final = f"litellm_encrypted_reasoning:{ORDER_ONE_BLOB}"
PROVIDER_KEY: Final = "integration-provider-key"
RESPONSES_MODEL: Final = "openai/gpt-5"
BRIDGED_MODEL: Final = "openai/gpt-5-codex"
AFFINITY_CHECK: Final = "encrypted_content_affinity"
HTTPX: Final = "httpx"
OPENAI_SYNC: Final = "openai-sync"
OPENAI_ASYNC: Final = "openai-async"
ANTHROPIC_SYNC: Final = "anthropic-sync"
ANTHROPIC_ASYNC: Final = "anthropic-async"
RESPONSES_CLIENTS: Final = (HTTPX, OPENAI_SYNC, OPENAI_ASYNC)
MESSAGES_CLIENTS: Final = (HTTPX, ANTHROPIC_SYNC, ANTHROPIC_ASYNC)


def _failure(message: str) -> dict[str, JsonValue]:
    return {"error": {"message": message, "type": "server_error", "code": "server_error", "param": None}}


def _responses_body(blob: str) -> dict[str, JsonValue]:
    return {
        "id": "resp_$UNIQUE_ID",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5-scripted",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_$UNIQUE_ID",
                "summary": [{"type": "summary_text", "text": "nineteen times twenty-one"}],
                "encrypted_content": blob,
            },
            {
                "type": "message",
                "id": "msg_$UNIQUE_ID",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "399", "annotations": []}],
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
    }


CHAT_BODY: Final[dict[str, JsonValue]] = {
    "id": "chatcmpl-$UNIQUE_ID",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5-scripted",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "399"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
}


def failing(message: str = "order one is down") -> StoredResponse:
    return JsonResponse(content_type="application/json", status=500, body=_failure(message))


def healthy_json(blob: str = ORDER_TWO_BLOB) -> StoredResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            "POST /responses": JsonResponse(content_type="application/json", body=_responses_body(blob)),
            "POST /chat/completions": JsonResponse(content_type="application/json", body=CHAT_BODY),
        },
    )


def _responses_frame(event: Mapping[str, JsonValue]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}"


def broken_responses_stream() -> StoredResponse:
    opened: Final = {**_responses_body(ORDER_ONE_BLOB), "status": "in_progress", "output": [], "usage": None}
    return SseResponse(
        content_type="text/event-stream",
        frames=(
            _responses_frame({"type": "response.created", "sequence_number": 0, "response": opened}),
            _responses_frame({"type": "response.in_progress", "sequence_number": 1, "response": opened}),
            _responses_frame({"type": "error", "sequence_number": 2, **_failure("order one broke mid-stream")}),
        ),
    )


def healthy_responses_stream(blob: str = ORDER_TWO_BLOB) -> StoredResponse:
    body: Final = _responses_body(blob)
    opened: Final = {**body, "status": "in_progress", "output": [], "usage": None}
    output: Final = body["output"]
    assert isinstance(output, list)
    reasoning, message = output
    assert isinstance(reasoning, dict) and isinstance(message, dict)
    part: Final = {"type": "output_text", "text": "", "annotations": []}
    events: Final = (
        {"type": "response.created", "response": opened},
        {"type": "response.in_progress", "response": opened},
        {"type": "response.output_item.added", "output_index": 0, "item": {**reasoning, "encrypted_content": None}},
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {**message, "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "output_index": 1,
            "content_index": 0,
            "item_id": "msg_$UNIQUE_ID",
            "part": part,
        },
        {
            "type": "response.output_text.delta",
            "output_index": 1,
            "content_index": 0,
            "item_id": "msg_$UNIQUE_ID",
            "delta": "399",
        },
        {
            "type": "response.output_text.done",
            "output_index": 1,
            "content_index": 0,
            "item_id": "msg_$UNIQUE_ID",
            "text": "399",
        },
        {
            "type": "response.content_part.done",
            "output_index": 1,
            "content_index": 0,
            "item_id": "msg_$UNIQUE_ID",
            "part": {**part, "text": "399"},
        },
        {"type": "response.output_item.done", "output_index": 1, "item": message},
        {"type": "response.completed", "response": body},
    )
    return SseResponse(
        content_type="text/event-stream",
        frames=tuple(_responses_frame({**event, "sequence_number": number}) for number, event in enumerate(events)),
    )


def _chat_chunk(delta: Mapping[str, JsonValue], finish_reason: str | None) -> str:
    chunk: Final = {
        "id": "chatcmpl-$UNIQUE_ID",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5-scripted",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}"


def healthy_chat_stream() -> StoredResponse:
    return SseResponse(
        content_type="text/event-stream",
        frames=(_chat_chunk({"role": "assistant", "content": "399"}, None), _chat_chunk({}, "stop"), "data: [DONE]"),
    )


@dataclass(frozen=True, slots=True)
class Target:
    name: str
    deployments: frozenset[str]


@dataclass(frozen=True, slots=True)
class OrderedGroup:
    name: str
    order_one: str
    order_two: str
    order_one_scenario: str
    order_two_scenario: str

    @property
    def target(self) -> Target:
        return Target(self.name, frozenset({self.order_one, self.order_two}))


@dataclass(frozen=True, slots=True)
class Lane(Gateway):
    pass


LANE_LIMITS: Final = httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=30)
CONVERGENCE_SECONDS: Final = 60


def deployment_ids(entries: JsonValue) -> frozenset[str]:
    assert isinstance(entries, list), entries
    return frozenset(string_value(object_value(object_value(entry)["model_info"])["id"]) for entry in entries)


def knows(lane: Gateway, deployments: frozenset[str]) -> bool:
    return deployments <= deployment_ids(lane.get("/model/info")["data"])


@contextmanager
def pinned_client(gateway: Gateway) -> Iterator[Lane]:
    with httpx.Client(base_url=base_url(gateway), timeout=15, trust_env=False, limits=LANE_LIMITS) as client:
        yield Lane(client, gateway.key, gateway.upstream_url)


@contextmanager
def lane_for(gateway: Gateway, target: Target) -> Iterator[Gateway]:
    if isinstance(gateway, Lane):
        yield gateway
        return
    with pinned_client(gateway) as lane:
        eventually(lambda: knows(lane, target.deployments), lambda converged: converged, seconds=CONVERGENCE_SECONDS)
        yield lane


@contextmanager
def lanes(gateway: Gateway, targets: Sequence[Target], count: int) -> Iterator[tuple[Lane, ...]]:
    wanted: Final = frozenset[str]().union(*(target.deployments for target in targets))
    with ExitStack() as stack:
        opened: Final = tuple(stack.enter_context(pinned_client(gateway)) for _ in range(count))
        with ThreadPoolExecutor(max_workers=count) as pool:
            _ = tuple(pool.map(lambda lane: knows(lane, wanted), opened))
        _ = eventually(lambda: tuple(knows(lane, wanted) for lane in opened), all, seconds=CONVERGENCE_SECONDS)
        yield opened


async def async_knows(transport: httpx.AsyncClient, key: str, deployments: frozenset[str]) -> bool:
    response: Final = await transport.get("/model/info", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200, response.text
    return deployments <= deployment_ids(JSON_OBJECT.validate_json(response.content)["data"])


@asynccontextmanager
async def async_lane(gateway: Gateway, target: Target) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        base_url=base_url(gateway), timeout=15, trust_env=False, limits=LANE_LIMITS
    ) as transport:
        deadline: Final = time.monotonic() + CONVERGENCE_SECONDS
        while not await async_knows(transport, gateway.key, target.deployments):
            assert time.monotonic() < deadline, f"{target} never reached this worker"
            await asyncio.sleep(0.1)
        yield transport


def scenario_api_base(scenario: Scenario, scenario_id: str, response: StoredResponse) -> str:
    handle: Final = register_scenario(scenario_id, response)
    scenario.cleanups.callback(delete_scenario, handle)
    return handle.api_base()


def deployment(
    scenario: Scenario,
    group: str,
    api_base: str,
    *,
    order: int | None,
    model: str = RESPONSES_MODEL,
    extra_params: Mapping[str, JsonValue] = MappingProxyType({}),
) -> str:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": group,
            "litellm_params": {
                "model": model,
                "api_key": PROVIDER_KEY,
                "api_base": api_base,
                **({} if order is None else {"order": order}),
                **extra_params,
            },
            "model_info": {},
        },
    )
    model_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, model_id)
    return model_id


def ordered_group(
    scenario: Scenario,
    *,
    order_one: StoredResponse,
    order_two: StoredResponse,
    model: str = RESPONSES_MODEL,
) -> OrderedGroup:
    name: Final = f"hop-{uuid.uuid4().hex[:12]}"
    one: Final = f"{name}-o1"
    two: Final = f"{name}-o2"
    one_base: Final = scenario_api_base(scenario, one, order_one)
    two_base: Final = scenario_api_base(scenario, two, order_two)
    return OrderedGroup(
        name,
        deployment(scenario, name, one_base, order=1, model=model),
        deployment(scenario, name, two_base, order=2, model=model),
        one,
        two,
    )


def observed(gateway: Gateway) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    with httpx.Client(base_url=gateway.upstream_url, trust_env=False, timeout=15) as upstream:
        payload: Final = object_value(upstream.get("/__observations").json())
    requests: Final = payload.get("requests")
    assert isinstance(requests, list), payload
    return tuple(
        (string_value(object_value(request)["path"]), object_value(object_value(request)["body"]))
        for request in requests
    )


def bodies_for(
    records: Sequence[tuple[str, dict[str, JsonValue]]], scenario_id: str
) -> tuple[dict[str, JsonValue], ...]:
    return tuple(body for path, body in records if path.startswith(f"/{scenario_id}/"))


@dataclass(frozen=True, slots=True)
class HopRecord:
    order_one: tuple[dict[str, JsonValue], ...]
    order_two: tuple[dict[str, JsonValue], ...]


def hop_record(gateway: Gateway, group: OrderedGroup) -> HopRecord:
    records: Final = observed(gateway)
    return HopRecord(bodies_for(records, group.order_one_scenario), bodies_for(records, group.order_two_scenario))


def user_item(text: str) -> dict[str, JsonValue]:
    return {"type": "message", "role": "user", "content": text}


def reasoning_item(encrypted_content: JsonValue, *, summary: bool = True) -> dict[str, JsonValue]:
    return {
        "type": "reasoning",
        "id": "rs_order1",
        "encrypted_content": encrypted_content,
        **({"summary": [{"type": "summary_text", "text": SUMMARY_TEXT}]} if summary else {}),
    }


ASSISTANT_ITEM: Final[dict[str, JsonValue]] = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "391"}],
}
STRIPPED_REASONING: Final[dict[str, JsonValue]] = {
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": SUMMARY_TEXT}],
}


def history(marker: str, *reasoning: dict[str, JsonValue]) -> list[JsonValue]:
    replayed: Final = reasoning or (reasoning_item(ORDER_ONE_BLOB),)
    return [user_item(f"What is 17*23? {marker}"), *replayed, ASSISTANT_ITEM, user_item("And 19*21?")]


def stripped_history(marker: str, *reasoning: dict[str, JsonValue]) -> list[JsonValue]:
    replayed: Final = reasoning or (STRIPPED_REASONING,)
    return [user_item(f"What is 17*23? {marker}"), *replayed, ASSISTANT_ITEM, user_item("And 19*21?")]


def chat_messages(marker: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": f"What is 17*23? {marker}"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": SUMMARY_TEXT, "signature": TAGGED_SIGNATURE},
                {"type": "text", "text": "391"},
            ],
        },
        {"role": "user", "content": "And 19*21?"},
    ]


def stripped_chat_messages(marker: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": f"What is 17*23? {marker}"},
        {"role": "assistant", "content": [{"type": "text", "text": "391"}]},
        {"role": "user", "content": "And 19*21?"},
    ]


def marker() -> str:
    return uuid.uuid4().hex


def base_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def data_frames(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        object_value(json.loads(line.removeprefix("data: ")))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


@dataclass(frozen=True, slots=True)
class Answer:
    status: int
    response_id: str | None
    text: str
    headers: Mapping[str, str]


def _httpx_answer(response: httpx.Response, response_id: Callable[[httpx.Response], str | None]) -> Answer:
    return Answer(
        response.status_code,
        response_id(response) if response.status_code == 200 else None,
        response.text,
        MappingProxyType(dict(response.headers)),
    )


def _json_id(response: httpx.Response) -> str | None:
    identity: Final = object_value(response.json()).get("id")
    return identity if isinstance(identity, str) else None


def _completed_stream_id(response: httpx.Response) -> str | None:
    frames: Final = data_frames(response.text)
    assert frames and frames[-1].get("type") == "response.completed", response.text
    identity: Final = object_value(frames[-1]["response"]).get("id")
    return identity if isinstance(identity, str) else None


def responses_answer(
    gateway: Gateway,
    client: str,
    target: Target,
    request_input: JsonValue,
    *,
    stream: bool = False,
    extra: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Answer:
    body: Final = {"model": target.name, "input": request_input, "store": False, **extra}
    if client == HTTPX:
        with lane_for(gateway, target) as lane:
            response: Final = lane.request("POST", "/v1/responses", {**body, "stream": stream})
        return _httpx_answer(response, _completed_stream_id if stream else _json_id)
    if client == OPENAI_SYNC:
        with lane_for(gateway, target) as lane:
            sdk: Final = openai.OpenAI(
                base_url=f"{base_url(gateway)}/v1", api_key=gateway.key, max_retries=0, http_client=lane.client
            )
            try:
                if stream:
                    streamed: Final = sdk.responses.with_raw_response.create(**body, stream=True)
                    events: Final = list(streamed.parse())
                    assert events and events[-1].type == "response.completed", events
                    return Answer(
                        streamed.status_code,
                        events[-1].response.id,
                        json.dumps([event.type for event in events]),
                        MappingProxyType(dict(streamed.headers)),
                    )
                raw: Final = sdk.responses.with_raw_response.create(**body)
                return Answer(raw.status_code, raw.parse().id, raw.text, MappingProxyType(dict(raw.headers)))
            except openai.APIStatusError as error:
                return Answer(
                    error.status_code, None, error.response.text, MappingProxyType(dict(error.response.headers))
                )
    assert client == OPENAI_ASYNC, client

    async def call() -> Answer:
        async with async_lane(gateway, target) as transport:
            sdk: Final = openai.AsyncOpenAI(
                base_url=f"{base_url(gateway)}/v1", api_key=gateway.key, max_retries=0, http_client=transport
            )
            try:
                if stream:
                    streamed: Final = await sdk.responses.with_raw_response.create(**body, stream=True)
                    events: Final = [event async for event in streamed.parse()]
                    assert events and events[-1].type == "response.completed", events
                    return Answer(
                        streamed.status_code,
                        events[-1].response.id,
                        json.dumps([event.type for event in events]),
                        MappingProxyType(dict(streamed.headers)),
                    )
                raw: Final = await sdk.responses.with_raw_response.create(**body)
                return Answer(raw.status_code, raw.parse().id, raw.text, MappingProxyType(dict(raw.headers)))
            except openai.APIStatusError as error:
                return Answer(
                    error.status_code, None, error.response.text, MappingProxyType(dict(error.response.headers))
                )

    return asyncio.run(call())


def chat_answer(
    gateway: Gateway, client: str, target: Target, messages: list[JsonValue], *, stream: bool = False
) -> Answer:
    body: Final = {"model": target.name, "messages": messages}
    if client == HTTPX:
        with lane_for(gateway, target) as lane:
            response: Final = lane.request("POST", "/v1/chat/completions", {**body, "stream": stream})
        if stream:
            return _httpx_answer(response, lambda served: string_value(data_frames(served.text)[-1]["id"]))
        return _httpx_answer(response, _json_id)
    if client == OPENAI_SYNC:
        with lane_for(gateway, target) as lane:
            sdk: Final = openai.OpenAI(
                base_url=f"{base_url(gateway)}/v1", api_key=gateway.key, max_retries=0, http_client=lane.client
            )
            try:
                raw: Final = sdk.chat.completions.with_raw_response.create(**body)
                return Answer(raw.status_code, raw.parse().id, raw.text, MappingProxyType(dict(raw.headers)))
            except openai.APIStatusError as error:
                return Answer(
                    error.status_code, None, error.response.text, MappingProxyType(dict(error.response.headers))
                )
    assert client == OPENAI_ASYNC, client

    async def call() -> Answer:
        async with async_lane(gateway, target) as transport:
            sdk: Final = openai.AsyncOpenAI(
                base_url=f"{base_url(gateway)}/v1", api_key=gateway.key, max_retries=0, http_client=transport
            )
            try:
                raw: Final = await sdk.chat.completions.with_raw_response.create(**body)
                return Answer(raw.status_code, raw.parse().id, raw.text, MappingProxyType(dict(raw.headers)))
            except openai.APIStatusError as error:
                return Answer(
                    error.status_code, None, error.response.text, MappingProxyType(dict(error.response.headers))
                )

    return asyncio.run(call())


def messages_answer(
    gateway: Gateway, client: str, target: Target, messages: list[JsonValue], *, stream: bool = False
) -> Answer:
    body: Final = {"model": target.name, "max_tokens": 64, "messages": messages}
    if client == HTTPX:
        with lane_for(gateway, target) as lane:
            response: Final = lane.request("POST", "/v1/messages", {**body, "stream": stream})
        if stream:
            return _httpx_answer(
                response, lambda served: string_value(object_value(data_frames(served.text)[0]["message"])["id"])
            )
        return _httpx_answer(response, _json_id)
    if client == ANTHROPIC_SYNC:
        with lane_for(gateway, target) as lane:
            sdk: Final = anthropic.Anthropic(
                base_url=base_url(gateway), api_key=gateway.key, max_retries=0, http_client=lane.client
            )
            try:
                raw: Final = sdk.messages.with_raw_response.create(**body)
                return Answer(raw.status_code, raw.parse().id, raw.text, MappingProxyType(dict(raw.headers)))
            except anthropic.APIStatusError as error:
                return Answer(
                    error.status_code, None, error.response.text, MappingProxyType(dict(error.response.headers))
                )
    assert client == ANTHROPIC_ASYNC, client

    async def call() -> Answer:
        async with async_lane(gateway, target) as transport:
            sdk: Final = anthropic.AsyncAnthropic(
                base_url=base_url(gateway), api_key=gateway.key, max_retries=0, http_client=transport
            )
            try:
                raw: Final = await sdk.messages.with_raw_response.create(**body)
                return Answer(raw.status_code, raw.parse().id, raw.text, MappingProxyType(dict(raw.headers)))
            except anthropic.APIStatusError as error:
                return Answer(
                    error.status_code, None, error.response.text, MappingProxyType(dict(error.response.headers))
                )

    return asyncio.run(call())


def assert_served_by(answer: Answer, model_id: str) -> None:
    assert answer.status == 200, answer.text
    assert answer.response_id is not None, answer.text
    assert answer.headers.get("x-litellm-model-id") == model_id, (model_id, answer.headers, answer.text)


def assert_served_by_order_two(answer: Answer, group: OrderedGroup) -> None:
    assert_served_by(answer, group.order_two)


def assert_hop_stripped(
    record: HopRecord, expected_order_one: list[JsonValue], expected_order_two: list[JsonValue]
) -> None:
    assert len(record.order_one) == 1, record
    assert record.order_one[0].get("input") == expected_order_one, record.order_one[0]
    assert len(record.order_two) == 1, record
    assert record.order_two[0].get("input") == expected_order_two, record.order_two[0]


def success_rows(call_ids: Sequence[str]) -> list[dict[str, JsonValue]]:
    placeholders: Final = ", ".join("%s" for _ in call_ids)
    rows: Final = eventually(
        lambda: read_rows(
            f'SELECT litellm_call_id, status, model_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id IN ({placeholders})',
            tuple(call_ids),
        ),
        lambda found: len({row["litellm_call_id"] for row in found if row["status"] == "success"}) == len(call_ids),
        seconds=90,
    )
    return [row for row in rows if row["status"] == "success"]


def spend_rows(call_id: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT litellm_call_id, status, model_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s',
            (call_id,),
        ),
        lambda rows: any(row["status"] == "success" for row in rows),
        seconds=70,
    )


def assert_logged_once_on_order_two(call_id: str, group: OrderedGroup) -> None:
    successes: Final = [row for row in spend_rows(call_id) if row["status"] == "success"]
    assert [row["model_id"] for row in successes] == [group.order_two], successes


def set_pre_call_checks(gateway: Gateway, checks: Sequence[str]) -> None:
    gateway.post("/config/update", {"router_settings": {"optional_pre_call_checks": list(checks)}})
    assert router_setting(gateway, "optional_pre_call_checks") == list(checks), router_setting(
        gateway, "optional_pre_call_checks"
    )


@pytest.fixture(scope="module", autouse=True)
def affinity_check_off() -> Iterator[None]:
    with gateway_from_environment() as gateway:
        original: Final = router_setting(gateway, "optional_pre_call_checks")
        set_pre_call_checks(gateway, ())
        try:
            yield
        finally:
            set_pre_call_checks(gateway, [str(check) for check in original] if isinstance(original, list) else ())


def enable_affinity_check(scenario: Scenario) -> None:
    scenario.cleanups.callback(set_pre_call_checks, scenario.gateway, ())
    set_pre_call_checks(scenario.gateway, (AFFINITY_CHECK,))


def router_setting(gateway: Gateway, name: str) -> JsonValue:
    return object_value(gateway.get("/router/settings")["current_values"]).get(name)


@pytest.mark.parametrize("client", RESPONSES_CLIENTS)
def test_responses_hop_drops_the_order_one_reasoning_order_two_cannot_decrypt(gateway: Gateway, client: str) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json())
        mark: Final = marker()
        answer: Final = responses_answer(gateway, client, group.target, history(mark))
        assert_served_by_order_two(answer, group)
        assert_hop_stripped(hop_record(gateway, group), history(mark), stripped_history(mark))
        assert_logged_once_on_order_two(answer.headers["x-litellm-call-id"], group)


@pytest.mark.parametrize("client", RESPONSES_CLIENTS)
def test_responses_mid_stream_hop_drops_the_order_one_reasoning(gateway: Gateway, client: str) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(
            scenario, order_one=broken_responses_stream(), order_two=healthy_responses_stream()
        )
        mark: Final = marker()
        answer: Final = responses_answer(gateway, client, group.target, history(mark), stream=True)
        assert_served_by_order_two(answer, group)
        assert_hop_stripped(hop_record(gateway, group), history(mark), stripped_history(mark))


def test_responses_stream_refused_before_any_frame_hops_stripped(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_responses_stream())
        mark: Final = marker()
        answer: Final = responses_answer(gateway, HTTPX, group.target, history(mark), stream=True)
        assert_served_by_order_two(answer, group)
        assert_hop_stripped(hop_record(gateway, group), history(mark), stripped_history(mark))


@pytest.mark.parametrize("client", RESPONSES_CLIENTS)
def test_chat_hop_drops_the_bridge_tagged_thinking_block(gateway: Gateway, client: str) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json())
        mark: Final = marker()
        answer: Final = chat_answer(gateway, client, group.target, chat_messages(mark))
        assert answer.status == 200, answer.text
        assert answer.response_id is not None and answer.response_id.startswith(
            f"chatcmpl-{group.order_two_scenario}-"
        ), answer.text
        record: Final = hop_record(gateway, group)
        assert len(record.order_one) == 1 and record.order_one[0].get("messages") == chat_messages(mark), record
        assert len(record.order_two) == 1 and record.order_two[0].get("messages") == stripped_chat_messages(mark), (
            record
        )


def test_chat_stream_hop_drops_the_bridge_tagged_thinking_block(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_chat_stream())
        mark: Final = marker()
        answer: Final = chat_answer(gateway, HTTPX, group.target, chat_messages(mark), stream=True)
        assert answer.status == 200, answer.text
        assert answer.response_id is not None and answer.response_id.startswith(
            f"chatcmpl-{group.order_two_scenario}-"
        ), answer.text
        record: Final = hop_record(gateway, group)
        assert len(record.order_one) == 1 and record.order_one[0].get("messages") == chat_messages(mark), record
        assert len(record.order_two) == 1 and record.order_two[0].get("messages") == stripped_chat_messages(mark), (
            record
        )


def assert_bridged_hop_stripped(record: HopRecord, mark: str) -> None:
    assert len(record.order_one) == 1, record
    assert ORDER_ONE_BLOB in json.dumps(record.order_one[0]), record.order_one[0]
    assert len(record.order_two) == 1, record
    order_two: Final = record.order_two[0]
    assert ORDER_ONE_BLOB not in json.dumps(order_two), order_two
    items: Final = order_two.get("input")
    assert isinstance(items, list) and mark in json.dumps(items), order_two
    assert not any(isinstance(item, dict) and "encrypted_content" in item for item in items), order_two


@pytest.mark.parametrize("client", MESSAGES_CLIENTS)
def test_messages_hop_drops_the_bridge_tagged_thinking_block(gateway: Gateway, client: str) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json(), model=BRIDGED_MODEL)
        mark: Final = marker()
        answer: Final = messages_answer(gateway, client, group.target, chat_messages(mark))
        assert answer.status == 200, answer.text
        assert "399" in answer.text, answer.text
        assert_bridged_hop_stripped(hop_record(gateway, group), mark)


def test_messages_mid_stream_hop_drops_the_bridge_tagged_thinking_block(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(
            scenario, order_one=broken_responses_stream(), order_two=healthy_responses_stream(), model=BRIDGED_MODEL
        )
        mark: Final = marker()
        answer: Final = messages_answer(gateway, HTTPX, group.target, chat_messages(mark), stream=True)
        assert answer.status == 200, answer.text
        assert "399" in answer.text, answer.text
        assert_bridged_hop_stripped(hop_record(gateway, group), mark)


def affinity_turn_one(gateway: Gateway, group: OrderedGroup) -> list[JsonValue]:
    answer: Final = responses_answer(
        gateway, HTTPX, group.target, f"hello affinity {marker()}", extra={"include": ["reasoning.encrypted_content"]}
    )
    assert answer.status == 200, answer.text
    assert answer.headers.get("x-litellm-model-id") == group.order_one, answer.headers
    output: Final = object_value(json.loads(answer.text)).get("output")
    assert isinstance(output, list), answer.text
    reasoning: Final = next(
        (object_value(item) for item in output if object_value(item).get("type") == "reasoning"), None
    )
    assert reasoning is not None and str(reasoning["id"]).startswith("encitem_"), answer.text
    message: Final = next((object_value(item) for item in output if object_value(item).get("type") == "message"), None)
    assert message is not None, answer.text
    return [reasoning, message]


def test_affinity_pin_yields_to_the_hop_and_strips_the_origins_reasoning(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        enable_affinity_check(scenario)
        group: Final = ordered_group(scenario, order_one=healthy_json(ORDER_ONE_BLOB), order_two=healthy_json())
        items: Final = affinity_turn_one(gateway, group)
        register_scenario(group.order_one_scenario, failing())
        mark: Final = marker()
        answer: Final = responses_answer(
            gateway, HTTPX, group.target, [user_item(f"hello affinity {mark}"), *items, user_item("continue")]
        )
        assert_served_by_order_two(answer, group)
        record: Final = hop_record(gateway, group)
        assert len(record.order_one) == 2 and ORDER_ONE_BLOB in json.dumps(record.order_one[1]), record.order_one
        assert len(record.order_two) == 1, record
        assert ORDER_ONE_BLOB not in json.dumps(record.order_two[0]), record.order_two[0]
        assert record.order_two[0].get("input") == [
            user_item(f"hello affinity {mark}"),
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "nineteen times twenty-one"}]},
            items[1],
            user_item("continue"),
        ], record.order_two[0]
    assert router_setting(gateway, "optional_pre_call_checks") == [], router_setting(
        gateway, "optional_pre_call_checks"
    )


def test_hop_inside_one_encryption_boundary_keeps_the_reasoning(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        shared: Final = f"shared-{uuid.uuid4().hex[:12]}"
        one: Final = scenario_api_base(scenario, f"{shared}-o1", failing()).rsplit("/", 1)[-1]
        two: Final = scenario_api_base(scenario, f"{shared}-o2", healthy_json()).rsplit("/", 1)[-1]
        shared_base: Final = f"{gateway.upstream_url}/{shared}"
        order_one: Final = deployment(
            scenario, shared, shared_base, order=1, extra_params={"extra_headers": {"x-scripted-scenario": one}}
        )
        order_two: Final = deployment(
            scenario, shared, shared_base, order=2, extra_params={"extra_headers": {"x-scripted-scenario": two}}
        )
        mark: Final = marker()
        answer: Final = responses_answer(
            gateway, HTTPX, Target(shared, frozenset({order_one, order_two})), history(mark)
        )
        assert answer.status == 200, answer.text
        assert answer.headers.get("x-litellm-model-id") == order_two, (order_one, answer.headers)
        bodies: Final = bodies_for(observed(gateway), shared)
        assert [body.get("input") for body in bodies] == [history(mark), history(mark)], bodies


def configured_fallback(scenario: Scenario, source: str, target: str) -> None:
    gateway: Final = scenario.gateway
    original: Final = router_setting(gateway, "fallbacks")
    restored: Final = original if isinstance(original, list) else []
    scenario.cleanups.callback(lambda: gateway.post("/config/update", {"router_settings": {"fallbacks": restored}}))
    gateway.post("/config/update", {"router_settings": {"fallbacks": [*restored, {source: [target]}]}})


def assert_cross_group_hop_stripped(gateway: Gateway, source_scenario: str, target_scenario: str, mark: str) -> None:
    records: Final = observed(gateway)
    assert [body.get("input") for body in bodies_for(records, source_scenario)] == [history(mark)], records
    assert [body.get("input") for body in bodies_for(records, target_scenario)] == [stripped_history(mark)], records


def test_configured_fallbacks_entry_hop_strips_the_source_groups_reasoning(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        source: Final = f"src-{uuid.uuid4().hex[:12]}"
        target: Final = f"dst-{uuid.uuid4().hex[:12]}"
        source_id: Final = deployment(scenario, source, scenario_api_base(scenario, source, failing()), order=None)
        target_id: Final = deployment(scenario, target, scenario_api_base(scenario, target, healthy_json()), order=None)
        configured_fallback(scenario, source, target)
        mark: Final = marker()
        answer: Final = responses_answer(
            gateway, HTTPX, Target(source, frozenset({source_id, target_id})), history(mark)
        )
        assert_served_by(answer, target_id)
        assert_cross_group_hop_stripped(gateway, source, target, mark)
    assert source not in json.dumps(router_setting(gateway, "fallbacks")), router_setting(gateway, "fallbacks")


def test_client_fallbacks_entry_hop_strips_the_source_groups_reasoning(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        source: Final = f"src-{uuid.uuid4().hex[:12]}"
        target: Final = f"dst-{uuid.uuid4().hex[:12]}"
        source_id: Final = deployment(scenario, source, scenario_api_base(scenario, source, failing()), order=None)
        target_id: Final = deployment(scenario, target, scenario_api_base(scenario, target, healthy_json()), order=None)
        mark: Final = marker()
        answer: Final = responses_answer(
            gateway,
            HTTPX,
            Target(source, frozenset({source_id, target_id})),
            history(mark),
            extra={"fallbacks": [target]},
        )
        assert_served_by(answer, target_id)
        assert_cross_group_hop_stripped(gateway, source, target, mark)


def assert_served_by_order_one_intact(gateway: Gateway, group: OrderedGroup, answer: Answer, mark: str) -> None:
    assert answer.status == 200, answer.text
    assert answer.headers.get("x-litellm-model-id") == group.order_one, answer.headers
    record: Final = hop_record(gateway, group)
    assert [body.get("input") for body in record.order_one] == [history(mark)], record
    assert record.order_two == (), record


@pytest.mark.parametrize(
    "target_order",
    [
        pytest.param(2, id="int"),
        pytest.param("2", id="str"),
        pytest.param([2], id="list"),
        pytest.param(True, id="bool"),
        pytest.param(99, id="unknown-order"),
        pytest.param(None, id="null"),
    ],
)
def test_client_sent_target_order_never_moves_a_request_off_order_one(
    gateway: Gateway, target_order: JsonValue
) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=healthy_json(ORDER_ONE_BLOB), order_two=healthy_json())
        mark: Final = marker()
        answer: Final = responses_answer(
            gateway, HTTPX, group.target, history(mark), extra={"_target_order": target_order}
        )
        assert_served_by_order_one_intact(gateway, group, answer, mark)


@pytest.mark.parametrize(
    "fallback_depth",
    [
        pytest.param(1, id="int"),
        pytest.param(True, id="bool"),
        pytest.param("1", id="str"),
        pytest.param([1], id="list"),
    ],
)
def test_client_sent_fallback_depth_never_strips_a_plain_request(gateway: Gateway, fallback_depth: JsonValue) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=healthy_json(ORDER_ONE_BLOB), order_two=healthy_json())
        mark: Final = marker()
        answer: Final = responses_answer(
            gateway, HTTPX, group.target, history(mark), extra={"fallback_depth": fallback_depth}
        )
        assert_served_by_order_one_intact(gateway, group, answer, mark)


def test_client_sent_fallback_depth_cannot_exhaust_the_fallback_budget(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json())
        mark: Final = marker()
        answer: Final = responses_answer(gateway, HTTPX, group.target, history(mark), extra={"fallback_depth": 5})
        assert_served_by_order_two(answer, group)
        assert_hop_stripped(hop_record(gateway, group), history(mark), stripped_history(mark))


def test_client_sent_attempted_targets_do_not_reach_the_hop(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json())
        mark: Final = marker()
        answer: Final = responses_answer(
            gateway, HTTPX, group.target, history(mark), extra={"attempted_targets": ["forged"]}
        )
        assert_served_by_order_two(answer, group)
        assert_hop_stripped(hop_record(gateway, group), history(mark), stripped_history(mark))


def test_client_sent_max_fallbacks_zero_still_stops_the_hop(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json())
        mark: Final = marker()
        answer: Final = responses_answer(gateway, HTTPX, group.target, history(mark), extra={"max_fallbacks": 0})
        assert answer.status == 500, answer.text
        assert "order one is down" in answer.text, answer.text
        record: Final = hop_record(gateway, group)
        assert [body.get("input") for body in record.order_one] == [history(mark)], record
        assert record.order_two == (), record


FIVE_KB: Final = "x" * 5120


@pytest.mark.parametrize(
    ("replayed", "expected"),
    [
        pytest.param((reasoning_item(7),), (STRIPPED_REASONING,), id="int"),
        pytest.param((reasoning_item(["a", "b"]),), (STRIPPED_REASONING,), id="list"),
        pytest.param((reasoning_item(FIVE_KB),), (STRIPPED_REASONING,), id="5kb"),
        pytest.param(
            (reasoning_item(ORDER_ONE_BLOB), reasoning_item(ORDER_ONE_BLOB)),
            (STRIPPED_REASONING, STRIPPED_REASONING),
            id="duplicated",
        ),
        pytest.param((reasoning_item(""),), (reasoning_item(""),), id="empty"),
    ],
)
def test_hop_strips_malformed_encrypted_content_without_failing(
    gateway: Gateway, replayed: tuple[dict[str, JsonValue], ...], expected: tuple[dict[str, JsonValue], ...]
) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json())
        mark: Final = marker()
        answer: Final = responses_answer(gateway, HTTPX, group.target, history(mark, *replayed))
        assert_served_by_order_two(answer, group)
        assert_hop_stripped(hop_record(gateway, group), history(mark, *replayed), stripped_history(mark, *expected))


def test_hop_drops_a_reasoning_item_with_nothing_readable_whole(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json())
        mark: Final = marker()
        unreadable: Final = reasoning_item(ORDER_ONE_BLOB, summary=False)
        answer: Final = responses_answer(gateway, HTTPX, group.target, history(mark, unreadable))
        assert_served_by_order_two(answer, group)
        assert_hop_stripped(
            hop_record(gateway, group),
            history(mark, unreadable),
            [user_item(f"What is 17*23? {mark}"), ASSISTANT_ITEM, user_item("And 19*21?")],
        )


def test_every_order_failing_reports_the_failure_to_the_caller(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(
            scenario, order_one=failing("order one is down"), order_two=failing("order two is down")
        )
        mark: Final = marker()
        answer: Final = responses_answer(gateway, HTTPX, group.target, history(mark))
        assert answer.status == 500, answer.text
        assert "is down" in answer.text, answer.text
        record: Final = hop_record(gateway, group)
        assert [body.get("input") for body in record.order_one] == [history(mark)], record
        assert len(record.order_two) == 1, record


def test_hop_strips_with_the_affinity_check_list_explicitly_empty(gateway: Gateway) -> None:
    assert router_setting(gateway, "optional_pre_call_checks") == [], router_setting(
        gateway, "optional_pre_call_checks"
    )
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json())
        mark: Final = marker()
        answer: Final = responses_answer(gateway, HTTPX, group.target, history(mark))
        assert_served_by_order_two(answer, group)
        assert_hop_stripped(hop_record(gateway, group), history(mark), stripped_history(mark))


def test_next_turn_without_a_hop_forwards_the_reasoning_untouched(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = ordered_group(scenario, order_one=healthy_json(ORDER_ONE_BLOB), order_two=healthy_json())
        mark: Final = marker()
        replayed: Final = reasoning_item(ORDER_TWO_BLOB)
        answer: Final = responses_answer(gateway, HTTPX, group.target, history(mark, replayed))
        assert answer.status == 200, answer.text
        assert answer.headers.get("x-litellm-model-id") == group.order_one, answer.headers
        record: Final = hop_record(gateway, group)
        assert [body.get("input") for body in record.order_one] == [history(mark, replayed)], record
        assert record.order_two == (), record


def test_affinity_pin_keeps_the_targets_own_marked_reasoning_and_the_unmarked_history(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        enable_affinity_check(scenario)
        group: Final = ordered_group(scenario, order_one=failing(), order_two=healthy_json())
        first: Final = responses_answer(
            gateway,
            HTTPX,
            group.target,
            f"hello affinity {marker()}",
            extra={"include": ["reasoning.encrypted_content"]},
        )
        assert_served_by_order_two(first, group)
        output: Final = object_value(json.loads(first.text)).get("output")
        assert isinstance(output, list), first.text
        marked: Final = next(
            (object_value(item) for item in output if object_value(item).get("type") == "reasoning"), None
        )
        assert marked is not None and str(marked["id"]).startswith("encitem_"), first.text
        mark: Final = marker()
        unmarked: Final = reasoning_item(ORDER_ONE_BLOB)
        answer: Final = responses_answer(
            gateway,
            HTTPX,
            group.target,
            [user_item(f"hello affinity {mark}"), marked, unmarked, ASSISTANT_ITEM, user_item("continue")],
        )
        assert_served_by_order_two(answer, group)
        record: Final = hop_record(gateway, group)
        assert len(record.order_one) == 1 and mark not in json.dumps(record.order_one), record
        assert len(record.order_two) == 2, record
        pinned: Final = record.order_two[-1].get("input")
        assert isinstance(pinned, list) and mark in json.dumps(pinned[0]), record
        own: Final = object_value(pinned[1])
        assert own.get("type") == "reasoning" and own.get("summary") == marked["summary"], pinned
        assert str(own.get("id")).startswith(f"rs_{group.order_two_scenario}-"), pinned
        assert own.get("encrypted_content") == ORDER_TWO_BLOB, pinned
        assert pinned[2] == unmarked, pinned
        assert pinned[3:] == [ASSISTANT_ITEM, user_item("continue")], pinned


RESPONSES_JSON: Final = "responses"
RESPONSES_STREAM: Final = "responses-stream"
CHAT_JSON: Final = "chat"
CHAT_STREAM: Final = "chat-stream"
MESSAGES_JSON: Final = "messages"
MESSAGES_STREAM: Final = "messages-stream"
BURST_KINDS: Final = (RESPONSES_JSON, CHAT_JSON, MESSAGES_JSON, RESPONSES_STREAM, CHAT_STREAM, MESSAGES_STREAM)


def burst_groups(scenario: Scenario) -> Mapping[str, OrderedGroup]:
    return MappingProxyType(
        {
            RESPONSES_JSON: ordered_group(scenario, order_one=failing(), order_two=healthy_json()),
            RESPONSES_STREAM: ordered_group(scenario, order_one=failing(), order_two=healthy_responses_stream()),
            CHAT_JSON: ordered_group(scenario, order_one=failing(), order_two=healthy_json()),
            CHAT_STREAM: ordered_group(scenario, order_one=failing(), order_two=healthy_chat_stream()),
            MESSAGES_JSON: ordered_group(scenario, order_one=failing(), order_two=healthy_json(), model=BRIDGED_MODEL),
            MESSAGES_STREAM: ordered_group(
                scenario, order_one=failing(), order_two=healthy_responses_stream(), model=BRIDGED_MODEL
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class Fired:
    kind: str
    mark: str
    status: int | None
    call_id: str | None
    text: str


def fire(gateway: Gateway, groups: Mapping[str, OrderedGroup], kind: str, mark: str) -> Fired:
    group: Final = groups[kind].target
    stream: Final = kind.endswith("-stream")
    try:
        if kind.startswith("responses"):
            answer: Final = responses_answer(gateway, HTTPX, group, history(mark), stream=stream)
        elif kind.startswith("chat"):
            answer = chat_answer(gateway, HTTPX, group, chat_messages(mark), stream=stream)
        else:
            answer = messages_answer(gateway, HTTPX, group, chat_messages(mark), stream=stream)
    except (httpx.HTTPError, AssertionError) as error:
        return Fired(kind, mark, None, None, repr(error))
    return Fired(kind, mark, answer.status, answer.headers.get("x-litellm-call-id"), answer.text)


def assert_burst_hopped_stripped(gateway: Gateway, groups: Mapping[str, OrderedGroup], fired: Sequence[Fired]) -> None:
    failures: Final = [shot for shot in fired if shot.status != 200]
    assert not failures, failures
    records: Final = observed(gateway)
    for shot in fired:
        group: Final = groups[shot.kind]
        order_one: Final = [
            body for body in bodies_for(records, group.order_one_scenario) if shot.mark in json.dumps(body)
        ]
        order_two: Final = [
            body for body in bodies_for(records, group.order_two_scenario) if shot.mark in json.dumps(body)
        ]
        assert len(order_one) == 1 and ORDER_ONE_BLOB in json.dumps(order_one[0]), (shot, order_one)
        assert len(order_two) == 1 and ORDER_ONE_BLOB not in json.dumps(order_two[0]), (shot, order_two)
    call_ids: Final = [shot.call_id for shot in fired if shot.call_id is not None]
    assert len(call_ids) == len(fired), fired
    successes: Final = success_rows(call_ids)
    assert sorted(string_value(row["litellm_call_id"]) for row in successes) == sorted(call_ids), successes
    order_two_ids: Final = {group.order_two for group in groups.values()}
    assert all(row["model_id"] in order_two_ids for row in successes), successes


@pytest.mark.timeout(300)
def test_concurrent_mixed_burst_hops_every_request_stripped_and_logs_each_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        groups: Final = burst_groups(scenario)
        shots: Final = tuple((BURST_KINDS[index % len(BURST_KINDS)], marker()) for index in range(30))
        with lanes(gateway, [group.target for group in groups.values()], 30) as pinned:
            with ThreadPoolExecutor(max_workers=30) as pool:
                fired: Final = tuple(
                    pool.map(
                        lambda shot: fire(shot[0], groups, shot[1][0], shot[1][1]), zip(pinned, shots, strict=True)
                    )
                )
        assert_burst_hopped_stripped(gateway, groups, fired)


def peer_reply(request: Request) -> Reply:
    identity: Final = f"resp_peer-{uuid.uuid4().hex[:8]}"
    body: Final = json.dumps({**_responses_body(ORDER_ONE_BLOB), "id": identity}).replace("$UNIQUE_ID", identity)
    return Reply(body=body.encode())


def served_by_order_one(answer: Answer, order_one: str) -> bool:
    return answer.status == 200 and answer.headers.get("x-litellm-model-id") == order_one


@pytest.mark.timeout(240)
def test_order_one_outage_mid_burst_hops_stripped_and_recovers_after_restart(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = f"hop-{uuid.uuid4().hex[:12]}"
        two_scenario: Final = f"{name}-o2"
        order_two: Final = deployment(
            scenario, name, scenario_api_base(scenario, two_scenario, healthy_json()), order=2
        )
        with wire_server(peer_reply) as peer:
            order_one: Final = deployment(scenario, name, peer.url, order=1)
            target: Final = Target(name, frozenset({order_one, order_two}))
            port: Final = int(peer.url.rsplit(":", 1)[-1])
            warm: Final = marker()
            assert served_by_order_one(responses_answer(gateway, HTTPX, target, history(warm)), order_one), warm
            assert [json.loads(request.body)["input"] for request in peer.drain()] == [history(warm)]
        shots: Final = tuple(marker() for _ in range(12))
        with lanes(gateway, [target], 12) as pinned:
            with ThreadPoolExecutor(max_workers=12) as pool:
                answers: Final = tuple(
                    pool.map(
                        lambda shot: responses_answer(shot[0], HTTPX, target, history(shot[1])),
                        zip(pinned, shots, strict=True),
                    )
                )
        for answer in answers:
            assert answer.status == 200 and answer.headers.get("x-litellm-model-id") == order_two, answer.text
        bodies: Final = bodies_for(observed(gateway), two_scenario)
        assert sorted(json.dumps(body.get("input")) for body in bodies) == sorted(
            json.dumps(stripped_history(mark)) for mark in shots
        ), bodies
        call_ids: Final = [answer.headers["x-litellm-call-id"] for answer in answers]
        assert sorted(string_value(row["litellm_call_id"]) for row in success_rows(call_ids)) == sorted(call_ids)
        with wire_server(peer_reply, port=port) as restarted:
            recovered: Final = marker()
            assert served_by_order_one(responses_answer(gateway, HTTPX, target, history(recovered)), order_one), (
                recovered
            )
            assert [json.loads(request.body)["input"] for request in restarted.drain()] == [history(recovered)]


WORKER_PID: Final = re.compile(r"Started server process \[(\d+)\]")


def worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(int(match) for match in WORKER_PID.findall(log.read_text()))


def assert_completed_shots_hopped_stripped(
    gateway: Gateway, groups: Mapping[str, OrderedGroup], fired: Sequence[Fired]
) -> None:
    completed: Final = [shot for shot in fired if shot.status == 200]
    assert completed, fired
    records: Final = observed(gateway)
    for shot in completed:
        group: Final = groups[shot.kind]
        order_one: Final = [
            body for body in bodies_for(records, group.order_one_scenario) if shot.mark in json.dumps(body)
        ]
        order_two: Final = [
            body for body in bodies_for(records, group.order_two_scenario) if shot.mark in json.dumps(body)
        ]
        assert len(order_one) == 1 and ORDER_ONE_BLOB in json.dumps(order_one[0]), (shot, order_one)
        assert len(order_two) == 1 and ORDER_ONE_BLOB not in json.dumps(order_two[0]), (shot, order_two)
    for shot in fired:
        assert shot.status in (200, None) or shot.status >= 500, shot


@pytest.mark.timeout(2 * graceful_stop_seconds() + 2 * CONVERGENCE_SECONDS + 120)
def test_worker_killed_mid_burst_leaves_the_survivor_hopping_stripped(tmp_path: Path) -> None:
    with gateway_from_environment() as upstream_gateway:
        with owned_proxy_process(upstream_gateway, tmp_path, {}, workers=2) as owned:
            owned.gateway.post("/config/update", {"router_settings": {"num_retries": 0}})
            with owned.gateway.scenario() as scenario:
                groups: Final = burst_groups(scenario)
                workers: Final = eventually(lambda: worker_pids(owned.log), lambda pids: len(pids) == 2, seconds=60)
                shots: Final = tuple((BURST_KINDS[index % len(BURST_KINDS)], marker()) for index in range(30))
                with lanes(owned.gateway, [group.target for group in groups.values()], 30) as pinned:
                    with ThreadPoolExecutor(max_workers=30) as pool:
                        futures: Final = [
                            pool.submit(fire, lane, groups, kind, mark)
                            for lane, (kind, mark) in zip(pinned, shots, strict=True)
                        ]
                        psutil.Process(workers[0]).send_signal(signal.SIGKILL)
                        fired: Final = tuple(future.result() for future in futures)
                assert_completed_shots_hopped_stripped(owned.gateway, groups, fired)
                probes: Final = tuple(fire(owned.gateway, groups, kind, marker()) for kind in BURST_KINDS)
                assert all(probe.status == 200 for probe in probes), probes
                assert_burst_hopped_stripped(owned.gateway, groups, probes)
