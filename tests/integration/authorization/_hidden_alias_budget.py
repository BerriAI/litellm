import json
import os
import uuid
from collections import Counter
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Final

import httpx
from anthropic import Anthropic, AsyncAnthropic
from integration._support.anthropic_thinking import JSON_LIST, JSON_OBJECT
from integration._support.client import (
    GATEWAY_LIMITS,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
)
from integration._support.database import read_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue

BUDGET: Final = 0.05
CHAT_REPLY: Final = "Hello! This is a mock response from the fake OpenAI endpoint."
RESPONSES_REPLY: Final = "free reply"
BUDGET_EXCEEDED: Final = 422
GATEWAY_BURST: Final = 24
PEER_BURST: Final = 4
SPEND_MARKER_HEADER: Final = "x-litellm-spend-logs-metadata"
PROXY_BUDGET_USER: Final = "litellm-proxy-budget"

_RESPONSE: Final[JsonValue] = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "id": "msg_$UNIQUE_ID",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": RESPONSES_REPLY, "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 20, "output_tokens": 20, "total_tokens": 40},
}
_RESPONSE_EVENTS: Final[tuple[Mapping[str, JsonValue], ...]] = (
    {
        "type": "response.created",
        "sequence_number": 0,
        "response": {**_RESPONSE, "status": "in_progress", "output": []},
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_$UNIQUE_ID",
        "output_index": 0,
        "content_index": 0,
        "delta": RESPONSES_REPLY,
    },
    {"type": "response.completed", "sequence_number": 2, "response": _RESPONSE},
)


@dataclass(frozen=True, slots=True)
class AliasRig:
    gateway: Gateway
    peer: Gateway
    free: str
    paid: str
    failing_free: str
    failing_provider_model: str
    hidden_free: str
    visible_free: str
    hidden_paid: str
    hidden_responses: str
    hidden_responses_stream: str
    hidden_failing: str
    shown_free: str
    null_hidden_free: str
    hidden_unpriced: str
    hidden_missing: str


def base_url(candidate: Gateway) -> str:
    return str(candidate.client.base_url).rstrip("/")


def spend_marker(marker: str) -> Mapping[str, str]:
    return {SPEND_MARKER_HEADER: json.dumps({"marker": marker})}


def fresh_post(candidate: Gateway, path: str, body: Mapping[str, JsonValue], key: str, marker: str) -> httpx.Response:
    return httpx.post(
        f"{base_url(candidate)}{path}",
        json=dict(body),
        headers={"Authorization": f"Bearer {key}", **spend_marker(marker)},
        timeout=60,
        trust_env=False,
    )


NO_EXTRA: Final[Mapping[str, JsonValue]] = MappingProxyType({})


def chat_body(model: JsonValue, marker: str, extra: Mapping[str, JsonValue] = NO_EXTRA) -> Mapping[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": marker}], **extra}


def fresh_chat(
    candidate: Gateway, model: str, key: str, marker: str, extra: Mapping[str, JsonValue] = NO_EXTRA
) -> httpx.Response:
    return fresh_post(candidate, "/v1/chat/completions", chat_body(model, marker, extra), key, marker)


def fresh_response(
    candidate: Gateway, model: str, key: str, marker: str, extra: Mapping[str, JsonValue] = NO_EXTRA
) -> httpx.Response:
    return fresh_post(candidate, "/v1/responses", {"model": model, "input": marker, **extra}, key, marker)


def fresh_message(
    candidate: Gateway, model: str, key: str, marker: str, extra: Mapping[str, JsonValue] = NO_EXTRA
) -> httpx.Response:
    return fresh_post(
        candidate,
        "/v1/messages",
        {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}], **extra},
        key,
        marker,
    )


def statuses(send: Callable[[str], httpx.Response], count: int) -> frozenset[int]:
    markers: Final = tuple(uuid.uuid4().hex for _ in range(count))
    with ThreadPoolExecutor(max_workers=count) as pool:
        return frozenset(response.status_code for response in pool.map(send, markers))


def error_type(response: httpx.Response) -> str:
    return str(object_value(JSON_OBJECT.validate_json(response.content)["error"])["type"])


def settle(
    send: Callable[[str], httpx.Response], status: int, *, burst: int = GATEWAY_BURST, seconds: float = 60
) -> None:
    eventually(
        lambda: statuses(send, burst) | statuses(send, burst),
        lambda seen: seen == frozenset({status}),
        seconds=seconds,
    )


def chat_statuses(
    candidate: Gateway, model: str, key: str, count: int, extra: Mapping[str, JsonValue] = NO_EXTRA
) -> frozenset[int]:
    return statuses(lambda marker: fresh_chat(candidate, model, key, marker, extra), count)


def settle_candidate(
    candidate: Gateway,
    model: str,
    key: str,
    status: int,
    *,
    burst: int = GATEWAY_BURST,
    seconds: float = 60,
    extra: Mapping[str, JsonValue] = NO_EXTRA,
) -> None:
    settle(lambda marker: fresh_chat(candidate, model, key, marker, extra), status, burst=burst, seconds=seconds)


def settle_chat(
    rig: AliasRig,
    model: str,
    key: str,
    status: int,
    *,
    seconds: float = 60,
    extra: Mapping[str, JsonValue] = NO_EXTRA,
) -> None:
    settle_candidate(rig.gateway, model, key, status, seconds=seconds, extra=extra)
    settle_candidate(rig.peer, model, key, status, burst=PEER_BURST, seconds=seconds, extra=extra)


def exhausted_key(rig: AliasRig, scenario: Scenario) -> str:
    key: Final = scenario.key(max_budget=BUDGET)
    first: Final = fresh_chat(rig.gateway, rig.paid, key, "exhaust-" + uuid.uuid4().hex)
    assert first.status_code == 200, first.text
    settle_chat(rig, rig.paid, key, BUDGET_EXCEEDED)
    return key


def upstream_requests(upstream_url: str) -> tuple[str, ...]:
    drained: Final = JSON_OBJECT.validate_json(
        httpx.get(f"{upstream_url}/__observations", timeout=15, trust_env=False).content
    )
    return tuple(json.dumps(entry) for entry in JSON_LIST.validate_python(drained["requests"]))


def upstream_hits(observed: tuple[str, ...], marker: str) -> int:
    return sum(1 for entry in observed if marker in entry)


def script_provider(rig: AliasRig, failures: int) -> None:
    scripted: Final = httpx.post(
        f"{rig.gateway.upstream_url}/__scripts/{rig.failing_provider_model}",
        json={"statuses": [500] * failures},
        timeout=15,
        trust_env=False,
    )
    assert scripted.status_code == 200, scripted.text


def clear_provider_script(rig: AliasRig) -> None:
    cleared: Final = httpx.delete(
        f"{rig.gateway.upstream_url}/__scripts/{rig.failing_provider_model}", timeout=15, trust_env=False
    )
    assert cleared.status_code in (200, 404), cleared.text


def spend_rows(key: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        read_rows(
            "SELECT request_id, spend, model_group, status, call_type, "
            "metadata->'spend_logs_metadata'->>'marker' AS marker "
            'FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
            (sha256(key.encode()).hexdigest(),),
        )
    )


def landed(key: str, marker: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(row for row in spend_rows(key) if row["marker"] == marker)


def landed_once(key: str, marker: str) -> Mapping[str, JsonValue]:
    rows: Final = eventually(lambda: landed(key, marker), lambda found: len(found) >= 1, seconds=70)
    assert len(rows) == 1, rows
    return rows[0]


def marker_counts(key: str, markers: frozenset[str]) -> Mapping[str, int]:
    return dict(Counter(str(row["marker"]) for row in spend_rows(key) if row["marker"] in markers))


def landed_all_once(key: str, markers: frozenset[str]) -> tuple[Mapping[str, JsonValue], ...]:
    counts: Final = eventually(
        lambda: marker_counts(key, markers), lambda found: frozenset(found) == markers, seconds=90
    )
    assert counts == dict.fromkeys(markers, 1), counts
    return tuple(row for row in spend_rows(key) if row["marker"] in markers)


def assert_free_row(row: Mapping[str, JsonValue], model_group: str) -> None:
    assert float(str(row["spend"])) == 0.0, row
    assert row["model_group"] == model_group, row
    assert row["status"] == "success", row


def alias_map(gateway: Gateway) -> Mapping[str, JsonValue]:
    current: Final = object_value(gateway.get("/router/settings")["current_values"]).get("model_group_alias")
    return object_value(current) if current is not None else {}


def write_aliases(gateway: Gateway, aliases: Mapping[str, JsonValue]) -> None:
    gateway.post("/config/update", {"router_settings": {"model_group_alias": dict(aliases)}})


def install_aliases(gateway: Gateway, aliases: Mapping[str, JsonValue]) -> None:
    write_aliases(gateway, {**alias_map(gateway), **aliases})


def remove_aliases(gateway: Gateway, names: frozenset[str]) -> None:
    write_aliases(gateway, {name: target for name, target in alias_map(gateway).items() if name not in names})


def hidden(group: str) -> JsonValue:
    return {"model": group, "hidden": True}


def openai_client(candidate: Gateway, key: str) -> OpenAI:
    return OpenAI(
        api_key=key,
        base_url=base_url(candidate) + "/v1",
        max_retries=0,
        http_client=httpx.Client(timeout=60, trust_env=False),
    )


def async_openai_client(candidate: Gateway, key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=key,
        base_url=base_url(candidate) + "/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=60, trust_env=False),
    )


def anthropic_client(candidate: Gateway, key: str) -> Anthropic:
    return Anthropic(
        api_key=key,
        base_url=base_url(candidate),
        max_retries=0,
        http_client=httpx.Client(timeout=60, trust_env=False),
    )


def async_anthropic_client(candidate: Gateway, key: str) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=key,
        base_url=base_url(candidate),
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=60, trust_env=False),
    )


def _zero_cost_responses_group(
    scenario: Scenario, gateway: Gateway, name: str, response: JsonResponse | SseResponse
) -> str:
    handle: Final = register_scenario(name, response, control_url=gateway.upstream_url)
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario.model(api_base=f"{handle.api_base()}/v1", input_cost_per_token=0, output_cost_per_token=0)


def settle_responses(
    rig: AliasRig, model: str, key: str, status: int, extra: Mapping[str, JsonValue] = NO_EXTRA
) -> None:
    settle(lambda marker: fresh_response(rig.gateway, model, key, marker, extra), status)
    settle(lambda marker: fresh_response(rig.peer, model, key, marker, extra), status, burst=PEER_BURST)


def _await_rig(rig: AliasRig) -> None:
    admin: Final = rig.gateway.key
    for model in (
        rig.hidden_free,
        rig.visible_free,
        rig.hidden_paid,
        rig.hidden_failing,
        rig.shown_free,
        rig.null_hidden_free,
        rig.hidden_unpriced,
    ):
        settle_chat(rig, model, admin, 200)
    settle_responses(rig, rig.hidden_responses, admin, 200)
    settle_responses(rig, rig.hidden_responses_stream, admin, 200, extra={"stream": True})


@contextmanager
def alias_rig() -> Generator[AliasRig]:
    suffix: Final = uuid.uuid4().hex[:12]
    with (
        gateway_from_environment() as gateway,
        httpx.Client(
            base_url=os.environ["INTEGRATION_PEER_URL"], timeout=15, trust_env=False, limits=GATEWAY_LIMITS
        ) as peer_client,
        gateway.scenario() as scenario,
    ):
        free: Final = scenario.model(input_cost_per_token=0, output_cost_per_token=0)
        paid: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        unpriced: Final = scenario.model()
        failing_provider_model: Final = f"hidden-alias-failing-{suffix}"
        failing_free: Final = scenario.model(
            model=f"openai/{failing_provider_model}", input_cost_per_token=0, output_cost_per_token=0
        )
        responses_free: Final = _zero_cost_responses_group(
            scenario,
            gateway,
            f"hidden-alias-json-{suffix}",
            JsonResponse(content_type="application/json", body=_RESPONSE),
        )
        responses_stream_free: Final = _zero_cost_responses_group(
            scenario,
            gateway,
            f"hidden-alias-sse-{suffix}",
            SseResponse(
                content_type="text/event-stream",
                frames=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}" for event in _RESPONSE_EVENTS),
            ),
        )
        rig: Final = AliasRig(
            gateway=gateway,
            peer=Gateway(peer_client, gateway.key, gateway.upstream_url),
            free=free,
            paid=paid,
            failing_free=failing_free,
            failing_provider_model=failing_provider_model,
            hidden_free=f"hidden-free-{suffix}",
            visible_free=f"visible-free-{suffix}",
            hidden_paid=f"hidden-paid-{suffix}",
            hidden_responses=f"hidden-responses-{suffix}",
            hidden_responses_stream=f"hidden-responses-stream-{suffix}",
            hidden_failing=f"hidden-failing-{suffix}",
            shown_free=f"shown-free-{suffix}",
            null_hidden_free=f"null-hidden-free-{suffix}",
            hidden_unpriced=f"hidden-unpriced-{suffix}",
            hidden_missing=f"hidden-missing-{suffix}",
        )
        aliases: Final[Mapping[str, JsonValue]] = {
            rig.hidden_free: hidden(free),
            rig.visible_free: free,
            rig.hidden_paid: hidden(paid),
            rig.hidden_responses: hidden(responses_free),
            rig.hidden_responses_stream: hidden(responses_stream_free),
            rig.hidden_failing: hidden(failing_free),
            rig.shown_free: {"model": free, "hidden": False},
            rig.null_hidden_free: {"model": free, "hidden": None},
            rig.hidden_unpriced: hidden(unpriced),
            rig.hidden_missing: hidden(f"missing-group-{suffix}"),
        }
        install_aliases(gateway, aliases)
        scenario.cleanups.callback(remove_aliases, gateway, frozenset(aliases))
        _await_rig(rig)
        yield rig
