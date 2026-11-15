import json
import os
import uuid
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import httpx
from integration._support.client import (
    GATEWAY_LIMITS,
    JSON_OBJECT,
    Gateway,
    Scenario,
    gateway_from_environment,
    object_value,
)
from integration._support.upstream import delete_scenario, register_scenario
from integration.authorization._hidden_alias_budget import (
    fresh_chat,
    fresh_message,
    fresh_post,
    fresh_response,
    settle,
)
from integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse
from pydantic import JsonValue

DENIED: Final = 403
STREAM: Final[Mapping[str, JsonValue]] = {"stream": True}
ROUTES: Final = (
    "chat",
    "chatstream",
    "completions",
    "embeddings",
    "responses",
    "responsesstream",
    "messages",
    "messagesstream",
    "deployments",
    "engines",
)
BURST_ROUTES: Final = ("chat", "chatstream", "responses", "responsesstream", "messages", "messagesstream")

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
            "content": [{"type": "output_text", "text": "served reply", "annotations": []}],
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
        "delta": "served reply",
    },
    {"type": "response.completed", "sequence_number": 2, "response": _RESPONSE},
)


@dataclass(frozen=True, slots=True)
class Models:
    chat: str
    responses: str
    responses_stream: str

    def names(self) -> list[JsonValue]:
        return [self.chat, self.responses, self.responses_stream]

    def serving(self, route: str) -> str:
        if route in ("responses", "messages"):
            return self.responses
        if route in ("responsesstream", "messagesstream"):
            return self.responses_stream
        return self.chat


@dataclass(frozen=True, slots=True)
class DenyRig:
    gateway: Gateway
    peer: Gateway
    allowed: Models
    denied: Models
    key: str
    open_key: str


def _responses_group(scenario: Scenario, gateway: Gateway, response: JsonResponse | SseResponse) -> str:
    handle: Final = register_scenario(f"key-disallowed-{uuid.uuid4().hex}", response, control_url=gateway.upstream_url)
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario.model(api_base=f"{handle.api_base()}/v1")


def _models(scenario: Scenario, gateway: Gateway) -> Models:
    return Models(
        chat=scenario.model(),
        responses=_responses_group(scenario, gateway, JsonResponse(content_type="application/json", body=_RESPONSE)),
        responses_stream=_responses_group(
            scenario,
            gateway,
            SseResponse(
                content_type="text/event-stream",
                frames=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}" for event in _RESPONSE_EVENTS),
            ),
        ),
    )


def send(candidate: Gateway, route: str, model: str, key: str, marker: str) -> httpx.Response:
    senders: Final[Mapping[str, Callable[[], httpx.Response]]] = {
        "chat": lambda: fresh_chat(candidate, model, key, marker),
        "chatstream": lambda: fresh_chat(candidate, model, key, marker, STREAM),
        "completions": lambda: fresh_post(
            candidate, "/v1/completions", {"model": model, "prompt": marker}, key, marker
        ),
        "embeddings": lambda: fresh_post(candidate, "/v1/embeddings", {"model": model, "input": marker}, key, marker),
        "responses": lambda: fresh_response(candidate, model, key, marker),
        "responsesstream": lambda: fresh_response(candidate, model, key, marker, STREAM),
        "messages": lambda: fresh_message(candidate, model, key, marker),
        "messagesstream": lambda: fresh_message(candidate, model, key, marker, STREAM),
        "deployments": lambda: fresh_post(
            candidate,
            f"/openai/deployments/{model}/chat/completions",
            {"messages": [{"role": "user", "content": marker}]},
            key,
            marker,
        ),
        "engines": lambda: fresh_post(
            candidate,
            f"/engines/{model}/chat/completions",
            {"messages": [{"role": "user", "content": marker}]},
            key,
            marker,
        ),
    }
    return senders[route]()


def denial(response: httpx.Response) -> tuple[int, JsonValue]:
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    return response.status_code, error.get("litellm_error_type", error.get("type"))


KEY_DENIAL: Final = (DENIED, "key_model_access_denied")
UPSTREAM_RECORDED_ROUTES: Final = frozenset(ROUTES) - {"embeddings"}


def send_all(send_one: Callable[[str], httpx.Response], markers: tuple[str, ...]) -> tuple[httpx.Response, ...]:
    with ThreadPoolExecutor(max_workers=len(markers)) as pool:
        return tuple(pool.map(send_one, markers))


def _settle_models(candidate: Gateway, models: Models, key: str, status: int, burst: int) -> None:
    settle(lambda marker: fresh_chat(candidate, models.chat, key, marker), status, burst=burst)
    settle(lambda marker: fresh_response(candidate, models.responses, key, marker), status, burst=burst)
    settle(lambda marker: fresh_response(candidate, models.responses_stream, key, marker, STREAM), status, burst=burst)


@contextmanager
def deny_rig() -> Generator[DenyRig]:
    with (
        gateway_from_environment() as gateway,
        httpx.Client(
            base_url=os.environ["INTEGRATION_PEER_URL"], timeout=15, trust_env=False, limits=GATEWAY_LIMITS
        ) as peer_client,
        gateway.scenario() as scenario,
    ):
        peer: Final = Gateway(peer_client, gateway.key, gateway.upstream_url)
        allowed: Final = _models(scenario, gateway)
        denied: Final = _models(scenario, gateway)
        for candidate, burst in ((gateway, 8), (peer, 4)):
            _settle_models(candidate, allowed, gateway.key, 200, burst)
            _settle_models(candidate, denied, gateway.key, 200, burst)
        yield DenyRig(
            gateway=gateway,
            peer=peer,
            allowed=allowed,
            denied=denied,
            key=scenario.key(disallowed_models=denied.names()),
            open_key=scenario.key(),
        )
