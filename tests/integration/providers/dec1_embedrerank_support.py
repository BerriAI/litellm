"""Shared rig for the `test_dec1_embedrerank_*` files: embeddings and rerank on the non-OpenAI providers.

Every test talks to the proxy and to one owned HTTP peer that plays the provider. The peer records each request
the proxy sends (method, path, query, headers, decoded JSON body) and answers with what the test's `respond`
function returns. Helpers below build the canned provider answers each provider family speaks:

    openai_list         an OpenAI embeddings list, one `VECTOR` per input, usage prompt and total tokens
    cohere_embed        a Cohere v2 embed body with `embeddings`, `meta.billed_units`
    ollama_embed        an Ollama `/api/embed` body with `embeddings` and `prompt_eval_count`
    cohere_rerank       a Cohere rerank body with `results` and `meta.billed_units.search_units`
    bedrock_rerank      a Bedrock Agent Runtime rerank body with `results[].relevanceScore`

Deployments are priced through `litellm_params` (`input_cost_per_token`, `input_cost_per_query`) so every expected
spend is a literal. `owned_rig` boots a second proxy with its own model list and router settings for the
routing contracts (ordered groups, fallbacks across providers, retries) that cannot be set per deployment on the
shared gateway. Inputs are made unique with `marker()` because the shared gateway runs the response cache in
`default_on` mode.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import httpx
import yaml
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

PROVIDER_KEY: Final = "dec1-embedrerank-provider-key"
INPUT_RATE: Final = 0.001
QUERY_RATE: Final = 0.002
VECTOR: Final = (0.125, 0.25, 0.5)
QUERY: Final = "which synthetic document names the gateway"
DOCUMENTS: Final = ("the gateway reranks documents", "an unrelated synthetic note", "a second unrelated note")
SCORES: Final = ((2, 0.88), (0, 0.61), (1, 0.02))
TOKEN_INPUT: Final = [[791, 4062, 14198, 2385], [40, 1093]]
FLAT_TOKEN_INPUT: Final = [791, 4062, 14198, 2385]
DECODED_TOKEN_TEXTS: Final = ["The quick brown base", "I like"]
LANDING_SECONDS: Final = 70
CALL_ID: Final = "x-litellm-call-id"
COST_HEADER: Final = "x-litellm-response-cost"
CACHE_KEY_HEADER: Final = "x-litellm-cache-key"
ROW_COLUMNS: Final = (
    "request_id, call_type, spend, total_tokens, prompt_tokens, completion_tokens, model, model_group, "
    'custom_llm_provider, api_base, "user", cache_hit, status'
)


@dataclass(frozen=True, slots=True)
class Call:
    """One request the proxy sent to the provider peer."""

    method: str
    path: str
    query: Mapping[str, Sequence[str]]
    headers: Mapping[str, str]
    body: Mapping[str, JsonValue]


Respond = Callable[[Call], Reply]


@dataclass(frozen=True, slots=True)
class Peer:
    wire: Wire
    seen: list[Call] = field(default_factory=list)  # mutable-ok: appended by the peer's handler threads
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        return self.wire.url

    def posts(self) -> tuple[Call, ...]:
        with self.lock:
            return tuple(call for call in self.seen if call.method == "POST")

    def only(self) -> Call:
        calls: Final = self.posts()
        assert len(calls) == 1, [(call.method, call.path) for call in calls]
        return calls[0]


def _decoded(request: Request) -> Mapping[str, JsonValue]:
    if not request.body:
        return {}
    try:
        return JSON_OBJECT.validate_json(request.body)
    except ValueError:
        return {}


@contextmanager
def peer(respond: Respond) -> Iterator[Peer]:
    """An owned provider peer that records every request and answers with `respond`."""
    holder: Final[list[Peer]] = []  # mutable-ok: the handler needs the Peer built from its own wire

    def handle(request: Request) -> Reply:
        path, _, query = request.target.partition("?")
        call: Final = Call(request.method, path, parse_qs(query), request.headers, _decoded(request))
        with holder[0].lock:
            holder[0].seen.append(call)
        return respond(call)

    with wire_server(handle) as wire:
        created: Final = Peer(wire)
        holder.append(created)
        yield created


def json_reply(body: JsonValue, status: int = 200, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode(), headers=headers or {})


def error_reply(status: int, message: str = "dec1 embedrerank scripted failure") -> Reply:
    kind: Final = "rate_limit_error" if status == 429 else "invalid_request_error" if status < 500 else "server_error"
    return json_reply({"error": {"message": message, "type": kind, "code": None}}, status)


def input_count(body: Mapping[str, JsonValue]) -> int:
    for name in ("input", "texts", "images", "inputs"):
        value = body.get(name)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, str):
            return 1
    return 1


def openai_list(prompt_tokens: int = 5, *, model: str | None = None) -> Respond:
    """An OpenAI-shaped embeddings answer with one `VECTOR` per input."""

    def respond(call: Call) -> Reply:
        return json_reply(
            {
                "object": "list",
                "model": model if model is not None else call.body.get("model"),
                "data": [
                    {"object": "embedding", "index": index, "embedding": list(VECTOR)}
                    for index in range(input_count(call.body))
                ],
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }
        )

    return respond


def cohere_embed(billed_units: Mapping[str, JsonValue] | None = None, *, embeddings: JsonValue = None) -> Respond:
    """A Cohere v2 embed answer; with no `embeddings` given, one `VECTOR` per text or image as a bare list."""

    def respond(call: Call) -> Reply:
        vectors: Final[JsonValue] = (
            [list(VECTOR) for _ in range(input_count(call.body))] if embeddings is None else embeddings
        )
        return json_reply(
            {
                "id": "dec1-embedrerank-cohere",
                "embeddings": vectors,
                "meta": {"billed_units": dict(billed_units) if billed_units is not None else {"input_tokens": 3}},
            }
        )

    return respond


def ollama_embed(prompt_eval_count: int | None = 4) -> Respond:
    def respond(call: Call) -> Reply:
        body: Final[dict[str, JsonValue]] = {
            "model": call.body.get("model"),
            "embeddings": [list(VECTOR) for _ in range(input_count(call.body))],
        }
        if prompt_eval_count is not None:
            body["prompt_eval_count"] = prompt_eval_count
        return json_reply(body)

    return respond


def cohere_rerank(search_units: int | None = 1, *, extra: Mapping[str, JsonValue] | None = None) -> Respond:
    """A Cohere-shaped rerank answer in `SCORES` order (index 2, then 0, then 1)."""

    def respond(call: Call) -> Reply:
        body: Final[dict[str, JsonValue]] = {
            "id": f"dec1-embedrerank-{uuid.uuid4().hex}",
            "results": [{"index": index, "relevance_score": score} for index, score in SCORES],
            **(extra or {}),
        }
        if search_units is not None:
            body["meta"] = {"api_version": {"version": "2"}, "billed_units": {"search_units": search_units}}
        return json_reply(body)

    return respond


def bedrock_rerank(call: Call) -> Reply:
    return json_reply({"results": [{"index": index, "relevanceScore": score} for index, score in SCORES]})


def failing(status: int, message: str = "dec1 embedrerank scripted failure") -> Respond:
    def respond(call: Call) -> Reply:
        return error_reply(status, message)

    return respond


def held(release: threading.Event, then: Respond, seconds: float = 5.0) -> Respond:
    """Hold every answer until `release` is set or `seconds` pass, then answer with `then`."""

    def respond(call: Call) -> Reply:
        release.wait(timeout=seconds)
        return then(call)

    return respond


def marker() -> str:
    """A unique string, so the shared gateway's response cache never answers a request by accident."""
    return f"dec1-embedrerank-{uuid.uuid4().hex}"


def embed(
    gateway: Gateway,
    model: str,
    value: JsonValue = None,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    path: str = "/v1/embeddings",
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": model, "input": marker() if value is None else value, **extra}
    return gateway.request("POST", path, body, key=key, headers=headers)


def rerank(
    gateway: Gateway,
    model: str,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    path: str = "/v1/rerank",
    query: str | None = None,
    documents: JsonValue = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "query": f"{QUERY} {uuid.uuid4().hex}" if query is None else query,
        "documents": list(DOCUMENTS) if documents is None else documents,
        **extra,
    }
    return gateway.request("POST", path, body, key=key, headers=headers)


def served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def error_message(response: httpx.Response) -> str:
    return string_value(object_value(JSON_OBJECT.validate_json(response.content)["error"])["message"])


def vectors(body: Mapping[str, JsonValue]) -> list[JsonValue]:
    data: Final = body["data"]
    assert isinstance(data, list), body
    return [object_value(entry)["embedding"] for entry in data]


def indexes(body: Mapping[str, JsonValue]) -> list[JsonValue]:
    data: Final = body["data"]
    assert isinstance(data, list), body
    return [object_value(entry)["index"] for entry in data]


def results(body: Mapping[str, JsonValue]) -> list[tuple[JsonValue, JsonValue]]:
    found: Final = body["results"]
    assert isinstance(found, list), body
    return [(object_value(entry)["index"], object_value(entry)["relevance_score"]) for entry in found]


def number(value: JsonValue) -> float:
    assert isinstance(value, (int, float, str)) and not isinstance(value, bool), value
    return float(value)


def hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def spend_row(gateway: Gateway, call_id: str) -> dict[str, JsonValue]:
    """The one spend log row whose `litellm_call_id` is the `x-litellm-call-id` the caller sent."""
    rows: Final = eventually(
        lambda: gateway.rows(f'SELECT {ROW_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s', (call_id,)),
        lambda found: len(found) == 1,
        seconds=LANDING_SECONDS,
    )
    return rows[0]


def call_header(call_id: str) -> dict[str, str]:
    return {CALL_ID: call_id}


def new_call_id() -> str:
    return str(uuid.uuid4())


def deployment(
    group: str,
    litellm_model: str,
    api_base: str,
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> dict[str, JsonValue]:
    """One `model_list` entry for `owned_rig`, with deployment id `dec1-er-<group>-<last part of the model>`."""
    return {
        "model_name": group,
        "litellm_params": {"model": litellm_model, "api_key": PROVIDER_KEY, "api_base": api_base, **params},
        "model_info": {"id": f"dec1-er-{group}-{litellm_model.rsplit('/', 1)[-1]}", **(model_info or {})},
    }


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    primary: Peer
    secondary: Peer


@contextmanager
def owned_rig(
    directory: Path,
    models: Callable[[str, str], Sequence[Mapping[str, JsonValue]]],
    *,
    primary: Respond,
    secondary: Respond,
    router_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[Rig]:
    """Two provider peers and one owned proxy whose `model_list` is `models(primary.url, secondary.url)`.

    Router defaults are `num_retries: 0` and `disable_cooldowns: true` unless `router_settings` overrides them,
    so each test sees exactly the attempts it asked for.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        first: Final = stack.enter_context(peer(primary))
        second: Final = stack.enter_context(peer(secondary))
        config: Final = {
            "model_list": [dict(entry) for entry in models(first.url, second.url)],
            "litellm_settings": dict(litellm_settings or {}),
            "general_settings": {
                "master_key": "os.environ/LITELLM_MASTER_KEY",
                "database_url": "os.environ/DATABASE_URL",
                "disable_model_info_refresh": True,
            },
            "router_settings": {"num_retries": 0, "disable_cooldowns": True, **(router_settings or {})},
        }
        path: Final = directory / f"dec1-embedrerank-{uuid.uuid4().hex[:8]}.yaml"
        path.write_text(yaml.safe_dump(config))
        gateway: Final = stack.enter_context(
            owned_proxy(shared, directory, {"STORE_MODEL_IN_DB": "False"}, config=path)
        )
        yield Rig(gateway, first, second)
