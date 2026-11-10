"""Shared rig for the `test_dec1_textapis_*` files: embeddings, rerank, moderations and legacy text completions.

Every test in those files talks to a real proxy and to one owned HTTP peer that plays the provider. The peer
records each request the proxy sends (method, path, query, headers, decoded JSON body) and answers with a small
canned reply chosen from the request path, unless the test passes its own `respond` function:

    .../embeddings        an OpenAI embeddings list with one `[0.25, 0.5, 0.75]` vector per input and
                          `usage.prompt_tokens` = `EMBEDDING_PROMPT_TOKENS` (7)
    .../rerank            a Cohere rerank body with two results (index 0 scores 0.91, index 1 scores 0.03) and
                          `meta.billed_units.search_units` = 1
    .../chat/completions  a chat completion with content `REPLY_TEXT` and usage 6 prompt + 2 completion tokens,
                          or the same as SSE chunks when `stream` is true
    .../completions       a text completion with text `REPLY_TEXT` and usage 6 prompt + 2 completion tokens,
                          or the same as SSE chunks when `stream` is true
    .../moderations       one unflagged moderation result

Deployments are priced through `litellm_params` at `INPUT_RATE` 0.001 per input token and `OUTPUT_RATE` 0.002 per
output token, so every expected spend is a literal: embeddings 7 * 0.001 = 0.007, text completion
6 * 0.001 + 2 * 0.002 = 0.01, Cohere rerank 1 search unit * `RERANK_QUERY_RATE` 0.002 = 0.002.

`owned_rig` boots a second proxy with its own config (model list, router settings, litellm settings) for the
contracts that cannot be set per deployment on the shared gateway: retries, fallbacks, cooldowns, global
`drop_params`, and the `generic_api` logging callback whose StandardLoggingPayload batches land in `rig.sink`.
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
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

PROVIDER_KEY: Final = "dec1-textapis-provider-key"
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
RERANK_QUERY_RATE: Final = 0.002
EMBEDDING_PROMPT_TOKENS: Final = 7
EMBEDDING_COST: Final = 0.007
TEXT_PROMPT_TOKENS: Final = 6
TEXT_COMPLETION_TOKENS: Final = 2
TEXT_COST: Final = 0.01
RERANK_COST: Final = 0.002
VECTOR: Final = (0.25, 0.5, 0.75)
REPLY_TEXT: Final = "dec1 textapis reply"
STREAM_PIECES: Final = ("dec1 ", "textapis ", "reply")
RERANK_SCORES: Final = ((0, 0.91), (1, 0.03))
QUERY: Final = "which document mentions the gateway"
DOCUMENTS: Final = ("the gateway proxies rerank calls", "unrelated synthetic text")
LANDING_SECONDS: Final = 70
MODEL_DENIED_TEMPLATE: Final = (
    "The requested model '{model}' is not available for this API key, or the model name is invalid. "
    "Check the models available to you and try again."
)
MODERATION_CATEGORIES: Final = ("harassment", "hate", "self-harm", "sexual", "violence")

ROW_COLUMNS: Final = (
    "request_id, call_type, api_key, spend, total_tokens, prompt_tokens, completion_tokens, model, model_id, "
    'model_group, custom_llm_provider, api_base, "user", metadata, cache_hit, cache_key, request_tags, team_id, '
    "end_user, status, litellm_call_id"
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
        """Every inference request received so far; model discovery GETs are left out."""
        with self.lock:
            return tuple(call for call in self.seen if call.method == "POST")

    def only(self) -> Call:
        calls: Final = self.posts()
        assert len(calls) == 1, [(call.method, call.path) for call in calls]
        return calls[0]

    def count(self, model: str | None = None) -> int:
        return len([call for call in self.posts() if model is None or call.body.get("model") == model])


def json_reply(body: Mapping[str, JsonValue], status: int = 200, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode(), headers=headers or {})


def error_reply(
    status: int,
    message: str = "dec1 textapis scripted failure",
    *,
    kind: str | None = None,
    code: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> Reply:
    """An OpenAI-shaped provider error body with the given HTTP status."""
    default_kind: Final = "rate_limit_error" if status == 429 else "invalid_request_error" if status < 500 else "server_error"
    return json_reply({"error": {"message": message, "type": kind or default_kind, "code": code}}, status, headers)


def _input_count(body: Mapping[str, JsonValue]) -> int:
    value: Final = body.get("input")
    if isinstance(value, list) and value and isinstance(value[0], (str, list, dict)):
        return len(value)
    return 1


def embedding_reply(body: Mapping[str, JsonValue], prompt_tokens: int = EMBEDDING_PROMPT_TOKENS) -> Reply:
    return json_reply(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": list(VECTOR)}
                for index in range(_input_count(body))
            ],
            "model": body.get("model"),
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
    )


def cohere_rerank_reply(search_units: int = 1) -> Reply:
    return json_reply(
        {
            "id": f"dec1-rerank-{uuid.uuid4().hex}",
            "results": [{"index": index, "relevance_score": score} for index, score in RERANK_SCORES],
            "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": search_units}},
        }
    )


def _text_usage() -> dict[str, JsonValue]:
    return {
        "prompt_tokens": TEXT_PROMPT_TOKENS,
        "completion_tokens": TEXT_COMPLETION_TOKENS,
        "total_tokens": TEXT_PROMPT_TOKENS + TEXT_COMPLETION_TOKENS,
    }


def sse(payload: Mapping[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def text_reply(body: Mapping[str, JsonValue]) -> Reply:
    """A native `/completions` answer: one JSON body, or three text chunks, a usage chunk and `[DONE]`."""
    identity: Final = f"cmpl-dec1-{uuid.uuid4().hex}"
    model: Final = body.get("model")
    if body.get("stream") is not True:
        return json_reply(
            {
                "id": identity,
                "object": "text_completion",
                "created": 1,
                "model": model,
                "choices": [{"text": REPLY_TEXT, "index": 0, "logprobs": None, "finish_reason": "stop"}],
                "usage": _text_usage(),
            }
        )
    base: Final[dict[str, JsonValue]] = {"id": identity, "object": "text_completion", "created": 1, "model": model}
    pieces: Final = tuple(
        sse({**base, "choices": [{"text": piece, "index": 0, "logprobs": None, "finish_reason": None}]})
        for piece in STREAM_PIECES
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(
            *pieces,
            sse({**base, "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}]}),
            sse({**base, "choices": [], "usage": _text_usage()}),
            b"data: [DONE]\n\n",
        ),
    )


def chat_reply(body: Mapping[str, JsonValue]) -> Reply:
    """A `/chat/completions` answer with the same text and usage as `text_reply`."""
    identity: Final = f"chatcmpl-dec1-{uuid.uuid4().hex}"
    model: Final = body.get("model")
    if body.get("stream") is not True:
        return json_reply(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": REPLY_TEXT}, "finish_reason": "stop"}
                ],
                "usage": _text_usage(),
            }
        )
    base: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
    }
    pieces: Final = tuple(
        sse({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
        for piece in STREAM_PIECES
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(
            *pieces,
            sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            sse({**base, "choices": [], "usage": _text_usage()}),
            b"data: [DONE]\n\n",
        ),
    )


def moderation_reply(body: Mapping[str, JsonValue], flagged: bool = False) -> Reply:
    return json_reply(
        {
            "id": f"modr-dec1-{uuid.uuid4().hex}",
            "model": body.get("model") or "omni-moderation-latest",
            "results": [
                {
                    "flagged": flagged,
                    "categories": {category: flagged and category == "violence" for category in MODERATION_CATEGORIES},
                    "category_scores": {
                        category: 0.93 if flagged and category == "violence" else 0.01
                        for category in MODERATION_CATEGORIES
                    },
                }
            ],
        }
    )


def default_respond(call: Call) -> Reply:
    """The canned answer for a provider path, as listed in the module docstring."""
    if call.method != "POST":
        return json_reply({"object": "list", "data": []})
    if call.path.endswith("/embeddings"):
        return embedding_reply(call.body)
    if call.path.endswith("/rerank"):
        return cohere_rerank_reply()
    if call.path.endswith("/chat/completions"):
        return chat_reply(call.body)
    if call.path.endswith("/completions"):
        return text_reply(call.body)
    if call.path.endswith("/moderations"):
        return moderation_reply(call.body)
    return error_reply(404, f"dec1 textapis peer has no route {call.path}")


def failing(status: int, message: str = "dec1 textapis scripted failure", **fields: str) -> Respond:
    """A provider that fails every inference call with `status` and an OpenAI-shaped error body."""

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return json_reply({"object": "list", "data": []})
        return error_reply(status, message, kind=fields.get("kind"), code=fields.get("code"))

    return respond


def by_model(call: Call) -> Reply:
    """Answer from the upstream model name: `status-<code>-<label>` always fails with that status, `flaky-<label>`
    fails with 500 on its first request only, anything else gets the canned answer."""
    model: Final = str(call.body.get("model", ""))
    if call.method == "POST" and model.startswith("status-"):
        status: Final = int(model.split("-")[1])
        return error_reply(status, f"scripted {status} from {model}")
    return default_respond(call)


def _decoded(request: Request) -> Mapping[str, JsonValue]:
    if not request.body or "json" not in request.headers.get("content-type", "json"):
        return {}
    try:
        return JSON_OBJECT.validate_json(request.body)
    except ValueError:
        return {}


@contextmanager
def peer(respond: Respond = default_respond) -> Iterator[Peer]:
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


def flaky_once(status: int = 500) -> Respond:
    """Fail the first inference request with `status`, then answer normally."""
    failed: Final[list[bool]] = []  # mutable-ok: one-shot flag shared across handler threads

    def respond(call: Call) -> Reply:
        if call.method == "POST" and not failed:
            failed.append(True)
            return error_reply(status, f"scripted one-off {status}")
        return default_respond(call)

    return respond


def openai_embedding_model(
    scenario: Scenario, provider: Peer, upstream: str = "text-embedding-3-small", **parameters: JsonValue
) -> str:
    """An `openai/<upstream>` embedding deployment priced at INPUT_RATE that calls the peer under `/v1`."""
    return scenario.model(
        **{
            "model": f"openai/{upstream}",
            "api_base": f"{provider.url}/v1",
            "api_key": PROVIDER_KEY,
            "input_cost_per_token": INPUT_RATE,
            "output_cost_per_token": 0.0,
            **parameters,
        }
    )


def text_model(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    """A `text-completion-openai/gpt-3.5-turbo-instruct` deployment that calls the peer's `/v1/completions`."""
    return scenario.model(
        **{
            "model": "text-completion-openai/gpt-3.5-turbo-instruct",
            "api_base": f"{provider.url}/v1",
            "api_key": PROVIDER_KEY,
            "input_cost_per_token": INPUT_RATE,
            "output_cost_per_token": OUTPUT_RATE,
            **parameters,
        }
    )


def chat_backed_text_model(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    """An `openai/gpt-4o-mini` chat deployment, which serves `/v1/completions` through `/chat/completions`."""
    return scenario.model(
        **{
            "model": "openai/gpt-4o-mini",
            "api_base": f"{provider.url}/v1",
            "api_key": PROVIDER_KEY,
            "input_cost_per_token": INPUT_RATE,
            "output_cost_per_token": OUTPUT_RATE,
            **parameters,
        }
    )


def cohere_rerank_model(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    """A `cohere/rerank-v3.5` deployment priced at RERANK_QUERY_RATE per search unit that calls the peer."""
    return scenario.model(
        model_info={"mode": "rerank"},
        **{
            "model": "cohere/rerank-v3.5",
            "api_base": provider.url,
            "api_key": PROVIDER_KEY,
            "input_cost_per_query": RERANK_QUERY_RATE,
            **parameters,
        },
    )


def moderation_model(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    return scenario.model(
        model_info={"mode": "moderation"},
        **{
            "model": "openai/omni-moderation-latest",
            "api_base": f"{provider.url}/v1",
            "api_key": PROVIDER_KEY,
            **parameters,
        },
    )


def marker() -> str:
    """A unique string, so the shared gateway's response cache never answers a request by accident."""
    return f"dec1-textapis-{uuid.uuid4().hex}"


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
        "query": marker() if query is None else query,
        "documents": list(DOCUMENTS) if documents is None else documents,
        **extra,
    }
    return gateway.request("POST", path, body, key=key, headers=headers)


def complete(
    gateway: Gateway,
    model: str,
    prompt: JsonValue = None,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    path: str = "/v1/completions",
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": model, "prompt": marker() if prompt is None else prompt, **extra}
    return gateway.request("POST", path, body, key=key, headers=headers)


def moderate(
    gateway: Gateway,
    model: str | None,
    value: JsonValue = None,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    path: str = "/v1/moderations",
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        **({} if model is None else {"model": model}),
        "input": marker() if value is None else value,
    }
    return gateway.request("POST", path, body, key=key, headers=headers)


def stream_text(
    gateway: Gateway, model: str, *, key: str | None = None, prompt: str | None = None, **extra: JsonValue
) -> tuple[httpx.Headers, tuple[str, ...]]:
    """Send a streaming `/v1/completions` request and return the response headers and every non-empty line."""
    body: Final[dict[str, JsonValue]] = {"model": model, "prompt": prompt or marker(), "stream": True, **extra}
    with gateway.client.stream(
        "POST", "/v1/completions", json=body, headers={"Authorization": f"Bearer {key or gateway.key}"}
    ) as response:
        assert response.status_code == 200, response.read().decode()
        return response.headers, tuple(line for line in response.iter_lines() if line)


def data_events(lines: Sequence[str]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data:").strip())
        for line in lines
        if line.startswith("data:") and not line.endswith("[DONE]")
    )


def served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def error_message(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def results_of(response: httpx.Response) -> tuple[tuple[JsonValue, JsonValue], ...]:
    """The `(index, relevance_score)` pairs of a rerank response, in the order returned."""
    results: Final = served(response)["results"]
    assert isinstance(results, list), response.text
    return tuple((object_value(entry)["index"], object_value(entry)["relevance_score"]) for entry in results)


def hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def number(value: JsonValue) -> float:
    assert isinstance(value, (int, float, str)) and not isinstance(value, bool), value
    return float(value)


def rows_of_key(gateway: Gateway, key: str, count: int) -> list[dict[str, JsonValue]]:
    """The key's spend log rows, oldest first, once exactly `count` of them have landed."""
    return eventually(
        lambda: gateway.rows(
            f'SELECT {ROW_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime"', (hashed(key),)
        ),
        lambda found: len(found) == count,
        seconds=LANDING_SECONDS,
    )


def key_spend(gateway: Gateway, key: str) -> float:
    rows: Final = gateway.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),))
    return number(rows[0]["spend"]) if rows else 0.0


def deployment(
    group: str,
    litellm_model: str,
    api_base: str,
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> dict[str, JsonValue]:
    """One `model_list` entry for `owned_rig`, with deployment id `dec1-<group>-<last part of the model>`."""
    return {
        "model_name": group,
        "litellm_params": {"model": litellm_model, "api_key": PROVIDER_KEY, "api_base": api_base, **params},
        "model_info": {"id": f"dec1-{group}-{litellm_model.rsplit('/', 1)[-1]}", **(model_info or {})},
    }


@dataclass
class Sink:
    """Every StandardLoggingPayload the `generic_api` callback delivered, kept across drains."""

    wire: Wire
    seen: list[dict[str, JsonValue]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def records(self) -> tuple[dict[str, JsonValue], ...]:
        with self.lock:
            for request in self.wire.drain():
                decoded = json.loads(request.body)
                self.seen.extend(decoded if isinstance(decoded, list) else [decoded])
            return tuple(self.seen)

    def event(self, call_id: str, status: str | None = None) -> dict[str, JsonValue]:
        """The single payload delivered for one `x-litellm-call-id`, optionally the one with `status`."""

        def matching() -> tuple[dict[str, JsonValue], ...]:
            return tuple(
                record
                for record in self.records()
                if record.get("litellm_call_id") == call_id and (status is None or record.get("status") == status)
            )

        found: Final = eventually(matching, lambda events: len(events) >= 1, seconds=30)
        assert len(found) == 1, [(event.get("id"), event.get("status")) for event in found]
        return found[0]


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    provider: Peer
    sink: Sink


@contextmanager
def owned_rig(
    directory: Path,
    models: Callable[[str], Sequence[Mapping[str, JsonValue]]],
    *,
    respond: Respond = by_model,
    router_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
) -> Iterator[Rig]:
    """One provider peer, one logging sink and one owned proxy whose `model_list` is `models(peer.url)`.

    The proxy has its own database and Redis. `generic_api` is always in `litellm_settings.callbacks`, pointed
    at the sink through GENERIC_LOGGER_ENDPOINT, and DEFAULT_FLUSH_INTERVAL_SECONDS=1 keeps batches short.
    Router defaults are `num_retries: 0` and `disable_cooldowns: true` unless `router_settings` overrides them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        provider: Final = stack.enter_context(peer(respond))
        sink_wire: Final = stack.enter_context(wire_server(lambda request: Reply()))
        settings: Final = dict(litellm_settings or {})
        config: Final = {
            "model_list": [dict(entry) for entry in models(provider.url)],
            "litellm_settings": {**settings, "callbacks": ["generic_api", *settings.get("callbacks", [])]},
            "general_settings": {
                "master_key": "os.environ/LITELLM_MASTER_KEY",
                "database_url": "os.environ/DATABASE_URL",
                "disable_model_info_refresh": True,
                "disable_spend_logs": False,
                "proxy_batch_write_at": 1,
                "proxy_batch_polling_interval": 1,
                **(general_settings or {}),
            },
            "router_settings": {"num_retries": 0, "disable_cooldowns": True, **(router_settings or {})},
        }
        path: Final = directory / f"dec1-textapis-{uuid.uuid4().hex[:8]}.yaml"
        path.write_text(yaml.safe_dump(config))
        overrides: Final = {
            "GENERIC_LOGGER_ENDPOINT": sink_wire.url,
            "DEFAULT_FLUSH_INTERVAL_SECONDS": "1",
            "STORE_MODEL_IN_DB": "False",
            **(environment or {}),
        }
        gateway: Final = stack.enter_context(owned_proxy(shared, directory, overrides, config=path))
        yield Rig(gateway, provider, Sink(sink_wire))


def new_call_id() -> str:
    return str(uuid.uuid4())


def with_call_id(call_id: str) -> dict[str, str]:
    return {"x-litellm-call-id": call_id}
