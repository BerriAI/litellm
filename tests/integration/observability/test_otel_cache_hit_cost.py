import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, list_value, object_value
from integration._support.database import read_rows
from integration._support.otlp_sink import Span, owned_sinks, recorded_spans
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter
from redis import Redis

pytestmark: Final = pytest.mark.shares_worker_state

INPUT_PRICE: Final = 0.001
OUTPUT_PRICE: Final = 0.002
PROMPT_TOKENS: Final = 7
COMPLETION_TOKENS: Final = 2
CHAT_PRICE: Final = PROMPT_TOKENS * INPUT_PRICE + COMPLETION_TOKENS * OUTPUT_PRICE
EMBEDDING_PRICE: Final = PROMPT_TOKENS * INPUT_PRICE
CLIENT: Final = 3
COST_HEADERS: Final = ("x-litellm-response-cost", "x-litellm-response-cost-input", "x-litellm-response-cost-output")
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
MARKER: Final = re.compile(rb"cache-hit-cost-[0-9a-f]{32}")


def _sse(frames: tuple[Mapping[str, JsonValue], ...]) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames), b"data: [DONE]\n\n"),
    )


def _answer(marker: str) -> str:
    return "answer to " + marker


def _embedding_value(marker: str) -> float:
    return float(int(marker[-12:], 16))


def _chat_reply(stream: bool, marker: str) -> Reply:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex
    message: Final = {"role": "assistant", "content": _answer(marker)}
    usage: Final = {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
    }
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": usage,
                }
            ).encode()
        )
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return _sse(
        (
            {**chunk, "choices": [{"index": 0, "delta": message}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {**chunk, "choices": [], "usage": usage},
        )
    )


def _responses_reply(marker: str) -> Reply:
    identity: Final = "resp_" + uuid.uuid4().hex
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "id": "msg_" + identity,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": _answer(marker), "annotations": []}],
                    }
                ],
                "usage": {
                    "input_tokens": PROMPT_TOKENS,
                    "output_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                },
            }
        ).encode()
    )


def _embedding_reply(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [_embedding_value(marker), 0.5, 0.75]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": PROMPT_TOKENS, "total_tokens": PROMPT_TOKENS},
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    found: Final = MARKER.search(request.body)
    assert found is not None, request.body
    marker: Final = found.group().decode()
    if request.target.endswith("/embeddings"):
        return _embedding_reply(marker)
    if request.target.endswith("/responses"):
        return _responses_reply(marker)
    return _chat_reply(JSON_OBJECT.validate_json(request.body).get("stream") is True, marker)


@dataclass(frozen=True, slots=True)
class Route:
    path: str
    stream: bool
    price: float
    body: Callable[[str, str], Mapping[str, JsonValue]]
    cached: Callable[[str], str] = _answer


ROUTES: Final[Mapping[str, Route]] = MappingProxyType(
    {
        "chat": Route(
            "/v1/chat/completions",
            False,
            CHAT_PRICE,
            lambda model, marker: {"model": model, "messages": [{"role": "user", "content": marker}]},
        ),
        "chat_stream": Route(
            "/v1/chat/completions",
            True,
            CHAT_PRICE,
            lambda model, marker: {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ),
        "messages": Route(
            "/v1/messages",
            False,
            CHAT_PRICE,
            lambda model, marker: {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]},
        ),
        "responses": Route("/v1/responses", False, CHAT_PRICE, lambda model, marker: {"model": model, "input": marker}),
        "embeddings": Route(
            "/v1/embeddings",
            False,
            EMBEDDING_PRICE,
            lambda model, marker: {"model": model, "input": marker},
            lambda marker: repr(_embedding_value(marker)),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Served:
    call_id: str
    headers: httpx.Headers


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    sink: str
    chat_model: str
    embedding_model: str
    upstream: Wire
    otel_v2: bool

    def send(self, route: str, marker: str) -> Served:
        chosen: Final = ROUTES[route]
        model: Final = self.embedding_model if route == "embeddings" else self.chat_model
        with self.proxy.client.stream(
            "POST",
            chosen.path,
            json=dict(chosen.body(model, marker)),
            headers={"Authorization": f"Bearer {self.proxy.key}"},
        ) as response:
            body: Final = response.read()
            assert response.status_code == 200, body
            return Served(response.headers["x-litellm-call-id"], response.headers)

    def upstream_calls(self, marker: str) -> int:
        return sum(1 for request in self.upstream.drain() if marker.encode() in request.body)

    def cost_span(self, call_id: str) -> Span:
        def matching() -> tuple[Span, ...]:
            return tuple(span for span in recorded_spans(self.sink)[1] if self._is_cost_span(span, call_id))

        found: Final = eventually(matching, lambda spans: len(spans) >= 1, seconds=30)
        assert len(found) == 1, found
        return found[0]

    def _is_cost_span(self, span: Span, call_id: str) -> bool:
        if span["attributes"].get("litellm.call_id") != call_id:
            return False
        if self.otel_v2:
            return span["kind"] == CLIENT and "gen_ai.operation.name" in span["attributes"]
        return span["name"] == "litellm_request"

    def total_cost(self, span: Span) -> float:
        return _number(span["attributes"].get("litellm.cost.total" if self.otel_v2 else "gen_ai.cost.total_cost"))


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float, str)) and not isinstance(value, bool), value
    return float(value)


def _config(provider: Wire, sink: str, chat_model: str, embedding_model: str) -> dict[str, JsonValue]:
    config: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    priced: Final = {"input_cost_per_token": INPUT_PRICE, "output_cost_per_token": OUTPUT_PRICE}
    return {
        **config,
        "model_list": [
            *list_value(config.get("model_list") or []),
            {
                "model_name": chat_model,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": provider.url + "/v1",
                    **priced,
                },
            },
            {
                "model_name": embedding_model,
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": "integration-provider-key",
                    "api_base": provider.url + "/v1",
                    **priced,
                },
            },
        ],
        "litellm_settings": {**object_value(config["litellm_settings"]), "callbacks": ["otel"]},
        "callback_settings": {"otel": {"exporter": "http/json", "endpoint": sink, "use_simple_processor": True}},
    }


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_provider) as wire:
        yield wire


def _rig(provider: Wire, directory: Path, otel_v2: bool) -> Iterator[Rig]:
    chat_model: Final = "cachecost-chat-" + uuid.uuid4().hex[:12]
    embedding_model: Final = "cachecost-embed-" + uuid.uuid4().hex[:12]
    with owned_sinks(directory / "sinks") as sinks:
        path: Final = directory / f"cache-hit-cost-{uuid.uuid4().hex}.yaml"
        path.write_text(yaml.safe_dump(_config(provider, sinks.operator, chat_model, embedding_model)))
        v1_exporter: Final = {"OTEL_EXPORTER": "http/json", "OTEL_ENDPOINT": sinks.operator}
        exporter: Final = {"LITELLM_OTEL_V2": "1"} if otel_v2 else v1_exporter
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway,
                directory,
                exporter,
                config=path,
                remove_environment=() if otel_v2 else ("LITELLM_OTEL_V2",),
                workers=2,
            ) as owned,
        ):
            yield Rig(owned.gateway, sinks.operator, chat_model, embedding_model, provider, otel_v2)


@pytest.fixture(scope="module")
def v1(provider: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from _rig(provider, tmp_path_factory.mktemp("cache-hit-cost-v1"), otel_v2=False)


@pytest.fixture(scope="module")
def v2(provider: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from _rig(provider, tmp_path_factory.mktemp("cache-hit-cost-v2"), otel_v2=True)


def _spend_row(call_id: str) -> Mapping[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, cache_hit, metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (call_id,)
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _breakdown_total(row: Mapping[str, JsonValue]) -> float:
    return _number(object_value(object_value(row["metadata"])["cost_breakdown"])["total_cost"])


def _cache_holds(needle: str) -> bool:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as store:
        return any(needle.encode() in _bytes(store.get(key)) for key in store.scan_iter(count=1000, _type="STRING"))


def _bytes(value: object) -> bytes:
    return value if isinstance(value, bytes) else b""


def _miss_then_hit(rig: Rig, route: str) -> tuple[Served, Served]:
    marker: Final = "cache-hit-cost-" + uuid.uuid4().hex
    miss: Final = rig.send(route, marker)
    eventually(lambda: _cache_holds(ROUTES[route].cached(marker)), bool, seconds=30)
    hit: Final = rig.send(route, marker)
    assert rig.upstream_calls(marker) == 1
    return miss, hit


@pytest.mark.parametrize("route", tuple(ROUTES))
def test_a_v1_cache_hit_exports_zero_cost_and_answers_zero_cost_headers(v1: Rig, route: str) -> None:
    price: Final = ROUTES[route].price
    miss, hit = _miss_then_hit(v1, route)

    hit_span: Final = v1.cost_span(hit.call_id)
    assert v1.total_cost(hit_span) == 0, hit_span
    assert hit_span["attributes"].get("litellm.cache_hit") is True, hit_span
    assert _number(hit_span["attributes"].get("litellm.saved_cache_cost")) == pytest.approx(price), hit_span

    assert {name: hit.headers.get(name) for name in COST_HEADERS} == dict.fromkeys(COST_HEADERS, "0.0"), hit.headers

    hit_row: Final = _spend_row(hit.call_id)
    assert (_number(hit_row["spend"]), hit_row["cache_hit"]) == (0, "True"), hit_row
    assert _breakdown_total(hit_row) == 0, hit_row
    assert _number(_spend_row(miss.call_id)["spend"]) == pytest.approx(price)

    miss_span: Final = v1.cost_span(miss.call_id)
    assert v1.total_cost(miss_span) == pytest.approx(price), miss_span
    assert miss_span["attributes"].get("litellm.cache_hit") is False, miss_span


@pytest.mark.parametrize("route", tuple(ROUTES))
def test_a_v2_cache_hit_exports_an_llm_span_at_zero_cost(v2: Rig, route: str) -> None:
    price: Final = ROUTES[route].price
    miss, hit = _miss_then_hit(v2, route)

    hit_span: Final = v2.cost_span(hit.call_id)
    assert v2.total_cost(hit_span) == 0, hit_span
    assert hit_span["attributes"].get("litellm.cache_hit") is True, hit_span
    assert _number(hit_span["attributes"].get("litellm.saved_cache_cost")) == pytest.approx(price), hit_span

    miss_span: Final = v2.cost_span(miss.call_id)
    assert v2.total_cost(miss_span) == pytest.approx(price), miss_span
    assert miss_span["attributes"].get("litellm.cache_hit") is False, miss_span
