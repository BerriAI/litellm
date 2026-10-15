import base64
import itertools
import json
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.responses_vendor import response_identities
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.responses import ResponseCompletedEvent
from pydantic import JsonValue, TypeAdapter


_BACKEND: Final = "gemini-3.1-flash-image"
_PROJECT: Final = "scripted-project"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/scripted/publishers/google/models/{_BACKEND}"
_GENERATE: Final = f"{_MODEL_PATH}:generateContent"
_STREAM: Final = f"{_MODEL_PATH}:streamGenerateContent?alt=sse"
_PROMPT_TOKENS: Final = 7
_IMAGE_TOKENS: Final = 1120
_CACHED_TOKENS: Final = 4
_UPLIFT: Final = 1.1
_UPLIFT_SOURCE: Final = (
    "Non-global column of https://cloud.google.com/vertex-ai/generative-ai/pricing for Gemini 3.1 Flash Image, "
    "read 2026-10-05"
)
_NO_CACHE: Final = {"cache": {"no-cache": True}}
_ONE_PIXEL_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ROW_QUERY: Final = 'SELECT request_id, spend, status, cache_hit FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime", request_id'


@dataclass(frozen=True, slots=True)
class _Location:
    configured: str | None
    present: bool
    uplifted: bool


_LOCATIONS: Final = {
    "global": _Location("global", True, False),
    "us": _Location("us", True, True),
    "europe-west4": _Location("europe-west4", True, True),
    "empty": _Location("", True, True),
    "null": _Location(None, True, True),
    "missing": _Location(None, False, True),
}
_GLOBAL: Final = _LOCATIONS["global"]
_US: Final = _LOCATIONS["us"]


def _usage(*, cached_tokens: int = 0) -> dict[str, JsonValue]:
    return {
        "promptTokenCount": _PROMPT_TOKENS,
        "candidatesTokenCount": _IMAGE_TOKENS,
        "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": _IMAGE_TOKENS}],
        "totalTokenCount": _PROMPT_TOKENS + _IMAGE_TOKENS,
        **({"cachedContentTokenCount": cached_tokens} if cached_tokens else {}),
    }


def _generated(usage: dict[str, JsonValue] | None) -> dict[str, JsonValue]:
    return {
        "responseId": f"scripted-{uuid.uuid4().hex}",
        "modelVersion": _BACKEND,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "scripted"}, {"inlineData": {"mimeType": "image/png", "data": "aW1n"}}],
                },
                "finishReason": "STOP",
            }
        ],
        **({"usageMetadata": usage} if usage is not None else {}),
    }


def _responder(
    usage: dict[str, JsonValue] | None = None, headers: dict[str, str] | None = None
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = json.dumps(_generated(_usage() if usage is None else usage))
        if request.target == _STREAM:
            return Reply(
                content_type="text/event-stream", chunks=(f"data: {body}\n\n".encode(),), headers=headers or {}
            )
        assert request.target == _GENERATE, request.target
        return Reply(body=body.encode(), headers=headers or {})

    return respond


_RESPOND: Final = _responder()


def _prompt() -> str:
    return f"a red square on a white background {uuid.uuid4().hex}"


def _deployment(scenario: Scenario, wire: Wire, location: _Location, *, images: bool = False) -> str:
    return scenario.model(
        model=f"vertex_ai/{_BACKEND}",
        api_base=f"{wire.url}{_GENERATE if images else _MODEL_PATH}",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_credentials=service_account_json(_PROJECT, scenario.gateway.upstream_url.rstrip("/")),
        **({"vertex_location": location.configured} if location.present else {}),
    )


def _published_prices(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    def listed() -> tuple[dict[str, JsonValue], ...]:
        entries: Final = gateway.get("/model/info")["data"]
        assert isinstance(entries, list), entries
        return tuple(_JSON_OBJECT.validate_python(entry) for entry in entries if entry["model_name"] == model)

    found: Final = eventually(listed, lambda entries: len(entries) == 1, seconds=30)
    return _JSON_OBJECT.validate_python(found[0]["model_info"])


def _rate(prices: dict[str, JsonValue], key: str) -> float:
    rate: Final = prices.get(key)
    assert isinstance(rate, int | float), (key, rate)
    return float(rate)


def _standard_cost(prices: dict[str, JsonValue]) -> float:
    return _PROMPT_TOKENS * _rate(prices, "input_cost_per_token") + _IMAGE_TOKENS * _rate(
        prices, "output_cost_per_image_token"
    )


def _uplift(prices: dict[str, JsonValue], location: _Location) -> float:
    if not location.uplifted:
        return 1.0
    assert prices.get("regional_endpoint_uplift_multiplier") == _UPLIFT, (_UPLIFT_SOURCE, prices)
    return _UPLIFT


def _expected(gateway: Gateway, model: str, location: _Location) -> float:
    prices: Final = _published_prices(gateway, model)
    return _standard_cost(prices) * _uplift(prices, location)


def _digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: read_rows(_ROW_QUERY, (_digest(key),)), lambda found: len(found) >= count, seconds=70
    )
    assert len(rows) == count, rows
    return rows


def _single_row(key: str, identities: tuple[str, ...]) -> dict[str, JsonValue]:
    row: Final = _rows(key, 1)[0]
    assert row["request_id"] in identities, (row, identities)
    assert row["status"] == "success", row
    return row


def _spend(row: dict[str, JsonValue]) -> float:
    return float(str(row["spend"]))


def _responses_identities(client_id: str) -> tuple[str, ...]:
    return (client_id, *response_identities(client_id))


def _only_generate_calls(wire: Wire, count: int, *, target: str = _GENERATE) -> tuple[Request, ...]:
    received: Final = wire.drain()
    assert [request.target for request in received] == [target] * count, received
    return received


@pytest.mark.parametrize("location", tuple(_LOCATIONS))
def test_chat_completion_spend_follows_the_vertex_location(gateway: Gateway, location: str) -> None:
    placed: Final = _LOCATIONS[location]
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, placed)
        key: Final = scenario.key(models=[model])
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        prompt: Final = _prompt()
        raw: Final = client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": prompt}], extra_body=_NO_CACHE
        )
        completion: Final = raw.parse()
        assert completion.usage is not None and (
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        ) == (
            _PROMPT_TOKENS,
            _IMAGE_TOKENS,
        ), completion.usage
        expected: Final = _expected(gateway, model, placed)
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(expected), raw.headers
        sent: Final = _only_generate_calls(wire, 1)[0]
        assert prompt in sent.body.decode(), sent.body
        assert _spend(_single_row(key, (completion.id,))) == pytest.approx(expected)


async def test_async_chat_completion_on_a_regional_endpoint_is_uplifted(gateway: Gateway) -> None:
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _US)
        key: Final = scenario.key(models=[model])
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        raw: Final = await client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": _prompt()}], extra_body=_NO_CACHE
        )
        completion: Final = raw.parse()
        expected: Final = _expected(gateway, model, _US)
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(expected), raw.headers
        _only_generate_calls(wire, 1)
        assert _spend(_single_row(key, (completion.id,))) == pytest.approx(expected)


@pytest.mark.parametrize("location", ("global", "us"))
def test_streamed_chat_completion_spend_follows_the_vertex_location(gateway: Gateway, location: str) -> None:
    placed: Final = _LOCATIONS[location]
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, placed)
        key: Final = scenario.key(models=[model])
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": _prompt()}],
                "stream": True,
                "stream_options": {"include_usage": True},
                **_NO_CACHE,
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
        ) as response:
            assert response.status_code == 200, response.read()
            lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
        assert lines[-1] == "data: [DONE]", lines[-2:]
        chunks: Final = tuple(_JSON_OBJECT.validate_json(line.removeprefix("data: ")) for line in lines[:-1])
        identities: Final = frozenset(str(chunk["id"]) for chunk in chunks)
        assert len(identities) == 1, identities
        usages: Final = tuple(chunk["usage"] for chunk in chunks if isinstance(chunk.get("usage"), dict))
        assert usages and usages[-1]["prompt_tokens"] == _PROMPT_TOKENS, usages
        assert usages[-1]["completion_tokens"] == _IMAGE_TOKENS, usages
        _only_generate_calls(wire, 1, target=_STREAM)
        expected: Final = _expected(gateway, model, placed)
        assert _spend(_single_row(key, tuple(identities))) == pytest.approx(expected)


@pytest.mark.parametrize("location", ("global", "us"))
def test_messages_spend_follows_the_vertex_location(gateway: Gateway, location: str) -> None:
    placed: Final = _LOCATIONS[location]
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, placed)
        key: Final = scenario.key(models=[model])
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=key, max_retries=0)
        raw: Final = client.messages.with_raw_response.create(
            model=model, max_tokens=2048, messages=[{"role": "user", "content": _prompt()}], extra_body=_NO_CACHE
        )
        message: Final = raw.parse()
        assert (message.usage.input_tokens, message.usage.output_tokens) == (_PROMPT_TOKENS, _IMAGE_TOKENS), message
        _only_generate_calls(wire, 1)
        assert _spend(_single_row(key, (message.id,))) == pytest.approx(_expected(gateway, model, placed))


def test_streamed_messages_on_a_regional_endpoint_are_uplifted(gateway: Gateway) -> None:
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _US)
        key: Final = scenario.key(models=[model])
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=key, max_retries=0)
        with client.messages.stream(
            model=model, max_tokens=2048, messages=[{"role": "user", "content": _prompt()}], extra_body=_NO_CACHE
        ) as stream:
            message: Final = stream.get_final_message()
        assert (message.usage.input_tokens, message.usage.output_tokens) == (_PROMPT_TOKENS, _IMAGE_TOKENS), message
        _only_generate_calls(wire, 1, target=_STREAM)
        expected: Final = _expected(gateway, model, _US)
        assert _spend(_single_row(key, (message.id,))) == pytest.approx(expected)


@pytest.mark.parametrize("location", ("global", "us"))
def test_responses_spend_follows_the_vertex_location(gateway: Gateway, location: str) -> None:
    placed: Final = _LOCATIONS[location]
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, placed)
        key: Final = scenario.key(models=[model])
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        raw: Final = client.responses.with_raw_response.create(model=model, input=_prompt(), extra_body=_NO_CACHE)
        response: Final = raw.parse()
        assert response.usage is not None and (response.usage.input_tokens, response.usage.output_tokens) == (
            _PROMPT_TOKENS,
            _IMAGE_TOKENS,
        ), response.usage
        expected: Final = _expected(gateway, model, placed)
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(expected), raw.headers
        _only_generate_calls(wire, 1)
        assert _spend(_single_row(key, _responses_identities(response.id))) == pytest.approx(expected)


async def test_streamed_responses_on_a_regional_endpoint_are_uplifted(gateway: Gateway) -> None:
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _US)
        key: Final = scenario.key(models=[model])
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        stream: Final = await client.responses.create(model=model, input=_prompt(), stream=True, extra_body=_NO_CACHE)
        events: Final = [event async for event in stream]
        completed: Final = events[-1]
        assert isinstance(completed, ResponseCompletedEvent), events
        usage: Final = completed.response.usage
        assert usage is not None and (usage.input_tokens, usage.output_tokens) == (_PROMPT_TOKENS, _IMAGE_TOKENS), usage
        _only_generate_calls(wire, 1, target=_STREAM)
        expected: Final = _expected(gateway, model, _US)
        assert _spend(_single_row(key, _responses_identities(completed.response.id))) == pytest.approx(expected)


@pytest.mark.parametrize("location", ("global", "us"))
def test_image_generation_spend_follows_the_vertex_location(gateway: Gateway, location: str) -> None:
    placed: Final = _LOCATIONS[location]
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, placed, images=True)
        key: Final = scenario.key(models=[model])
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        raw: Final = client.images.with_raw_response.generate(model=model, prompt=_prompt(), extra_body=_NO_CACHE)
        images: Final = raw.parse()
        assert images.data is not None and [image.b64_json for image in images.data] == ["aW1n"], images
        assert images.usage is not None and (images.usage.input_tokens, images.usage.output_tokens) == (
            _PROMPT_TOKENS,
            _IMAGE_TOKENS,
        ), images.usage
        expected: Final = _expected(gateway, model, placed)
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(expected), raw.headers
        _only_generate_calls(wire, 1)
        assert _spend(_single_row(key, (raw.headers["x-litellm-call-id"],))) == pytest.approx(expected)


async def test_async_image_generation_on_a_regional_endpoint_is_uplifted(gateway: Gateway) -> None:
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _US, images=True)
        key: Final = scenario.key(models=[model])
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        raw: Final = await client.images.with_raw_response.generate(model=model, prompt=_prompt(), extra_body=_NO_CACHE)
        images: Final = raw.parse()
        assert images.data is not None and len(images.data) == 1, images
        expected: Final = _expected(gateway, model, _US)
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(expected), raw.headers
        _only_generate_calls(wire, 1)
        assert _spend(_single_row(key, (raw.headers["x-litellm-call-id"],))) == pytest.approx(expected)


@pytest.mark.parametrize("location", ("global", "us"))
def test_image_generation_without_usage_charges_the_flat_image_price(gateway: Gateway, location: str) -> None:
    placed: Final = _LOCATIONS[location]
    with wire_server(_responder(usage={})) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, placed, images=True)
        key: Final = scenario.key(models=[model])
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        raw: Final = client.images.with_raw_response.generate(model=model, prompt=_prompt(), extra_body=_NO_CACHE)
        images: Final = raw.parse()
        assert images.data is not None and len(images.data) == 1, images
        prices: Final = _published_prices(gateway, model)
        expected: Final = _rate(prices, "output_cost_per_image") * _uplift(prices, placed)
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(expected), raw.headers
        _only_generate_calls(wire, 1)
        assert _spend(_single_row(key, (raw.headers["x-litellm-call-id"],))) == pytest.approx(expected)


@pytest.mark.parametrize("location", ("global", "us"))
def test_image_edit_spend_follows_the_vertex_location(gateway: Gateway, location: str) -> None:
    placed: Final = _LOCATIONS[location]
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, placed, images=True)
        key: Final = scenario.key(models=[model])
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        prompt: Final = _prompt()
        raw: Final = client.images.with_raw_response.edit(
            model=model, image=("square.png", _ONE_PIXEL_PNG, "image/png"), prompt=prompt
        )
        images: Final = raw.parse()
        assert images.data is not None and [image.b64_json for image in images.data] == ["aW1n"], images
        prices: Final = _published_prices(gateway, model)
        expected: Final = _rate(prices, "output_cost_per_image") * _uplift(prices, placed)
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(expected), raw.headers
        (sent,) = _only_generate_calls(wire, 1)
        forwarded: Final = sent.body.decode()
        assert prompt in forwarded and base64.b64encode(_ONE_PIXEL_PNG).decode() in forwarded, forwarded[:300]
        assert _spend(_single_row(key, (raw.headers["x-litellm-call-id"],))) == pytest.approx(expected)


def _chat_spend(gateway: Gateway, scenario: Scenario, wire: Wire, location: _Location) -> tuple[float, str]:
    model: Final = _deployment(scenario, wire, location)
    key: Final = scenario.key(models=[model])
    client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
    completion: Final = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": _prompt()}], extra_body=_NO_CACHE
    )
    _only_generate_calls(wire, 1)
    return _spend(_single_row(key, (completion.id,))), model


def test_cached_prompt_tokens_on_a_regional_endpoint_are_uplifted_with_the_rest(gateway: Gateway) -> None:
    with wire_server(_responder(usage=_usage(cached_tokens=_CACHED_TOKENS))) as wire, gateway.scenario() as scenario:
        global_spend, global_model = _chat_spend(gateway, scenario, wire, _GLOBAL)
        regional_spend, regional_model = _chat_spend(gateway, scenario, wire, _US)
        prices: Final = _published_prices(gateway, global_model)
        assert global_spend == pytest.approx(
            (_PROMPT_TOKENS - _CACHED_TOKENS) * _rate(prices, "input_cost_per_token")
            + _CACHED_TOKENS * _rate(prices, "cache_read_input_token_cost")
            + _IMAGE_TOKENS * _rate(prices, "output_cost_per_image_token")
        )
        assert regional_spend == pytest.approx(global_spend * _uplift(_published_prices(gateway, regional_model), _US))


def test_flex_tier_on_a_regional_endpoint_is_uplifted_with_the_rest(gateway: Gateway) -> None:
    with wire_server(_responder(headers={"x-gemini-service-tier": "flex"})) as wire, gateway.scenario() as scenario:
        global_spend, global_model = _chat_spend(gateway, scenario, wire, _GLOBAL)
        regional_spend, regional_model = _chat_spend(gateway, scenario, wire, _US)
        prices: Final = _published_prices(gateway, global_model)
        assert global_spend == pytest.approx(
            _PROMPT_TOKENS * _rate(prices, "input_cost_per_token_flex")
            + _IMAGE_TOKENS * _rate(prices, "output_cost_per_image_token")
        )
        assert global_spend < _standard_cost(prices)
        assert regional_spend == pytest.approx(global_spend * _uplift(_published_prices(gateway, regional_model), _US))


def test_an_uppercase_vertex_location_is_refused_before_any_provider_call(gateway: Gateway) -> None:
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _Location("GLOBAL", True, False))
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _prompt()}], **_NO_CACHE},
            key=key,
        )
        assert response.status_code == 500, response.text
        assert "Invalid vertex_location format" in response.text, response.text
        assert wire.drain() == (), "the location validator runs before any provider call"
        row: Final = _rows(key, 1)[0]
        assert row["status"] == "failure" and _spend(row) == 0, row
        assert row["request_id"] == response.headers["x-litellm-call-id"], (row, dict(response.headers))


def test_a_response_cache_hit_on_a_regional_endpoint_is_not_charged(gateway: Gateway) -> None:
    with wire_server(_RESPOND) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _US)
        key: Final = scenario.key(models=[model])
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        messages: Final = [{"role": "user", "content": _prompt()}]
        first: Final = client.chat.completions.create(model=model, messages=messages)
        served_from_cache: Final = client.chat.completions.create(model=model, messages=messages)
        assert served_from_cache.id == first.id, (served_from_cache.id, first.id)
        _only_generate_calls(wire, 1)
        expected: Final = _expected(gateway, model, _US)
        rows: Final = _rows(key, 2)
        charged: Final = tuple(row for row in rows if row["cache_hit"] != "True")
        assert len(charged) == 1 and charged[0]["request_id"] == first.id, rows
        assert _spend(charged[0]) == pytest.approx(expected), rows
        hits: Final = tuple(row for row in rows if row["cache_hit"] == "True")
        assert len(hits) == 1 and str(hits[0]["request_id"]).startswith(f"{first.id}_cache_hit"), rows
        assert _spend(hits[0]) == 0, hits


@pytest.mark.parametrize(
    ("reply", "status"),
    ((Reply(drop_connection=True), 500), (Reply(body=b"not json"), 422)),
    ids=("dropped", "garbage"),
)
def test_a_provider_failure_on_a_regional_endpoint_is_not_charged(gateway: Gateway, reply: Reply, status: int) -> None:
    with wire_server(lambda request: reply) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _US)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _prompt()}], **_NO_CACHE},
            key=key,
        )
        assert response.status_code == status, response.text
        assert "error" in response.json(), response.text
        _only_generate_calls(wire, 1)
        row: Final = _rows(key, 1)[0]
        assert row["status"] == "failure" and _spend(row) == 0, row
        assert row["request_id"] == response.headers["x-litellm-call-id"], (row, dict(response.headers))


@dataclass(frozen=True, slots=True)
class _Outcome:
    kind: str
    status: int
    identities: tuple[str, ...]
    expected: float


@dataclass(frozen=True, slots=True)
class _Job:
    kind: str
    location: str
    copy: int
    prompt: str

    @property
    def dropped(self) -> bool:
        return self.copy == 1 and self.kind in _DROPPABLE_KINDS


_BURST_KINDS: Final = ("chat", "chat_stream", "messages", "messages_stream", "responses", "images")
_DROPPABLE_KINDS: Final = frozenset(_BURST_KINDS) - {"messages_stream"}


def _burst_call(
    client: httpx.Client, key: str, job: _Job, chat_model: str, image_model: str, expected: float
) -> _Outcome:
    headers: Final = {"Authorization": f"Bearer {key}"}
    kind: Final = job.kind
    if kind == "images":
        response: Final = client.post(
            "/v1/images/generations", json={"model": image_model, "prompt": job.prompt, **_NO_CACHE}, headers=headers
        )
        return _Outcome(kind, response.status_code, (response.headers["x-litellm-call-id"],), expected)
    if kind == "responses":
        answered: Final = client.post(
            "/v1/responses", json={"model": chat_model, "input": job.prompt, **_NO_CACHE}, headers=headers
        )
        if answered.status_code != 200:
            return _Outcome(kind, answered.status_code, (answered.headers["x-litellm-call-id"],), expected)
        return _Outcome(kind, 200, _responses_identities(str(answered.json()["id"])), expected)
    path: Final = "/v1/messages" if kind.startswith("messages") else "/v1/chat/completions"
    body: Final = {
        "model": chat_model,
        "messages": [{"role": "user", "content": job.prompt}],
        "stream": kind.endswith("_stream"),
        **({"max_tokens": 2048} if kind.startswith("messages") else {}),
        **_NO_CACHE,
    }
    with client.stream("POST", path, json=body, headers=headers) as streamed:
        if streamed.status_code != 200:
            streamed.read()
            return _Outcome(kind, streamed.status_code, (streamed.headers["x-litellm-call-id"],), expected)
        if not kind.endswith("_stream"):
            return _Outcome(kind, 200, (str(_JSON_OBJECT.validate_json(streamed.read())["id"]),), expected)
        frames: Final = tuple(
            line.removeprefix("data: ") for line in streamed.iter_lines() if line.startswith("data: ")
        )
    events: Final = tuple(_JSON_OBJECT.validate_json(frame) for frame in frames if frame != "[DONE]")
    if kind == "messages_stream":
        started: Final = tuple(event for event in events if event.get("type") == "message_start")
        assert len(started) == 1, events
        message: Final = started[0]["message"]
        assert isinstance(message, dict), message
        return _Outcome(kind, 200, (str(message["id"]),), expected)
    identities: Final = frozenset(str(event["id"]) for event in events)
    assert len(identities) == 1, identities
    return _Outcome(kind, 200, tuple(identities), expected)


def test_a_provider_outage_mid_burst_charges_every_served_request_once(gateway: Gateway) -> None:
    jobs: Final = tuple(
        _Job(kind, name, copy, _prompt())
        for kind, name, copy in itertools.product(_BURST_KINDS, ("global", "us"), (0, 1))
    )
    dropped_prompts: Final = frozenset(job.prompt for job in jobs if job.dropped)

    def respond(request: Request) -> Reply:
        body: Final = request.body.decode()
        if any(prompt in body for prompt in dropped_prompts):
            return Reply(drop_connection=True)
        return _RESPOND(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        chat_models: Final = {name: _deployment(scenario, wire, _LOCATIONS[name]) for name in ("global", "us")}
        image_models: Final = {
            name: _deployment(scenario, wire, _LOCATIONS[name], images=True) for name in ("global", "us")
        }
        key: Final = scenario.key(models=[*chat_models.values(), *image_models.values()])
        expected: Final = {name: _expected(gateway, chat_models[name], _LOCATIONS[name]) for name in ("global", "us")}

        def run(job: _Job) -> _Outcome:
            with httpx.Client(base_url=str(gateway.client.base_url), timeout=120) as client:
                return _burst_call(
                    client, key, job, chat_models[job.location], image_models[job.location], expected[job.location]
                )

        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            outcomes: Final = tuple(pool.map(run, jobs))
        assert len(wire.drain()) == len(jobs), outcomes
        assert len(dropped_prompts) == len(_DROPPABLE_KINDS) * 2, dropped_prompts
        rows: Final = _rows(key, len(jobs))
        by_identity: Final = {str(row["request_id"]): row for row in rows}
        assert len(by_identity) == len(jobs), rows
        for job, outcome in zip(jobs, outcomes, strict=True):
            if job.dropped:
                assert outcome.status >= 500, (job, outcome)
                failed: Final = by_identity[outcome.identities[0]]
                assert failed["status"] == "failure" and _spend(failed) == 0, (job, outcome, failed)
                continue
            assert outcome.status == 200, (job, outcome)
            matched: Final = tuple(identity for identity in outcome.identities if identity in by_identity)
            assert len(matched) == 1, (job, outcome, rows)
            assert by_identity[matched[0]]["status"] == "success", (job, outcome, by_identity[matched[0]])
            assert _spend(by_identity[matched[0]]) == pytest.approx(outcome.expected), (
                job,
                outcome,
                by_identity[matched[0]],
            )
