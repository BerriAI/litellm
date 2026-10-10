import asyncio
import base64
import json
import os
import re
import signal
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final
from urllib.parse import urlsplit

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_IMAGE_BACKEND: Final = "gemini-scripted-image"
_CHAT_BACKEND: Final = "gemini-scripted-chat"
_PROJECT: Final = "scripted-project"
_REGION: Final = "us-central1"
_PROMPT: Final = "a scripted lighthouse at dusk"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_REGION}/publishers/google/models"
_IMAGE_TARGET: Final = f"{_MODEL_PATH}/{_IMAGE_BACKEND}:generateContent"
_CHAT_TARGET: Final = f"{_MODEL_PATH}/{_CHAT_BACKEND}:generateContent"
_PNG_B64: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
_PNG: Final = base64.b64decode(_PNG_B64)
_GLOBAL_TOKEN_COST: Final = 12 * 2e-6 + 1290 * 4e-5
_REGIONAL_TOKEN_COST: Final = (12 * 2e-6 + 1290 * 4e-5) * 1.1
_GLOBAL_IMAGE_COST: Final = 0.05
_REGIONAL_IMAGE_COST: Final = 0.05 * 1.1
_REGIONAL_CHAT_COST: Final = (12 * 2e-6 + 7 * 4e-5) * 1.1
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_CONFIG_MODEL: Final = "vertex-image-chaos"
_BURST: Final = 30


def _image_body(*, images: int, usage: bool) -> bytes:
    parts: Final = [{"inlineData": {"mimeType": "image/png", "data": _PNG_B64}} for _ in range(images)]
    usage_metadata: Final = {
        "promptTokenCount": 12,
        "candidatesTokenCount": 1290,
        "totalTokenCount": 1302,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 12}],
    }
    return json.dumps(
        {
            "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
            **({"usageMetadata": usage_metadata} if usage else {}),
        }
    ).encode()


_CHAT_BODY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "scripted answer"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 7, "totalTokenCount": 19},
    }
).encode()


def _scripted(*, images: int = 1, usage: bool = True, delay: float = 0) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _IMAGE_TARGET), (request.method, request.target)
        assert _PROMPT in request.body.decode(), request.body
        if delay:
            time.sleep(delay)
        return Reply(body=_image_body(images=images, usage=usage))

    return respond


def _scripted_chat(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", _CHAT_TARGET), (request.method, request.target)
    assert request.headers["authorization"] == "Bearer scripted-token", request.headers
    return Reply(body=_CHAT_BODY)


def _deployment(
    gateway: Gateway,
    scenario: Scenario,
    wire: Wire,
    *,
    location: Mapping[str, JsonValue],
    multiplier: JsonValue = 1.1,
    backend: str = _IMAGE_BACKEND,
) -> str:
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_base=f"{wire.url}{_MODEL_PATH}/{backend}:generateContent",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        input_cost_per_token=2e-6,
        output_cost_per_token=4e-5,
        output_cost_per_image=0.05,
        **({} if multiplier is None else {"regional_endpoint_uplift_multiplier": multiplier}),
        **location,
    )


def _chat_deployment(gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"vertex_ai/{_CHAT_BACKEND}",
        api_base=f"{wire.url}{_MODEL_PATH}/{_CHAT_BACKEND}",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_REGION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        input_cost_per_token=2e-6,
        output_cost_per_token=4e-5,
        regional_endpoint_uplift_multiplier=1.1,
    )


def _spend_rows(call_ids: tuple[str, ...], *, seconds: float = 70) -> dict[str, float]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)', (list(call_ids),)
        ),
        lambda found: len(found) == len(call_ids),
        seconds=seconds,
    )
    assert len({str(row["request_id"]) for row in rows}) == len(rows), rows
    return {str(row["request_id"]): float(str(row["spend"])) for row in rows}


def _spend(call_id: str) -> float:
    return _spend_rows((call_id,))[call_id]


def _spend_by_litellm_call_id(call_id: str) -> float:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT spend FROM \"LiteLLM_SpendLogs\" WHERE metadata->>'litellm_call_id' = %s", (call_id,)
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return float(str(rows[0]["spend"]))


def _assert_image_response(response: httpx.Response, expected: float, *, images: int = 1) -> None:
    assert response.status_code == 200, response.text
    data: Final = _JSON_OBJECT.validate_json(response.content)["data"]
    assert isinstance(data, list) and len(data) == images, response.text
    assert all(object_value(item)["b64_json"] == _PNG_B64 for item in data), response.text
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected), response.headers


def _assert_images(response: httpx.Response, call_id: str, expected: float, *, images: int = 1) -> None:
    _assert_image_response(response, expected, images=images)
    assert _spend(call_id) == pytest.approx(expected)


def _assert_one_upstream_image_call(wire: Wire, *, prompt: str = _PROMPT) -> None:
    received: Final = wire.drain()
    assert len(received) == 1, [request.target for request in received]
    body: Final = _JSON_OBJECT.validate_json(received[0].body)
    contents: Final = body["contents"]
    assert isinstance(contents, list) and prompt in json.dumps(contents), body
    assert "IMAGE" in json.dumps(body["generationConfig"]), body


def _generate(
    gateway: Gateway, model: str, *, key: str | None = None, extra: Mapping[str, JsonValue] | None = None
) -> tuple[str, httpx.Response]:
    call_id: Final = uuid.uuid4().hex
    response: Final = gateway.request(
        "POST",
        "/v1/images/generations",
        {"model": model, "prompt": _PROMPT, "n": 1, **(extra or {})},
        key=key,
        headers={"x-litellm-call-id": call_id},
    )
    return call_id, response


def _edit(gateway: Gateway, model: str) -> tuple[str, httpx.Response]:
    call_id: Final = uuid.uuid4().hex
    response: Final = gateway.client.post(
        "/v1/images/edits",
        data={"model": model, "prompt": _PROMPT},
        files={"image": ("red.png", _PNG, "image/png")},
        headers={"Authorization": f"Bearer {gateway.key}", "x-litellm-call-id": call_id},
    )
    return call_id, response


@pytest.mark.parametrize(
    ("location", "multiplier", "usage", "images", "expected"),
    [
        pytest.param({"vertex_location": "global"}, 1.1, True, 1, _GLOBAL_TOKEN_COST, id="global-usage"),
        pytest.param({"vertex_location": _REGION}, 1.1, True, 1, _REGIONAL_TOKEN_COST, id="regional-usage"),
        pytest.param({"vertex_location": _REGION}, 1.1, False, 1, _REGIONAL_IMAGE_COST, id="regional-per-image"),
        pytest.param({"vertex_location": _REGION}, 1.1, False, 2, 2 * _REGIONAL_IMAGE_COST, id="regional-two-images"),
        pytest.param({"vertex_location": "GLOBAL"}, 1.1, True, 1, _GLOBAL_TOKEN_COST, id="upper-case-global"),
        pytest.param({"vertex_location": "us"}, 1.1, True, 1, _REGIONAL_TOKEN_COST, id="multi-region"),
        pytest.param({"vertex_location": _REGION}, None, True, 1, _GLOBAL_TOKEN_COST, id="regional-without-multiplier"),
        pytest.param({"vertex_location": _REGION}, "1.1", True, 1, _REGIONAL_TOKEN_COST, id="string-multiplier"),
        pytest.param({}, 1.1, True, 1, _REGIONAL_TOKEN_COST, id="missing-location-defaults-to-us-central1"),
        pytest.param(
            {"vertex_location": ""}, 1.1, True, 1, _REGIONAL_TOKEN_COST, id="empty-location-defaults-to-us-central1"
        ),
    ],
)
def test_image_generation_bills_the_deployment_location(
    gateway: Gateway,
    location: Mapping[str, JsonValue],
    multiplier: JsonValue,
    usage: bool,
    images: int,
    expected: float,
) -> None:
    with wire_server(_scripted(images=images, usage=usage)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, location=location, multiplier=multiplier)
        call_id, response = _generate(gateway, model, extra={"n": images})
        _assert_images(response, call_id, expected, images=images)
        _assert_one_upstream_image_call(wire)


@pytest.mark.parametrize(
    ("location", "usage", "expected"),
    [
        pytest.param("global", False, _GLOBAL_IMAGE_COST, id="global-edit"),
        pytest.param(_REGION, False, _REGIONAL_IMAGE_COST, id="regional-edit"),
        pytest.param(_REGION, True, _REGIONAL_IMAGE_COST, id="regional-edit-usage-is-per-image"),
    ],
)
def test_image_edit_bills_the_deployment_location(
    gateway: Gateway, location: str, usage: bool, expected: float
) -> None:
    with wire_server(_scripted(usage=usage)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": location})
        call_id, response = _edit(gateway, model)
        _assert_images(response, call_id, expected)
        received: Final = wire.drain()
        assert len(received) == 1, [request.target for request in received]
        assert _PNG_B64 in received[0].body.decode(), received[0].body[:200]


def test_openai_sdk_image_generation_bills_the_regional_uplift(gateway: Gateway) -> None:
    with wire_server(_scripted()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": _REGION})
        call_id: Final = uuid.uuid4().hex
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, timeout=60)
        raw: Final = client.images.with_raw_response.generate(
            model=model, prompt=_PROMPT, extra_headers={"x-litellm-call-id": call_id}
        )
        assert raw.status_code == 200, raw.text
        (image,) = raw.parse().data or ()
        assert image.b64_json == _PNG_B64
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(_REGIONAL_TOKEN_COST), raw.headers
        assert _spend(call_id) == pytest.approx(_REGIONAL_TOKEN_COST)
        _assert_one_upstream_image_call(wire)


async def test_async_openai_sdk_image_generation_keeps_global_flat(gateway: Gateway) -> None:
    with wire_server(_scripted()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": "global"})
        call_id: Final = uuid.uuid4().hex
        async with openai.AsyncOpenAI(
            base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, timeout=60
        ) as client:
            raw: Final = await client.images.with_raw_response.generate(
                model=model, prompt=_PROMPT, extra_headers={"x-litellm-call-id": call_id}
            )
        assert raw.status_code == 200, raw.text
        (image,) = raw.parse().data or ()
        assert image.b64_json == _PNG_B64
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(_GLOBAL_TOKEN_COST), raw.headers
        assert await asyncio.to_thread(_spend, call_id) == pytest.approx(_GLOBAL_TOKEN_COST)
        _assert_one_upstream_image_call(wire)


def test_openai_sdk_image_edit_bills_the_regional_uplift(gateway: Gateway) -> None:
    with wire_server(_scripted(usage=False)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": _REGION})
        call_id: Final = uuid.uuid4().hex
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, timeout=60)
        raw: Final = client.images.with_raw_response.edit(
            model=model,
            image=("red.png", _PNG, "image/png"),
            prompt=_PROMPT,
            extra_headers={"x-litellm-call-id": call_id},
        )
        assert raw.status_code == 200, raw.text
        (image,) = raw.parse().data or ()
        assert image.b64_json == _PNG_B64
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(_REGIONAL_IMAGE_COST), raw.headers
        assert _spend(call_id) == pytest.approx(_REGIONAL_IMAGE_COST)
        received: Final = wire.drain()
        assert len(received) == 1 and _PNG_B64 in received[0].body.decode(), received


def test_model_info_reads_back_the_multiplier_the_price_follows(gateway: Gateway) -> None:
    with wire_server(_scripted()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": _REGION})
        entries: Final = gateway.get("/model/info")["data"]
        assert isinstance(entries, list)
        (entry,) = (object_value(item) for item in entries if object_value(item)["model_name"] == model)
        info: Final = object_value(entry["model_info"])
        assert info["regional_endpoint_uplift_multiplier"] == 1.1, info
        assert object_value(entry["litellm_params"])["vertex_location"] == _REGION, entry


def test_invalid_multiplier_is_refused_at_registration(gateway: Gateway) -> None:
    with wire_server(_scripted()) as wire:
        refused: Final = gateway.request(
            "POST",
            "/model/new",
            {
                "model_name": f"integration-{uuid.uuid4().hex}",
                "litellm_params": {
                    "model": f"vertex_ai/{_IMAGE_BACKEND}",
                    "api_base": f"{wire.url}{_IMAGE_TARGET}",
                    "vertex_project": _PROJECT,
                    "vertex_location": _REGION,
                    "output_cost_per_image": 0.05,
                    "regional_endpoint_uplift_multiplier": "abc",
                },
                "model_info": {},
            },
        )
        assert refused.status_code == 422, refused.text
        assert "regional_endpoint_uplift_multiplier" in refused.text, refused.text
        assert wire.drain() == ()


def _body_without_created(response: httpx.Response) -> dict[str, object]:
    return {key: value for key, value in response.json().items() if key != "created"}


def test_identical_regional_generations_each_bill_the_uplift_once(gateway: Gateway) -> None:
    with wire_server(_scripted()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": _REGION})
        first_id, first = _generate(gateway, model)
        second_id, second = _generate(gateway, model)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert _body_without_created(first) == _body_without_created(second), (first.text, second.text)
        rows: Final = _spend_rows((first_id, second_id))
        assert rows == {first_id: pytest.approx(_REGIONAL_TOKEN_COST), second_id: pytest.approx(_REGIONAL_TOKEN_COST)}
        assert len(wire.drain()) == 2


@pytest.mark.parametrize(
    ("deployment_location", "extra", "expected"),
    [
        pytest.param(_REGION, {"vertex_location": "global"}, _GLOBAL_TOKEN_COST, id="body-global-wins-over-regional"),
        pytest.param("global", {"vertex_location": 123}, _REGIONAL_TOKEN_COST, id="body-int"),
        pytest.param("global", {"vertex_location": ["us-central1"]}, _REGIONAL_TOKEN_COST, id="body-list"),
        pytest.param("global", {"vertex_location": ""}, _REGIONAL_TOKEN_COST, id="body-empty"),
        pytest.param("global", {"vertex_location": "a" * 5000}, _REGIONAL_TOKEN_COST, id="body-5kb"),
        pytest.param(
            "global",
            {"vertex_location": "global", "vertex_ai_location": _REGION},
            _GLOBAL_TOKEN_COST,
            id="body-both-aliases",
        ),
    ],
)
def test_request_body_location_overrides_the_deployment(
    gateway: Gateway, deployment_location: str, extra: Mapping[str, JsonValue], expected: float
) -> None:
    with wire_server(_scripted()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": deployment_location})
        call_id, response = _generate(gateway, model, extra=extra)
        _assert_images(response, call_id, expected)
        _assert_one_upstream_image_call(wire)
        control_id, control = _generate(gateway, model)
        _assert_images(
            control, control_id, _GLOBAL_TOKEN_COST if deployment_location == "global" else _REGIONAL_TOKEN_COST
        )


def test_stream_flag_on_image_generation_prices_the_regional_uplift(gateway: Gateway) -> None:
    with wire_server(_scripted()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": _REGION})
        _, response = _generate(gateway, model, extra={"stream": True})
        _assert_image_response(response, _REGIONAL_TOKEN_COST)
        _assert_one_upstream_image_call(wire)


def test_openai_style_image_deployment_is_untouched(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/images/generations"), request
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _PNG_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3", api_base=wire.url, api_key="synthetic-image-key", output_cost_per_image=0.25
        )
        call_id, response = _generate(gateway, model)
        _assert_images(response, call_id, 0.25)
        assert len(wire.drain()) == 1


def test_chat_completions_on_a_regional_chat_deployment(gateway: Gateway) -> None:
    with wire_server(_scripted_chat) as wire, gateway.scenario() as scenario:
        model: Final = _chat_deployment(gateway, scenario, wire)
        call_id: Final = uuid.uuid4().hex
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}], "cache": {"no-cache": True}},
            headers={"x-litellm-call-id": call_id},
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        assert "scripted answer" in response.text and response.headers["x-litellm-call-id"] == call_id, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(_REGIONAL_CHAT_COST), (
            response.headers
        )
        assert _spend(str(body["id"])) == pytest.approx(_REGIONAL_CHAT_COST)
        assert len(wire.drain()) == 1


def test_messages_on_a_regional_chat_deployment(gateway: Gateway) -> None:
    with wire_server(_scripted_chat) as wire, gateway.scenario() as scenario:
        model: Final = _chat_deployment(gateway, scenario, wire)
        call_id: Final = uuid.uuid4().hex
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, timeout=60)
        raw: Final = client.messages.with_raw_response.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": _PROMPT}],
            extra_headers={"x-litellm-call-id": call_id},
            extra_body={"cache": {"no-cache": True}},
        )
        assert raw.status_code == 200, raw.text
        assert "scripted answer" in raw.text and raw.headers["x-litellm-call-id"] == call_id, raw.text
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(_REGIONAL_CHAT_COST), raw.headers
        assert _spend(str(_JSON_OBJECT.validate_json(raw.content)["id"])) == pytest.approx(_REGIONAL_CHAT_COST)
        assert len(wire.drain()) == 1


def test_responses_on_a_regional_chat_deployment(gateway: Gateway) -> None:
    with wire_server(_scripted_chat) as wire, gateway.scenario() as scenario:
        model: Final = _chat_deployment(gateway, scenario, wire)
        call_id: Final = uuid.uuid4().hex
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, timeout=60)
        raw: Final = client.responses.with_raw_response.create(
            model=model,
            input=_PROMPT,
            extra_headers={"x-litellm-call-id": call_id},
            extra_body={"cache": {"no-cache": True}},
        )
        assert raw.status_code == 200, raw.text
        assert "scripted answer" in raw.text and raw.headers["x-litellm-call-id"] == call_id, raw.text
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(_REGIONAL_CHAT_COST), raw.headers
        assert _spend_by_litellm_call_id(call_id) == pytest.approx(_REGIONAL_CHAT_COST)
        assert len(wire.drain()) == 1


@dataclass(frozen=True, slots=True)
class _Call:
    call_id: str
    route: str
    model: str
    expected: float


def _burst_calls(global_model: str, regional_model: str) -> tuple[_Call, ...]:
    kinds: Final = (
        ("/v1/images/generations", global_model, _GLOBAL_TOKEN_COST),
        ("/v1/images/generations", regional_model, _REGIONAL_TOKEN_COST),
        ("/v1/images/edits", global_model, _GLOBAL_IMAGE_COST),
        ("/v1/images/edits", regional_model, _REGIONAL_IMAGE_COST),
    )
    return tuple(
        _Call(call_id=uuid.uuid4().hex, route=route, model=model, expected=expected)
        for route, model, expected in (kinds[index % len(kinds)] for index in range(_BURST))
    )


def _send(client: httpx.Client, key: str, call: _Call) -> httpx.Response:
    headers: Final = {"Authorization": f"Bearer {key}", "x-litellm-call-id": call.call_id}
    if call.route == "/v1/images/edits":
        return client.post(
            call.route,
            data={"model": call.model, "prompt": _PROMPT},
            files={"image": ("red.png", _PNG, "image/png")},
            headers=headers,
            timeout=120,
        )
    return client.post(call.route, json={"model": call.model, "prompt": _PROMPT, "n": 1}, headers=headers, timeout=120)


def test_mixed_burst_against_a_slow_upstream_bills_every_call_once(gateway: Gateway) -> None:
    with wire_server(_scripted(delay=0.3)) as wire, gateway.scenario() as scenario:
        global_model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": "global"})
        regional_model: Final = _deployment(gateway, scenario, wire, location={"vertex_location": _REGION})
        calls: Final = _burst_calls(global_model, regional_model)
        with ThreadPoolExecutor(max_workers=_BURST) as pool:
            responses: Final = tuple(pool.map(lambda call: _send(gateway.client, gateway.key, call), calls))
        assert [response.status_code for response in responses] == [200] * _BURST, [r.text[:120] for r in responses]
        rows: Final = _spend_rows(tuple(call.call_id for call in calls), seconds=120)
        assert rows == {call.call_id: pytest.approx(call.expected) for call in calls}
        assert len(wire.drain()) == _BURST


@dataclass(frozen=True, slots=True)
class _Served:
    call_id: str
    status: int
    text: str


async def _async_burst(base_url: str, key: str, call_ids: tuple[str, ...]) -> tuple[_Served, ...]:
    async def send(client: httpx.AsyncClient, call_id: str) -> _Served:
        response: Final = await client.post(
            "/v1/images/generations",
            json={"model": _CONFIG_MODEL, "prompt": _PROMPT, "n": 1},
            headers={"Authorization": f"Bearer {key}", "x-litellm-call-id": call_id},
        )
        return _Served(call_id=call_id, status=response.status_code, text=response.text)

    async with httpx.AsyncClient(base_url=base_url, timeout=120, trust_env=False) as client:
        results: Final = await asyncio.gather(*(send(client, call_id) for call_id in call_ids), return_exceptions=True)
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _chaos_config(wire: Wire, tmp_path: Path, upstream_url: str) -> Path:
    config: Final = {
        **_JSON_OBJECT.validate_python(
            yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
        ),
        "model_list": [
            {
                "model_name": _CONFIG_MODEL,
                "litellm_params": {
                    "model": f"vertex_ai/{_IMAGE_BACKEND}",
                    "api_base": f"{wire.url}{_IMAGE_TARGET}",
                    "vertex_project": _PROJECT,
                    "vertex_location": _REGION,
                    "vertex_credentials": service_account_json(_PROJECT, upstream_url.rstrip("/")),
                    "input_cost_per_token": 2e-6,
                    "output_cost_per_token": 4e-5,
                    "output_cost_per_image": 0.05,
                    "regional_endpoint_uplift_multiplier": 1.1,
                },
            }
        ],
    }
    path: Final = tmp_path / "vertex-image-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _artifact(tmp_path: Path, name: str, payload: Mapping[str, JsonValue]) -> None:
    (Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(tmp_path))) / name).write_text(json.dumps(payload))


def _worker_startups(log: Path) -> tuple[tuple[int, ...], int]:
    text: Final = log.read_text()
    return tuple(int(pid) for pid in _STARTED_WORKER.findall(text)), text.count("Application startup complete.")


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _gated(release: threading.Event, held: SimpleQueue[str]) -> Callable[[Request], Reply]:
    scripted: Final = _scripted()

    def respond(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=120), "The burst was never released"
        return scripted(request)

    return respond


@asynccontextmanager
async def _gated_burst(
    base_url: str, key: str, call_ids: tuple[str, ...], release: threading.Event
) -> AsyncIterator[asyncio.Task[tuple[_Served, ...]]]:
    burst: Final = asyncio.create_task(_async_burst(base_url, key, call_ids))
    try:
        yield burst
    finally:
        release.set()
        if not burst.done():
            burst.cancel()
        await asyncio.gather(burst, return_exceptions=True)


@pytest.mark.timeout(420)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_billing_the_uplift(gateway: Gateway, tmp_path: Path) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    call_ids: Final = tuple(uuid.uuid4().hex for _ in range(20))
    with wire_server(_gated(release, held)) as wire:
        config: Final = _chaos_config(wire, tmp_path, gateway.upstream_url)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            base_url: Final = str(owned.gateway.client.base_url)
            workers, _ = eventually(
                lambda: _worker_startups(owned.log), lambda found: len(found[0]) == 2 and found[1] == 2, seconds=240
            )
            async with _gated_burst(base_url, gateway.key, call_ids, release) as burst:
                await asyncio.to_thread(eventually, held.qsize, lambda size: size == len(call_ids), 120)
                held_by: Final = {pid: _open_upstream_connections(pid, wire.url) for pid in workers}
                assert sum(held_by.values()) == len(call_ids), held_by
                victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
                psutil.Process(victim_pid).send_signal(signal.SIGKILL)
                release.set()
                served: Final = await burst
                succeeded: Final = tuple(item.call_id for item in served if item.status == 200)
                _artifact(
                    tmp_path,
                    "worker-sigkill.json",
                    {
                        "held_by_killed_worker": held_by[victim_pid],
                        "held_by_surviving_worker": held_by[survivor_pid],
                        "served": len(served),
                        "succeeded": len(succeeded),
                        "transport_errors": len(call_ids) - len(served),
                    },
                )
                assert len(succeeded) == held_by[survivor_pid], (
                    held_by,
                    [(item.status, item.text[:80]) for item in served],
                )
                follow_up_id, follow_up = _generate(owned.gateway, _CONFIG_MODEL)
                _assert_images(follow_up, follow_up_id, _REGIONAL_TOKEN_COST)
                rows: Final = await asyncio.to_thread(_spend_rows, succeeded, seconds=120)
                assert rows == {call_id: pytest.approx(_REGIONAL_TOKEN_COST) for call_id in succeeded}
                await asyncio.to_thread(
                    eventually,
                    lambda: _worker_startups(owned.log),
                    lambda found: len(found[0]) == 3 and found[1] == 3,
                    240,
                )


@pytest.mark.timeout(600)
async def test_proxy_restart_mid_burst_bills_each_landed_call_once(gateway: Gateway, tmp_path: Path) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    call_ids: Final = tuple(uuid.uuid4().hex for _ in range(20))
    with wire_server(_gated(release, held)) as wire:
        config: Final = _chaos_config(wire, tmp_path, gateway.upstream_url)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            base_url: Final = str(owned.gateway.client.base_url)
            eventually(
                lambda: _worker_startups(owned.log), lambda found: len(found[0]) == 2 and found[1] == 2, seconds=240
            )
            async with _gated_burst(base_url, gateway.key, call_ids, release) as burst:
                await asyncio.to_thread(eventually, held.qsize, lambda size: size == len(call_ids), 120)
                owned.process.terminate()
                release.set()
                served: Final = await burst
                await asyncio.to_thread(owned.process.wait, 240)
        succeeded: Final = tuple(item.call_id for item in served if item.status == 200)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as rebooted:
            follow_up_id, follow_up = _generate(rebooted.gateway, _CONFIG_MODEL)
            _assert_images(follow_up, follow_up_id, _REGIONAL_TOKEN_COST)
            landed: Final = eventually(
                lambda: read_rows(
                    'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)', (list(call_ids),)
                ),
                lambda rows: set(succeeded) <= {str(row["request_id"]) for row in rows},
                seconds=120,
            )
        landed_ids: Final = tuple(str(row["request_id"]) for row in landed)
        _artifact(
            tmp_path,
            "restart-loss.json",
            {"served": len(served), "succeeded": len(succeeded), "landed": len(landed_ids)},
        )
        assert sorted(landed_ids) == sorted(succeeded), (succeeded, landed_ids)
        assert all(float(str(row["spend"])) == pytest.approx(_REGIONAL_TOKEN_COST) for row in landed), landed


async def test_gated_burst_releases_its_handlers_and_finishes_after_a_precondition_failure() -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    call_ids: Final = tuple(uuid.uuid4().hex for _ in range(2))

    def respond(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=5), "The failed burst did not release its handlers"
        return Reply()

    with wire_server(respond) as wire:
        tasks_before: Final = asyncio.all_tasks()

        async def fail_precondition() -> None:
            async with _gated_burst(wire.url, "owned-burst-key", call_ids, release):
                await asyncio.to_thread(eventually, held.qsize, lambda size: size == len(call_ids), 5)
                raise AssertionError("injected precondition failure")

        with pytest.raises(AssertionError, match="injected precondition failure"):
            await fail_precondition()
        assert release.is_set()
        assert asyncio.all_tasks() == tasks_before
