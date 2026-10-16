import json
import uuid
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, held
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

Endpoint: TypeAlias = Literal["chat", "responses"]
Deploy: TypeAlias = Callable[[Gateway, Scenario, Wire], str]

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROJECT: Final = "scripted-project"
_REPLY_TEXT: Final = "header probe reply"
_USAGE: Final = {"prompt_tokens": 13, "completion_tokens": 4, "total_tokens": 17}
_UPSTREAM_COST: Final = "0.0421"
_SPEND_SELECT: Final = 'SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s'


def _keyed(model: str, **extra: JsonValue) -> Deploy:
    def create(_gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
        return scenario.model(model=model, api_base=wire.url, api_key="synthetic-provider-key", **extra)

    return create


def _vertex(model: str, location: str) -> Deploy:
    def create(gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
        return scenario.model(
            model=f"vertex_ai/{model}",
            api_base=wire.url,
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=location,
            vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        )

    return create


_MEMBERS: Final[Mapping[str, Deploy]] = MappingProxyType(
    {
        "bedrock_mantle": _keyed("bedrock_mantle/google.gemma-4-31b", aws_region_name="us-west-2"),
        "groq": _keyed("groq/openai/gpt-oss-20b"),
        "watsonx": _keyed("watsonx/ibm/granite-3-8b-instruct", zen_api_key="synthetic-zen-key", project_id=_PROJECT),
        "vertex_model_garden": _vertex("xai/grok-4.20-non-reasoning", "global"),
        "maritalk": _keyed("maritalk/sabia-3"),
    }
)


def _reply_json() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "scripted",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": _REPLY_TEXT}, "finish_reason": "stop"}
            ],
            "usage": _USAGE,
            "service_tier": "on_demand",
        }
    ).encode()


def _chunk(response_id: str, choices: list[JsonValue], **extra: JsonValue) -> bytes:
    body: Final = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "scripted",
        "choices": choices,
        **extra,
    }
    return f"data: {json.dumps(body)}\n\n".encode()


def _frames() -> tuple[bytes, ...]:
    response_id: Final = f"chatcmpl-{uuid.uuid4().hex}"
    return (
        _chunk(
            response_id, [{"index": 0, "delta": {"role": "assistant", "content": _REPLY_TEXT}, "finish_reason": None}]
        ),
        _chunk(response_id, [{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        _chunk(response_id, [], usage=_USAGE),
        b"data: [DONE]\n\n",
    )


def _peer(headers: Mapping[str, str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=404)
        if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_frames(), headers=headers)
        return Reply(body=_reply_json(), headers=headers)

    return respond


def _request_id_headers(request_id: str) -> Mapping[str, str]:
    return MappingProxyType({"x-request-id": request_id, "x-amzn-requestid": request_id})


def _send(gateway: Gateway, endpoint: Endpoint, model: str, key: str, *, stream: bool = False) -> httpx.Response:
    prompt: Final = f"header probe {uuid.uuid4().hex}"
    body: Final[dict[str, JsonValue]] = (
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "cache": {"no-cache": True},
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }
        if endpoint == "chat"
        else {"model": model, "input": prompt, "stream": stream}
    )
    return gateway.request("POST", f"/v1/{'chat/completions' if endpoint == 'chat' else 'responses'}", body, key=key)


def _text(endpoint: Endpoint, response: httpx.Response) -> str:
    body: Final = _JSON_OBJECT.validate_json(response.content)
    if endpoint == "chat":
        return str(body["choices"][0]["message"]["content"])
    return "".join(
        str(part.get("text") or "")
        for item in body["output"]
        for part in item.get("content") or []  # comprehension-ok: output item then content part
        if isinstance(part, dict)
    )


def _llm_provider_request_ids(response: httpx.Response) -> dict[str, str | None]:
    return {name: response.headers.get(name) for name in ("llm_provider-x-request-id", "llm_provider-x-amzn-requestid")}


def _spend(key: str) -> float:
    digest: Final = sha256(key.encode()).hexdigest()
    eventually(lambda: read_rows(_SPEND_SELECT, (digest,)), lambda rows: len(rows) >= 1, seconds=70)
    rows: Final = held(lambda: read_rows(_SPEND_SELECT, (digest,)), lambda found: len(found) == 1, holding=3)
    spend: Final = rows[0]["spend"]
    assert isinstance(spend, (int, float)), rows
    return float(spend)


@pytest.mark.parametrize(
    ("endpoint", "member"),
    [
        ("chat", "bedrock_mantle"),
        ("chat", "groq"),
        ("chat", "watsonx"),
        ("chat", "vertex_model_garden"),
        ("chat", "maritalk"),
        ("responses", "groq"),
    ],
)
def test_non_stream_returns_the_provider_request_id(gateway: Gateway, endpoint: Endpoint, member: str) -> None:
    request_id: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(_peer(_request_id_headers(request_id))) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS[member](gateway, scenario, wire)
        response: Final = _send(gateway, endpoint, model, scenario.key(models=[model]))
        assert response.status_code == 200, response.text
        assert _text(endpoint, response) == _REPLY_TEXT, response.text
        assert _llm_provider_request_ids(response) == {
            "llm_provider-x-request-id": request_id,
            "llm_provider-x-amzn-requestid": request_id,
        }, dict(response.headers)
        assert len(wire.drain()) == 1


def test_stream_chat_returns_the_provider_request_id(gateway: Gateway) -> None:
    request_id: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(_peer(_request_id_headers(request_id))) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS["bedrock_mantle"](gateway, scenario, wire)
        response: Final = _send(gateway, "chat", model, scenario.key(models=[model]), stream=True)
        assert response.status_code == 200, response.text
        assert _REPLY_TEXT in response.text, response.text
        assert _llm_provider_request_ids(response) == {
            "llm_provider-x-request-id": request_id,
            "llm_provider-x-amzn-requestid": request_id,
        }, dict(response.headers)
        assert len(wire.drain()) == 1


def test_upstream_cannot_overwrite_litellm_owned_headers(gateway: Gateway) -> None:
    spoofed: Final = {
        "x-litellm-call-id": "spoofed-call-id",
        "x-litellm-model-id": "spoofed-model-id",
        "x-litellm-attempted-fallbacks": "7",
    }
    with wire_server(_peer(spoofed)) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS["groq"](gateway, scenario, wire)
        response: Final = _send(gateway, "chat", model, scenario.key(models=[model]))
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"] != spoofed["x-litellm-call-id"], dict(response.headers)
        assert response.headers["x-litellm-model-id"] != spoofed["x-litellm-model-id"], dict(response.headers)
        assert response.headers["x-litellm-attempted-fallbacks"] == "0", dict(response.headers)
        assert len(wire.drain()) == 1


@pytest.mark.parametrize(("endpoint", "member"), [("chat", "bedrock_mantle"), ("responses", "groq")])
def test_non_stream_spend_matches_stream_spend_under_an_upstream_cost_header(
    gateway: Gateway, endpoint: Endpoint, member: str
) -> None:
    with wire_server(_peer({"x-litellm-response-cost": _UPSTREAM_COST})) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS[member](gateway, scenario, wire)
        stream_key: Final = scenario.key(models=[model])
        non_stream_key: Final = scenario.key(models=[model])
        streamed: Final = _send(gateway, endpoint, model, stream_key, stream=True)
        assert streamed.status_code == 200, streamed.text
        assert _REPLY_TEXT in streamed.text, streamed.text
        whole: Final = _send(gateway, endpoint, model, non_stream_key)
        assert whole.status_code == 200, whole.text
        assert _text(endpoint, whole) == _REPLY_TEXT, whole.text
        stream_spend: Final = _spend(stream_key)
        assert stream_spend > 0
        assert _spend(non_stream_key) == pytest.approx(stream_spend)
        assert len(wire.drain()) == 2


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_provider_header_outside_latin1_does_not_fail_the_call(gateway: Gateway, tmp_path: Path) -> None:
    request_id: Final = f"req_{uuid.uuid4().hex}"
    utf8_bytes_as_wire_text: Final = "東京リージョン".encode().decode("latin-1")
    headers: Final = {**_request_id_headers(request_id), "x-provider-region": utf8_bytes_as_wire_text}
    with (
        wire_server(_peer(headers)) as wire,
        owned_proxy(gateway, tmp_path, {"DISABLE_AIOHTTP_TRANSPORT": "True"}) as httpx_transport,
        httpx_transport.scenario() as scenario,
    ):
        model: Final = _MEMBERS["bedrock_mantle"](httpx_transport, scenario, wire)
        key: Final = scenario.key(models=[model])
        whole: Final = _send(httpx_transport, "chat", model, key)
        streamed: Final = _send(httpx_transport, "chat", model, key, stream=True)
        assert {"non_stream": whole.status_code, "stream": streamed.status_code} == {
            "non_stream": 200,
            "stream": 200,
        }, (whole.text, streamed.text)
        assert _text("chat", whole) == _REPLY_TEXT, whole.text
        assert _REPLY_TEXT in streamed.text, streamed.text
        assert _llm_provider_request_ids(whole) == {
            "llm_provider-x-request-id": request_id,
            "llm_provider-x-amzn-requestid": request_id,
        }, dict(whole.headers)
        assert len(wire.drain()) == 2
