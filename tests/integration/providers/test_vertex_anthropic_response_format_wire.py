import asyncio
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue, TypeAdapter

_PROJECT: Final = "scripted-project"
_LOCATION: Final = "global"
_SONNET_5_5: Final = "claude-sonnet-5-5"
_SONNET_4_6: Final = "claude-sonnet-4-6"
_FORCED_TOOL_CHOICE_ERROR: Final = 'tool_choice: type "tool" and "any" are not supported for this model.'
_JSON_TOOL_INPUT: Final = {"name": "Paris", "country": "France"}
_RESPONSE_FORMAT: Final = {
    "type": "json_schema",
    "json_schema": {
        "name": "city",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "country": {"type": "string"}},
            "required": ["name", "country"],
            "additionalProperties": False,
        },
    },
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _prompt() -> str:
    return f"Give me Paris as JSON. Trace {uuid.uuid4().hex[:12]}"


def _service_account_json(token_url: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": _PROJECT,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def _model_path(model: str) -> str:
    return f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/{model}:rawPredict"


def _anthropic_tool_use_message(model: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "json_tool_call", "input": _JSON_TOOL_INPUT}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def _model(gateway: Gateway, scenario: Scenario, wire_url: str, backend: str, **extra: JsonValue) -> str:
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_base=wire_url,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=_service_account_json(gateway.upstream_url.rstrip("/")),
        **extra,
    )


def _request(gateway: Gateway, model: str) -> tuple[int, dict[str, JsonValue], str]:
    response: Final = gateway.client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": _prompt()}],
            "response_format": _RESPONSE_FORMAT,
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=30,
    )
    return response.status_code, response.json(), response.text


def test_vertex_sonnet_5_5_response_format_sends_json_tool_without_forced_tool_choice(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        tool_choice: Final = body.get("tool_choice")
        if isinstance(tool_choice, dict) and tool_choice.get("type") in ("tool", "any"):
            # Observed live from Vertex claude-sonnet-5-5 on 2026-09-29, LIT-8983
            return Reply(
                status=400,
                body=json.dumps(
                    {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": _FORCED_TOOL_CHOICE_ERROR},
                    }
                ).encode(),
            )
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        status, payload, text = _request(gateway, model)
        assert status == 200, text
        assert json.loads(str(payload["choices"][0]["message"]["content"])) == _JSON_TOOL_INPUT
        received: Final = wire.drain()
        assert len(received) == 1
        upstream_body: Final = _JSON_OBJECT.validate_json(received[0].body)
        assert [tool["name"] for tool in upstream_body["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream_body


def test_vertex_sonnet_4_6_response_format_still_forces_json_tool_call(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_4_6)
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_4_6)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_4_6)
        status, payload, text = _request(gateway, model)
        assert status == 200, text
        assert json.loads(str(payload["choices"][0]["message"]["content"])) == _JSON_TOOL_INPUT
        received: Final = wire.drain()
        assert len(received) == 1
        upstream_body: Final = _JSON_OBJECT.validate_json(received[0].body)
        assert upstream_body["tool_choice"] == {"type": "tool", "name": "json_tool_call"}
        assert [tool["name"] for tool in upstream_body["tools"]] == ["json_tool_call"]


_OPUS_5_5: Final = "claude-opus-5-5"
_SONNET_5_5_DEFAULT: Final = "claude-sonnet-5-5@default"
_GET_WEATHER: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_ANTHROPIC_WEATHER: Final = {
    "name": "get_weather",
    "description": "Get weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
_MESSAGES_PATH: Final = "/v1/messages"


def _chat(gateway: Gateway, model: str, **extra: JsonValue) -> httpx.Response:
    return gateway.client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": _prompt()}],
            "max_tokens": 1024,
            **extra,
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=60,
    )


def _responses(gateway: Gateway, model: str, **extra: JsonValue) -> httpx.Response:
    return gateway.client.post(
        "/v1/responses",
        json={
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": _prompt()}]}],
            "max_output_tokens": 1024,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "city",
                    "strict": True,
                    "schema": _RESPONSE_FORMAT["json_schema"]["schema"],
                }
            },
            **extra,
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=60,
    )


def _anthropic_text_message(model: str, text: str) -> bytes:
    return json.dumps(
        {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }
    ).encode()


def _sse_frame(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _anthropic_tool_use_stream(model: str) -> tuple[bytes, ...]:
    return (
        _sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": f"msg_{uuid.uuid4().hex[:12]}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ),
        _sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "json_tool_call", "input": {}},
            },
        ),
        _sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"name": "Paris", '},
            },
        ),
        _sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"country": "France"}'},
            },
        ),
        _sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse_frame(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
        ),
        _sse_frame("message_stop", {"type": "message_stop"}),
    )


def _sse_events(body: bytes) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: ").encode())
        for line in body.decode().splitlines()
        if line.startswith("data: ") and line.removeprefix("data: ") != "[DONE]"
    )


def _spend_count(request_id: str) -> int:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
        lambda rows: len(rows) >= 1,
        seconds=70,
    )
    return len(rows)


def _spend_count_by_call_id(call_id: str) -> int:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id FROM \"LiteLLM_SpendLogs\" WHERE metadata->>'litellm_call_id'=%s", (call_id,)
        ),
        lambda rows: len(rows) >= 1,
        seconds=70,
    )
    return len(rows)


def _rejects_forced_tool_choice(body: dict[str, JsonValue]) -> Reply | None:
    tool_choice: Final = body.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") in ("tool", "any"):
        # Observed live from Vertex claude-sonnet-5-5 on 2026-09-29, LIT-8983
        return Reply(
            status=400,
            body=json.dumps(
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": _FORCED_TOOL_CHOICE_ERROR},
                }
            ).encode(),
        )
    return None


def _forced_tool_choice_reply(model: str, request: Request, path: str) -> Reply:
    assert request.target == path
    body: Final = _JSON_OBJECT.validate_json(request.body)
    rejected: Final = _rejects_forced_tool_choice(body)
    if rejected is not None:
        return rejected
    return Reply(body=json.dumps(_anthropic_tool_use_message(model)).encode())


def _upstream_bodies(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(_JSON_OBJECT.validate_json(request.body) for request in wire.drain())


def _output_text(payload: dict[str, JsonValue]) -> str:
    texts: Final = []

    def walk(node: JsonValue) -> None:
        if isinstance(node, dict):
            if node.get("type") == "output_text":
                texts.append(node.get("text", ""))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return "".join(str(text) for text in texts)


def test_vertex_sonnet_5_5_response_format_json_tool_unforced_openai_sdk_sync(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return _forced_tool_choice_reply(_SONNET_5_5, request, _model_path(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        client: Final = OpenAI(
            base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, http_client=httpx.Client(trust_env=False)
        )
        raw: Final = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": _prompt()}],
            max_tokens=1024,
            response_format=_RESPONSE_FORMAT,
        )
        completion: Final = raw.parse()
        assert json.loads(str(completion.choices[0].message.content)) == _JSON_TOOL_INPUT
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream[0]
        assert _spend_count(completion.id) == 1


def test_vertex_sonnet_5_5_response_format_json_tool_unforced_openai_sdk_async(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return _forced_tool_choice_reply(_SONNET_5_5, request, _model_path(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)

        async def call():
            client: Final = AsyncOpenAI(
                base_url=f"{gateway.client.base_url}/v1",
                api_key=gateway.key,
                http_client=httpx.AsyncClient(trust_env=False),
            )
            return await client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{"role": "user", "content": _prompt()}],
                max_tokens=1024,
                response_format=_RESPONSE_FORMAT,
            )

        raw: Final = asyncio.run(call())
        completion: Final = raw.parse()
        assert json.loads(str(completion.choices[0].message.content)) == _JSON_TOOL_INPUT
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream[0]
        assert _spend_count(completion.id) == 1


def test_vertex_sonnet_5_5_response_format_json_tool_unforced_stream(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5).replace(":rawPredict", ":streamRawPredict") + "?alt=sse"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        tool_choice: Final = body.get("tool_choice")
        assert not (isinstance(tool_choice, dict) and tool_choice.get("type") in ("tool", "any")), body
        return Reply(content_type="text/event-stream", chunks=_anthropic_tool_use_stream(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        client: Final = OpenAI(
            base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, http_client=httpx.Client(trust_env=False)
        )
        raw_stream: Final = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": _prompt()}],
            max_tokens=1024,
            response_format=_RESPONSE_FORMAT,
            stream=True,
        )
        chunks: Final = list(raw_stream.parse())
        joined: Final = "".join(
            choice.delta.content or "" for chunk in chunks for choice in chunk.choices if choice.delta
        )
        assert json.loads(joined) == _JSON_TOOL_INPUT, joined
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream[0]
        assert _spend_count(chunks[0].id) == 1


def test_vertex_sonnet_5_5_responses_text_format_sends_json_tool_unforced(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return _forced_tool_choice_reply(_SONNET_5_5, request, _model_path(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = _responses(gateway, model)
        assert response.status_code == 200, response.text
        payload: Final = response.json()
        assert json.loads(_output_text(payload)) == _JSON_TOOL_INPUT, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream[0]
        assert _spend_count_by_call_id(response.headers["x-litellm-call-id"]) == 1


def test_vertex_sonnet_5_5_responses_stream_sends_json_tool_unforced(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5).replace(":rawPredict", ":streamRawPredict") + "?alt=sse"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        tool_choice: Final = body.get("tool_choice")
        assert not (isinstance(tool_choice, dict) and tool_choice.get("type") in ("tool", "any")), body
        return Reply(content_type="text/event-stream", chunks=_anthropic_tool_use_stream(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        with gateway.client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": model,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": _prompt()}]}],
                "max_output_tokens": 1024,
                "stream": True,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "city",
                        "strict": True,
                        "schema": _RESPONSE_FORMAT["json_schema"]["schema"],
                    }
                },
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=60,
        ) as response:
            call_id: Final = response.headers["x-litellm-call-id"]
            raw: Final = response.read()
        assert response.status_code == 200, raw.decode()
        events: Final = _sse_events(raw)
        deltas: Final = "".join(
            str(event["delta"])
            for event in events
            if event.get("type") == "response.output_text.delta" and isinstance(event.get("delta"), str)
        )
        completed: Final = [event for event in events if event.get("type") == "response.completed"]
        joined: Final = deltas or (_output_text(completed[0]) if completed else "")
        assert json.loads(joined) == _JSON_TOOL_INPUT, raw.decode()[:2000]
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert "tool_choice" not in upstream[0]
        assert _spend_count_by_call_id(call_id) == 1


def test_vertex_sonnet_5_5_response_format_with_user_tool_sends_both_tools_unforced(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return _forced_tool_choice_reply(_SONNET_5_5, request, _model_path(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT, tools=[_GET_WEATHER])
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call", "get_weather"]
        assert "tool_choice" not in upstream[0]


def test_vertex_sonnet_5_5_response_format_drop_params_drops_temperature(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return _forced_tool_choice_reply(_SONNET_5_5, request, _model_path(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5, drop_params=True)
        response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT, temperature=0.2)
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream[0]
        assert "temperature" not in upstream[0]


def test_vertex_sonnet_5_5_response_format_drop_params_downgrades_required_tool_choice(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return _forced_tool_choice_reply(_SONNET_5_5, request, _model_path(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5, drop_params=True)
        response: Final = _chat(
            gateway, model, response_format=_RESPONSE_FORMAT, tools=[_GET_WEATHER], tool_choice="required"
        )
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call", "get_weather"]
        assert upstream[0]["tool_choice"] == {"type": "auto"}


def test_vertex_sonnet_5_5_response_format_rejects_temperature_before_upstream(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT, temperature=0.2)
        assert response.status_code == 400, response.text
        assert "does not support temperature=0.2" in response.text, response.text
        assert wire.drain() == ()


def test_vertex_sonnet_5_5_response_format_rejects_forced_tool_choice_before_upstream(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = _chat(
            gateway, model, response_format=_RESPONSE_FORMAT, tools=[_GET_WEATHER], tool_choice="required"
        )
        assert response.status_code == 400, response.text
        assert "does not support forced tool use" in response.text, response.text
        assert wire.drain() == ()


def test_vertex_sonnet_4_6_response_format_reasoning_effort_maps_to_real_model(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_4_6)
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_4_6)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_4_6)
        response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT, reasoning_effort="low")
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert upstream[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert upstream[0]["output_config"] == {"effort": "low"}
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream[0]


def test_vertex_sonnet_4_6_response_format_adaptive_thinking_is_forwarded(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_4_6)
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_4_6)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_4_6)
        response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT, thinking={"type": "adaptive"})
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert upstream[0]["thinking"] == {"type": "adaptive"}
        assert upstream[0]["tool_choice"] == {"type": "tool", "name": "json_tool_call"}
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call"]


def test_vertex_opus_5_5_response_format_uses_native_output_format(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_OPUS_5_5)
        return Reply(body=json.dumps(_anthropic_tool_use_message(_OPUS_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _OPUS_5_5)
        response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT)
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert upstream[0]["output_format"] == {
            "type": "json_schema",
            "schema": _RESPONSE_FORMAT["json_schema"]["schema"],
        }
        assert "tools" not in upstream[0]
        assert "tool_choice" not in upstream[0]


def test_vertex_sonnet_5_5_plain_request_sends_no_tools(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5)
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = _chat(gateway, model)
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert "tools" not in upstream[0]
        assert "tool_choice" not in upstream[0]


def test_vertex_sonnet_5_5_response_format_text_is_noop(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5)
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = _chat(gateway, model, response_format={"type": "text"})
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert "tools" not in upstream[0]
        assert "tool_choice" not in upstream[0]


def test_vertex_sonnet_5_5_response_format_json_object_sends_no_tool(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5)
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = _chat(gateway, model, response_format={"type": "json_object"})
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert "tools" not in upstream[0]
        assert "tool_choice" not in upstream[0]
        assert "output_format" not in upstream[0]
        assert "output_config" not in upstream[0]


def test_vertex_sonnet_5_5_messages_endpoint_forwards_tool_choice_auto(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5)
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = gateway.client.post(
            _MESSAGES_PATH,
            json={
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": [{"type": "text", "text": _prompt()}]}],
                "tools": [_ANTHROPIC_WEATHER],
                "tool_choice": {"type": "auto"},
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=60,
        )
        assert response.status_code == 200, response.text
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["get_weather"]
        assert upstream[0]["tool_choice"] == {"type": "auto"}


def test_vertex_sonnet_5_5_response_format_surfaces_peer_500(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        rejected: Final = _rejects_forced_tool_choice(body)
        if rejected is not None:
            return rejected
        return Reply(
            status=500,
            body=json.dumps(
                {"type": "error", "error": {"type": "api_error", "message": "scripted vertex 500"}}
            ).encode(),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5, num_retries=0)
        response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT)
        assert response.status_code >= 400, response.text
        assert "scripted vertex 500" in response.text, response.text
        assert len(wire.drain()) == 1


def test_vertex_sonnet_5_5_response_format_returns_plain_text_verbatim(gateway: Gateway) -> None:
    peer_text: Final = "unforced tools are a hint; here is prose instead"

    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        rejected: Final = _rejects_forced_tool_choice(body)
        if rejected is not None:
            return rejected
        return Reply(body=_anthropic_text_message(_SONNET_5_5, peer_text))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == peer_text


def test_vertex_sonnet_5_5_response_format_missing_schema_returns_500(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = _chat(
            gateway,
            model,
            response_format={"type": "json_schema", "json_schema": {"name": "city", "strict": True}},
        )
        assert response.status_code == 500, response.text
        assert "'schema'" in response.text, response.text
        assert wire.drain() == ()


def test_vertex_sonnet_5_5_response_format_string_value_is_rejected(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return _forced_tool_choice_reply(_SONNET_5_5, request, _model_path(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        bad: Final = _chat(gateway, model, response_format="json")
        assert bad.status_code == 500, bad.text
        assert "Unsupported response_format type" in bad.text
        good: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT)
        assert good.status_code == 200, good.text
        assert json.loads(str(good.json()["choices"][0]["message"]["content"])) == _JSON_TOOL_INPUT


def test_vertex_sonnet_5_5_response_format_requires_auth(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": _prompt()}],
                "max_tokens": 1024,
                "response_format": _RESPONSE_FORMAT,
            },
            timeout=60,
        )
        assert response.status_code == 401, response.text
        assert wire.drain() == ()


def test_vertex_sonnet_5_5_response_format_repeated_requests_get_own_spend_rows(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return _forced_tool_choice_reply(_SONNET_5_5, request, _model_path(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        ids: Final = []
        for _ in range(3):
            response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT)
            assert response.status_code == 200, response.text
            ids.append(str(response.json()["id"]))
        assert len(set(ids)) == 3, ids
        for request_id in ids:
            assert _spend_count(request_id) == 1, request_id
        assert len(wire.drain()) == 3


def test_vertex_sonnet_5_5_default_alias_response_format_sends_json_tool_unforced(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return _forced_tool_choice_reply(_SONNET_5_5_DEFAULT, request, _model_path(_SONNET_5_5_DEFAULT))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5_DEFAULT)
        response: Final = _chat(gateway, model, response_format=_RESPONSE_FORMAT)
        assert response.status_code == 200, response.text
        assert json.loads(str(response.json()["choices"][0]["message"]["content"])) == _JSON_TOOL_INPUT
        upstream: Final = _upstream_bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream[0]


def test_vertex_sonnet_5_5_response_format_outage_burst_recovers(gateway: Gateway) -> None:
    outage: Final = threading.BoundedSemaphore(8)

    def respond(request: Request) -> Reply:
        is_stream: Final = ":streamRawPredict" in request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        tool_choice: Final = body.get("tool_choice")
        if isinstance(tool_choice, dict) and tool_choice.get("type") in ("tool", "any"):
            return Reply(
                status=400,
                body=json.dumps(
                    {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": _FORCED_TOOL_CHOICE_ERROR},
                    }
                ).encode(),
            )
        if outage.acquire(blocking=False):
            return Reply(
                status=503,
                body=json.dumps(
                    {"type": "error", "error": {"type": "overloaded_error", "message": "scripted vertex 503"}}
                ).encode(),
            )
        if is_stream:
            return Reply(content_type="text/event-stream", chunks=_anthropic_tool_use_stream(_SONNET_5_5))
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    def send_chat(model: str, stream: bool) -> tuple[int, str, str]:
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": _prompt()}],
                "max_tokens": 1024,
                "response_format": _RESPONSE_FORMAT,
                "stream": stream,
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=120,
        )
        if stream and response.status_code == 200:
            events: Final = _sse_events(response.read())
            joined: Final = "".join(
                "".join(str(choice.get("delta", {}).get("content") or "") for choice in event.get("choices", ()))
                for event in events
            )
            identity: Final = next((str(event["id"]) for event in events if isinstance(event.get("id"), str)), "")
            return response.status_code, joined, identity
        if response.status_code != 200:
            return response.status_code, response.text, ""
        return (
            response.status_code,
            str(response.json()["choices"][0]["message"]["content"]),
            str(response.json()["id"]),
        )

    def send_responses(model: str) -> tuple[int, str, str]:
        response: Final = _responses(gateway, model)
        if response.status_code != 200:
            return response.status_code, response.text, ""
        payload: Final = response.json()
        return response.status_code, _output_text(payload), "call:" + str(response.headers["x-litellm-call-id"])

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5, num_retries=0)
        calls: Final = (
            [(send_chat, (model, False)) for _ in range(8)]
            + [(send_chat, (model, True)) for _ in range(8)]
            + [(send_responses, (model,)) for _ in range(8)]
        )
        with ThreadPoolExecutor(max_workers=24) as pool:
            outcomes: Final = list(pool.map(lambda call: call[0](*call[1]), calls))
        failures: Final = [outcome for outcome in outcomes if outcome[0] != 200]
        successes: Final = [outcome for outcome in outcomes if outcome[0] == 200]
        assert len(failures) == 8, [outcome[0] for outcome in outcomes]
        assert all("scripted vertex 503" in outcome[1] for outcome in failures), failures
        for _, content, _ in successes:
            assert json.loads(content) == _JSON_TOOL_INPUT, content
        assert len(wire.drain()) == 24
        for _, _, request_id in successes:
            assert request_id
            if request_id.startswith("call:"):
                assert _spend_count_by_call_id(request_id.removeprefix("call:")) == 1, request_id
            else:
                assert _spend_count(request_id) == 1, request_id
        health: Final = gateway.client.get("/health/liveliness", headers={"Authorization": f"Bearer {gateway.key}"})
        assert health.status_code == 200, health.text
