from __future__ import annotations

import json
import uuid
from typing import Final, Literal

import anthropic
import httpx
import openai
import pytest
import websockets
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue


def _marker() -> str:
    return uuid.uuid4().hex


def _register_scenario(gateway: Gateway, scenario_id: str, response: dict[str, JsonValue]) -> None:
    with httpx.Client(timeout=5, trust_env=False) as client:
        result: Final = client.post(
            f"{gateway.upstream_url}/__scenarios",
            json={"scenario_id": scenario_id, "response": response},
        )
    assert result.status_code == 200, result.text


_ANTHROPIC_MESSAGE: Final = {
    "id": "msg_$UNIQUE_ID",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "scripted reply"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 9, "output_tokens": 5},
}

_ANTHROPIC_EVENTS: Final = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_$UNIQUE_ID","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","content":[],"usage":{"input_tokens":9,"output_tokens":0}}}',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"scripted reply"}}',
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
    'event: message_stop\ndata: {"type":"message_stop"}',
)

_RESPONSES_BODY: Final = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "created_at": 1700000000,
    "status": "completed",
    "model": "openai/gpt-4o-mini",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
}

_RESPONSES_EVENTS: Final = (
    'event: response.created\ndata: {"type":"response.created","response":{"id":"$UNIQUE_ID","object":"response","status":"in_progress"}}',
    'event: response.completed\ndata: {"type":"response.completed","response":{"id":"$UNIQUE_ID","object":"response","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}}',
)


def _anthropic_model(gateway: Gateway, *, stream: bool) -> str:
    scenario_id: Final = "audit-anthropic-stream" if stream else "audit-anthropic"
    if stream:
        _register_scenario(gateway, scenario_id, {"content_type": "text/event-stream", "frames": _ANTHROPIC_EVENTS})
    else:
        _register_scenario(
            gateway,
            scenario_id,
            {
                "content_type": "application/x-routed",
                "routes": {"POST /v1/messages": {"content_type": "application/json", "body": _ANTHROPIC_MESSAGE}},
            },
        )
    name: Final = f"integration-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "api_key": "integration-provider-key",
                "api_base": f"{gateway.upstream_url}/{scenario_id}",
            },
        },
    )
    assert isinstance(created["model_info"], dict) and created["model_info"]["id"], created
    return name


def _responses_model(gateway: Gateway, *, stream: bool) -> str:
    scenario_id: Final = "audit-resp-stream" if stream else "audit-resp"
    if stream:
        _register_scenario(gateway, scenario_id, {"content_type": "text/event-stream", "frames": _RESPONSES_EVENTS})
    else:
        _register_scenario(
            gateway,
            scenario_id,
            {
                "content_type": "application/x-routed",
                "routes": {"POST /responses": {"content_type": "application/json", "body": _RESPONSES_BODY}},
            },
        )
    name: Final = f"integration-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "integration-provider-key",
                "api_base": f"{gateway.upstream_url}/{scenario_id}",
            },
        },
    )
    assert isinstance(created["model_info"], dict) and created["model_info"]["id"], created
    return name


def _openai_client(gateway: Gateway, key: str) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=key,
        http_client=httpx.Client(trust_env=False),
    )


def _openai_async_client(gateway: Gateway, key: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=key,
        http_client=httpx.AsyncClient(trust_env=False),
    )


def _anthropic_client(gateway: Gateway, key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url).rstrip("/"),
        api_key=key,
        http_client=httpx.Client(trust_env=False),
    )


def _anthropic_async_client(gateway: Gateway, key: str) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(gateway.client.base_url).rstrip("/"),
        api_key=key,
        http_client=httpx.AsyncClient(trust_env=False),
    )


def _sse_chunk_models(text: str) -> tuple[str, ...]:
    models: list[str] = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        value = json.loads(line.removeprefix("data: "))
        if isinstance(value, dict) and isinstance(value.get("model"), str):
            models.append(value["model"])
    return tuple(models)


@pytest.mark.parametrize("client_kind", ("httpx", "openai", "openai_async"))
@pytest.mark.parametrize("stream", (False, True))
def test_allowed_key_chat_completions_restamps_alias(
    gateway: Gateway, client_kind: Literal["httpx", "openai", "openai_async"], stream: bool
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        body: Final = {
            "model": model,
            "messages": [{"role": "user", "content": _marker()}],
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }
        if client_kind == "httpx":
            response: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
            assert response.status_code == 200, response.text
            if stream:
                models: Final = _sse_chunk_models(response.text)
                assert models and all(chunk_model == model for chunk_model in models), response.text
            else:
                assert response.json()["model"] == model, response.text
        elif client_kind == "openai":
            client: Final = _openai_client(gateway, key)
            completion: Final = client.chat.completions.create(**body)
            if stream:
                chunk_models: list[str] = []
                for chunk in completion:
                    if chunk.model:
                        chunk_models.append(chunk.model)
                assert chunk_models and all(chunk_model == model for chunk_model in chunk_models)
            else:
                assert completion.model == model
        else:

            async def _run() -> None:
                client_async: Final = _openai_async_client(gateway, key)
                completion_async: Final = await client_async.chat.completions.create(**body)
                if stream:
                    seen: list[str] = []
                    async for chunk in completion_async:
                        if chunk.model:
                            seen.append(chunk.model)
                    assert seen and all(chunk_model == model for chunk_model in seen)
                else:
                    assert completion_async.model == model

            import asyncio

            asyncio.run(_run())


@pytest.mark.parametrize("client_kind", ("httpx", "anthropic", "anthropic_async"))
@pytest.mark.parametrize("stream", (False, True))
def test_allowed_key_messages(gateway: Gateway, client_kind: str, stream: bool) -> None:
    model: Final = _anthropic_model(gateway, stream=stream)
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[model])
        body: Final = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": _marker()}],
            "stream": stream,
        }
        if client_kind == "httpx":
            response: Final = gateway.request("POST", "/v1/messages", body, key=key)
            assert response.status_code == 200, response.text
            if stream:
                assert "message_start" in response.text and "message_stop" in response.text, response.text
            else:
                assert response.json()["type"] == "message", response.text
        elif client_kind == "anthropic":
            client: Final = _anthropic_client(gateway, key)
            if stream:
                with client.messages.stream(**{k: v for k, v in body.items() if k != "stream"}) as event_stream:
                    message: Final = event_stream.get_final_message()
                assert message.type == "message" and message.content, message
            else:
                message = client.messages.create(**{k: v for k, v in body.items() if k != "stream"})
                assert message.type == "message" and message.content, message
        else:

            async def _run() -> None:
                client_async: Final = _anthropic_async_client(gateway, key)
                params: Final = {k: v for k, v in body.items() if k != "stream"}
                if stream:
                    async with client_async.messages.stream(**params) as event_stream:
                        message_async: Final = await event_stream.get_final_message()
                else:
                    message_async = await client_async.messages.create(**params)
                assert message_async.type == "message" and message_async.content, message_async

            import asyncio

            asyncio.run(_run())


@pytest.mark.parametrize("client_kind", ("httpx", "openai", "openai_async"))
@pytest.mark.parametrize("stream", (False, True))
def test_allowed_key_responses(gateway: Gateway, client_kind: str, stream: bool) -> None:
    model: Final = _responses_model(gateway, stream=stream)
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[model])
        body: Final = {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": _marker()}]}],
            "stream": stream,
        }
        if client_kind == "httpx":
            response: Final = gateway.request("POST", "/v1/responses", body, key=key)
            assert response.status_code == 200, response.text
            if stream:
                assert "response.completed" in response.text, response.text
            else:
                assert response.json()["object"] == "response", response.text
        elif client_kind == "openai":
            client: Final = _openai_client(gateway, key)
            result: Final = client.responses.create(**body)
            if stream:
                events: list[str] = []
                for event in result:
                    events.append(event.type)
                assert "response.completed" in events, events
            else:
                assert result.object == "response", result
        else:

            async def _run() -> None:
                client_async: Final = _openai_async_client(gateway, key)
                result_async: Final = await client_async.responses.create(**body)
                if stream:
                    events_async: list[str] = []
                    async for event in result_async:
                        events_async.append(event.type)
                    assert "response.completed" in events_async, events_async
                else:
                    assert result_async.object == "response", result_async

            import asyncio

            asyncio.run(_run())


def test_denied_key_rejected(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        other_model: Final = scenario.model()
        key: Final = scenario.key(models=[other_model])
        for path, body in (
            ("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "x"}]}),
            ("/v1/messages", {"model": model, "max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}),
            ("/v1/responses", {"model": model, "input": "x"}),
        ):
            response: Final = gateway.request("POST", path, body, key=key)
            assert response.status_code in (401, 403), f"{path}: {response.status_code} {response.text}"


async def test_realtime_websocket_allowed_and_denied(gateway: Gateway) -> None:
    rt_scenario: Final = f"auditrt{uuid.uuid4().hex[:8]}"
    _register_scenario(
        gateway,
        rt_scenario,
        {
            "content_type": "application/x-realtime",
            "session_model": "gpt-4o-realtime-preview",
            "events": (
                {"type": "response.done", "response": {"id": "resp_$UNIQUE_ID", "model": "gpt-4o-realtime-preview"}},
            ),
        },
    )
    name: Final = f"integration-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "openai/gpt-4o-realtime-preview",
                "api_key": rt_scenario,
                "api_base": gateway.upstream_url,
            },
        },
    )
    assert isinstance(created["model_info"], dict) and created["model_info"]["id"], created
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[name])
        other_model: Final = scenario.model()
        denied_key: Final = scenario.key(models=[other_model])
        url: Final = f"{str(gateway.client.base_url).replace('http://', 'ws://')}/v1/realtime?model={name}"
        async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {key}"}) as websocket:
            session: Final = json.loads(await websocket.recv())
            assert session["type"] == "session.created", session
            assert session["session"]["model"] == "gpt-4o-realtime-preview", session
            await websocket.send(json.dumps({"type": "response.create"}))
            event: Final = json.loads(await websocket.recv())
            assert event["type"] == "response.done", event
        with pytest.raises(websockets.exceptions.InvalidHandshake):
            async with websockets.connect(
                url, additional_headers={"Authorization": f"Bearer {denied_key}"}
            ) as denied_socket:
                await denied_socket.recv()


def test_max_budget_allows_then_denies_and_records_spend(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], max_budget=0.0000001)
        body: Final = {"model": model, "messages": [{"role": "user", "content": _marker()}]}
        first: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert first.status_code == 200, first.text
        request_id: Final = first.json()["id"]
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows, first.text
        second: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert second.status_code == 422, second.text
        assert "budget_exceeded" in second.text, second.text


_PATH_MODEL_BACKEND: Final = "gpt-5.4-mini"
_PATH_MODEL_ROUTES: Final = (
    ("/openai/deployments/{model}/chat/completions", "chat"),
    ("/engines/{model}/chat/completions", "chat"),
    ("/openai/deployments/{model}/completions", "text"),
    ("/engines/{model}/completions", "text"),
)


def _path_model_reply(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/chat/completions"), request
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + _marker(),
                "object": "chat.completion",
                "created": 1,
                "model": _PATH_MODEL_BACKEND,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "path"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


def _path_model_body(kind: str, prompt: str) -> dict[str, JsonValue]:
    return {"messages": [{"role": "user", "content": prompt}]} if kind == "chat" else {"prompt": prompt}


def _upstream_bodies(wire: Wire) -> list[dict[str, JsonValue]]:
    return [json.loads(request.body) for request in wire.drain()]


@pytest.mark.parametrize(("route", "kind"), _PATH_MODEL_ROUTES)
def test_path_only_model_alias_refuses_an_unallowed_deployment(gateway: Gateway, route: str, kind: str) -> None:
    with wire_server(_path_model_reply) as allowed_wire, wire_server(_path_model_reply) as other_wire:
        with gateway.scenario() as scenario:
            allowed: Final = scenario.model(model=f"openai/{_PATH_MODEL_BACKEND}", api_base=allowed_wire.url)
            other: Final = scenario.model(model=f"openai/{_PATH_MODEL_BACKEND}", api_base=other_wire.url)
            key: Final = scenario.key(models=[allowed])
            response: Final = gateway.request(
                "POST",
                route.format(model=other),
                _path_model_body(kind, _marker()),
                key=key,
                params={"api-version": "2024-10-21"},
            )
            assert (_upstream_bodies(other_wire), _upstream_bodies(allowed_wire)) == ([], []), response.text
            assert response.status_code == 403, response.text
            assert f"The requested model '{other}' is not available for this API key" in response.text, response.text


@pytest.mark.parametrize(("route", "kind"), _PATH_MODEL_ROUTES)
def test_mixed_body_and_path_model_is_refused_or_served_by_the_body_model(
    gateway: Gateway, route: str, kind: str
) -> None:
    pytest.skip(
        "BUG: a key scoped to model A that sends model A in the body and an unallowed model B in the "
        "/openai/deployments or /engines path is served by deployment B"
    )
    with wire_server(_path_model_reply) as allowed_wire, wire_server(_path_model_reply) as other_wire:
        with gateway.scenario() as scenario:
            allowed: Final = scenario.model(model=f"openai/{_PATH_MODEL_BACKEND}", api_base=allowed_wire.url)
            other: Final = scenario.model(model=f"openai/{_PATH_MODEL_BACKEND}", api_base=other_wire.url)
            key: Final = scenario.key(models=[allowed])
            prompt: Final = _marker()
            response: Final = gateway.request(
                "POST",
                route.format(model=other),
                {"model": allowed, **_path_model_body(kind, prompt)},
                key=key,
                params={"api-version": "2024-10-21"},
            )
            other_bodies: Final = _upstream_bodies(other_wire)
            allowed_bodies: Final = _upstream_bodies(allowed_wire)
            outcome: Final = (response.status_code, other_bodies, allowed_bodies)
            refused: Final = outcome[0] in (401, 403) and outcome[1:] == ([], [])
            served_by_body_model: Final = outcome == (
                200,
                [],
                [{"model": _PATH_MODEL_BACKEND, "messages": [{"role": "user", "content": prompt}]}],
            )
            assert refused or served_by_body_model, (outcome, response.text)


@pytest.mark.parametrize(("route", "kind"), _PATH_MODEL_ROUTES)
def test_path_only_model_alias_reaches_the_allowed_deployment(gateway: Gateway, route: str, kind: str) -> None:
    with wire_server(_path_model_reply) as allowed_wire, wire_server(_path_model_reply) as other_wire:
        with gateway.scenario() as scenario:
            allowed: Final = scenario.model(model=f"openai/{_PATH_MODEL_BACKEND}", api_base=allowed_wire.url)
            scenario.model(model=f"openai/{_PATH_MODEL_BACKEND}", api_base=other_wire.url)
            key: Final = scenario.key(models=[allowed])
            prompt: Final = _marker()
            response: Final = gateway.request(
                "POST",
                route.format(model=allowed),
                _path_model_body(kind, prompt),
                key=key,
                params={"api-version": "2024-10-21"},
            )
            assert response.status_code == 200, response.text
            payload: Final = response.json()
            assert [
                choice["message"]["content"] if kind == "chat" else choice["text"] for choice in payload["choices"]
            ] == ["path"], response.text
            assert payload["object"] == ("chat.completion" if kind == "chat" else "text_completion"), response.text
            assert _upstream_bodies(other_wire) == [], response.text
            assert _upstream_bodies(allowed_wire) == [
                {"model": _PATH_MODEL_BACKEND, "messages": [{"role": "user", "content": prompt}]}
            ], response.text
