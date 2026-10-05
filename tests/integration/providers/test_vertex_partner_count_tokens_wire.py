import json
import re
import signal
import socket
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually
from integration._support.process import owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "claude-sonnet-4-6"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_MODELS_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models"
_COUNT_TARGET: Final = f"{_MODELS_PATH}/count-tokens:rawPredict"
_MESSAGE_TARGET: Final = f"{_MODELS_PATH}/{_BACKEND}:rawPredict"
_STREAM_TARGET: Final = f"{_MODELS_PATH}/{_BACKEND}:streamRawPredict?alt=sse"
_PEER_COUNT: Final = 4242
_REJECTION: Final = "scripted partner rejection"
_REJECT_TEXT: Final = "The peer must reject this message"
_REPLY_TEXT: Final = "scripted reply"
_OWNED_MODEL: Final = "partner-claude"
_OWNED_UNREACHABLE_MODEL: Final = "partner-claude-unreachable"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")

_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": "Count this message"}]
_SYSTEM: Final = "You are a terse assistant that answers in one sentence"
_SYSTEM_BLOCKS: Final[list[JsonValue]] = [
    {"type": "text", "text": "You are a terse assistant"},
    {"type": "text", "text": "Answer in one sentence"},
]
_WEATHER_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City to look up"}},
    "required": ["city"],
}
_TOOLS: Final[list[JsonValue]] = [
    {"name": "get_weather", "description": "Look up the current weather for a city", "input_schema": _WEATHER_SCHEMA}
]
_OPENAI_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the current weather for a city",
            "parameters": _WEATHER_SCHEMA,
        },
    }
]
_RESPONSES_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Look up the current weather for a city",
        "parameters": _WEATHER_SCHEMA,
    }
]
_PEER_BARE: Final[dict[str, JsonValue]] = {"model": _BACKEND, "messages": _MESSAGES}
_PEER_FULL: Final[dict[str, JsonValue]] = {**_PEER_BARE, "system": _SYSTEM, "tools": _TOOLS}
_GEMINI_BODY: Final[dict[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": "Count this"}]}]}
_GEMINI_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": "Count this"}]

_REPLY: Final[dict[str, JsonValue]] = {
    "id": "msg_scripted",
    "type": "message",
    "role": "assistant",
    "model": _BACKEND,
    "content": [{"type": "text", "text": _REPLY_TEXT}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 3},
}
_EVENTS: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
    (
        "message_start",
        {
            "type": "message_start",
            "message": {**_REPLY, "content": [], "stop_reason": None, "usage": {"input_tokens": 5, "output_tokens": 1}},
        },
    ),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _REPLY_TEXT}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
)
_SSE: Final = tuple(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in _EVENTS)


def _counted(_request: Request) -> Reply:
    return Reply(body=json.dumps({"input_tokens": _PEER_COUNT}).encode())


def _rejected(status: int) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": _REJECTION}}).encode(),
    )


def _rejecting(status: int) -> Callable[[Request], Reply]:
    def count(_request: Request) -> Reply:
        return _rejected(status)

    return count


def _anthropic_message(message: JsonValue) -> bool:
    return isinstance(message, dict) and message.get("role") in ("user", "assistant")


def _anthropic_tool(tool: JsonValue) -> bool:
    return isinstance(tool, dict) and isinstance(tool.get("name"), str) and isinstance(tool.get("input_schema"), dict)


def _strict(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    messages: Final = body.get("messages")
    tools: Final = body.get("tools", [])
    accepted: Final = (
        isinstance(messages, list)
        and all(map(_anthropic_message, messages))
        and isinstance(body.get("system", ""), (str, list))
        and isinstance(tools, list)
        and all(map(_anthropic_tool, tools))
    )
    return _counted(request) if accepted else _rejected(400)


def _rejecting_marked_messages(request: Request) -> Reply:
    return _rejected(400) if _REJECT_TEXT in request.body.decode() else _counted(request)


def _peer(count: Callable[[Request], Reply] = _counted) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target.endswith("/count-tokens:rawPredict"):
            return count(request)
        if request.target.endswith(":streamRawPredict?alt=sse"):
            return Reply(content_type="text/event-stream", chunks=_SSE)
        if request.target.endswith(f"/{_BACKEND}:rawPredict"):
            return Reply(body=json.dumps(_REPLY).encode())
        return Reply(status=404, body=json.dumps({"error": f"unscripted target {request.target}"}).encode())

    return respond


def _count_requests(requests: Sequence[Request]) -> tuple[Request, ...]:
    return tuple(request for request in requests if "count-tokens" in request.target)


def _count_bodies(requests: Sequence[Request], target: str = _COUNT_TARGET) -> tuple[dict[str, JsonValue], ...]:
    counts: Final = _count_requests(requests)
    for request in counts:
        assert (request.method, request.target) == ("POST", target), request.target
        assert request.headers["authorization"] == "Bearer scripted-token", request.headers
    return tuple(_JSON_OBJECT.validate_json(request.body) for request in counts)


def _counted_bodies(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return _count_bodies(wire.drain())


def _bare(model: str) -> dict[str, JsonValue]:
    return {"model": model, "messages": _MESSAGES}


def _full(model: str) -> dict[str, JsonValue]:
    return {**_bare(model), "system": _SYSTEM, "tools": _TOOLS}


def _deployment(gateway: Gateway, scenario: Scenario, api_base: str, **overrides: JsonValue) -> str:
    return scenario.model(
        **{
            "model": f"vertex_ai/{_BACKEND}",
            "api_base": api_base,
            "api_key": None,
            "vertex_project": _PROJECT,
            "vertex_location": _LOCATION,
            "vertex_credentials": service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
            **overrides,
        }
    )


def _count(gateway: Gateway, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/v1/messages/count_tokens", body)


def _payload(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def _local_count(gateway: Gateway, body: Mapping[str, JsonValue]) -> int:
    response: Final = gateway.request("POST", "/utils/token_counter", body, params={"call_endpoint": "false"})
    payload: Final = _payload(response)
    total: Final = payload["total_tokens"]
    assert payload["tokenizer_type"] != "vertex_ai_partner_models", response.text
    assert isinstance(total, int) and 0 < total != _PEER_COUNT, response.text
    return total


def _closed_port_url() -> str:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{reserve.getsockname()[1]}"


def _clients(stack: ExitStack, base_url: str, count: int) -> tuple[httpx.Client, ...]:
    return tuple(
        stack.enter_context(httpx.Client(base_url=base_url, timeout=30, trust_env=False)) for _ in range(count)
    )


def _counted_on(client: httpx.Client, key: str, body: Mapping[str, JsonValue]) -> tuple[int, JsonValue]:
    response: Final = client.post(
        "/v1/messages/count_tokens", json=dict(body), headers={"Authorization": f"Bearer {key}"}
    )
    return response.status_code, _JSON_OBJECT.validate_json(response.content).get("input_tokens")


def _generated_then_counted(
    client: httpx.Client, key: str, model: str, body: Mapping[str, JsonValue]
) -> tuple[int, int, JsonValue]:
    generated: Final = client.post(
        "/v1/messages",
        json={
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": f"Generate before counting {uuid.uuid4().hex}"}],
        },
        headers={"Authorization": f"Bearer {key}"},
    )
    return generated.status_code, *_counted_on(client, key, body)


def _local_port(client: httpx.Client) -> int:
    with client.stream("GET", "/health/liveliness") as response:
        port: Final = int(response.extensions["network_stream"].get_extra_info("client_addr")[1])
        response.read()
    assert response.status_code == 200, response.text
    return port


def _counted_or_dropped(client: httpx.Client, key: str, body: Mapping[str, JsonValue]) -> tuple[int, JsonValue] | None:
    try:
        return _counted_on(client, key, body)
    except httpx.TransportError:
        return None


def _accepted_client_ports(pid: int, proxy_port: int) -> frozenset[int]:
    return frozenset(
        connection.raddr.port
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.raddr and connection.laddr.port == proxy_port
    )


def _owned_config(
    path: Path, gateway: Gateway, api_bases: Mapping[str, str], settings: Mapping[str, JsonValue]
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": name,
                        "litellm_params": {
                            "model": f"vertex_ai/{_BACKEND}",
                            "api_base": api_base,
                            "vertex_project": _PROJECT,
                            "vertex_location": _LOCATION,
                            "vertex_credentials": service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
                        },
                    }
                    for name, api_base in api_bases.items()
                ],
                "litellm_settings": {**config["litellm_settings"], **settings},
            }
        )
    )
    return path


@pytest.mark.parametrize("system", [_SYSTEM, _SYSTEM_BLOCKS], ids=["string", "blocks"])
def test_messages_count_tokens_forwards_system_and_tools(gateway: Gateway, system: JsonValue) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = _count(gateway, {**_bare(model), "system": system, "tools": _TOOLS})
        assert _payload(response) == {"input_tokens": _PEER_COUNT}, response.text
        assert _counted_bodies(wire) == ({**_PEER_BARE, "system": system, "tools": _TOOLS},)


def test_messages_count_tokens_without_system_or_tools_sends_bare_body(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = _count(gateway, _bare(model))
        assert _payload(response) == {"input_tokens": _PEER_COUNT}, response.text
        assert _counted_bodies(wire) == (_PEER_BARE,)


def test_utils_token_counter_call_endpoint_forwards_system_and_tools(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request(
            "POST", "/utils/token_counter", _full(model), params={"call_endpoint": "true"}
        )
        payload: Final = _payload(response)
        assert (payload["total_tokens"], payload["tokenizer_type"]) == (_PEER_COUNT, "vertex_ai_partner_models")
        assert (payload["request_model"], payload["model_used"]) == (model, _BACKEND), response.text
        assert _counted_bodies(wire) == (_PEER_FULL,)


def test_utils_token_counter_local_mode_never_calls_the_peer(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        assert _local_count(gateway, _full(model)) > 0
        assert wire.drain() == ()


def test_utils_token_counter_falls_back_locally_when_peer_rejects_openai_tools(gateway: Gateway) -> None:
    with wire_server(_peer(_strict)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        body: Final = {**_bare(model), "tools": _OPENAI_TOOLS}
        response: Final = gateway.request("POST", "/utils/token_counter", body, params={"call_endpoint": "true"})
        payload: Final = _payload(response)
        assert _counted_bodies(wire) == ({**_PEER_BARE, "tools": _OPENAI_TOOLS},)
        assert payload["total_tokens"] == _local_count(gateway, body), response.text
        assert payload["tokenizer_type"] != "vertex_ai_partner_models", response.text


def test_responses_input_tokens_counts_through_the_partner_peer(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request(
            "POST", "/v1/responses/input_tokens", {"model": model, "input": "Count this message"}
        )
        assert _payload(response) == {"object": "response.input_tokens", "input_tokens": _PEER_COUNT}, response.text
        assert _counted_bodies(wire) == (_PEER_BARE,)


def test_responses_input_tokens_falls_back_locally_when_peer_rejects(gateway: Gateway) -> None:
    with wire_server(_peer(_strict)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/responses/input_tokens",
            {"model": model, "input": "Count this message", "instructions": "Be terse", "tools": _RESPONSES_TOOLS},
        )
        payload: Final = _payload(response)
        (sent,) = _counted_bodies(wire)
        assert sent["tools"] == _RESPONSES_TOOLS, sent
        local: Final = _local_count(gateway, {"model": model, "messages": sent["messages"], "tools": _RESPONSES_TOOLS})
        assert payload == {"object": "response.input_tokens", "input_tokens": local}, response.text


def test_gemini_count_tokens_route_reaches_the_partner_peer(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request("POST", f"/v1beta/models/{model}:countTokens", _GEMINI_BODY)
        assert "totalTokens" in _payload(response), response.text
        assert _counted_bodies(wire) == ({"model": _BACKEND, "messages": _GEMINI_MESSAGES},)


def test_gemini_count_tokens_route_falls_back_locally_when_peer_rejects(gateway: Gateway) -> None:
    with wire_server(_peer(_rejecting(400))) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request("POST", f"/v1beta/models/{model}:countTokens", _GEMINI_BODY)
        payload: Final = _payload(response)
        assert _counted_bodies(wire) == ({"model": _BACKEND, "messages": _GEMINI_MESSAGES},)
        assert payload["totalTokens"] == _local_count(gateway, {"model": model, "messages": _GEMINI_MESSAGES})


@pytest.mark.parametrize("status", [400, 500, 503])
def test_messages_count_tokens_falls_back_locally_when_peer_errors(gateway: Gateway, status: int) -> None:
    with wire_server(_peer(_rejecting(status))) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = _count(gateway, _full(model))
        payload: Final = _payload(response)
        assert _counted_bodies(wire) == (_PEER_FULL,)
        assert payload == {"input_tokens": _local_count(gateway, _full(model))}, response.text


def test_messages_count_tokens_falls_back_locally_when_token_endpoint_rejects(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        if request.target == "/_oauth/token":
            return Reply(
                status=400,
                body=json.dumps({"error": "invalid_grant", "error_description": "scripted refusal"}).encode(),
            )
        return _peer()(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(
            gateway, scenario, wire.url, vertex_credentials=service_account_json(_PROJECT, wire.url)
        )
        response: Final = _count(gateway, _full(model))
        payload: Final = _payload(response)
        targets: Final = frozenset(request.target for request in wire.drain())
        assert targets == {"/_oauth/token"}, targets
        assert payload == {"input_tokens": _local_count(gateway, _full(model))}, response.text


def test_messages_count_tokens_falls_back_locally_when_peer_is_unreachable(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, _closed_port_url())
        response: Final = _count(gateway, _full(model))
        assert _payload(response) == {"input_tokens": _local_count(gateway, _full(model))}, response.text


@pytest.mark.parametrize(
    "tools",
    [5, "", "x" * 5120, ["get_weather"]],
    ids=["int", "empty_string", "5kb_string", "list_of_strings"],
)
def test_messages_count_tokens_rejects_malformed_tools_without_calling_the_peer(
    gateway: Gateway, tools: JsonValue
) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        refused: Final = _count(gateway, {**_bare(model), "tools": tools})
        assert 400 <= refused.status_code < 600, refused.text
        assert "input_tokens" not in refused.text, refused.text
        assert wire.drain() == ()
        assert _generated_then_counted(gateway.client, gateway.key, model, _bare(model)) == (200, 200, _PEER_COUNT)
        assert _counted_bodies(wire) == (_PEER_BARE,)


def test_messages_count_tokens_forwards_an_empty_tools_list(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = _count(gateway, {**_bare(model), "tools": []})
        assert _payload(response) == {"input_tokens": _PEER_COUNT}, response.text
        assert _counted_bodies(wire) == ({**_PEER_BARE, "tools": []},)


def test_messages_count_tokens_forwards_an_empty_system_string(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = _count(gateway, {**_bare(model), "system": ""})
        assert _payload(response) == {"input_tokens": _PEER_COUNT}, response.text
        assert _counted_bodies(wire) == ({**_PEER_BARE, "system": ""},)


def test_messages_count_tokens_leaves_null_system_and_tools_out(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = _count(gateway, {**_bare(model), "system": None, "tools": None})
        assert _payload(response) == {"input_tokens": _PEER_COUNT}, response.text
        assert _counted_bodies(wire) == (_PEER_BARE,)


def test_messages_count_tokens_falls_back_locally_when_peer_rejects_a_non_text_system(gateway: Gateway) -> None:
    with wire_server(_peer(_strict)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = _count(gateway, {**_bare(model), "system": 5})
        payload: Final = _payload(response)
        assert _counted_bodies(wire) == ({**_PEER_BARE, "system": 5},)
        assert payload == {"input_tokens": _local_count(gateway, _bare(model))}, response.text


def test_messages_count_tokens_forwards_a_5kb_system_verbatim(gateway: Gateway) -> None:
    system: Final = "Answer in one sentence. " * 214
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = _count(gateway, {**_bare(model), "system": system})
        assert _payload(response) == {"input_tokens": _PEER_COUNT}, response.text
        assert _counted_bodies(wire) == ({**_PEER_BARE, "system": system},)


def test_messages_count_tokens_duplicate_system_and_tools_keys_forward_one_value_each(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        fields: Final = f'"system": {json.dumps(_SYSTEM)}, "tools": {json.dumps(_TOOLS)}'
        response: Final = gateway.client.post(
            "/v1/messages/count_tokens",
            content=f'{{"model": "{model}", "messages": {json.dumps(_MESSAGES)}, {fields}, {fields}}}',
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        assert _payload(response) == {"input_tokens": _PEER_COUNT}, response.text
        (sent,) = _count_requests(wire.drain())
        assert _count_bodies((sent,)) == (_PEER_FULL,)
        assert (sent.body.count(b'"system"'), sent.body.count(b'"tools"')) == (1, 1), sent.body


def test_messages_count_tokens_unauthenticated_request_never_reaches_the_peer(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request(
            "POST", "/v1/messages/count_tokens", _full(model), key="sk-not-a-key-this-proxy-issued"
        )
        assert response.status_code == 401, response.text
        assert "input_tokens" not in response.text, response.text
        assert wire.drain() == ()


@pytest.mark.parametrize("fields", [{}, {"messages": []}], ids=["missing", "empty"])
def test_messages_count_tokens_without_messages_is_rejected(gateway: Gateway, fields: dict[str, JsonValue]) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = _count(gateway, {"model": model, "system": _SYSTEM, "tools": _TOOLS, **fields})
        assert response.status_code == 400, response.text
        assert "messages parameter is required" in response.text, response.text
        assert wire.drain() == ()


def test_count_tokens_location_override_targets_the_count_region(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(
            gateway, scenario, wire.url, vertex_location="global", vertex_count_tokens_location="europe-west1"
        )
        response: Final = _count(gateway, _bare(model))
        assert _payload(response) == {"input_tokens": _PEER_COUNT}, response.text
        target: Final = _COUNT_TARGET.replace(f"/locations/{_LOCATION}/", "/locations/europe-west1/")
        assert _count_bodies(wire.drain(), target) == (_PEER_BARE,)


def test_messages_count_tokens_repeated_request_reaches_the_peer_each_time(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        answers: Final = tuple(_payload(_count(gateway, _bare(model))) for _ in range(2))
        assert answers == ({"input_tokens": _PEER_COUNT},) * 2
        assert _counted_bodies(wire) == (_PEER_BARE,) * 2


@pytest.mark.timeout(240)  # boots an owned two-worker proxy with litellm_settings.disable_token_counter
def test_disabled_token_counter_surfaces_provider_failures_instead_of_counting_locally(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_peer(_rejecting_marked_messages)) as wire:
        config: Final = _owned_config(
            tmp_path / "disabled-token-counter.yaml",
            gateway,
            {_OWNED_MODEL: wire.url, _OWNED_UNREACHABLE_MODEL: _closed_port_url()},
            {"disable_token_counter": True},
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            counted: Final = _count(owned.gateway, _full(_OWNED_MODEL))
            assert _payload(counted) == {"input_tokens": _PEER_COUNT}, counted.text
            rejected: Final = _count(
                owned.gateway, {**_full(_OWNED_MODEL), "messages": [{"role": "user", "content": _REJECT_TEXT}]}
            )
            assert rejected.status_code == 400, rejected.text
            assert _REJECTION in rejected.text and "input_tokens" not in rejected.text, rejected.text
            unreachable: Final = _count(owned.gateway, _full(_OWNED_UNREACHABLE_MODEL))
            assert 500 <= unreachable.status_code < 600, unreachable.text
            assert "input_tokens" not in unreachable.text, unreachable.text
            local: Final = owned.gateway.request(
                "POST", "/utils/token_counter", _full(_OWNED_MODEL), params={"call_endpoint": "false"}
            )
            assert local.status_code == 503, local.text
            assert len(_counted_bodies(wire)) == 2


def test_peer_outage_between_concurrent_waves_falls_back_then_recovers(gateway: Gateway) -> None:
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 8)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        scenario: Final = stack.enter_context(gateway.scenario())
        with wire_server(_peer()) as wire:
            port: Final = int(wire.url.rsplit(":", 1)[1])
            model: Final = _deployment(gateway, scenario, wire.url)
            body: Final = _full(model)
            local: Final = _local_count(gateway, body)

            def generate_then_count(client: httpx.Client) -> tuple[int, int, JsonValue]:
                return _generated_then_counted(client, gateway.key, model, body)

            assert tuple(pool.map(generate_then_count, clients)) == ((200, 200, _PEER_COUNT),) * len(clients)
            assert _counted_bodies(wire) == (_PEER_FULL,) * len(clients)
        outage: Final = tuple(pool.map(lambda client: _counted_on(client, gateway.key, body), clients))
        assert outage == ((200, local),) * len(clients)
        with wire_server(_peer(), port=port) as revived:
            assert tuple(pool.map(generate_then_count, clients)) == ((200, 200, _PEER_COUNT),) * len(clients)
            assert _counted_bodies(revived) == (_PEER_FULL,) * len(clients)


def test_slow_peer_holds_concurrent_counts_without_stalling_the_proxy(gateway: Gateway) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()

    def hold(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=20), "Held count was never released"
        return _counted(request)

    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 6)
        wire: Final = stack.enter_context(wire_server(_peer(hold)))
        scenario: Final = stack.enter_context(gateway.scenario())
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        model: Final = _deployment(gateway, scenario, wire.url)
        try:
            futures: Final = tuple(
                pool.submit(_generated_then_counted, client, gateway.key, model, _full(model)) for client in clients
            )
            eventually(held.qsize, lambda size: size == len(clients), seconds=30)
            assert gateway.request("GET", "/health/liveliness").status_code == 200
            assert _local_count(gateway, _full(model)) > 0
            assert not any(future.done() for future in futures)
        finally:
            release.set()
        assert tuple(future.result(timeout=30) for future in futures) == ((200, 200, _PEER_COUNT),) * len(clients)
        assert len(_counted_bodies(wire)) == len(clients)


@pytest.mark.timeout(300)  # boots an owned two-worker proxy, kills one worker, and waits for its replacement
def test_worker_sigkill_mid_burst_leaves_the_sibling_counting(gateway: Gateway, tmp_path: Path) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()

    def hold(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=60), "Held count was never released"
        return _counted(request)

    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_peer(hold)))
        stack.callback(release.set)
        config: Final = _owned_config(tmp_path / "worker-kill.yaml", gateway, {_OWNED_MODEL: wire.url}, {})
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2))
        proxy_url: Final = owned.gateway.client.base_url
        workers: Final = eventually(
            lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
            lambda pids: len(pids) == 2,
            seconds=30,
        )
        clients: Final = _clients(stack, str(proxy_url), 12)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        stack.callback(release.set)
        ports: Final = tuple(_local_port(client) for client in clients)
        futures: Final = tuple(
            pool.submit(_counted_or_dropped, client, gateway.key, _full(_OWNED_MODEL)) for client in clients
        )
        eventually(held.qsize, lambda size: size == len(clients), seconds=30)
        shares: Final = {pid: _accepted_client_ports(pid, proxy_url.port or 0) & frozenset(ports) for pid in workers}
        assert sum(map(len, shares.values())) == len(clients), shares
        victim: Final = min((pid for pid in workers if shares[pid]), key=lambda pid: len(shares[pid]))
        psutil.Process(victim).send_signal(signal.SIGKILL)
        release.set()
        results: Final = tuple(future.result(timeout=60) for future in futures)
        for port, result in zip(ports, results, strict=True):
            assert result == (None if port in shares[victim] else (200, _PEER_COUNT)), (port, result, shares)
        second_wave: Final = _clients(stack, str(proxy_url), 6)
        assert tuple(_counted_on(client, gateway.key, _full(_OWNED_MODEL)) for client in second_wave) == (
            (200, _PEER_COUNT),
        ) * len(second_wave)
        assert len(_counted_bodies(wire)) == len(clients) + len(second_wave)
        eventually(lambda: len(_STARTED_WORKER.findall(owned.log.read_text())), lambda started: started >= 3, 120)
        assert owned.process.poll() is None


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_completions_on_the_same_deployment_still_generate(gateway: Gateway, stream: bool) -> None:
    marker: Final = f"chat control {uuid.uuid4().hex}"
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "stream": stream,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert _REPLY_TEXT in response.text, response.text
        assert not stream or response.text.rstrip().endswith("data: [DONE]"), response.text
        (sent,) = wire.drain()
        assert sent.target == (_STREAM_TARGET if stream else _MESSAGE_TARGET), sent.target
        assert marker in sent.body.decode(), sent.body


def test_messages_endpoint_on_the_same_deployment_still_generates(gateway: Gateway) -> None:
    marker: Final = f"messages control {uuid.uuid4().hex}"
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]},
        )
        assert response.status_code == 200, response.text
        assert _REPLY_TEXT in response.text, response.text
        (sent,) = wire.drain()
        assert sent.target == _MESSAGE_TARGET, sent.target
        assert marker in sent.body.decode(), sent.body


def test_responses_endpoint_on_the_same_deployment_still_generates(gateway: Gateway) -> None:
    marker: Final = f"responses control {uuid.uuid4().hex}"
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": marker})
        assert response.status_code == 200, response.text
        assert _REPLY_TEXT in response.text, response.text
        (sent,) = wire.drain()
        assert sent.target == _MESSAGE_TARGET, sent.target
        assert marker in sent.body.decode(), sent.body
