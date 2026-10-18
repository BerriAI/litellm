import asyncio
import json
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "llama3-custom-template-chaos"
_API_KEY: Final = "synthetic-ollama-key"
_CONFIG_MODEL: Final = "ollama-custom-template-chaos"
_SYSTEM: Final = "Answer in one word."
_ROLES: Final[dict[str, JsonValue]] = {
    "system": {"pre_message": "<sys>", "post_message": "</sys>"},
    "user": {"pre_message": "<user>", "post_message": "</user>"},
}
_REJECTION: Final = "initial_prompt_value: Input should be a valid string"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120

Endpoint = Literal["chat", "messages", "responses"]


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    marker: str
    rejected: bool = False


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _question(marker: str) -> str:
    return f"What colour is the sky? marker-{marker}"


def _prompt(marker: str) -> str:
    return f"<sys>{_SYSTEM}</sys><user>{_question(marker)}</user>"


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(model: str, call: _Call) -> dict[str, JsonValue]:
    common: Final[dict[str, JsonValue]] = {
        "model": model,
        "stream": call.stream,
        "num_retries": 0,
        "cache": {"no-cache": True},
        **({"initial_prompt_value": 5} if call.rejected else {}),
    }
    match call.endpoint:
        case "chat":
            return {
                **common,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _question(call.marker)},
                ],
            }
        case "messages":
            return {
                **common,
                "max_tokens": 64,
                "system": _SYSTEM,
                "messages": [{"role": "user", "content": _question(call.marker)}],
            }
        case "responses":
            return {**common, "store": False, "instructions": _SYSTEM, "input": _question(call.marker)}


def _generate_reply(marker: str, stream: bool, drop_connection: bool = False) -> Reply:
    done: Final = {
        "model": _BACKEND,
        "created_at": "2026-10-15T00:00:00Z",
        "response": "",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 30,
        "eval_count": 12,
    }
    if not stream:
        return Reply(
            body=json.dumps({**done, "response": f"blue marker-{marker}"}).encode(), drop_connection=drop_connection
        )
    frames: Final = (
        *(
            {"model": _BACKEND, "created_at": "2026-10-15T00:00:00Z", "response": piece, "done": False}
            for piece in ("blue ", f"marker-{marker}")
        ),
        done,
    )
    return Reply(
        content_type="application/x-ndjson",
        chunks=tuple(json.dumps(frame).encode() + b"\n" for frame in frames),
        drop_connection=drop_connection,
    )


def _is_generate(request: Request) -> bool:
    return (request.method, request.target) == ("POST", "/api/generate")


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _echo(request: Request) -> Reply:
    if not _is_generate(request):
        return Reply(body=b"{}")
    stream: Final = _JSON_OBJECT.validate_json(request.body).get("stream") is True
    return _generate_reply(_marker_of(request), stream)


def _assert_each_prompt_rendered_once(received: tuple[Request, ...], markers: frozenset[str]) -> None:
    generates: Final = tuple(request for request in received if _is_generate(request))
    assert sorted(_marker_of(request) for request in generates) == sorted(markers)
    for request in generates:
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["prompt"] == _prompt(_marker_of(request)), body["prompt"]
        assert "roles" not in body and "initial_prompt_value" not in body, sorted(body)


def _response_id(served: _Served) -> str | None:
    if served.call.endpoint == "responses":
        return None
    if not served.call.stream:
        identity: Final = _JSON_OBJECT.validate_json(served.text)["id"]
        assert isinstance(identity, str)
        return identity
    for line in served.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload: Final = _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        if served.call.endpoint == "chat":
            first: Final = payload["id"]
            assert isinstance(first, str)
            return first
        if payload.get("type") == "message_start":
            message: Final = payload["message"]
            assert isinstance(message, dict) and isinstance(message["id"], str)
            return message["id"]
    raise AssertionError(served.text)


def _spend_statuses(model: str, expected: int) -> MappingProxyType[str, str]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    statuses: Final = MappingProxyType({str(row["request_id"]): str(row["status"]) for row in rows})
    assert len(statuses) == len(rows) == expected, rows
    return statuses


def _successes(statuses: Mapping[str, str]) -> frozenset[str]:
    return frozenset(identity for identity, status in statuses.items() if status == "success")


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(model, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call=call, status=response.status_code, text=raw.decode())


async def _burst(
    base_url: str, key: str, model: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _calls(count: int, endpoints: tuple[Endpoint, ...], stream: Callable[[int], bool]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoints[index % len(endpoints)], stream=stream(index), marker=uuid.uuid4().hex)
        for index in range(count)
    )


def _assert_answered_with_its_own_marker(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(_MARKER.findall(served.text)) == {served.call.marker}, served.text


def _assert_rejected_by_name(served: _Served) -> None:
    assert served.status == 500, served.text
    assert _REJECTION in served.text and "marker-" not in served.text, served.text


async def test_concurrent_burst_renders_each_template_and_rejects_only_the_bad_requests(gateway: Gateway) -> None:
    calls: Final = tuple(
        _Call(endpoint=call.endpoint, stream=call.stream, marker=call.marker, rejected=index % 5 == 4)
        for index, call in enumerate(_calls(30, ("chat", "messages", "responses"), lambda index: index % 2 == 0))
    )
    accepted: Final = tuple(call for call in calls if not call.rejected)
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY, roles=_ROLES)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 30
        for item in served:
            if item.call.rejected:
                _assert_rejected_by_name(item)
            else:
                _assert_answered_with_its_own_marker(item)
        _assert_each_prompt_rendered_once(wire.drain(), frozenset(call.marker for call in accepted))
        answered: Final = tuple(item for item in served if not item.call.rejected)
        known: Final = frozenset(identity for identity in map(_response_id, answered) if identity is not None)
        assert len(known) == 16, known
        statuses: Final = _spend_statuses(model, 30)
        assert len(_successes(statuses)) == len(accepted), statuses
        assert known <= _successes(statuses)


async def test_dropped_ollama_connections_fail_their_callers_and_the_rest_keep_their_template(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(12, ("chat",), lambda index: index % 2 == 1)
    dropped: Final = frozenset(call.marker for index, call in enumerate(calls) if index % 3 == 0)

    def respond(request: Request) -> Reply:
        if not _is_generate(request):
            return Reply(body=b"{}")
        marker: Final = _marker_of(request)
        stream: Final = _JSON_OBJECT.validate_json(request.body).get("stream") is True
        return _generate_reply(marker, stream, drop_connection=marker in dropped)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY, roles=_ROLES)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 12
        for item in served:
            if item.call.marker in dropped:
                assert item.status == 500, item.text
                assert "APIConnectionError" in item.text and "marker-" not in item.text, item.text
            else:
                _assert_answered_with_its_own_marker(item)
        recovery: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
        (recovered,) = await _burst(str(gateway.client.base_url), gateway.key, model, (recovery,))
        _assert_answered_with_its_own_marker(recovered)
        _assert_each_prompt_rendered_once(wire.drain(), frozenset(call.marker for call in (*calls, recovery)))
        answered: Final = tuple(item for item in served if item.call.marker not in dropped)
        survivors: Final = frozenset(
            identity for identity in map(_response_id, (*answered, recovered)) if identity is not None
        )
        assert len(survivors) == 9, survivors
        statuses: Final = _spend_statuses(model, 13)
        assert _successes(statuses) == survivors, statuses
        assert sum(status == "failure" for status in statuses.values()) == len(dropped), statuses


def _owned_config(
    wires: tuple[Wire, ...],
    tmp_path: Path,
    template: Mapping[str, JsonValue],
    litellm_settings: Mapping[str, JsonValue],
    router_settings: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    deployments: Final = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {
                "model": f"ollama/{_BACKEND}",
                "api_base": wire.url,
                "api_key": _API_KEY,
                "order": order,
                **template,
            },
        }
        for order, wire in enumerate(wires, start=1)
    ]
    config: Final = {
        **base,
        "model_list": deployments,
        "litellm_settings": {**base.get("litellm_settings", {}), **litellm_settings},
        "router_settings": {**base.get("router_settings", {}), **router_settings},
    }
    path: Final = tmp_path / "ollama-custom-template-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_rendering_templates(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("chat",), lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        if not _is_generate(request):
            return Reply(body=b"{}")
        held_markers.put(_marker_of(request))
        assert release.wait(timeout=60), "The burst was never released"
        return _echo(request)

    with wire_server(held) as wire:
        path: Final = _owned_config((wire,), tmp_path, {"roles": _ROLES}, {})
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(
                    str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, calls, tolerate_transport_errors=True
                )
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                served: Final = await burst
                assert held_by[survivor_pid] >= 10, held_by
                assert len(served) == held_by[survivor_pid], (held_by, len(served))
                for item in served:
                    _assert_answered_with_its_own_marker(item)
            follow_up: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            _assert_answered_with_its_own_marker(answered)
            _assert_each_prompt_rendered_once(wire.drain(), frozenset(call.marker for call in (*calls, follow_up)))


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
async def test_rejected_templates_never_cool_down_the_preferred_deployment(gateway: Gateway, tmp_path: Path) -> None:
    rejected: Final = tuple(
        _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex, rejected=True) for _ in range(4)
    )
    valid: Final = _calls(8, ("chat",), lambda _: False)
    with wire_server(_echo) as preferred, wire_server(_echo) as standby:
        path: Final = _owned_config(
            (preferred, standby), tmp_path, {"roles": _ROLES}, {}, {"disable_cooldowns": False, "allowed_fails": 0}
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            base_url: Final = str(owned.gateway.client.base_url)
            for served in await _burst(base_url, owned.gateway.key, _CONFIG_MODEL, rejected):
                _assert_rejected_by_name(served)
            for served in await _burst(base_url, owned.gateway.key, _CONFIG_MODEL, valid):
                _assert_answered_with_its_own_marker(served)
            _assert_each_prompt_rendered_once(preferred.drain(), frozenset(call.marker for call in valid))
            assert not any(_is_generate(request) for request in standby.drain())


def _cached_chat(candidate: Gateway, marker: str, field: str, value: str) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": _CONFIG_MODEL,
            "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": _question(marker)}],
            field: value,
        },
    )


def _served_from_cache(candidate: Gateway, field: str) -> bool:
    marker: Final = uuid.uuid4().hex
    first: Final = _cached_chat(candidate, marker, field, "<first>")
    assert first.status_code == 200, first.text
    assert "x-litellm-cache-key" not in first.headers, sorted(first.headers)
    eventually(
        lambda: _cached_chat(candidate, marker, field, "<first>"),
        lambda found: "x-litellm-cache-key" in found.headers,
        seconds=20,
    )
    changed: Final = _cached_chat(candidate, marker, field, "<second>")
    assert changed.status_code == 200, changed.text
    return "x-litellm-cache-key" in changed.headers


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_initial_and_final_prompt_values_key_the_provider_specific_cache_alike(
    gateway: Gateway, tmp_path: Path
) -> None:
    template: Final[dict[str, JsonValue]] = {
        "roles": _ROLES,
        "initial_prompt_value": "<i>",
        "final_prompt_value": "<f>",
    }
    with wire_server(_echo) as wire:
        path: Final = _owned_config(
            (wire,), tmp_path, template, {"enable_caching_on_provider_specific_optional_params": True}
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            initial: Final = _served_from_cache(owned.gateway, "initial_prompt_value")
            final: Final = _served_from_cache(owned.gateway, "final_prompt_value")
            assert initial == final, {"initial_prompt_value": initial, "final_prompt_value": final}


def _gemini_reply(_: Request) -> Reply:
    candidate: Final[dict[str, JsonValue]] = {
        "content": {"role": "model", "parts": [{"text": '{"colour": "blue"}'}]},
        "finishReason": "STOP",
    }
    usage: Final[dict[str, JsonValue]] = {"promptTokenCount": 30, "candidatesTokenCount": 12, "totalTokenCount": 42}
    return Reply(body=json.dumps({"candidates": [candidate], "usageMetadata": usage}).encode())


def _texts(contents: JsonValue) -> tuple[str, ...]:
    assert isinstance(contents, list), contents
    parts: Final = tuple(part for content in contents if isinstance(content, dict) for part in _parts(content))
    return tuple(text for part in parts if isinstance(text := part.get("text"), str))


def _parts(content: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    parts: Final = content.get("parts")
    assert isinstance(parts, list), content
    return tuple(part for part in parts if isinstance(part, dict))


def _anthropic_text_reply(_: Request) -> Reply:
    completion: Final[dict[str, JsonValue]] = {
        "type": "completion",
        "id": "compl_scripted",
        "completion": "blue",
        "stop_reason": "stop_sequence",
    }
    return Reply(body=json.dumps(completion).encode())


def _boot_deployment(name: str, model: str, wire: Wire) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, "api_base": wire.url, "api_key": _API_KEY}}


def _boot_chat(candidate: Gateway, model: str, extra: Mapping[str, JsonValue]) -> None:
    response: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": _question(model)}],
            "cache": {"no-cache": True},
            **extra,
        },
    )
    assert response.status_code == 200, response.text


def _anthropic_text_prompts(wire: Wire) -> MappingProxyType[str, JsonValue]:
    bodies: Final = tuple(_JSON_OBJECT.validate_json(request.body) for request in wire.drain())
    return MappingProxyType({str(body["model"]): body["prompt"] for body in bodies})


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_boot_time_custom_prompt_dict_renders_partial_templates(gateway: Gateway, tmp_path: Path) -> None:
    templates: Final[dict[str, JsonValue]] = {
        "response_schema_prompt": {"roles": {"user": {"pre_message": "<schema>", "post_message": "</schema>"}}},
        "claude-2.1": {"roles": _ROLES},
        "claude-instant-1.2": {"initial_prompt_value": "<i>", "final_prompt_value": "<f>"},
    }
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {"colour": {"type": "string"}},
        "required": ["colour"],
    }
    with wire_server(_gemini_reply) as gemini, wire_server(_anthropic_text_reply) as anthropic_text:
        base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        deployments: Final = [
            _boot_deployment("gemini-schema", "gemini/scripted-template-model", gemini),
            _boot_deployment("anthropic-text-roles", "anthropic_text/claude-2.1", anthropic_text),
            _boot_deployment("anthropic-text-values", "anthropic_text/claude-instant-1.2", anthropic_text),
        ]
        settings: Final = {**base.get("litellm_settings", {}), "custom_prompt_dict": templates}
        path: Final = tmp_path / "boot-time-custom-prompt-dict.yaml"
        path.write_text(yaml.safe_dump({**base, "model_list": deployments, "litellm_settings": settings}))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            _boot_chat(
                owned.gateway,
                "gemini-schema",
                {"response_format": {"type": "json_schema", "json_schema": {"name": "sky", "schema": schema}}},
            )
            posts: Final = tuple(request for request in gemini.drain() if request.method == "POST")
            assert len(posts) == 1, [request.target for request in posts]
            texts: Final = _texts(_JSON_OBJECT.validate_json(posts[0].body)["contents"])
            wrapped: Final = tuple(text for text in texts if text.startswith("<schema>"))
            assert len(wrapped) == 1, texts
            assert wrapped[0].endswith("</schema>") and "colour" in wrapped[0], wrapped[0]
            _boot_chat(owned.gateway, "anthropic-text-roles", {})
            _boot_chat(owned.gateway, "anthropic-text-values", {})
            assert _anthropic_text_prompts(anthropic_text) == {
                "claude-2.1": f"<sys>{_SYSTEM}</sys><user>{_question('anthropic-text-roles')}</user>",
                "claude-instant-1.2": f"<i>{_SYSTEM}{_question('anthropic-text-values')}<f>",
            }
