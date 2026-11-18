import itertools
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, assert_never
from urllib.parse import urlsplit

import httpx
import pytest
from integration._support import anthropic_sse as sse
from integration._support import codex_vendor as cv
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, gateway_from_environment, object_value, string_value
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

Endpoint: TypeAlias = Literal["chat", "messages"]

_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages")
_PATHS: Final[Mapping[Endpoint, str]] = {"chat": "/v1/chat/completions", "messages": "/v1/messages"}
_RESPONSES_ROUTED: Final = "openai/responses/"
_OPENAI: Final = "openai/"
_ENDPOINT_PREFIX: Final[Mapping[Endpoint, str]] = {"chat": _RESPONSES_ROUTED, "messages": _OPENAI}
_AZURE_API_VERSION: Final = "2025-04-01-preview"
_SYSTEM: Final = "You are a careful reviewer. Cite the file you read."
_SOL: Final = "gpt-6.2-sol"
_PREVIOUS: Final = "gpt-6.1-sol"
_ALIAS: Final = "house-reasoner"
_NO_FIELDS: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_CHAT_TOOL: Final[Mapping[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
}
_MESSAGES_TOOL: Final[Mapping[str, JsonValue]] = {
    "name": "lookup",
    "description": "Look a file up",
    "input_schema": {"type": "object", "properties": {}},
}


@dataclass(frozen=True, slots=True)
class _Target:
    name: str
    base_model: str | None = None
    model_info: Mapping[str, JsonValue] | None = None


_TARGETS: Final[Mapping[str, _Target]] = {
    _SOL: _Target(_SOL),
    _PREVIOUS: _Target(_PREVIOUS),
    "astra": _Target("gpt-6.2-astra"),
    "dated-snapshot": _Target("gpt-6.2-sol-2026-11-05"),
    "unlisted-minor": _Target("gpt-6.3-sol"),
    "unlisted-major": _Target("gpt-7-sol"),
    "alias-model-info": _Target(_ALIAS, model_info={"base_model": _SOL}),
    "alias-litellm-param": _Target(_ALIAS, base_model=_SOL),
    "older-base-model-info": _Target(_SOL, model_info={"base_model": _PREVIOUS}),
    "older-base-litellm-param": _Target(_SOL, base_model=_PREVIOUS),
}
_LATER_MODELS: Final = ("astra", "dated-snapshot", "unlisted-minor", "unlisted-major")
_ALIASES: Final = ("alias-model-info", "alias-litellm-param")
_OLDER_BASES: Final = ("older-base-model-info", "older-base-litellm-param")


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire
    deployments: Mapping[tuple[str, str], str]
    azure: str

    def model(self, label: str, endpoint: Endpoint) -> str:
        return self.deployments[(label, _ENDPOINT_PREFIX[endpoint])]


@dataclass(frozen=True, slots=True)
class _ChatGptRig:
    gateway: Gateway
    wire: Wire
    model: str


@pytest.fixture(scope="module")
def rig() -> Iterator[_Rig]:
    with wire_server(pcb.respond) as wire, gateway_from_environment() as gateway, gateway.scenario() as scenario:
        yield _Rig(
            gateway,
            wire,
            {
                (label, prefix): scenario.model(
                    model=f"{prefix}{target.name}",
                    api_base=f"{wire.url}/v1",
                    model_info=target.model_info,
                    base_model=target.base_model,
                )
                for (label, target), prefix in itertools.product(_TARGETS.items(), (_RESPONSES_ROUTED, _OPENAI))
            },
            scenario.model(model=f"azure/responses/{_SOL}", api_base=wire.url, api_version=_AZURE_API_VERSION),
        )


@pytest.fixture(scope="module")
def chatgpt(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ChatGptRig]:
    directory: Final = tmp_path_factory.mktemp("chatgpt-system-prompt-rig")
    with gateway_from_environment() as gateway, wire_server(cv.CodexVendor().respond) as wire:
        overrides: Final = {"CHATGPT_TOKEN_DIR": str(cv.login(directory)), "CHATGPT_API_BASE": wire.url}
        config: Final = cv.proxy_config(directory, probe=False)
        with (
            owned_proxy_process(gateway, directory, overrides, config=config, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            yield _ChatGptRig(
                owned.gateway, wire, scenario.model(model=f"chatgpt/{_SOL}", api_base=wire.url, api_key=None)
            )


def _system(text: str) -> Mapping[str, JsonValue]:
    return {"role": "system", "content": text}


def _user(marker: str, suffix: str = "") -> Mapping[str, JsonValue]:
    return {"role": "user", "content": f"{pcb.prompt(marker)}{suffix}"}


def _post_chat(
    gateway: Gateway,
    model: str,
    messages: Sequence[Mapping[str, JsonValue]],
    *,
    stream: bool = False,
    fields: Mapping[str, JsonValue] = _NO_FIELDS,
) -> httpx.Response:
    return gateway.request(
        "POST",
        _PATHS["chat"],
        {
            "model": model,
            "messages": [dict(message) for message in messages],
            "stream": stream,
            **pcb.NO_CACHE,
            **fields,
        },
    )


def _post_messages(
    gateway: Gateway,
    model: str,
    marker: str,
    *,
    stream: bool = False,
    suffix: str = "",
    fields: Mapping[str, JsonValue] = _NO_FIELDS,
) -> httpx.Response:
    return gateway.request(
        "POST",
        _PATHS["messages"],
        {
            "model": model,
            "max_tokens": 64,
            "messages": [dict(_user(marker, suffix))],
            "stream": stream,
            **pcb.NO_CACHE,
            **fields,
        },
    )


def _send(
    gateway: Gateway,
    endpoint: Endpoint,
    model: str,
    marker: str,
    *,
    system: str | None = _SYSTEM,
    stream: bool = False,
    tools: bool = False,
    suffix: str = "",
    fields: Mapping[str, JsonValue] = _NO_FIELDS,
) -> httpx.Response:
    match endpoint:
        case "chat":
            return _post_chat(
                gateway,
                model,
                [*(() if system is None else (_system(system),)), _user(marker, suffix)],
                stream=stream,
                fields={**({"tools": [dict(_CHAT_TOOL)]} if tools else {}), **fields},
            )
        case "messages":
            return _post_messages(
                gateway,
                model,
                marker,
                stream=stream,
                suffix=suffix,
                fields={
                    **({} if system is None else {"system": system}),
                    **({"tools": [dict(_MESSAGES_TOOL)]} if tools else {}),
                    **fields,
                },
            )
        case _:
            assert_never(endpoint)


def _wire_body(wire: Wire, marker: str, *, stream: bool = False) -> Mapping[str, JsonValue]:
    (request,) = pcb.with_marker(pcb.drained_posts(wire), marker)
    assert urlsplit(request.target).path.endswith("/responses"), request.target
    body: Final = pcb.body_of(request)
    assert (body.get("stream") is True) is stream, body
    return body


def _items(body: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(rv.ITEMS.validate_python(body["input"]))


def _texts(item: Mapping[str, JsonValue]) -> tuple[str, ...]:
    parts: Final = rv.ITEMS.validate_python(item["content"])
    assert all(part.get("type") == "input_text" for part in parts), item
    return tuple(string_value(part["text"]) for part in parts)


def _assert_message(item: Mapping[str, JsonValue], role: str, text: str) -> None:
    assert (item.get("type"), item.get("role")) == ("message", role), item
    assert "\n".join(_texts(item)) == text, item


def _developer_item(texts: Sequence[str]) -> Mapping[str, JsonValue]:
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": text} for text in texts],
    }


def _assert_developer_item_leads(
    body: Mapping[str, JsonValue], texts: Sequence[str]
) -> tuple[Mapping[str, JsonValue], ...]:
    assert not body.get("instructions"), body
    first, *rest = _items(body)
    assert first == _developer_item(texts), body
    return tuple(rest)


def _assert_placed_for_caching(
    body: Mapping[str, JsonValue], marker: str, texts: Sequence[str] = (_SYSTEM,), suffix: str = ""
) -> None:
    (user,) = _assert_developer_item_leads(body, texts)
    _assert_message(user, "user", f"{pcb.prompt(marker)}{suffix}")


def _assert_kept_in_instructions(body: Mapping[str, JsonValue], marker: str) -> None:
    assert body.get("instructions") == _SYSTEM, body
    (user,) = _items(body)
    _assert_message(user, "user", pcb.prompt(marker))


def _assert_system_item_leads(body: Mapping[str, JsonValue], marker: str) -> None:
    assert not body.get("instructions"), body
    first, user = _items(body)
    _assert_message(first, "system", _SYSTEM)
    _assert_message(user, "user", pcb.prompt(marker))


def _chat_stream_choices(text: str) -> Iterator[Mapping[str, JsonValue]]:
    return itertools.chain.from_iterable(
        rv.ITEMS.validate_python(event.data.get("choices", [])) for event in sse.parse_sse(text)
    )


def _delta_text(choice: Mapping[str, JsonValue]) -> str:
    content: Final = object_value(choice["delta"]).get("content")
    return content if isinstance(content, str) else ""


def _chat_answer(response: httpx.Response, stream: bool) -> str:
    if stream:
        return "".join(map(_delta_text, _chat_stream_choices(response.text)))
    (choice,) = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(response.content)["choices"])
    return string_value(object_value(choice["message"])["content"])


def _messages_answer(response: httpx.Response, stream: bool) -> str:
    if stream:
        return sse.delta_text(sse.parse_sse(response.text))
    (block,) = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(response.content)["content"])
    return string_value(block["text"])


def _answer(response: httpx.Response, endpoint: Endpoint, stream: bool) -> str:
    match endpoint:
        case "chat":
            return _chat_answer(response, stream)
        case "messages":
            return _messages_answer(response, stream)
        case _:
            assert_never(endpoint)


def _assert_answered(response: httpx.Response, endpoint: Endpoint, marker: str, *, stream: bool = False) -> None:
    assert response.status_code == 200, response.text
    assert _answer(response, endpoint, stream) == rv.answer(marker), response.text


def _tool_names(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    return tuple(tool["name"] for tool in rv.ITEMS.validate_python(body["tools"]))


@pytest.mark.parametrize("tools", (False, True), ids=("no-tools", "tools"))
@pytest.mark.parametrize("stream", (False, True), ids=("unary", "stream"))
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_system_prompt_is_the_leading_developer_item(rig: _Rig, endpoint: Endpoint, stream: bool, tools: bool) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig.gateway, endpoint, rig.model(_SOL, endpoint), marker, stream=stream, tools=tools)
    body: Final = _wire_body(rig.wire, marker, stream=stream)
    assert body["model"] == _SOL, body
    _assert_placed_for_caching(body, marker)
    assert not tools or _tool_names(body) == ("lookup",), body
    _assert_answered(response, endpoint, marker, stream=stream)


def test_chat_bridged_by_function_tools_and_reasoning_leads_with_the_developer_item(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post_chat(
        rig.gateway,
        rig.deployments[(_SOL, _OPENAI)],
        [_system(_SYSTEM), _user(marker)],
        fields={"tools": [dict(_CHAT_TOOL)], "reasoning_effort": "low"},
    )
    body: Final = _wire_body(rig.wire, marker)
    _assert_placed_for_caching(body, marker)
    assert _tool_names(body) == ("lookup",), body
    _assert_answered(response, "chat", marker)


@pytest.mark.parametrize("label", (*_LATER_MODELS, *_ALIASES))
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_every_model_from_gpt_6_2_on_leads_with_the_developer_item(rig: _Rig, endpoint: Endpoint, label: str) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig.gateway, endpoint, rig.model(label, endpoint), marker)
    body: Final = _wire_body(rig.wire, marker)
    assert body["model"] == _TARGETS[label].name, body
    _assert_placed_for_caching(body, marker)
    _assert_answered(response, endpoint, marker)


def test_two_leading_chat_system_messages_become_one_developer_item(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post_chat(
        rig.gateway,
        rig.model(_SOL, "chat"),
        [_system("You are a careful reviewer."), _system("Cite the file you read."), _user(marker)],
    )
    _assert_placed_for_caching(_wire_body(rig.wire, marker), marker)
    _assert_answered(response, "chat", marker)


def test_json_object_request_sends_the_system_text_as_a_developer_item(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post_chat(
        rig.gateway,
        rig.model(_SOL, "chat"),
        [_system(_SYSTEM), _user(marker)],
        fields={"response_format": {"type": "json_object"}},
    )
    _assert_placed_for_caching(_wire_body(rig.wire, marker), marker)
    _assert_answered(response, "chat", marker)


def test_system_only_chat_request_sends_a_developer_item(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    text: Final = f"{_SYSTEM} {pcb.prompt(marker)}"
    response: Final = _post_chat(rig.gateway, rig.model(_SOL, "chat"), [_system(text)])
    assert _assert_developer_item_leads(_wire_body(rig.wire, marker), (text,)) == ()
    _assert_answered(response, "chat", marker)


@pytest.mark.parametrize("stream", (False, True), ids=("unary", "stream"))
def test_anthropic_system_text_blocks_become_the_leading_developer_item(rig: _Rig, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post_messages(
        rig.gateway,
        rig.model(_SOL, "messages"),
        marker,
        stream=stream,
        fields={"system": [{"type": "text", "text": "First rule"}, {"type": "text", "text": "Second rule"}]},
    )
    _assert_placed_for_caching(_wire_body(rig.wire, marker, stream=stream), marker, ("First rule", "Second rule"))
    _assert_answered(response, "messages", marker, stream=stream)


@pytest.mark.parametrize("stream", (False, True), ids=("unary", "stream"))
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_chatgpt_keeps_the_system_prompt_after_its_default_instructions(
    chatgpt: _ChatGptRig, endpoint: Endpoint, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(chatgpt.gateway, endpoint, chatgpt.model, marker, stream=stream)
    (request,) = pcb.with_marker(pcb.drained_posts(chatgpt.wire), marker)
    body: Final = cv.forwarded(request, marker)
    assert string_value(body["instructions"]).endswith(f"\n\n{_SYSTEM}"), body
    (user,) = _items(body)
    _assert_message(user, "user", pcb.prompt(marker))
    _assert_answered(response, endpoint, marker, stream=stream)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_gpt_6_1_keeps_the_system_prompt_in_instructions(rig: _Rig, endpoint: Endpoint) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig.gateway, endpoint, rig.model(_PREVIOUS, endpoint), marker)
    _assert_kept_in_instructions(_wire_body(rig.wire, marker), marker)
    _assert_answered(response, endpoint, marker)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_azure_host_of_gpt_6_2_keeps_the_system_prompt_in_instructions(rig: _Rig, endpoint: Endpoint) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig.gateway, endpoint, rig.azure, marker)
    _assert_kept_in_instructions(_wire_body(rig.wire, marker), marker)
    _assert_answered(response, endpoint, marker)


@pytest.mark.parametrize("label", _OLDER_BASES)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_older_declared_base_model_keeps_the_system_prompt_in_instructions(
    rig: _Rig, endpoint: Endpoint, label: str
) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig.gateway, endpoint, rig.model(label, endpoint), marker)
    body: Final = _wire_body(rig.wire, marker)
    assert body["model"] == _SOL, body
    _assert_kept_in_instructions(body, marker)
    _assert_answered(response, endpoint, marker)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_auto_truncation_keeps_the_system_prompt_in_instructions(rig: _Rig, endpoint: Endpoint) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig.gateway, endpoint, rig.model(_SOL, endpoint), marker, fields={"truncation": "auto"})
    body: Final = _wire_body(rig.wire, marker)
    assert body["truncation"] == "auto", body
    _assert_kept_in_instructions(body, marker)
    _assert_answered(response, endpoint, marker)


def test_chat_with_previous_response_id_keeps_the_system_prompt_in_instructions(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post_chat(
        rig.gateway,
        rig.model(_SOL, "chat"),
        [_system(_SYSTEM), _user(marker)],
        fields={"previous_response_id": "resp_prior"},
    )
    body: Final = _wire_body(rig.wire, marker)
    assert body["previous_response_id"] == "resp_prior", body
    _assert_kept_in_instructions(body, marker)
    _assert_answered(response, "chat", marker)


def test_json_object_with_previous_response_id_keeps_the_system_role(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post_chat(
        rig.gateway,
        rig.model(_SOL, "chat"),
        [_system(_SYSTEM), _user(marker)],
        fields={"previous_response_id": "resp_prior", "response_format": {"type": "json_object"}},
    )
    body: Final = _wire_body(rig.wire, marker)
    assert body["previous_response_id"] == "resp_prior", body
    _assert_system_item_leads(body, marker)
    _assert_answered(response, "chat", marker)


@pytest.mark.parametrize("prefix", (_RESPONSES_ROUTED, _OPENAI), ids=("responses-routed", "openai"))
@pytest.mark.parametrize("shape", ("string", "items"))
def test_direct_responses_request_is_forwarded_unchanged(rig: _Rig, shape: str, prefix: str) -> None:
    marker: Final = uuid.uuid4().hex
    prompt: Final = pcb.prompt(marker)
    sent: Final[JsonValue] = (
        prompt
        if shape == "string"
        else [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}]
    )
    response: Final = rig.gateway.request(
        "POST",
        "/v1/responses",
        {"model": rig.deployments[(_SOL, prefix)], "input": sent, "instructions": _SYSTEM, **pcb.NO_CACHE},
    )
    body: Final = _wire_body(rig.wire, marker)
    assert body["instructions"] == _SYSTEM, body
    assert body["input"] == sent, body
    assert response.status_code == 200, response.text
    assert rv.answer(marker) in response.text, response.text


@pytest.mark.parametrize("system", (None, ""), ids=("absent", "empty"))
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_no_developer_item_is_invented_without_a_system_prompt(
    rig: _Rig, endpoint: Endpoint, system: str | None
) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig.gateway, endpoint, rig.model(_SOL, endpoint), marker, system=system)
    body: Final = _wire_body(rig.wire, marker)
    assert not body.get("instructions"), body
    (user,) = _items(body)
    _assert_message(user, "user", pcb.prompt(marker))
    _assert_answered(response, endpoint, marker)


def test_json_object_on_gpt_6_1_keeps_the_leading_system_item(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post_chat(
        rig.gateway,
        rig.model(_PREVIOUS, "chat"),
        [_system(_SYSTEM), _user(marker)],
        fields={"response_format": {"type": "json_object"}},
    )
    _assert_system_item_leads(_wire_body(rig.wire, marker), marker)
    _assert_answered(response, "chat", marker)


@pytest.mark.parametrize("status", (400, 500))
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_upstream_error_reaches_the_caller_after_the_prompt_was_placed_for_caching(
    rig: _Rig, endpoint: Endpoint, status: int
) -> None:
    marker: Final = uuid.uuid4().hex
    suffix: Final = f" fail-{status}"
    response: Final = _send(rig.gateway, endpoint, rig.model(_SOL, endpoint), marker, suffix=suffix)
    posts: Final = pcb.with_marker(pcb.drained_posts(rig.wire), marker)
    assert posts, response.text
    for request in posts:
        _assert_placed_for_caching(pcb.body_of(request), marker, suffix=suffix)
    assert response.status_code // 100 == status // 100, response.text
    assert rv.answer(marker) not in response.text, response.text


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_malformed_messages_field_is_rejected_before_the_wire(rig: _Rig, endpoint: Endpoint) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.gateway.request(
        "POST",
        _PATHS[endpoint],
        {
            "model": rig.model(_SOL, endpoint),
            "messages": f"not a list {pcb.prompt(marker)}",
            **({"max_tokens": 64, "system": _SYSTEM} if endpoint == "messages" else {}),
            **pcb.NO_CACHE,
        },
    )
    assert response.status_code >= 400, response.text
    assert pcb.with_marker(pcb.drained_posts(rig.wire), marker) == (), response.text
