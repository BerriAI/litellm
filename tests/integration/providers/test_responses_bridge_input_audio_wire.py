import asyncio
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import anthropic
import httpx
import openai
import pytest
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, eventually, gateway_from_environment, string_value
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(120)

Mode: TypeAlias = Literal["on", "off"]

_MODES: Final[tuple[Mode, ...]] = ("on", "off")
_CAPABLE_MODEL: Final = "openai/responses/gpt-audio-mini"
_CAPABLE_WIRE_MODEL: Final = "gpt-audio-mini"
_HOSTILE: Final[tuple[tuple[str, JsonValue], ...]] = (
    ("int", 7),
    ("list", ["Zm9v"]),
    ("empty-string", ""),
    ("5kb-string", "x" * 5000),
)
_HOSTILE_IDS: Final = tuple(name for name, _ in _HOSTILE)
_HOSTILE_VALUES: Final = tuple(value for _, value in _HOSTILE)


@dataclass(frozen=True, slots=True)
class _Bridge:
    gateway: Gateway
    wire: Wire
    on: str
    off: str
    null: str
    capable_on: str
    base_model_param_on: str
    base_model_info_on: str
    injecting_off: str
    spend: pcb.SpendLogs

    def model(self, mode: Mode) -> str:
        return self.on if mode == "on" else self.off


@pytest.fixture(scope="module")
def bridge() -> Iterator[_Bridge]:
    with (
        wire_server(pcb.respond) as wire,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
        pcb.spend_logs() as spend,
    ):
        api_base: Final = f"{wire.url}/v1"
        yield _Bridge(
            gateway,
            wire,
            scenario.model(model=pcb.MODEL, api_base=api_base, drop_params=True),
            scenario.model(model=pcb.MODEL, api_base=api_base),
            scenario.model(model=pcb.MODEL, api_base=api_base, drop_params=None),
            scenario.model(model=_CAPABLE_MODEL, api_base=api_base, drop_params=True),
            scenario.model(model=pcb.MODEL, api_base=api_base, drop_params=True, base_model=_CAPABLE_WIRE_MODEL),
            scenario.model(
                model=pcb.MODEL,
                api_base=api_base,
                drop_params=True,
                model_info={"base_model": _CAPABLE_WIRE_MODEL},
            ),
            scenario.model(model=pcb.MODEL, api_base=api_base, **pcb.INJECTION),
            spend,
        )


def _v1(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/") + "/v1"


def _text(marker: str) -> dict[str, JsonValue]:
    return {"type": "input_text", "text": pcb.prompt(marker)}


def _audio_on_wire(payload: JsonValue) -> dict[str, JsonValue]:
    return {"type": "input_audio", "input_audio": payload}


def _user(marker: str, *parts: JsonValue) -> dict[str, JsonValue]:
    return {"role": "user", "content": [pcb.text(pcb.prompt(marker)), *parts]}


def _chat(bridge: _Bridge, model: str, messages: Sequence[JsonValue], *, stream: bool = False) -> httpx.Response:
    return bridge.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": list(messages), "stream": stream, **pcb.NO_CACHE},
    )


def _completion(response: httpx.Response, marker: str) -> str:
    assert response.status_code == 200, response.text
    body: Final = rv.JSON_OBJECT.validate_json(response.text)
    assert pcb.answers(string_value(body["id"]), marker), body
    (choice,) = rv.ITEMS.validate_python(body["choices"])
    assert rv.JSON_OBJECT.validate_python(choice["message"])["content"] == rv.answer(marker), body
    return response.headers["x-litellm-call-id"]


def _wire_request(bridge: _Bridge, marker: str, *, model: str = "gpt-6.1-sol", stream: bool = False) -> Request:
    request: Final = pcb.posted(bridge.wire, marker)
    body: Final = pcb.body_of(request)
    assert body["model"] == model, body
    assert (body.get("stream") is True) is stream, body
    return request


def _user_content_on_wire(
    bridge: _Bridge, marker: str, *, model: str = "gpt-6.1-sol", stream: bool = False
) -> list[dict[str, JsonValue]]:
    return pcb.content_of(pcb.input_items(_wire_request(bridge, marker, model=model, stream=stream)), "user")


def _expected(mode: Mode, marker: str, *forwarded: JsonValue) -> list[JsonValue]:
    return [_text(marker)] if mode == "on" else [_text(marker), *forwarded]


def test_audio_part_is_dropped_under_drop_params(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.on, [_user(marker, pcb.audio())]), marker)
    assert _user_content_on_wire(bridge, marker) == [_text(marker)]
    bridge.spend.landed(bridge.on, call_id, marker)


def test_audio_part_is_forwarded_as_input_audio_without_drop_params(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.off, [_user(marker, pcb.audio())]), marker)
    assert _user_content_on_wire(bridge, marker) == [_text(marker), _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))]
    bridge.spend.landed(bridge.off, call_id, marker)


@pytest.mark.parametrize("mode", _MODES)
def test_openai_sdk_stream_shapes_the_audio_part_by_mode(bridge: _Bridge, mode: Mode) -> None:
    marker: Final = uuid.uuid4().hex
    with openai.OpenAI(api_key=bridge.gateway.key, base_url=_v1(bridge.gateway), max_retries=0) as client:
        raw: Final = client.chat.completions.with_raw_response.create(
            model=bridge.model(mode),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": pcb.prompt(marker)},
                        {"type": "input_audio", "input_audio": {"data": "Zm9v", "format": "wav"}},
                    ],
                }
            ],
            stream=True,
            extra_body=dict(pcb.NO_CACHE),
        )
        chunks: Final = tuple(raw.parse())
    assert chunks and pcb.answers(chunks[0].id, marker), chunks
    streamed: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert streamed == rv.answer(marker), chunks
    content: Final = _user_content_on_wire(bridge, marker, stream=True)
    assert content == _expected(mode, marker, _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))), content
    bridge.spend.landed(bridge.model(mode), raw.headers["x-litellm-call-id"], marker)


@pytest.mark.parametrize("mode", _MODES)
async def test_async_openai_sdk_shapes_the_audio_part_by_mode(bridge: _Bridge, mode: Mode) -> None:
    marker: Final = uuid.uuid4().hex
    async with openai.AsyncOpenAI(api_key=bridge.gateway.key, base_url=_v1(bridge.gateway), max_retries=0) as client:
        raw: Final = await client.chat.completions.with_raw_response.create(
            model=bridge.model(mode),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": pcb.prompt(marker)},
                        {"type": "input_audio", "input_audio": {"data": "Zm9v", "format": "wav"}},
                    ],
                }
            ],
            extra_body=dict(pcb.NO_CACHE),
        )
    completion: Final = raw.parse()
    assert pcb.answers(completion.id, marker), completion
    assert completion.choices[0].message.content == rv.answer(marker), completion
    content: Final = _user_content_on_wire(bridge, marker)
    assert content == _expected(mode, marker, _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))), content
    bridge.spend.landed(bridge.model(mode), raw.headers["x-litellm-call-id"], marker)


def test_audio_capable_model_keeps_the_audio_part_under_drop_params(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.capable_on, [_user(marker, pcb.audio())]), marker)
    content: Final = _user_content_on_wire(bridge, marker, model=_CAPABLE_WIRE_MODEL)
    assert content == [_text(marker), _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))], content
    bridge.spend.landed(bridge.capable_on, call_id, marker)


def test_base_model_in_litellm_params_keeps_the_audio_part_under_drop_params(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.base_model_param_on, [_user(marker, pcb.audio())]), marker)
    assert _user_content_on_wire(bridge, marker) == [_text(marker), _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))]
    bridge.spend.landed(bridge.base_model_param_on, call_id, marker)


def test_base_model_in_model_info_keeps_the_audio_part_under_drop_params(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.base_model_info_on, [_user(marker, pcb.audio())]), marker)
    assert _user_content_on_wire(bridge, marker) == [_text(marker), _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))]
    bridge.spend.landed(bridge.base_model_info_on, call_id, marker)


@pytest.mark.parametrize("mode", _MODES)
def test_tool_output_audio_part_follows_the_mode(bridge: _Bridge, mode: Mode) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": pcb.prompt(marker)},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "record", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": [pcb.text("heard it"), pcb.audio()]},
    ]
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), messages), marker)
    output: Final = pcb.function_output(pcb.input_items(_wire_request(bridge, marker)), "call_1")
    heard: Final[dict[str, JsonValue]] = {"type": "input_text", "text": "heard it"}
    expected: Final[list[JsonValue]] = [heard] if mode == "on" else [heard, _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))]
    assert output == expected, output
    bridge.spend.landed(bridge.model(mode), call_id, marker)


def test_injected_system_marker_lands_on_a_trailing_audio_part_without_drop_params(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    system: Final[dict[str, JsonValue]] = {"role": "system", "content": [pcb.text("sys"), pcb.audio()]}
    call_id: Final = _completion(_chat(bridge, bridge.injecting_off, [system, _user(marker)]), marker)
    request: Final = _wire_request(bridge, marker)
    assert pcb.body_of(request)["prompt_cache_options"] == {"mode": "explicit"}, request.body
    system_content: Final = pcb.content_of(pcb.input_items(request), "system")
    assert system_content == [
        {"type": "input_text", "text": "sys"},
        pcb.marked(_audio_on_wire(dict(pcb.AUDIO_PAYLOAD)), pcb.EXPLICIT),
    ], system_content
    bridge.spend.landed(bridge.injecting_off, call_id, marker)


@pytest.mark.parametrize("mode", _MODES)
def test_assistant_audio_part_follows_the_mode(bridge: _Bridge, mode: Mode) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": pcb.prompt(marker)},
        {"role": "assistant", "content": [pcb.text("earlier answer"), pcb.audio()]},
        {"role": "user", "content": "and again"},
    ]
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), messages), marker)
    earlier: Final = pcb.content_of(pcb.input_items(_wire_request(bridge, marker)), "assistant")
    spoken: Final[dict[str, JsonValue]] = {"type": "output_text", "text": "earlier answer"}
    expected: Final[list[JsonValue]] = [spoken] if mode == "on" else [spoken, _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))]
    assert earlier == expected, earlier
    bridge.spend.landed(bridge.model(mode), call_id, marker)


def test_anthropic_sdk_request_carries_no_audio_part_into_the_bridge(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    client: Final = anthropic.Anthropic(
        base_url=str(bridge.gateway.client.base_url), api_key=bridge.gateway.key, max_retries=0
    )
    with client:
        raw: Final = client.messages.with_raw_response.create(
            model=bridge.on,
            max_tokens=64,
            messages=[{"role": "user", "content": [pcb.text(pcb.prompt(marker)), pcb.audio()]}],
            extra_body=dict(pcb.NO_CACHE),
        )
    (content,) = raw.parse().content
    assert content.type == "text" and content.text == rv.answer(marker), content
    assert _user_content_on_wire(bridge, marker) == [_text(marker)]
    bridge.spend.landed(bridge.on, raw.headers["x-litellm-call-id"], None)


def test_native_responses_audio_part_never_enters_the_bridge(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    audio: Final = _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))
    response: Final = bridge.gateway.request(
        "POST",
        "/v1/responses",
        {
            "model": bridge.on,
            "input": [{"type": "message", "role": "user", "content": [_text(marker), audio]}],
            **pcb.NO_CACHE,
        },
    )
    assert response.status_code == 200, response.text
    body: Final = rv.JSON_OBJECT.validate_json(response.text)
    assert pcb.answers(string_value(body["id"]), marker), body
    assert rv.answer(marker) in response.text, response.text
    assert _user_content_on_wire(bridge, marker) == [_text(marker), audio]
    bridge.spend.landed(bridge.on, response.headers["x-litellm-call-id"], marker)


@pytest.mark.parametrize("value", _HOSTILE_VALUES, ids=_HOSTILE_IDS)
def test_hostile_audio_value_is_forwarded_verbatim_without_drop_params(bridge: _Bridge, value: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    hostile: Final[dict[str, JsonValue]] = {"type": "input_audio", "input_audio": value}
    call_id: Final = _completion(_chat(bridge, bridge.off, [_user(marker, hostile)]), marker)
    assert _user_content_on_wire(bridge, marker) == [_text(marker), _audio_on_wire(value)]
    bridge.spend.landed(bridge.off, call_id, marker)


@pytest.mark.parametrize("value", _HOSTILE_VALUES, ids=_HOSTILE_IDS)
def test_hostile_audio_value_is_dropped_under_drop_params(bridge: _Bridge, value: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    hostile: Final[dict[str, JsonValue]] = {"type": "input_audio", "input_audio": value}
    call_id: Final = _completion(_chat(bridge, bridge.on, [_user(marker, hostile)]), marker)
    assert _user_content_on_wire(bridge, marker) == [_text(marker)]
    bridge.spend.landed(bridge.on, call_id, marker)


@pytest.mark.parametrize("mode", _MODES)
def test_audio_part_without_a_payload_follows_the_mode(bridge: _Bridge, mode: Mode) -> None:
    marker: Final = uuid.uuid4().hex
    bare: Final[dict[str, JsonValue]] = {"type": "input_audio"}
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), [_user(marker, bare)]), marker)
    content: Final = _user_content_on_wire(bridge, marker)
    assert content == _expected(mode, marker, _audio_on_wire(None)), content
    bridge.spend.landed(bridge.model(mode), call_id, marker)


@pytest.mark.parametrize("mode", _MODES)
def test_two_identical_audio_parts_follow_the_mode(bridge: _Bridge, mode: Mode) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), [_user(marker, pcb.audio(), pcb.audio())]), marker)
    audio: Final = _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))
    content: Final = _user_content_on_wire(bridge, marker)
    assert content == _expected(mode, marker, audio, audio), content
    bridge.spend.landed(bridge.model(mode), call_id, marker)


def test_audio_only_message_is_forwarded_with_empty_content_under_drop_params(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": [pcb.audio()]},
        {"role": "assistant", "content": "I could not hear that"},
        {"role": "user", "content": pcb.prompt(marker)},
    ]
    call_id: Final = _completion(_chat(bridge, bridge.on, messages), marker)
    items: Final = pcb.input_items(_wire_request(bridge, marker))
    users: Final = [item for item in items if item.get("role") == "user"]
    assert [item["content"] for item in users] == [[], [_text(marker)]], items
    bridge.spend.landed(bridge.on, call_id, marker)


def test_leading_audio_marker_has_no_preceding_part_to_carry_to(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final[list[JsonValue]] = [pcb.marked(pcb.audio(), pcb.EXPLICIT), pcb.text(pcb.prompt(marker))]
    call_id: Final = _completion(_chat(bridge, bridge.on, [{"role": "user", "content": content}]), marker)
    assert _user_content_on_wire(bridge, marker) == [_text(marker)]
    bridge.spend.landed(bridge.on, call_id, marker)


def test_each_dropped_audio_marker_moves_to_its_own_preceding_text(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final[list[JsonValue]] = [
        pcb.text(pcb.prompt(marker)),
        pcb.marked(pcb.audio(), pcb.EXPLICIT),
        pcb.text("second clip follows"),
        pcb.marked(pcb.audio(), pcb.EXPLICIT_30M),
    ]
    call_id: Final = _completion(_chat(bridge, bridge.on, [{"role": "user", "content": content}]), marker)
    assert _user_content_on_wire(bridge, marker) == [
        pcb.marked(_text(marker), pcb.EXPLICIT),
        {"type": "input_text", "text": "second clip follows", "prompt_cache_breakpoint": pcb.EXPLICIT_30M},
    ]
    bridge.spend.landed(bridge.on, call_id, marker)


def test_text_marker_wins_over_the_dropped_audio_marker(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final[list[JsonValue]] = [
        pcb.marked(pcb.text(pcb.prompt(marker)), pcb.EXPLICIT_30M),
        pcb.marked(pcb.audio(), pcb.EXPLICIT),
    ]
    call_id: Final = _completion(_chat(bridge, bridge.on, [{"role": "user", "content": content}]), marker)
    assert _user_content_on_wire(bridge, marker) == [pcb.marked(_text(marker), pcb.EXPLICIT_30M)]
    bridge.spend.landed(bridge.on, call_id, marker)


def test_null_drop_params_forwards_the_audio_part(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.null, [_user(marker, pcb.audio())]), marker)
    assert _user_content_on_wire(bridge, marker) == [_text(marker), _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))]
    bridge.spend.landed(bridge.null, call_id, marker)


@dataclass(frozen=True, slots=True)
class _Call:
    mode: Mode
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str
    call_id: str


def _calls(count: int) -> tuple[_Call, ...]:
    return tuple(_Call(_MODES[index % 2], index % 4 >= 2, uuid.uuid4().hex) for index in range(count))


async def _send(client: httpx.AsyncClient, bridge: _Bridge, model: str, call: _Call) -> _Served:
    body: Final[Mapping[str, JsonValue]] = {
        "model": model,
        "messages": [_user(call.marker, pcb.audio())],
        "stream": call.stream,
        **pcb.NO_CACHE,
    }
    async with client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {bridge.gateway.key}"}
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode(), response.headers["x-litellm-call-id"])


async def _burst(bridge: _Bridge, calls: Sequence[_Call], model_for: Mapping[Mode, str]) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(bridge.gateway.client.base_url), timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, bridge, model_for[call.mode], call) for call in calls)))


def _answered_in_its_own_shape(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(rv.MARKER.findall(served.text)) == {served.call.marker}, served.text
    assert served.text.startswith("data:") == served.call.stream, served.text
    assert rv.answer(served.call.marker) in served.text, served.text


def _attempts_by_marker(bridge: _Bridge, calls: Sequence[_Call]) -> Mapping[str, tuple[Request, ...]]:
    posts: Final = pcb.drained_posts(bridge.wire)
    marked: Final = tuple((rv.newest_marker(request.body.decode()), request) for request in posts)
    attempts: Final = {
        call.marker: tuple(request for marker, request in marked if marker == call.marker) for call in calls
    }
    assert sum(len(group) for group in attempts.values()) == len(posts), [request.body for request in posts]
    assert all(attempts.values()), sorted(marker for marker, group in attempts.items() if not group)
    return attempts


def _posts_by_marker(bridge: _Bridge, calls: Sequence[_Call]) -> Mapping[str, Request]:
    attempts: Final = _attempts_by_marker(bridge, calls)
    assert all(len(group) == 1 for group in attempts.values()), {
        marker: len(group) for marker, group in attempts.items()
    }
    return {marker: group[0] for marker, group in attempts.items()}


def _landed_once(bridge: _Bridge, model: str, served: Sequence[_Served], *, status: str = "success") -> None:
    rows: Final = eventually(
        lambda: bridge.spend.rows_for(model),
        lambda found: {string_value(row["litellm_call_id"]) for row in found} >= {item.call_id for item in served},
        seconds=70,
    )
    by_call: Final = {string_value(row["litellm_call_id"]): row for row in rows}
    assert len(by_call) == len(rows), rows
    for item in served:
        assert by_call[item.call_id]["status"] == status, (item.call_id, by_call[item.call_id])


async def test_mixed_audio_burst_shapes_every_upstream_request_by_its_mode(bridge: _Bridge) -> None:
    calls: Final = _calls(24)
    served: Final = await _burst(bridge, calls, {"on": bridge.on, "off": bridge.off})
    assert len(served) == 24
    for item in served:
        _answered_in_its_own_shape(item)
    by_marker: Final = _posts_by_marker(bridge, calls)
    for call in calls:
        body: Final = pcb.body_of(by_marker[call.marker])
        assert (body.get("stream") is True) is call.stream, body
        content: Final = pcb.content_of(pcb.input_items(by_marker[call.marker]), "user")
        assert content == _expected(call.mode, call.marker, _audio_on_wire(dict(pcb.AUDIO_PAYLOAD))), content
    _landed_once(bridge, bridge.on, tuple(item for item in served if item.call.mode == "on"))
    _landed_once(bridge, bridge.off, tuple(item for item in served if item.call.mode == "off"))


@dataclass(frozen=True, slots=True)
class _Doomed:
    markers: frozenset[str]

    def respond(self, request: Request) -> Reply:
        marker: Final = rv.newest_marker(request.body.decode()) if request.method == "POST" else None
        if marker in self.markers:
            return Reply(drop_connection=True)
        return pcb.respond(request)


async def test_dropped_upstream_connections_fail_only_their_own_calls(bridge: _Bridge) -> None:
    calls: Final = tuple(_Call("on", False, uuid.uuid4().hex) for _ in range(16))
    doomed: Final = _Doomed(frozenset(call.marker for index, call in enumerate(calls) if index % 4 == 0))
    with wire_server(doomed.respond) as wire, bridge.gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1", drop_params=True)
        rig: Final = _Bridge(
            bridge.gateway,
            wire,
            model,
            model,
            model,
            model,
            model,
            model,
            model,
            bridge.spend,
        )
        served: Final = await _burst(rig, calls, {"on": model, "off": model})
        failed: Final = tuple(item for item in served if item.call.marker in doomed.markers)
        answered: Final = tuple(item for item in served if item.call.marker not in doomed.markers)
        assert (len(failed), len(answered)) == (4, 12), [(item.call.marker, item.status) for item in served]
        for item in failed:
            assert item.status >= 500, (item.status, item.text)
            assert "answer marker" not in item.text, item.text
        for item in answered:
            _answered_in_its_own_shape(item)
        attempts: Final = _attempts_by_marker(rig, calls)
        for item in answered:
            assert len(attempts[item.call.marker]) == 1, attempts[item.call.marker]
        for call in calls:
            for request in attempts[call.marker]:
                assert pcb.content_of(pcb.input_items(request), "user") == [_text(call.marker)], request.body
        _landed_once(rig, model, answered)
        _landed_once(rig, model, failed, status="failure")
