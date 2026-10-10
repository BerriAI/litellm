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
from integration._support.wire import Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(120)

Mode: TypeAlias = Literal["on", "off"]

_UNKNOWN_KEY: Final[Mapping[str, JsonValue]] = {"mode": "explicit", "note": "kept"}
_MALFORMED: Final[tuple[tuple[str, JsonValue], ...]] = (
    ("string", "yes"),
    ("int", 1),
    ("list", ["explicit"]),
    ("empty-string", ""),
    ("5kb-string", "x" * 5000),
    ("empty-object", {}),
    ("bogus-mode", {"mode": "bogus"}),
    ("bad-ttl", {"mode": "explicit", "ttl": "1h"}),
)
_MALFORMED_IDS: Final = tuple(name for name, _ in _MALFORMED)
_MALFORMED_VALUES: Final = tuple(value for _, value in _MALFORMED)
_CASES: Final[tuple[tuple[str, Mode, JsonValue, JsonValue], ...]] = (
    ("valid-on", "on", pcb.EXPLICIT, pcb.EXPLICIT),
    ("valid-off", "off", pcb.EXPLICIT, pcb.EXPLICIT),
    ("malformed-on", "on", "yes", None),
    ("malformed-off", "off", "yes", "yes"),
)
_CASE_IDS: Final = tuple(case[0] for case in _CASES)
_CASE_VALUES: Final = tuple(case[1:] for case in _CASES)


@dataclass(frozen=True, slots=True)
class _Bridge:
    gateway: Gateway
    wire: Wire
    on: str
    off: str
    injecting_on: str
    injecting_off: str
    spend: pcb.SpendLogs

    def model(self, mode: Mode) -> str:
        return self.on if mode == "on" else self.off

    def injecting(self, mode: Mode) -> str:
        return self.injecting_on if mode == "on" else self.injecting_off

    @property
    def api_base(self) -> str:
        return f"{self.wire.url}/v1"


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
            scenario.model(model=pcb.MODEL, api_base=api_base, drop_params=True, **pcb.INJECTION),
            scenario.model(model=pcb.MODEL, api_base=api_base, **pcb.INJECTION),
            spend,
        )


def _v1(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/") + "/v1"


def _user(marker: str, breakpoint: JsonValue) -> dict[str, JsonValue]:
    return {"role": "user", "content": [pcb.marked(pcb.text(pcb.prompt(marker)), breakpoint)]}


def _chat(
    bridge: _Bridge, model: str, messages: Sequence[JsonValue], *, stream: bool = False, key: str | None = None
) -> httpx.Response:
    return bridge.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": list(messages), "stream": stream, **pcb.NO_CACHE},
        key=key,
    )


def _completion(response: httpx.Response, marker: str) -> str:
    assert response.status_code == 200, response.text
    body: Final = rv.JSON_OBJECT.validate_json(response.text)
    assert pcb.answers(string_value(body["id"]), marker), body
    (choice,) = rv.ITEMS.validate_python(body["choices"])
    assert rv.JSON_OBJECT.validate_python(choice["message"])["content"] == rv.answer(marker), body
    return response.headers["x-litellm-call-id"]


def _wire_body(request: Request, *, stream: bool = False) -> dict[str, JsonValue]:
    body: Final = pcb.body_of(request)
    assert body["model"] == "gpt-6.1-sol", body
    assert (body.get("stream") is True) is stream, body
    return body


def _user_block_on_wire(bridge: _Bridge, marker: str, *, stream: bool = False) -> dict[str, JsonValue]:
    request: Final = pcb.posted(bridge.wire, marker)
    block: Final = pcb.single_block(pcb.input_items(request), "user")
    _wire_body(request, stream=stream)
    assert block["type"] == "input_text" and block["text"] == pcb.prompt(marker), block
    return block


def test_openai_sdk_sends_a_valid_marker_through_the_bridge(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    with openai.OpenAI(api_key=bridge.gateway.key, base_url=_v1(bridge.gateway), max_retries=0) as client:
        raw: Final = client.chat.completions.with_raw_response.create(
            model=bridge.on, messages=[_user(marker, pcb.EXPLICIT)], extra_body=dict(pcb.NO_CACHE)
        )
    completion: Final = raw.parse()
    assert pcb.answers(completion.id, marker), completion
    assert completion.choices[0].message.content == rv.answer(marker), completion
    pcb.assert_marker(_user_block_on_wire(bridge, marker), pcb.EXPLICIT)
    bridge.spend.landed(bridge.on, raw.headers["x-litellm-call-id"], marker)


def test_openai_sdk_stream_carries_the_system_list_marker(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    system: Final[dict[str, JsonValue]] = {"role": "system", "content": [pcb.marked(pcb.text("sys"), pcb.EXPLICIT)]}
    with openai.OpenAI(api_key=bridge.gateway.key, base_url=_v1(bridge.gateway), max_retries=0) as client:
        raw: Final = client.chat.completions.with_raw_response.create(
            model=bridge.on,
            messages=[system, {"role": "user", "content": pcb.prompt(marker)}],
            stream=True,
            extra_body=dict(pcb.NO_CACHE),
        )
        chunks: Final = tuple(raw.parse())
    assert chunks and pcb.answers(chunks[0].id, marker), chunks
    streamed: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert streamed == rv.answer(marker), chunks
    request: Final = pcb.posted(bridge.wire, marker)
    _wire_body(request, stream=True)
    (system_block,) = pcb.content_of(pcb.input_items(request), "system")
    assert system_block["type"] == "input_text" and system_block["text"] == "sys", system_block
    pcb.assert_marker(system_block, pcb.EXPLICIT)
    bridge.spend.landed(bridge.on, raw.headers["x-litellm-call-id"], marker)


async def test_async_openai_sdk_keeps_the_ttl_without_drop_params(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    async with openai.AsyncOpenAI(api_key=bridge.gateway.key, base_url=_v1(bridge.gateway), max_retries=0) as client:
        raw: Final = await client.chat.completions.with_raw_response.create(
            model=bridge.off, messages=[_user(marker, pcb.EXPLICIT_30M)], extra_body=dict(pcb.NO_CACHE)
        )
    completion: Final = raw.parse()
    assert pcb.answers(completion.id, marker), completion
    assert completion.choices[0].message.content == rv.answer(marker), completion
    pcb.assert_marker(_user_block_on_wire(bridge, marker), pcb.EXPLICIT_30M)
    bridge.spend.landed(bridge.off, raw.headers["x-litellm-call-id"], marker)


@pytest.mark.parametrize("mode", ("on", "off"))
@pytest.mark.parametrize("breakpoint", (pcb.EXPLICIT, pcb.EXPLICIT_30M), ids=("explicit", "ttl"))
def test_valid_marker_shapes_reach_the_wire_unchanged(bridge: _Bridge, mode: Mode, breakpoint: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), [_user(marker, breakpoint)]), marker)
    pcb.assert_marker(_user_block_on_wire(bridge, marker), breakpoint)
    bridge.spend.landed(bridge.model(mode), call_id, marker)


@pytest.mark.parametrize(
    ("mode", "expected"), (("on", pcb.EXPLICIT), ("off", _UNKNOWN_KEY)), ids=("normalized-on", "verbatim-off")
)
def test_marker_with_an_unknown_key(bridge: _Bridge, mode: Mode, expected: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), [_user(marker, _UNKNOWN_KEY)]), marker)
    pcb.assert_marker(_user_block_on_wire(bridge, marker), expected)
    bridge.spend.landed(bridge.model(mode), call_id, marker)


def _second_block_on_wire(bridge: _Bridge, marker: str) -> dict[str, JsonValue]:
    request: Final = pcb.posted(bridge.wire, marker)
    _wire_body(request)
    first, second = pcb.content_of(pcb.input_items(request), "user")
    assert first == {"type": "input_text", "text": pcb.prompt(marker)}, first
    return second


@pytest.mark.parametrize("mode", ("on", "off"))
@pytest.mark.parametrize("kind", pcb.KINDS)
def test_valid_marker_is_carried_on_every_block_kind(bridge: _Bridge, kind: pcb.Kind, mode: Mode) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final[list[JsonValue]] = [
        pcb.text(pcb.prompt(marker)),
        pcb.marked(pcb.block(kind, "second"), pcb.EXPLICIT),
    ]
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), [{"role": "user", "content": content}]), marker)
    second: Final = _second_block_on_wire(bridge, marker)
    assert second["type"] == pcb.WIRE_TYPE[kind], second
    pcb.assert_marker(second, pcb.EXPLICIT)
    bridge.spend.landed(bridge.model(mode), call_id, marker)


@pytest.mark.parametrize("kind", pcb.KINDS)
def test_malformed_marker_is_dropped_on_every_block_kind(bridge: _Bridge, kind: pcb.Kind) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final[list[JsonValue]] = [pcb.text(pcb.prompt(marker)), pcb.marked(pcb.block(kind, "second"), "yes")]
    call_id: Final = _completion(_chat(bridge, bridge.on, [{"role": "user", "content": content}]), marker)
    second: Final = _second_block_on_wire(bridge, marker)
    assert second["type"] == pcb.WIRE_TYPE[kind], second
    pcb.assert_marker(second, None)
    bridge.spend.landed(bridge.on, call_id, marker)


def test_input_audio_block_is_forwarded_with_its_marker_without_drop_params(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final[list[JsonValue]] = [pcb.text(pcb.prompt(marker)), pcb.marked(pcb.audio(), pcb.EXPLICIT)]
    call_id: Final = _completion(_chat(bridge, bridge.off, [{"role": "user", "content": content}]), marker)
    second: Final = _second_block_on_wire(bridge, marker)
    assert second["type"] == "input_audio" and second["input_audio"] == pcb.AUDIO_PAYLOAD, second
    pcb.assert_marker(second, pcb.EXPLICIT)
    bridge.spend.landed(bridge.off, call_id, marker)


def test_input_audio_block_is_dropped_under_drop_params_and_its_marker_moves_to_the_text(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final[list[JsonValue]] = [pcb.text(pcb.prompt(marker)), pcb.marked(pcb.audio(), pcb.EXPLICIT)]
    call_id: Final = _completion(_chat(bridge, bridge.on, [{"role": "user", "content": content}]), marker)
    assert _user_block_on_wire(bridge, marker) == {
        "type": "input_text",
        "text": pcb.prompt(marker),
        "prompt_cache_breakpoint": pcb.EXPLICIT,
    }
    bridge.spend.landed(bridge.on, call_id, marker)


@pytest.mark.parametrize(("mode", "breakpoint", "expected"), _CASE_VALUES, ids=_CASE_IDS)
def test_tool_output_marker(bridge: _Bridge, mode: Mode, breakpoint: JsonValue, expected: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": pcb.prompt(marker)},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": [pcb.marked(pcb.text("found it"), breakpoint)]},
    ]
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), messages), marker)
    request: Final = pcb.posted(bridge.wire, marker)
    _wire_body(request)
    (output,) = pcb.function_output(pcb.input_items(request), "call_1")
    assert output["type"] == "input_text" and output["text"] == "found it", output
    pcb.assert_marker(output, expected)
    bridge.spend.landed(bridge.model(mode), call_id, marker)


@pytest.mark.parametrize(("mode", "breakpoint", "expected"), _CASE_VALUES, ids=_CASE_IDS)
def test_assistant_list_marker(bridge: _Bridge, mode: Mode, breakpoint: JsonValue, expected: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": pcb.prompt(marker)},
        {"role": "assistant", "content": [pcb.marked(pcb.text("earlier answer"), breakpoint)]},
        {"role": "user", "content": "and again"},
    ]
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), messages), marker)
    request: Final = pcb.posted(bridge.wire, marker)
    _wire_body(request)
    (earlier,) = pcb.content_of(pcb.input_items(request), "assistant")
    assert earlier["type"] == "output_text" and earlier["text"] == "earlier answer", earlier
    pcb.assert_marker(earlier, expected)
    bridge.spend.landed(bridge.model(mode), call_id, marker)


def test_injected_system_marker_survives_a_trailing_audio_block(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    system: Final[dict[str, JsonValue]] = {"role": "system", "content": [pcb.text("sys"), pcb.audio()]}
    messages: Final[list[JsonValue]] = [system, {"role": "user", "content": pcb.prompt(marker)}]
    call_id: Final = _completion(_chat(bridge, bridge.injecting_on, messages), marker)
    request: Final = pcb.posted(bridge.wire, marker)
    body: Final = _wire_body(request)
    assert body["prompt_cache_options"] == {"mode": "explicit"}, body
    (system_block,) = pcb.content_of(pcb.input_items(request), "system")
    assert system_block == {"type": "input_text", "text": "sys", "prompt_cache_breakpoint": pcb.EXPLICIT}, system_block
    bridge.spend.landed(bridge.injecting_on, call_id, marker)


def test_injected_marker_on_a_string_system_message(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "system", "content": "Answer briefly"},
        {"role": "user", "content": pcb.prompt(marker)},
    ]
    call_id: Final = _completion(_chat(bridge, bridge.injecting_off, messages), marker)
    request: Final = pcb.posted(bridge.wire, marker)
    body: Final = _wire_body(request)
    assert body["prompt_cache_options"] == {"mode": "explicit"}, body
    system_block: Final = pcb.single_block(pcb.input_items(request), "system")
    assert system_block == {"type": "input_text", "text": "Answer briefly", "prompt_cache_breakpoint": pcb.EXPLICIT}
    bridge.spend.landed(bridge.injecting_off, call_id, marker)


def _anthropic(bridge: _Bridge) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(bridge.gateway.client.base_url), api_key=bridge.gateway.key, max_retries=0)


def _anthropic_text(message: anthropic.types.Message, marker: str) -> None:
    (content,) = message.content
    assert content.type == "text" and content.text == rv.answer(marker), message


@pytest.mark.parametrize(("mode", "breakpoint", "expected"), _CASE_VALUES, ids=_CASE_IDS)
def test_anthropic_sdk_marker_on_user_text(
    bridge: _Bridge, mode: Mode, breakpoint: JsonValue, expected: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    with _anthropic(bridge) as client:
        raw: Final = client.messages.with_raw_response.create(
            model=bridge.model(mode),
            max_tokens=64,
            messages=[{"role": "user", "content": [pcb.marked(pcb.text(pcb.prompt(marker)), breakpoint)]}],
            extra_body=dict(pcb.NO_CACHE),
        )
    _anthropic_text(raw.parse(), marker)
    pcb.assert_marker(_user_block_on_wire(bridge, marker), expected)
    bridge.spend.landed(bridge.model(mode), raw.headers["x-litellm-call-id"], None)


@pytest.mark.parametrize(("mode", "breakpoint", "expected"), _CASE_VALUES, ids=_CASE_IDS)
def test_anthropic_sdk_marker_on_a_system_list(
    bridge: _Bridge, mode: Mode, breakpoint: JsonValue, expected: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    with _anthropic(bridge) as client:
        raw: Final = client.messages.with_raw_response.create(
            model=bridge.model(mode),
            max_tokens=64,
            system=[pcb.marked(pcb.text("Answer briefly"), breakpoint)],
            messages=[{"role": "user", "content": pcb.prompt(marker)}],
            extra_body=dict(pcb.NO_CACHE),
        )
    _anthropic_text(raw.parse(), marker)
    request: Final = pcb.posted(bridge.wire, marker)
    body: Final = _wire_body(request)
    if expected is None:
        assert body["instructions"] == "Answer briefly", body
        assert "prompt_cache_breakpoint" not in request.body.decode(), body
    else:
        instruction: Final = pcb.instruction_block(pcb.input_items(request))
        assert instruction == {"type": "input_text", "text": "Answer briefly", "prompt_cache_breakpoint": expected}
    bridge.spend.landed(bridge.model(mode), raw.headers["x-litellm-call-id"], None)


@pytest.mark.parametrize(("mode", "expected"), (("off", "yes"), ("on", None)), ids=("verbatim-off", "dropped-on"))
def test_anthropic_sdk_upstream_400_reaches_the_caller_once(bridge: _Bridge, mode: Mode, expected: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    failing: Final = pcb.marked(pcb.text(f"{pcb.prompt(marker)} fail-400"), "yes")
    with _anthropic(bridge) as client, pytest.raises(anthropic.BadRequestError) as caught:
        client.messages.create(
            model=bridge.model(mode),
            max_tokens=64,
            messages=[{"role": "user", "content": [failing]}],
            extra_body=dict(pcb.NO_CACHE),
        )
    assert f"scripted 400 marker-{marker}" in caught.value.response.text, caught.value.response.text
    (request,) = pcb.with_marker(pcb.drained_posts(bridge.wire), marker)
    pcb.assert_marker(pcb.single_block(pcb.input_items(request), "user"), expected)
    call_id: Final = caught.value.response.headers["x-litellm-call-id"]
    bridge.spend.landed(bridge.model(mode), call_id, None, status="failure")
    follow_up: Final = uuid.uuid4().hex
    with _anthropic(bridge) as client:
        raw: Final = client.messages.with_raw_response.create(
            model=bridge.model(mode),
            max_tokens=64,
            messages=[{"role": "user", "content": [pcb.marked(pcb.text(pcb.prompt(follow_up)), "yes")]}],
            extra_body=dict(pcb.NO_CACHE),
        )
    _anthropic_text(raw.parse(), follow_up)
    pcb.assert_marker(_user_block_on_wire(bridge, follow_up), expected)
    bridge.spend.landed(bridge.model(mode), raw.headers["x-litellm-call-id"], None)


def test_anthropic_sdk_system_string_gets_the_injected_marker(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    with _anthropic(bridge) as client:
        raw: Final = client.messages.with_raw_response.create(
            model=bridge.injecting_off,
            max_tokens=64,
            system="Answer briefly",
            messages=[{"role": "user", "content": pcb.prompt(marker)}],
            extra_body=dict(pcb.NO_CACHE),
        )
    _anthropic_text(raw.parse(), marker)
    request: Final = pcb.posted(bridge.wire, marker)
    body: Final = _wire_body(request)
    assert body["prompt_cache_options"] == {"mode": "explicit"}, body
    instruction: Final = pcb.instruction_block(pcb.input_items(request))
    assert instruction == {"type": "input_text", "text": "Answer briefly", "prompt_cache_breakpoint": pcb.EXPLICIT}
    bridge.spend.landed(bridge.injecting_off, raw.headers["x-litellm-call-id"], None)


def test_native_responses_request_never_enters_the_bridge(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    block: Final[dict[str, JsonValue]] = {"type": "input_text", "text": pcb.prompt(marker)}
    response: Final = bridge.gateway.request(
        "POST",
        "/v1/responses",
        {
            "model": bridge.on,
            "input": [{"type": "message", "role": "user", "content": [pcb.marked(block, _UNKNOWN_KEY)]}],
            **pcb.NO_CACHE,
        },
    )
    assert response.status_code == 200, response.text
    body: Final = rv.JSON_OBJECT.validate_json(response.text)
    assert pcb.answers(string_value(body["id"]), marker), body
    assert rv.answer(marker) in response.text, response.text
    request: Final = pcb.posted(bridge.wire, marker)
    _wire_body(request)
    on_wire: Final = pcb.single_block(pcb.input_items(request), "user")
    assert on_wire == pcb.marked(block, _UNKNOWN_KEY), on_wire
    bridge.spend.landed(bridge.on, response.headers["x-litellm-call-id"], marker)


@pytest.mark.parametrize("breakpoint", _MALFORMED_VALUES, ids=_MALFORMED_IDS)
def test_malformed_marker_is_dropped_under_drop_params(bridge: _Bridge, breakpoint: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.on, [_user(marker, breakpoint)]), marker)
    pcb.assert_marker(_user_block_on_wire(bridge, marker), None)
    bridge.spend.landed(bridge.on, call_id, marker)


@pytest.mark.parametrize("breakpoint", _MALFORMED_VALUES, ids=_MALFORMED_IDS)
def test_malformed_marker_passes_verbatim_without_drop_params(bridge: _Bridge, breakpoint: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.off, [_user(marker, breakpoint)]), marker)
    pcb.assert_marker(_user_block_on_wire(bridge, marker), breakpoint)
    bridge.spend.landed(bridge.off, call_id, marker)


def test_two_marked_blocks_are_both_carried(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final[list[JsonValue]] = [
        pcb.marked(pcb.text(pcb.prompt(marker)), pcb.EXPLICIT),
        pcb.marked(pcb.text("and more"), pcb.EXPLICIT_30M),
    ]
    call_id: Final = _completion(_chat(bridge, bridge.on, [{"role": "user", "content": content}]), marker)
    request: Final = pcb.posted(bridge.wire, marker)
    _wire_body(request)
    first, second = pcb.content_of(pcb.input_items(request), "user")
    assert first == {"type": "input_text", "text": pcb.prompt(marker), "prompt_cache_breakpoint": pcb.EXPLICIT}
    assert second == {"type": "input_text", "text": "and more", "prompt_cache_breakpoint": pcb.EXPLICIT_30M}
    bridge.spend.landed(bridge.on, call_id, marker)


def test_wrong_key_is_refused_before_the_wire(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _chat(bridge, bridge.on, [_user(marker, pcb.EXPLICIT)], key="sk-wrong")
    assert response.status_code == 401, response.text
    assert pcb.with_marker(pcb.drained_posts(bridge.wire), marker) == (), marker


@pytest.mark.parametrize(
    ("mode", "status"), (("on", 400), ("off", 400), ("on", 401)), ids=("400-on", "400-off", "401-on")
)
def test_upstream_error_reaches_the_caller_once(bridge: _Bridge, mode: Mode, status: int) -> None:
    marker: Final = uuid.uuid4().hex
    failing: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [pcb.marked(pcb.text(f"{pcb.prompt(marker)} fail-{status}"), pcb.EXPLICIT)],
    }
    response: Final = _chat(bridge, bridge.model(mode), [failing])
    assert response.status_code == status, response.text
    assert f"scripted {status} marker-{marker}" in response.text, response.text
    (request,) = pcb.with_marker(pcb.drained_posts(bridge.wire), marker)
    pcb.assert_marker(pcb.single_block(pcb.input_items(request), "user"), pcb.EXPLICIT)
    bridge.spend.landed(bridge.model(mode), response.headers["x-litellm-call-id"], None, status="failure")
    follow_up: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), [_user(follow_up, pcb.EXPLICIT)]), follow_up)
    pcb.assert_marker(_user_block_on_wire(bridge, follow_up), pcb.EXPLICIT)
    bridge.spend.landed(bridge.model(mode), call_id, follow_up)


def test_null_drop_params_on_the_deployment_means_off(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    with bridge.gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=bridge.api_base, drop_params=None)
        call_id: Final = _completion(_chat(bridge, model, [_user(marker, "yes")]), marker)
        pcb.assert_marker(_user_block_on_wire(bridge, marker), "yes")
        bridge.spend.landed(model, call_id, marker)


@pytest.mark.parametrize("mode", ("on", "off"))
@pytest.mark.parametrize("shape", ("null", "missing"))
def test_null_or_missing_marker_sends_a_plain_block(bridge: _Bridge, shape: str, mode: Mode) -> None:
    marker: Final = uuid.uuid4().hex
    block: Final = pcb.marked(pcb.text(pcb.prompt(marker)), None) if shape == "null" else pcb.text(pcb.prompt(marker))
    call_id: Final = _completion(_chat(bridge, bridge.model(mode), [{"role": "user", "content": [block]}]), marker)
    assert _user_block_on_wire(bridge, marker) == {"type": "input_text", "text": pcb.prompt(marker)}
    bridge.spend.landed(bridge.model(mode), call_id, marker)


async def _send_marked(client: httpx.AsyncClient, key: str, model: str, marker: str) -> httpx.Response:
    return await client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [_user(marker, "yes")], **pcb.NO_CACHE},
        headers={"Authorization": f"Bearer {key}"},
    )


def _probe_marker(bridge: _Bridge, model: str) -> JsonValue:
    marker: Final = uuid.uuid4().hex
    _completion(_chat(bridge, model, [_user(marker, "yes")]), marker)
    return _user_block_on_wire(bridge, marker).get("prompt_cache_breakpoint")


@pytest.mark.timeout(180)
async def test_flipping_drop_params_mid_burst_keeps_every_marked_request_answered(bridge: _Bridge) -> None:
    gateway: Final = bridge.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=bridge.api_base, drop_params=True)
        identity: Final = pcb.model_id(gateway.get("/model/info")["data"], model)
        markers: Final = tuple(uuid.uuid4().hex for _ in range(20))
        async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
            burst: Final = asyncio.gather(*(_send_marked(client, gateway.key, model, marker) for marker in markers))
            updated: Final = await asyncio.to_thread(
                gateway.request,
                "POST",
                "/model/update",
                {
                    "model_name": model,
                    "litellm_params": {"model": pcb.MODEL, "drop_params": False},
                    "model_info": {"id": identity},
                },
            )
            responses: Final = await burst
        assert updated.status_code == 200, updated.text
        for marker, response in zip(markers, responses, strict=True):
            _completion(response, marker)
        posts: Final = pcb.drained_posts(bridge.wire)
        for marker in markers:
            (request,) = pcb.with_marker(posts, marker)
            seen: Final = pcb.single_block(pcb.input_items(request), "user").get("prompt_cache_breakpoint")
            assert seen in (None, "yes"), request.body
        flipped: Final = eventually(lambda: _probe_marker(bridge, model), lambda seen: seen == "yes", seconds=70)
        assert flipped == "yes"
        for marker, response in zip(markers, responses, strict=True):
            bridge.spend.landed(model, response.headers["x-litellm-call-id"], marker)


def test_three_identical_marked_requests_are_each_sent_and_logged(bridge: _Bridge) -> None:
    marker: Final = uuid.uuid4().hex
    responses: Final = tuple(_chat(bridge, bridge.on, [_user(marker, pcb.EXPLICIT)]) for _ in range(3))
    call_ids: Final = tuple(_completion(response, marker) for response in responses)
    assert len(set(call_ids)) == 3, call_ids
    posts: Final = pcb.with_marker(pcb.drained_posts(bridge.wire), marker)
    assert len(posts) == 3, [request.body for request in posts]
    for request in posts:
        pcb.assert_marker(pcb.single_block(pcb.input_items(request), "user"), pcb.EXPLICIT)
    for call_id in call_ids:
        bridge.spend.landed(bridge.on, call_id, marker)
