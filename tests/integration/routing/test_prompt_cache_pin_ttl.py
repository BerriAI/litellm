import asyncio
import json
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import eventually, gateway_from_environment, held
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.wire import Wire, wire_server
from integration.routing._prompt_cache_pin_support import (
    BEDROCK_FIVE_MINUTE_MODEL,
    BEDROCK_ONE_HOUR_MODEL,
    DEFAULT_MARK,
    FIVE_MINUTE_MARK,
    INFERENCE_PROFILE_ARN,
    ONE_HOUR_MARK,
    OVERLOAD_TOKEN,
    PROVIDER_KEY,
    Pin,
    anthropic_deployment,
    ask_turn,
    bedrock_deployment,
    bedrock_runtime_peer,
    chat_body,
    claude_peer,
    deployment,
    end_user_row_lands,
    follow_up,
    lives_five_minutes,
    lives_one_hour,
    marked,
    marker_of,
    new_marker,
    no_pin_written,
    one_new_pin,
    openai_peer,
    pin_at,
    pin_config,
    pins_since,
    post_chat,
    question,
    read_pins,
    reference,
    sent_with,
    slow_claude_peer,
    spend_row_lands,
    spend_rows,
    system_turn,
    vertex_deployment,
    vertex_peer,
    wire_carries_one_hour,
)
from pydantic import JsonValue
from redis import Redis

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 300), pytest.mark.shares_worker_state]

_CLAUDE: Final = "pin-claude"
_SLOW: Final = "pin-slow"
_MIXED: Final = "pin-mixed"
_OPENAI: Final = "pin-openai"
_LEFT: Final = "pin-claude-left"
_RIGHT: Final = "pin-claude-right"
_PRIMARY: Final = "pin-mixed-primary"
_SECONDARY: Final = "pin-mixed-secondary"
_MALFORMED_TTLS: Final[tuple[JsonValue, ...]] = (3600, "2h", "1H", "", None, ["1h"], "1h" * 2500)


@dataclass(frozen=True, slots=True)
class _Family:
    group: str
    identity: str
    one_hour: bool
    wire_one_hour: bool | None


_FAMILIES: Final = (
    _Family("pin-bedrock-converse-1h", "pin-bedrock-converse-1h", one_hour=True, wire_one_hour=True),
    _Family("pin-bedrock-converse-5m", "pin-bedrock-converse-5m", one_hour=False, wire_one_hour=False),
    _Family("pin-bedrock-invoke-1h", "pin-bedrock-invoke-1h", one_hour=True, wire_one_hour=True),
    _Family("pin-bedrock-arn-base", "pin-bedrock-arn-base", one_hour=True, wire_one_hour=True),
    _Family("pin-bedrock-arn", "pin-bedrock-arn", one_hour=False, wire_one_hour=False),
    _Family("pin-azure-ai", "pin-azure-ai", one_hour=True, wire_one_hour=True),
    _Family("pin-vertex", "pin-vertex", one_hour=True, wire_one_hour=True),
    _Family(_OPENAI, "pin-openai", one_hour=False, wire_one_hour=None),
    _Family("pin-unmapped", "pin-unmapped", one_hour=False, wire_one_hour=True),
    _Family("pin-zero-one-hour-price", "pin-zero-one-hour-price", one_hour=False, wire_one_hour=None),
)


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    redis: Redis
    left: Wire
    right: Wire
    misc: Wire
    bedrock: Wire
    openai: Wire
    primary: Wire
    slow: Wire

    def wires(self) -> tuple[Wire, ...]:
        return (self.left, self.right, self.misc, self.bedrock, self.openai, self.primary, self.slow)

    def base_url(self) -> str:
        return str(self.owned.gateway.client.base_url).rstrip("/")

    def served_by(self, marker: str) -> str:
        identities: Final = {_LEFT: self.left, _RIGHT: self.right}
        found: Final = [identity for identity, wire in identities.items() if sent_with((wire,), marker)]
        assert len(found) == 1, found
        return found[0]


def _second_in_order(entry: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    params: Final = entry["litellm_params"]
    assert isinstance(params, dict)
    return {**entry, "litellm_params": {**params, "order": 2}}


def _model_list(rig_wires: Mapping[str, str]) -> list[JsonValue]:
    bedrock: Final = rig_wires["bedrock"]
    misc: Final = rig_wires["misc"]
    return [
        anthropic_deployment(_CLAUDE, _LEFT, rig_wires["left"]),
        anthropic_deployment(_CLAUDE, _RIGHT, rig_wires["right"]),
        anthropic_deployment(_SLOW, "pin-slow-left", rig_wires["slow"]),
        anthropic_deployment(_SLOW, "pin-slow-right", rig_wires["slow"]),
        anthropic_deployment(_MIXED, _PRIMARY, rig_wires["primary"], order=1),
        _second_in_order(
            bedrock_deployment(_MIXED, _SECONDARY, f"bedrock/converse/{BEDROCK_FIVE_MINUTE_MODEL}", bedrock)
        ),
        bedrock_deployment(
            "pin-bedrock-converse-1h", "pin-bedrock-converse-1h", f"bedrock/converse/{BEDROCK_ONE_HOUR_MODEL}", bedrock
        ),
        bedrock_deployment(
            "pin-bedrock-converse-5m",
            "pin-bedrock-converse-5m",
            f"bedrock/converse/{BEDROCK_FIVE_MINUTE_MODEL}",
            bedrock,
        ),
        bedrock_deployment(
            "pin-bedrock-invoke-1h", "pin-bedrock-invoke-1h", f"bedrock/invoke/{BEDROCK_ONE_HOUR_MODEL}", bedrock
        ),
        bedrock_deployment(
            "pin-bedrock-arn-base",
            "pin-bedrock-arn-base",
            f"bedrock/converse/{INFERENCE_PROFILE_ARN}",
            bedrock,
            model_info={"base_model": f"bedrock/{BEDROCK_ONE_HOUR_MODEL}"},
        ),
        bedrock_deployment("pin-bedrock-arn", "pin-bedrock-arn", f"bedrock/converse/{INFERENCE_PROFILE_ARN}", bedrock),
        deployment(
            "pin-azure-ai", "pin-azure-ai", model="azure_ai/claude-opus-5-5", api_base=misc, api_key=PROVIDER_KEY
        ),
        vertex_deployment("pin-vertex", "pin-vertex", misc),
        deployment(
            _OPENAI, "pin-openai", model="openai/gpt-5.6", api_base=f"{rig_wires['openai']}/v1", api_key=PROVIDER_KEY
        ),
        deployment(
            "pin-unmapped",
            "pin-unmapped",
            model="anthropic/claude-pin-model-the-cost-map-lacks",
            api_base=misc,
            api_key=PROVIDER_KEY,
        ),
        deployment(
            "pin-zero-one-hour-price",
            "pin-zero-one-hour-price",
            model="sarvam/sarvam-m",
            api_base=f"{rig_wires['openai']}/v1",
            api_key=PROVIDER_KEY,
        ),
    ]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("prompt-cache-pin-ttl")
    with (
        gateway_from_environment() as gateway,
        owned_redis(directory) as cache,
        wire_server(claude_peer) as left,
        wire_server(claude_peer) as right,
        wire_server(vertex_peer) as misc,
        wire_server(bedrock_runtime_peer) as bedrock,
        wire_server(openai_peer) as openai_wire,
        wire_server(claude_peer) as primary,
        wire_server(slow_claude_peer) as slow,
    ):
        urls: Final = {
            "left": left.url,
            "right": right.url,
            "misc": misc.url,
            "bedrock": bedrock.url,
            "openai": openai_wire.url,
            "primary": primary.url,
            "slow": slow.url,
        }
        config: Final = pin_config(directory, _model_list(urls))
        with (
            owned_proxy_process(
                gateway,
                directory,
                {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)},
                config=config,
                workers=2,
            ) as owned,
            Redis(host=cache.host, port=cache.port, socket_timeout=2) as redis,
        ):
            yield _Rig(owned, redis, left, right, misc, bedrock, openai_wire, primary, slow)


def _before(rig: _Rig) -> frozenset[str]:
    for wire in rig.wires():
        wire.drain()
    return frozenset(read_pins(rig.redis))


def _first_turn(session: str, marker: str, mark: JsonValue) -> list[JsonValue]:
    return [system_turn(session, mark), ask_turn(marker)]


def _breakpoint_shapes(session: str, marker: str) -> Mapping[str, list[JsonValue]]:
    return {
        "system-1h": _first_turn(session, marker, ONE_HOUR_MARK),
        "system-5m": _first_turn(session, marker, FIVE_MINUTE_MARK),
        "system-default-ttl": _first_turn(session, marker, DEFAULT_MARK),
        "system-1h-then-turn-5m": [
            system_turn(session, ONE_HOUR_MARK),
            {"role": "user", "content": "Earlier question."},
            {"role": "assistant", "content": "sunny"},
            ask_turn(marker, FIVE_MINUTE_MARK),
        ],
        "earlier-user-turn-1h": [
            system_turn(session, None),
            {"role": "user", "content": [marked("Earlier question.", ONE_HOUR_MARK)]},
            {"role": "assistant", "content": "sunny"},
            ask_turn(marker),
        ],
        "message-level-1h-on-string-content": [
            {"role": "system", "content": reference(session), "cache_control": ONE_HOUR_MARK},
            ask_turn(marker),
        ],
    }


_ONE_HOUR_SHAPES: Final = frozenset(
    {"system-1h", "system-1h-then-turn-5m", "earlier-user-turn-1h", "message-level-1h-on-string-content"}
)


@pytest.mark.parametrize("shape", sorted(_breakpoint_shapes("s", "m")))
def test_chat_pin_lifetime_follows_the_longest_breakpoint_ttl_in_the_cacheable_prefix(rig: _Rig, shape: str) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    response_id, served = post_chat(
        rig.owned.gateway, chat_body(_CLAUDE, _breakpoint_shapes(new_marker(), marker)[shape])
    )
    _, pin = one_new_pin(rig.redis, before)
    assert served == rig.served_by(marker) == pin.model_id, (served, pin)
    assert (lives_one_hour if shape in _ONE_HOUR_SHAPES else lives_five_minutes)(pin), (shape, pin)
    spend_row_lands(response_id)


def test_chat_one_hour_mark_on_a_tool_definition_alone_pins_the_served_deployment_for_an_hour(rig: _Rig) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "function": {"name": "lookup_weather", "parameters": {"type": "object"}},
        "cache_control": ONE_HOUR_MARK,
    }
    response_id, served = post_chat(
        rig.owned.gateway, chat_body(_CLAUDE, _first_turn(new_marker(), marker, None), tools=[tool])
    )
    _, pin = one_new_pin(rig.redis, before)
    assert served == rig.served_by(marker) == pin.model_id, (served, pin)
    assert lives_one_hour(pin), pin
    spend_row_lands(response_id)


def _openai_client(rig: _Rig) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{rig.base_url()}/v1", api_key=rig.owned.gateway.key, max_retries=0)


def _openai_chat(rig: _Rig, messages: list[JsonValue], *, asynchronous: bool, stream: bool) -> str:
    body: Final = chat_body(_CLAUDE, messages)
    if asynchronous:
        return asyncio.run(_openai_chat_async(rig, body, stream=stream))
    with _openai_client(rig) as client:
        if not stream:
            return client.chat.completions.create(**body).id
        chunks: Final = tuple(client.chat.completions.create(**body, stream=True))
        return chunks[0].id


async def _openai_chat_async(rig: _Rig, body: Mapping[str, JsonValue], *, stream: bool) -> str:
    async with openai.AsyncOpenAI(
        base_url=f"{rig.base_url()}/v1", api_key=rig.owned.gateway.key, max_retries=0
    ) as client:
        if not stream:
            return (await client.chat.completions.create(**body)).id
        chunks: Final = [chunk async for chunk in await client.chat.completions.create(**body, stream=True)]
        return chunks[0].id


def _messages_body(session: str, marker: str) -> dict[str, JsonValue]:
    return {
        "model": _CLAUDE,
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": [marked(reference(session), ONE_HOUR_MARK), marked(question(marker), None)]}
        ],
    }


def _anthropic_messages(rig: _Rig, body: Mapping[str, JsonValue], *, asynchronous: bool, stream: bool) -> str:
    if asynchronous:
        return asyncio.run(_anthropic_messages_async(rig, body, stream=stream))
    with anthropic.Anthropic(base_url=rig.base_url(), api_key=rig.owned.gateway.key, max_retries=0) as client:
        if not stream:
            return client.messages.create(**body).id
        with client.messages.stream(**body) as events:
            return events.get_final_message().id


async def _anthropic_messages_async(rig: _Rig, body: Mapping[str, JsonValue], *, stream: bool) -> str:
    async with anthropic.AsyncAnthropic(
        base_url=rig.base_url(), api_key=rig.owned.gateway.key, max_retries=0
    ) as client:
        if not stream:
            return (await client.messages.create(**body)).id
        async with client.messages.stream(**body) as events:
            return (await events.get_final_message()).id


_CLIENT_MODES: Final = ("sync", "sync-stream", "async", "async-stream")


@pytest.mark.parametrize("mode", _CLIENT_MODES)
def test_openai_sdk_chat_pins_a_one_hour_breakpoint_for_an_hour(rig: _Rig, mode: str) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    response_id: Final = _openai_chat(
        rig,
        _first_turn(new_marker(), marker, ONE_HOUR_MARK),
        asynchronous=mode.startswith("async"),
        stream=mode.endswith("stream"),
    )
    _, pin = one_new_pin(rig.redis, before)
    assert pin.model_id == rig.served_by(marker), pin
    assert lives_one_hour(pin), pin
    spend_row_lands(response_id)


@pytest.mark.parametrize("mode", _CLIENT_MODES)
def test_anthropic_sdk_messages_pins_a_one_hour_breakpoint_for_an_hour(rig: _Rig, mode: str) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    response_id: Final = _anthropic_messages(
        rig,
        _messages_body(new_marker(), marker),
        asynchronous=mode.startswith("async"),
        stream=mode.endswith("stream"),
    )
    assert response_id == f"msg_{marker}", response_id
    _, pin = one_new_pin(rig.redis, before)
    assert pin.model_id == rig.served_by(marker), pin
    assert lives_one_hour(pin), pin


def _responses_id(rig: _Rig, body: Mapping[str, JsonValue], *, stream: bool) -> str:
    with _openai_client(rig) as client:
        if not stream:
            return client.responses.create(**body).id
        events: Final = tuple(client.responses.create(**body, stream=True))
        return next(event.response.id for event in events if event.type == "response.completed")


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_responses_api_one_hour_breakpoint_answers_and_writes_no_pin(rig: _Rig, stream: bool) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    user: Final = f"pin-responses-{marker}"
    body: Final[dict[str, JsonValue]] = {
        "model": _CLAUDE,
        "max_output_tokens": 16,
        "user": user,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": reference(new_marker()), "cache_control": ONE_HOUR_MARK}],
            },
            {"role": "user", "content": question(marker)},
        ],
    }
    assert _responses_id(rig, body, stream=stream).startswith("resp_")
    assert rig.served_by(marker) in {_LEFT, _RIGHT}
    end_user_row_lands(user)
    held(lambda: pins_since(rig.redis, before), lambda found: not found, holding=3, seconds=20)


def _ask_surface(rig: _Rig, surface: str, session: str, marker: str) -> str:
    if surface == "messages":
        return _anthropic_messages(rig, _messages_body(session, marker), asynchronous=False, stream=False)
    return _openai_chat(
        rig, _first_turn(session, marker, ONE_HOUR_MARK), asynchronous=False, stream=surface == "chat-stream"
    )


@pytest.mark.parametrize("surface", ["chat", "chat-stream", "messages"])
def test_response_cache_hit_rewrites_the_pin_with_the_breakpoint_lifetime(rig: _Rig, surface: str) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    session: Final = new_marker()
    first: Final = _ask_surface(rig, surface, session, marker)
    key, pin = one_new_pin(rig.redis, before)
    assert lives_one_hour(pin), pin
    assert len(sent_with(rig.wires(), marker)) == 1
    spend_row_lands(first)
    rig.redis.delete(key)
    second: Final = _ask_surface(rig, surface, session, marker)
    assert sent_with(rig.wires(), marker) == ()
    assert second == first
    rewritten: Final = eventually(lambda: pin_at(rig.redis, key), lambda found: found is not None, seconds=30)
    assert rewritten is not None and rewritten.model_id in {_LEFT, _RIGHT}, rewritten
    assert lives_one_hour(rewritten), rewritten


@pytest.mark.parametrize("family", _FAMILIES, ids=[family.group for family in _FAMILIES])
def test_provider_family_pin_lifetime_agrees_with_the_ttl_the_wire_carries(rig: _Rig, family: _Family) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    response_id, served = post_chat(
        rig.owned.gateway, chat_body(family.group, _first_turn(new_marker(), marker, ONE_HOUR_MARK))
    )
    _, pin = one_new_pin(rig.redis, before)
    sent: Final = sent_with(rig.wires(), marker)
    assert served == pin.model_id == family.identity, (served, pin)
    assert lives_one_hour(pin) is family.one_hour, pin
    if family.wire_one_hour is not None:
        assert [wire_carries_one_hour(request) for request in sent] == [family.wire_one_hour], sent
    spend_row_lands(response_id)


@pytest.mark.parametrize("group", ["pin-bedrock-invoke-1h", "pin-bedrock-arn-base"])
def test_messages_through_bedrock_pins_an_hour_and_sends_the_hour(rig: _Rig, group: str) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    response: Final = rig.owned.gateway.request(
        "POST", "/v1/messages", {**_messages_body(new_marker(), marker), "model": group}
    )
    assert response.status_code == 200, response.text
    _, pin = one_new_pin(rig.redis, before)
    assert pin.model_id == group, pin
    assert lives_one_hour(pin), pin
    assert [wire_carries_one_hour(request) for request in sent_with((rig.bedrock,), marker)] == [True]


def test_response_cache_hit_on_an_application_profile_rewrites_the_pin_for_its_base_model_hour(rig: _Rig) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    body: Final = chat_body("pin-bedrock-arn-base", _first_turn(new_marker(), marker, ONE_HOUR_MARK))
    first, served = post_chat(rig.owned.gateway, body)
    key, pin = one_new_pin(rig.redis, before)
    assert served == pin.model_id == "pin-bedrock-arn-base", (served, pin)
    assert lives_one_hour(pin), pin
    assert [wire_carries_one_hour(request) for request in sent_with(rig.wires(), marker)] == [True]
    spend_row_lands(first)
    rig.redis.delete(key)
    second, _ = post_chat(rig.owned.gateway, body)
    assert sent_with(rig.wires(), marker) == ()
    assert second == first
    rewritten: Final = eventually(lambda: pin_at(rig.redis, key), lambda found: found is not None, seconds=30)
    assert rewritten is not None and rewritten.model_id == "pin-bedrock-arn-base", rewritten
    assert lives_one_hour(rewritten), rewritten


@pytest.mark.parametrize("ttl", _MALFORMED_TTLS, ids=["int", "2h", "upper-1H", "empty", "null", "list", "5kb"])
def test_malformed_ttl_answers_and_pins_five_minutes_without_a_wire_ttl(rig: _Rig, ttl: JsonValue) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    response_id, _ = post_chat(
        rig.owned.gateway,
        chat_body("pin-bedrock-converse-1h", _first_turn(new_marker(), marker, {"type": "ephemeral", "ttl": ttl})),
    )
    _, pin = one_new_pin(rig.redis, before)
    assert lives_five_minutes(pin), pin
    assert [wire_carries_one_hour(request) for request in sent_with((rig.bedrock,), marker)] == [False]
    spend_row_lands(response_id)


@pytest.mark.parametrize("order", [("5m", "1h"), ("1h", "5m")], ids=["5m-then-1h", "1h-then-5m"])
def test_duplicated_ttl_key_takes_the_last_value_for_pin_and_wire(rig: _Rig, order: tuple[str, str]) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    placeholder: Final = new_marker()
    body: Final = json.dumps(
        chat_body(
            "pin-bedrock-converse-1h", _first_turn(new_marker(), marker, {"type": "ephemeral", "ttl": placeholder})
        )
    ).replace(f'"ttl": "{placeholder}"', f'"ttl": "{order[0]}", "ttl": "{order[1]}"')
    response: Final = rig.owned.gateway.client.post(
        "/v1/chat/completions",
        content=body.encode(),
        headers={"Authorization": f"Bearer {rig.owned.gateway.key}", "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    _, pin = one_new_pin(rig.redis, before)
    assert lives_one_hour(pin) is (order[1] == "1h"), pin
    assert [wire_carries_one_hour(request) for request in sent_with((rig.bedrock,), marker)] == [order[1] == "1h"]


@pytest.mark.parametrize(
    "mark", ["ephemeral", {"type": "persistent", "ttl": "1h"}], ids=["string", "non-ephemeral-type-1h"]
)
def test_non_ephemeral_cache_control_never_pins_an_hour(rig: _Rig, mark: JsonValue) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    response: Final = rig.owned.gateway.request(
        "POST", "/v1/chat/completions", chat_body(_CLAUDE, _first_turn(new_marker(), marker, mark))
    )
    assert response.status_code == 200, response.text
    no_pin_written(rig.redis, before, str(response.json()["id"]))


def test_failed_first_turn_reports_the_error_and_writes_no_pin(rig: _Rig) -> None:
    before: Final = _before(rig)
    response: Final = rig.owned.gateway.request(
        "POST",
        "/v1/chat/completions",
        chat_body(_CLAUDE, _first_turn(new_marker(), f"{new_marker()} {OVERLOAD_TOKEN}", ONE_HOUR_MARK)),
    )
    assert response.status_code == 500, response.text
    assert "overloaded_error" in response.text, response.text
    healthy, _ = post_chat(rig.owned.gateway, chat_body(_CLAUDE, _first_turn(new_marker(), new_marker(), None)))
    spend_row_lands(healthy)
    assert pins_since(rig.redis, before) == {}


def test_one_hour_breakpoint_after_a_five_minute_one_reports_the_provider_400_and_writes_no_pin(rig: _Rig) -> None:
    before: Final = _before(rig)
    marker: Final = new_marker()
    messages: Final = [
        system_turn(new_marker(), FIVE_MINUTE_MARK),
        ask_turn(marker, ONE_HOUR_MARK),
    ]
    response: Final = rig.owned.gateway.request("POST", "/v1/chat/completions", chat_body(_CLAUDE, messages))
    assert response.status_code == 400, response.text
    assert "must not come after" in response.text, response.text
    healthy, _ = post_chat(rig.owned.gateway, chat_body(_CLAUDE, _first_turn(new_marker(), new_marker(), None)))
    spend_row_lands(healthy)
    assert pins_since(rig.redis, before) == {}


@pytest.mark.parametrize(
    "marks", [(FIVE_MINUTE_MARK, ONE_HOUR_MARK), (ONE_HOUR_MARK, FIVE_MINUTE_MARK)], ids=["5m-then-1h", "1h-then-5m"]
)
def test_rewriting_one_prefix_moves_its_pin_to_the_latest_lifetime(
    rig: _Rig, marks: tuple[Mapping[str, JsonValue], Mapping[str, JsonValue]]
) -> None:
    before: Final = _before(rig)
    session: Final = new_marker()
    first_id, _ = post_chat(rig.owned.gateway, chat_body(_CLAUDE, _first_turn(session, new_marker(), dict(marks[0]))))
    key, first = one_new_pin(rig.redis, before)
    spend_row_lands(first_id)
    second_id, served = post_chat(
        rig.owned.gateway, chat_body(_CLAUDE, _first_turn(session, new_marker(), dict(marks[1])))
    )
    spend_row_lands(second_id)
    second: Final = pin_at(rig.redis, key)
    assert second is not None and second.model_id == first.model_id == served, (first, second)
    assert lives_one_hour(first) is (marks[0] == ONE_HOUR_MARK), first
    assert lives_one_hour(second) is (marks[1] == ONE_HOUR_MARK), second
    assert frozenset(pins_since(rig.redis, before)) == {key}


def test_mixed_group_pin_takes_the_lifetime_of_the_deployment_that_served_last(rig: _Rig) -> None:
    before: Final = _before(rig)
    session: Final = new_marker()
    first_turn: Final = _first_turn(session, new_marker(), ONE_HOUR_MARK)
    first_id, first_served = post_chat(rig.owned.gateway, chat_body(_MIXED, first_turn))
    key, first = one_new_pin(rig.redis, before)
    assert first_served == first.model_id == _PRIMARY, first
    assert lives_one_hour(first), first
    spend_row_lands(first_id)
    marker: Final = new_marker()
    second_id, second_served = post_chat(
        rig.owned.gateway, chat_body(_MIXED, follow_up(first_turn, f"{marker} {OVERLOAD_TOKEN}"))
    )
    spend_row_lands(second_id)
    assert second_served == _SECONDARY
    assert [wire_carries_one_hour(request) for request in sent_with((rig.bedrock,), marker)] == [False]
    second: Final = pin_at(rig.redis, key)
    assert second is not None and second.model_id == _SECONDARY, second
    assert lives_five_minutes(second), second


def test_same_prefix_moving_from_a_five_minute_group_to_a_one_hour_group_repins_for_an_hour(rig: _Rig) -> None:
    before: Final = _before(rig)
    session: Final = new_marker()
    first_id, _ = post_chat(rig.owned.gateway, chat_body(_OPENAI, _first_turn(session, new_marker(), ONE_HOUR_MARK)))
    key, first = one_new_pin(rig.redis, before)
    assert first.model_id == "pin-openai" and lives_five_minutes(first), first
    spend_row_lands(first_id)
    second_id, served = post_chat(
        rig.owned.gateway, chat_body(_CLAUDE, _first_turn(session, new_marker(), ONE_HOUR_MARK))
    )
    spend_row_lands(second_id)
    second: Final = pin_at(rig.redis, key)
    assert second is not None and second.model_id == served, (second, served)
    assert lives_one_hour(second), second


def test_follow_ups_stick_to_the_pinned_deployment_and_keep_an_hour(rig: _Rig) -> None:
    before: Final = _before(rig)
    first_turn: Final = _first_turn(new_marker(), new_marker(), ONE_HOUR_MARK)
    first_id, first_served = post_chat(rig.owned.gateway, chat_body(_CLAUDE, first_turn))
    key, first = one_new_pin(rig.redis, before)
    spend_row_lands(first_id)
    follow_ups: Final = tuple(
        post_chat(rig.owned.gateway, chat_body(_CLAUDE, follow_up(first_turn, new_marker()))) for _ in range(5)
    )
    assert [served for _, served in follow_ups] == [first_served] * 5
    for response_id, _ in follow_ups:
        spend_row_lands(response_id)
    pin: Final = pin_at(rig.redis, key)
    assert pin is not None and pin.model_id == first.model_id == first_served, pin
    assert lives_one_hour(pin), pin
    assert frozenset(pins_since(rig.redis, before)) == {key}


def test_concurrent_first_turns_on_one_prefix_write_one_hour_long_pin(rig: _Rig) -> None:
    before: Final = _before(rig)
    session: Final = new_marker()
    bodies: Final = tuple(chat_body(_CLAUDE, _first_turn(session, new_marker(), ONE_HOUR_MARK)) for _ in range(8))
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers: Final = tuple(pool.map(lambda body: post_chat(rig.owned.gateway, body), bodies))
    for response_id, _ in answers:
        spend_row_lands(response_id)
    pins: Final = pins_since(rig.redis, before)
    assert len(pins) == 1, pins
    pin: Final[Pin] = next(iter(pins.values()))
    assert pin.model_id in {served for _, served in answers}, pin
    assert lives_one_hour(pin), pin


def test_slow_upstream_burst_writes_every_one_hour_pin(rig: _Rig) -> None:
    before: Final = _before(rig)
    markers: Final = tuple(new_marker() for _ in range(24))
    bodies: Final = tuple(chat_body(_SLOW, _first_turn(new_marker(), marker, ONE_HOUR_MARK)) for marker in markers)
    with ThreadPoolExecutor(max_workers=24) as pool:
        answers: Final = tuple(pool.map(lambda body: post_chat(rig.owned.gateway, body), bodies))
    for response_id, _ in answers:
        spend_row_lands(response_id)
    pins: Final = pins_since(rig.redis, before)
    assert len(pins) == 24, pins
    assert sorted(pin.model_id for pin in pins.values()) == sorted(served for _, served in answers)
    assert all(lives_one_hour(pin) for pin in pins.values()), pins
    assert sorted(marker_of(request) for request in rig.slow.drain()) == sorted(markers)
