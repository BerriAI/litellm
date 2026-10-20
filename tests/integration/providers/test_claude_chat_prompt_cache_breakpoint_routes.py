import json
from collections.abc import Iterable
from typing import Final, cast

import anthropic
import openai
import pytest
from anthropic.types import MessageParam
from integration._support import bedrock_runtime_peer
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ANTHROPIC_MODEL,
    ASK,
    ASK_LABEL,
    BEDROCK_MODEL,
    CITIES,
    PROVIDER_KEY,
    SYSTEM,
    SYSTEM_LABEL,
    Mark,
    anthropic_labels,
    anthropic_marks,
    anthropic_peer,
    bedrock_labels,
    call_id,
    chat_body,
    final_text,
    gateway_injected,
    marker_of,
    new_marker,
    post_chat,
    tool_call,
    tool_use_label,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import ResponseInputParam
from pydantic import JsonValue, TypeAdapter

_EXPLICIT: Final[dict[str, JsonValue]] = {"mode": "explicit"}
_THIRTY_MINUTES: Final[dict[str, JsonValue]] = {"mode": "explicit", "ttl": "30m"}
_CONVERSE_SYSTEM: Final = f"system:text:{SYSTEM}"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MALFORMED: Final[tuple[JsonValue, ...]] = (
    5,
    [],
    "",
    "x" * 5000,
    {},
    {"mode": "explicit", "ttl": 5},
    {"mode": "explicit", "ttl": ""},
    {"mode": ["explicit"]},
)
_MALFORMED_IDS: Final = ("int", "list", "empty_string", "5kb_string", "empty_object", "ttl_int", "ttl_empty", "mode_list")


def _anthropic(scenario: Scenario, wire: Wire, **fields: JsonValue) -> str:
    return scenario.model(
        model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=PROVIDER_KEY, model_info=None, **fields
    )


def _converse(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"bedrock/converse/{BEDROCK_MODEL}",
        api_key=PROVIDER_KEY,
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=wire.url,
    )


def _turns(marker: str, **system_fields: JsonValue) -> list[JsonValue]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM, **system_fields}]},
        {"role": "user", "content": final_text(marker)},
    ]


def _messages_turns(marker: str, breakpoint: JsonValue) -> list[JsonValue]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ASK, "prompt_cache_breakpoint": breakpoint},
                {"type": "text", "text": final_text(marker)},
            ],
        }
    ]


def _posts(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST")


def _only_request(wire: Wire) -> Request:
    received: Final = _posts(wire)
    assert len(received) == 1, [request.target for request in received]
    return received[0]


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)


def _anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _chat_messages(turns: list[JsonValue]) -> Iterable[ChatCompletionMessageParam]:
    return cast(Iterable[ChatCompletionMessageParam], turns)


def _sdk_messages(turns: list[JsonValue]) -> Iterable[MessageParam]:
    return cast(Iterable[MessageParam], turns)


def _messages_text(client: anthropic.Anthropic, model: str, turns: list[JsonValue], *, stream: bool) -> str:
    if not stream:
        message: Final = client.messages.create(model=model, max_tokens=64, messages=_sdk_messages(turns))
        return "".join(block.text for block in message.content if block.type == "text")
    with client.messages.stream(model=model, max_tokens=64, messages=_sdk_messages(turns)) as events:
        return "".join(events.text_stream)


def _spend_row_lands(response_id: str) -> None:
    eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)),
        lambda rows: len(rows) == 1,
        seconds=70,
    )


async def test_async_openai_sdk_stream_to_claude_carries_the_system_breakpoint(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic(scenario, wire)
        async with openai.AsyncOpenAI(
            base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0
        ) as client:
            stream: Final = await client.chat.completions.create(
                model=model,
                messages=_chat_messages(_turns(marker, prompt_cache_breakpoint=_EXPLICIT)),
                max_tokens=64,
                stream=True,
            )
            chunks: Final = [chunk async for chunk in stream]
        request: Final = _only_request(wire)
    assert len({chunk.id for chunk in chunks}) == 1, chunks
    assert marker_of(request) == marker
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "sunny"
    assert _JSON_OBJECT.validate_json(request.body)["stream"] is True
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert anthropic_marks(request) == (Mark(SYSTEM_LABEL, None),)


def test_sync_openai_sdk_chat_to_converse_carries_the_system_breakpoint_as_a_cache_point(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(bedrock_runtime_peer.respond) as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        with _openai_client(gateway) as client:
            completion: Final = client.chat.completions.create(
                model=model, messages=_chat_messages(_turns(marker, prompt_cache_breakpoint=_EXPLICIT)), max_tokens=64
            )
        request: Final = _only_request(wire)
    assert completion.choices[0].message.content == bedrock_runtime_peer.answer(marker)
    assert request.target.endswith("/converse"), request.target
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert bedrock_labels(request) == [_CONVERSE_SYSTEM]


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_anthropic_sdk_messages_to_converse_carry_the_breakpoint_as_a_cache_point(
    gateway: Gateway, stream: bool
) -> None:
    marker: Final = new_marker()
    with wire_server(bedrock_runtime_peer.respond) as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        with _anthropic_client(gateway) as client:
            text: Final = _messages_text(client, model, _messages_turns(marker, _EXPLICIT), stream=stream)
        request: Final = _only_request(wire)
    assert text == bedrock_runtime_peer.answer(marker)
    assert request.target.endswith("/converse-stream" if stream else "/converse"), request.target
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert bedrock_labels(request) == [ASK_LABEL]


def test_anthropic_sdk_messages_to_converse_with_a_ttl_claude_lacks_is_a_400_before_any_provider_call(
    gateway: Gateway,
) -> None:
    marker: Final = new_marker()
    with wire_server(bedrock_runtime_peer.respond) as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        with _anthropic_client(gateway) as client, pytest.raises(anthropic.BadRequestError) as rejected:
            _messages_text(client, model, _messages_turns(marker, _THIRTY_MINUTES), stream=False)
        received: Final = _posts(wire)
    assert rejected.value.status_code == 400
    assert "30m" in str(rejected.value) and "drop_params" in str(rejected.value), rejected.value
    assert received == (), [request.target for request in received]


def test_messages_to_converse_with_a_ttl_claude_lacks_caches_under_request_drop_params(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(bedrock_runtime_peer.respond) as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": _messages_turns(marker, _THIRTY_MINUTES),
                "drop_params": True,
            },
        )
        request: Final = _only_request(wire)
    assert response.status_code == 200, response.text
    assert bedrock_runtime_peer.answer(marker) in response.text, response.text
    assert bedrock_labels(request) == [ASK_LABEL]


def test_messages_to_a_native_claude_deployment_forward_the_content_as_sent(gateway: Gateway) -> None:
    marker: Final = new_marker()
    turns: Final = _messages_turns(marker, _EXPLICIT)
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic(scenario, wire)
        with _anthropic_client(gateway) as client:
            message: Final = client.messages.create(model=model, max_tokens=64, messages=_sdk_messages(turns))
        request: Final = _only_request(wire)
    assert message.id == f"msg_{marker}"
    assert _JSON_OBJECT.validate_json(request.body)["messages"] == turns


def test_responses_bridge_to_claude_answers_a_breakpoint_with_a_ttl_claude_lacks(gateway: Gateway) -> None:
    marker: Final = new_marker()
    items: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": ASK, "prompt_cache_breakpoint": _THIRTY_MINUTES},
                {"type": "input_text", "text": final_text(marker)},
            ],
        }
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic(scenario, wire)
        with _openai_client(gateway) as client:
            response: Final = client.responses.create(
                model=model, input=cast(ResponseInputParam, items), max_output_tokens=64
            )
        request: Final = _only_request(wire)
    assert response.output_text == "sunny"
    assert f"marker-{marker}" in request.body.decode()
    assert anthropic_labels(request) == []


def _unauthorized(_: Request) -> Reply:
    return Reply(
        status=401,
        body=json.dumps(
            {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "code": "invalid_api_key"}}
        ).encode(),
    )


def test_a_fallback_to_claude_carries_the_breakpoint(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with (
        wire_server(_unauthorized) as primary_wire,
        wire_server(anthropic_peer) as claude_wire,
        gateway.scenario() as scenario,
    ):
        primary: Final = scenario.model(api_base=f"{primary_wire.url}/v1")
        claude: Final = _anthropic(scenario, claude_wire)
        status, _, text = post_chat(
            gateway,
            chat_body(primary, _turns(marker, prompt_cache_breakpoint=_EXPLICIT), fallbacks=[claude], num_retries=0),
        )
        primary_posts: Final = _posts(primary_wire)
        request: Final = _only_request(claude_wire)
    assert status == 200, text
    assert marker_of(request) == marker
    assert len(primary_posts) == 1, [post.target for post in primary_posts]
    assert anthropic_marks(request) == (Mark(SYSTEM_LABEL, None),)


def test_a_fallback_to_claude_with_a_ttl_claude_lacks_never_reaches_claude(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with (
        wire_server(_unauthorized) as primary_wire,
        wire_server(anthropic_peer) as claude_wire,
        gateway.scenario() as scenario,
    ):
        primary: Final = scenario.model(api_base=f"{primary_wire.url}/v1")
        claude: Final = _anthropic(scenario, claude_wire)
        status, _, text = post_chat(
            gateway,
            chat_body(
                primary, _turns(marker, prompt_cache_breakpoint=_THIRTY_MINUTES), fallbacks=[claude], num_retries=0
            ),
        )
        primary_posts: Final = _posts(primary_wire)
        claude_posts: Final = _posts(claude_wire)
    assert status == 401, text
    assert "UnsupportedParamsError" in text and "30m" in text, text
    assert len(primary_posts) == 1, [post.target for post in primary_posts]
    assert claude_posts == (), [post.target for post in claude_posts]


def test_auto_caching_stands_down_for_one_breakpoint_on_a_tool_call(gateway: Gateway) -> None:
    marker: Final = new_marker()
    city: Final = CITIES[0]
    turns: Final[list[JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ASK},
        {"role": "assistant", "content": "", "tool_calls": [tool_call(city, prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "tool", "tool_call_id": call_id(city), "content": "sunny."},
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic(scenario, wire)
        key: Final = scenario.key(metadata={"enable_prompt_caching": True})
        status, response_id, text = post_chat(gateway, chat_body(model, turns), key=key)
        request: Final = _only_request(wire)
    assert status == 200, text
    assert marker_of(request) == marker
    assert anthropic_labels(request) == [tool_use_label(city)]
    assert gateway_injected(response_id) is False


def test_a_repeated_breakpoint_request_is_answered_from_the_response_cache(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic(scenario, wire)
        body: Final = chat_body(model, _turns(marker, prompt_cache_breakpoint=_EXPLICIT))
        first: Final = post_chat(gateway, body)
        _spend_row_lands(first[1])
        second: Final = post_chat(gateway, body)
        request: Final = _only_request(wire)
    assert first[0] == 200 and second[0] == 200, (first, second)
    assert first[1] == second[1], (first, second)
    assert marker_of(request) == marker
    assert anthropic_marks(request) == (Mark(SYSTEM_LABEL, None),)


@pytest.mark.parametrize("value", _MALFORMED, ids=_MALFORMED_IDS)
def test_a_malformed_breakpoint_is_a_400_and_the_deployment_keeps_serving(gateway: Gateway, value: JsonValue) -> None:
    marker: Final = new_marker()
    plain: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic(scenario, wire)
        status, _, text = post_chat(gateway, chat_body(model, _turns(marker, prompt_cache_breakpoint=value)))
        rejected: Final = _posts(wire)
        plain_status, _, plain_text = post_chat(gateway, chat_body(model, _turns(plain)))
        served: Final = _only_request(wire)
    assert status == 400, text
    assert "prompt_cache_breakpoint" in text and "drop_params" in text, text
    assert rejected == (), [request.target for request in rejected]
    assert plain_status == 200, plain_text
    assert marker_of(served) == plain
    assert anthropic_labels(served) == []


@pytest.mark.parametrize(
    ("first", "last", "status", "labels"),
    (
        ({"mode": "implicit"}, _EXPLICIT, 200, [SYSTEM_LABEL]),
        (_EXPLICIT, {"mode": "implicit"}, 400, []),
    ),
    ids=("last_explicit", "last_implicit"),
)
def test_a_breakpoint_sent_twice_in_one_block_takes_the_last_value(
    gateway: Gateway, first: JsonValue, last: JsonValue, status: int, labels: list[str]
) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic(scenario, wire)
        raw: Final = json.dumps(chat_body(model, _turns(marker, duplicated=0))).replace(
            '"duplicated": 0',
            f'"prompt_cache_breakpoint": {json.dumps(first)}, "prompt_cache_breakpoint": {json.dumps(last)}',
        )
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            content=raw,
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        received: Final = _posts(wire)
    assert response.status_code == status, response.text
    assert [anthropic_labels(request) for request in received] == ([labels] if status == 200 else [])


@pytest.mark.parametrize("upstream_status", (500, 401))
def test_an_upstream_error_on_a_breakpoint_request_reaches_the_caller_after_one_attempt(
    gateway: Gateway, upstream_status: int
) -> None:
    marker: Final = new_marker()
    failure: Final = Reply(
        status=upstream_status,
        body=json.dumps(
            {"type": "error", "error": {"type": "api_error", "message": f"scripted {upstream_status} {marker}"}}
        ).encode(),
    )
    with wire_server(lambda _: failure) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic(scenario, wire)
        status, _, text = post_chat(
            gateway, chat_body(model, _turns(marker, prompt_cache_breakpoint=_EXPLICIT), num_retries=0)
        )
        request: Final = _only_request(wire)
    assert status == upstream_status, text
    assert f"scripted {upstream_status} {marker}" in text, text
    assert anthropic_marks(request) == (Mark(SYSTEM_LABEL, None),)


@pytest.mark.parametrize(
    ("value", "marks"),
    (
        (None, ()),
        ({"mode": "explicit", "ttl": None}, (Mark(SYSTEM_LABEL, None),)),
        ({"mode": "explicit"}, (Mark(SYSTEM_LABEL, None),)),
    ),
    ids=("null", "ttl_null", "ttl_missing"),
)
def test_a_null_breakpoint_is_no_mark_and_a_null_or_missing_ttl_is_the_default_ttl(
    gateway: Gateway, value: JsonValue, marks: tuple[Mark, ...]
) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic(scenario, wire)
        status, _, text = post_chat(gateway, chat_body(model, _turns(marker, prompt_cache_breakpoint=value)))
        request: Final = _only_request(wire)
    assert status == 200, text
    assert marker_of(request) == marker
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert anthropic_marks(request) == marks
