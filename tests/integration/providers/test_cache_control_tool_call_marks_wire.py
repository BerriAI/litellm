import asyncio
import json
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, cast

import anthropic
import httpx
import openai
import pytest
import yaml
from anthropic.types import MessageParam, TextBlockParam, ToolParam
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ANTHROPIC_MODEL,
    ASK,
    ASK_LABEL,
    BEDROCK_MODEL,
    CITIES,
    EPHEMERAL,
    POINTS,
    PROVIDER_KEY,
    SYSTEM,
    SYSTEM_LABEL,
    TOOL,
    anthropic_labels,
    anthropic_marks,
    anthropic_peer,
    bedrock_labels,
    bedrock_peer,
    call_id,
    chat_body,
    client_marked,
    conversation,
    final_label,
    final_text,
    gateway_injected,
    marked_calls,
    messages_body,
    new_marker,
    post_chat,
    responses_body,
    tool_call,
    tool_use_label,
)
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.responses import ResponseInputParam
from openai.types.responses import ToolParam as ResponsesToolParam
from pydantic import JsonValue, TypeAdapter

from litellm.utils import get_prompt_cache_min_tokens

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_LANE_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_CLIENT_MARKED: Final = client_marked()
_GEMINI_REPLY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "sunny"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1600, "candidatesTokenCount": 1, "totalTokenCount": 1601},
    }
).encode()
_OPENAI_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl-cache-census",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-6-sol",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "sunny"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
    }
).encode()


def _uncapped_cache_config(directory: Path) -> Path:
    lane: Final = object_value(yaml.safe_load(_LANE_CONFIG.read_text()))
    settings: Final = object_value(lane["litellm_settings"])
    cache_params: Final = object_value(settings["cache_params"])
    config: Final = {**lane, "litellm_settings": {**settings, "cache_params": {**cache_params, "max_messages": None}}}
    path: Final = directory / "proxy_config_uncapped_cache.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def uncapped_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """A proxy whose response cache has no `max_messages` cap, so a seven-message tool conversation is cached"""
    with gateway_from_environment() as lane:
        directory: Final = tmp_path_factory.mktemp("uncapped_cache")
        with owned_proxy(lane, directory, {}, config=_uncapped_cache_config(directory)) as owned:
            yield owned


def _anthropic_deployment(scenario: Scenario, wire: Wire, **fields: JsonValue) -> str:
    return scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=PROVIDER_KEY, **fields)


def _only_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, [request.target for request in received]
    return received[0]


def _stream(gateway: Gateway, path: str, body: dict[str, JsonValue], *, key: str | None = None) -> tuple[int, str]:
    headers: Final = {"Authorization": f"Bearer {gateway.key if key is None else key}"}
    with gateway.client.stream("POST", path, json=body, headers=headers) as response:
        return response.status_code, response.read().decode()


def _chat_stream_chunks(text: str) -> list[dict[str, JsonValue]]:
    return [
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _chat_stream_content(chunks: list[dict[str, JsonValue]]) -> str:
    return "".join(
        str(object_value(object_value(choice).get("delta") or {}).get("content") or "")
        for chunk in chunks
        for choice in (chunk.get("choices") if isinstance(chunk.get("choices"), list) else [])
    )


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_chat_points_skip_injection_when_client_marks_fill_the_cap(gateway: Gateway, stream: bool) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        body: Final = chat_body(model, conversation(marker, marked_calls()), stream=stream)
        if stream:
            status, text = _stream(gateway, "/v1/chat/completions", body)
            assert status == 200, text
            chunks: Final = _chat_stream_chunks(text)
            assert _chat_stream_content(chunks) == "sunny", text
            assert "data: [DONE]" in text, text
            response_id = str(chunks[0]["id"])
        else:
            status, response_id, text = post_chat(gateway, body)
            assert status == 200, text
        assert anthropic_labels(_only_request(wire)) == _CLIENT_MARKED
    assert gateway_injected(response_id) is False


def test_chat_points_inject_system_and_last_message_without_client_marks(gateway: Gateway) -> None:
    marker: Final = new_marker()
    unmarked: Final = [tool_call(city) for city in CITIES]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, response_id, text = post_chat(
            gateway, chat_body(model, conversation(marker, unmarked, ask_marked=False))
        )
        assert status == 200, text
        assert anthropic_labels(_only_request(wire)) == [SYSTEM_LABEL, final_label(marker)]
    assert gateway_injected(response_id) is True


def _prompt_caching_rows(gateway: Gateway, cursor: dict[str, str]) -> list[dict[str, JsonValue]]:
    now: Final = datetime.now(timezone.utc)
    page: Final = gateway.get(
        "/cost_optimization/prompt_caching/requests",
        {
            "start_date": (now - timedelta(minutes=10)).isoformat(),
            "end_date": (now + timedelta(minutes=10)).isoformat(),
            "page_size": "100",
            "filter": "injected",
            **cursor,
        },
    )
    rows: Final = [object_value(row) for row in page["requests"]] if isinstance(page["requests"], list) else []
    following: Final = page.get("next_cursor")
    if not page.get("has_more") or not isinstance(following, dict):
        return rows
    return [
        *rows,
        *_prompt_caching_rows(
            gateway,
            {"cursor_start_time": str(following["start_time"]), "cursor_request_id": str(following["request_id"])},
        ),
    ]


def _listed_as_injected(gateway: Gateway, response_id: str) -> bool:
    return any(row["request_id"] == response_id for row in _prompt_caching_rows(gateway, {}))


@pytest.mark.parametrize(
    ("calls", "ask_marked", "expected", "injected"),
    (
        pytest.param(marked_calls(), False, [tool_use_label(city) for city in CITIES], False, id="three-tool-calls"),
        pytest.param([tool_call(city) for city in CITIES], False, None, True, id="no-client-marks"),
        pytest.param(
            [*marked_calls(cities=CITIES[:2]), tool_call(CITIES[2])],
            False,
            [tool_use_label(city) for city in CITIES[:2]],
            False,
            id="two-tool-calls",
        ),
    ),
)
def test_auto_prompt_caching_stands_down_when_client_marks_only_tool_calls(
    gateway: Gateway,
    calls: list[JsonValue],
    ask_marked: bool,
    expected: list[str] | None,
    injected: bool,
) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        key: Final = scenario.key(metadata={"enable_prompt_caching": True})
        status, response_id, text = post_chat(
            gateway, chat_body(model, conversation(marker, calls, ask_marked=ask_marked)), key=key
        )
        assert status == 200, text
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == (expected if expected is not None else [SYSTEM_LABEL, final_label(marker)])
    assert gateway_injected(response_id) is injected
    assert (
        eventually(lambda: _listed_as_injected(gateway, response_id), lambda listed: listed is injected, 70) is injected
    )


def test_assistant_point_skips_message_whose_tool_call_carries_a_one_hour_mark(gateway: Gateway) -> None:
    marker: Final = new_marker()
    hour: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
    calls: Final = [tool_call(CITIES[0], cache_control=hour)]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(
            scenario, wire, cache_control_injection_points=[{"location": "message", "role": "assistant"}]
        )
        status, _, text = post_chat(
            gateway,
            chat_body(model, conversation(marker, calls, ask_marked=False, assistant={"content": "I will check."})),
        )
        assert status == 200, text
        marks: Final = anthropic_marks(_only_request(wire))
    assert [(mark.label, mark.ttl) for mark in marks] == [(tool_use_label(CITIES[0]), "1h")]


@pytest.mark.parametrize(
    "mark",
    (
        pytest.param("ephemeral", id="string"),
        pytest.param(1, id="int"),
        pytest.param(["ephemeral"], id="list"),
        pytest.param("", id="empty-string"),
        pytest.param("x" * 5120, id="5kb-string"),
    ),
)
def test_non_object_tool_call_marks_count_against_the_cap(gateway: Gateway, mark: JsonValue) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, _, text = post_chat(gateway, chat_body(model, conversation(marker, marked_calls(mark))))
        assert status == 200, text
        assert anthropic_labels(_only_request(wire)) == [ASK_LABEL]


@pytest.mark.parametrize(
    ("calls", "expected"),
    (
        pytest.param(marked_calls({}), _CLIENT_MARKED, id="empty-object"),
        pytest.param(marked_calls(None), None, id="null"),
        pytest.param(
            [tool_call(CITIES[0], cache_control=EPHEMERAL)] * 2,
            [SYSTEM_LABEL, ASK_LABEL, tool_use_label(CITIES[0])],
            id="same-call-twice",
        ),
    ),
)
def test_tool_call_mark_shapes_keep_the_request_within_the_cap(
    gateway: Gateway, calls: list[JsonValue], expected: list[str] | None
) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, _, text = post_chat(gateway, chat_body(model, conversation(marker, calls)))
        assert status == 200, text
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == (expected if expected is not None else [SYSTEM_LABEL, ASK_LABEL, final_label(marker)])


def _untyped_call(city: str) -> dict[str, JsonValue]:
    return {name: value for name, value in tool_call(city, cache_control=EPHEMERAL).items() if name != "type"}


_BEDROCK_CLIENT_MARKED: Final = [f"user:text:{ASK}", *(f"assistant:toolUse:{call_id(city)}" for city in CITIES)]


@pytest.mark.parametrize(
    ("calls", "expected"),
    (
        pytest.param(marked_calls(), _BEDROCK_CLIENT_MARKED, id="object-marks"),
        pytest.param(marked_calls("ephemeral"), _BEDROCK_CLIENT_MARKED, id="string-marks"),
        pytest.param([_untyped_call(city) for city in CITIES], _BEDROCK_CLIENT_MARKED, id="calls-without-type"),
        pytest.param(
            [tool_call(CITIES[0], cache_control=EPHEMERAL)] * 2,
            [f"system:text:{SYSTEM}", ASK_LABEL, f"assistant:toolUse:{call_id(CITIES[0])}", "assistant:other"],
            id="same-call-twice",
        ),
    ),
)
def test_bedrock_converse_cache_points_stay_within_the_cap(
    gateway: Gateway, calls: list[JsonValue], expected: list[str]
) -> None:
    marker: Final = new_marker()
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/converse/{BEDROCK_MODEL}",
            api_key=PROVIDER_KEY,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            cache_control_injection_points=POINTS,
        )
        status, _, text = post_chat(gateway, chat_body(model, conversation(marker, calls)))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert request.target.endswith("/converse"), request.target
    assert bedrock_labels(request) == expected


_SERVER_CALL: Final = tool_call(
    "web", cache_control=EPHEMERAL, id="srvtoolu_web", function={"name": "web_search", "arguments": "{}"}
)
_WEB_RESULTS: Final[dict[str, JsonValue]] = {
    "provider_specific_fields": {
        "web_search_results": [{"type": "web_search_tool_result", "tool_use_id": "srvtoolu_web", "content": []}]
    }
}


@pytest.mark.parametrize(
    ("assistant", "expected"),
    (
        pytest.param(
            _WEB_RESULTS,
            [SYSTEM_LABEL, ASK_LABEL, tool_use_label(CITIES[0]), tool_use_label(CITIES[1])],
            id="server-tool-with-result",
        ),
        pytest.param(
            {},
            [ASK_LABEL, tool_use_label(CITIES[0]), tool_use_label(CITIES[1]), "assistant:tool_use:srvtoolu_web"],
            id="server-tool-without-result",
        ),
    ),
)
def test_server_tool_call_mark_counts_only_when_forwarded(
    gateway: Gateway, assistant: dict[str, JsonValue], expected: list[str]
) -> None:
    marker: Final = new_marker()
    calls: Final[list[JsonValue]] = [*marked_calls(cities=CITIES[:2]), _SERVER_CALL]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, _, text = post_chat(gateway, chat_body(model, conversation(marker, calls, assistant=assistant)))
        assert status == 200, text
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == expected


def test_azure_ai_claude_points_stay_within_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure_ai/{ANTHROPIC_MODEL}",
            api_base=wire.url,
            api_key=PROVIDER_KEY,
            cache_control_injection_points=POINTS,
        )
        status, _, text = post_chat(gateway, chat_body(model, conversation(marker, marked_calls())))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert request.target == "/anthropic/v1/messages", request.target
    assert anthropic_labels(request) == _CLIENT_MARKED


def _gemini_peer(request: Request) -> Reply:
    if "cachedContents" in request.target and request.method == "GET":
        return Reply(body=b'{"cachedContents":[]}')
    if "cachedContents" in request.target:
        return Reply(
            body=json.dumps(
                {
                    "name": "cachedContents/census",
                    "model": "models/gemini-3.8-flash",
                    "expireTime": "2099-01-01T00:00:00Z",
                }
            ).encode()
        )
    return Reply(body=_GEMINI_REPLY)


_GEMINI: Final = "gemini/gemini-3.8-flash"
_GEMINI_CACHE_WRITE: Final = [
    ("GET", "/models/gemini-3.8-flash:cachedContents"),
    ("POST", "/models/gemini-3.8-flash:cachedContents"),
    ("POST", "/models/gemini-3.8-flash:generateContent"),
]


@pytest.mark.parametrize(
    ("calls", "expected"),
    (
        pytest.param(
            marked_calls(), [("POST", "/models/gemini-3.8-flash:generateContent")], id="tool-calls-fill-the-cap"
        ),
        pytest.param([tool_call(city) for city in CITIES], _GEMINI_CACHE_WRITE, id="unmarked-tool-calls"),
    ),
)
def test_gemini_context_cache_follows_the_cap_census(
    gateway: Gateway, calls: list[JsonValue], expected: list[tuple[str, str]]
) -> None:
    marker: Final = new_marker()
    long_system: Final[JsonValue] = {"role": "system", "content": "lorem " * (2 * get_prompt_cache_min_tokens(_GEMINI))}
    messages: Final = [long_system, *conversation(marker, calls)[1:]]
    with wire_server(_gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_GEMINI,
            api_base=wire.url,
            api_key=PROVIDER_KEY,
            cache_control_injection_points=POINTS,
        )
        status, _, text = post_chat(gateway, chat_body(model, messages))
        assert status == 200, text
        targets: Final = [(request.method, request.target.split("?")[0]) for request in wire.drain()]
    assert targets == expected


def _openai_marks(request: Request) -> list[str]:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    messages: Final = body["messages"] if isinstance(body["messages"], list) else []
    return [label for message in messages if isinstance(message, dict) for label in _openai_message_marks(message)]


def _openai_message_marks(message: dict[str, JsonValue]) -> list[str]:
    role: Final = str(message.get("role"))
    content: Final = message.get("content")
    calls: Final = message.get("tool_calls")
    blocks: Final = content if isinstance(content, list) else []
    return [
        *(f"{key}@{role}:message" for key in ("cache_control", "prompt_cache_breakpoint") if key in message),
        *(
            f"{key}@{role}:text:{block.get('text')}"
            for block in blocks
            if isinstance(block, dict)
            for key in ("cache_control", "prompt_cache_breakpoint")
            if key in block
        ),
        *(
            f"cache_control@tool_call:{call.get('id')}"
            for call in (calls if isinstance(calls, list) else [])
            if isinstance(call, dict) and "cache_control" in call
        ),
    ]


@pytest.mark.parametrize(
    "options",
    (
        pytest.param({"prompt_cache_options": {"mode": "explicit"}}, id="breakpoint-dialect"),
        pytest.param({}, id="plain"),
    ),
)
def test_openai_points_skip_injection_when_client_marks_fill_the_cap(
    gateway: Gateway, options: dict[str, JsonValue]
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: Reply(body=_OPENAI_REPLY)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-6-sol",
            api_base=f"{wire.url}/v1",
            api_key=PROVIDER_KEY,
            cache_control_injection_points=POINTS,
            **options,
        )
        status, _, text = post_chat(gateway, chat_body(model, conversation(marker, marked_calls())))
        assert status == 200, text
        marks: Final = _openai_marks(_only_request(wire))
    assert marks == [f"cache_control@user:text:{ASK}", *(f"cache_control@tool_call:{call_id(city)}" for city in CITIES)]


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_messages_endpoint_points_skip_injection_when_tool_use_marks_fill_the_cap(
    gateway: Gateway, stream: bool
) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, text = _stream(gateway, "/v1/messages", messages_body(model, marker, stream=stream))
        assert status == 200, text
        assert (
            ("event: message_stop" in text) if stream else (_JSON_OBJECT.validate_json(text)["id"] == f"msg_{marker}")
        )
        assert anthropic_labels(_only_request(wire)) == _CLIENT_MARKED


def test_responses_bridge_keeps_system_and_user_marks_within_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        response: Final = gateway.request("POST", "/v1/responses", responses_body(model, marker))
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["status"] == "completed", response.text
        assert anthropic_labels(_only_request(wire)) == [SYSTEM_LABEL, ASK_LABEL]


@pytest.mark.timeout(240)
def test_response_cache_serves_the_capped_request_once(uncapped_gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, uncapped_gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        body: Final = chat_body(model, conversation(marker, marked_calls()))
        first: Final = post_chat(uncapped_gateway, body)
        second: Final = post_chat(uncapped_gateway, body)
        received: Final = wire.drain()
    assert (first[0], second[0]) == (200, 200), (first[2], second[2])
    assert first[1].startswith("chatcmpl-"), first[2]
    assert first[1] == second[1], (first[2], second[2])
    assert [anthropic_labels(request) for request in received] == [_CLIENT_MARKED]


def test_unauthenticated_request_never_reaches_the_provider(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, _, text = post_chat(
            gateway, chat_body(model, conversation(marker, marked_calls())), key=f"sk-not-a-key-{marker}"
        )
        received: Final = wire.drain()
    assert status == 401, text
    assert "error" in _JSON_OBJECT.validate_json(text), text
    assert received == ()


@pytest.mark.parametrize(
    "calls",
    (
        pytest.param({"tool_calls": []}, id="empty"),
        pytest.param({"tool_calls": None}, id="null"),
        pytest.param({}, id="missing"),
    ),
)
def test_assistant_without_tool_calls_keeps_configured_points(gateway: Gateway, calls: dict[str, JsonValue]) -> None:
    marker: Final = new_marker()
    messages: Final[list[JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": ASK, "cache_control": EPHEMERAL}]},
        {"role": "assistant", "content": "I will check.", **calls},
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, _, text = post_chat(gateway, chat_body(model, messages))
        assert status == 200, text
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == [SYSTEM_LABEL, ASK_LABEL, final_label(marker)]


def test_responses_stream_bridge_keeps_system_and_user_marks_within_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, text = _stream(gateway, "/v1/responses", {**responses_body(model, marker), "stream": True})
        assert status == 200, text
        assert '"type":"response.completed"' in text, text
        assert anthropic_labels(_only_request(wire)) == [SYSTEM_LABEL, ASK_LABEL]


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def _anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _async_anthropic_client(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def _sdk_messages(marker: str) -> list[ChatCompletionMessageParam]:
    return cast(list[ChatCompletionMessageParam], conversation(marker, marked_calls()))


_SDK_TOOLS: Final = cast(list[ChatCompletionToolParam], [TOOL])


def test_openai_sdk_sync_chat_keeps_the_capped_request_within_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        with _openai_client(gateway) as client:
            completion: Final = client.chat.completions.create(
                model=model, messages=_sdk_messages(marker), tools=_SDK_TOOLS, max_tokens=64
            )
        assert completion.choices[0].message.content == "sunny", completion.model_dump()
        assert anthropic_labels(_only_request(wire)) == _CLIENT_MARKED
    assert gateway_injected(completion.id) is False


async def test_openai_sdk_async_chat_stream_keeps_the_capped_request_within_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        async with _async_openai_client(gateway) as client:
            stream: Final = await client.chat.completions.create(
                model=model, messages=_sdk_messages(marker), tools=_SDK_TOOLS, max_tokens=64, stream=True
            )
            chunks: Final = [chunk async for chunk in stream]
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "sunny", chunks
        assert anthropic_labels(_only_request(wire)) == _CLIENT_MARKED
    assert gateway_injected(chunks[0].id) is False


def _messages_system(model: str, marker: str) -> Iterable[TextBlockParam]:
    return cast(Iterable[TextBlockParam], messages_body(model, marker, stream=False)["system"])


def _messages_tools(model: str, marker: str) -> Iterable[ToolParam]:
    return cast(Iterable[ToolParam], messages_body(model, marker, stream=False)["tools"])


def _messages_turns(model: str, marker: str) -> Iterable[MessageParam]:
    return cast(Iterable[MessageParam], messages_body(model, marker, stream=False)["messages"])


def test_anthropic_sdk_sync_messages_keeps_the_capped_request_within_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        with _anthropic_client(gateway) as client:
            message: Final = client.messages.create(
                model=model,
                max_tokens=64,
                system=_messages_system(model, marker),
                tools=_messages_tools(model, marker),
                messages=_messages_turns(model, marker),
            )
        assert message.id == f"msg_{marker}", message.model_dump()
        assert [block.text for block in message.content if block.type == "text"] == ["sunny"], message.model_dump()
        assert anthropic_labels(_only_request(wire)) == _CLIENT_MARKED
    assert gateway_injected(message.id) is False


async def test_anthropic_sdk_async_messages_stream_keeps_the_capped_request_within_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        async with _async_anthropic_client(gateway) as client:
            stream: Final = await client.messages.create(
                model=model,
                max_tokens=64,
                system=_messages_system(model, marker),
                tools=_messages_tools(model, marker),
                messages=_messages_turns(model, marker),
                stream=True,
            )
            events: Final = [event async for event in stream]
        assert [event.type for event in events][-1] == "message_stop", events
        starts: Final = [event for event in events if event.type == "message_start"]
        assert [start.message.id for start in starts] == [f"msg_{marker}"], events
        assert anthropic_labels(_only_request(wire)) == _CLIENT_MARKED
    assert gateway_injected(f"msg_{marker}") is False


async def test_openai_sdk_async_responses_keeps_system_and_user_marks_within_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        body: Final = responses_body(model, marker)
        async with _async_openai_client(gateway) as client:
            response: Final = await client.responses.create(
                model=model,
                instructions=SYSTEM,
                max_output_tokens=64,
                tools=cast(Iterable[ResponsesToolParam], body["tools"]),
                input=cast(ResponseInputParam, body["input"]),
            )
        assert response.status == "completed", response.model_dump()
        assert anthropic_labels(_only_request(wire)) == [SYSTEM_LABEL, ASK_LABEL]


def test_request_level_points_skip_injection_when_client_marks_fill_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        body: Final = chat_body(model, conversation(marker, marked_calls()), cache_control_injection_points=POINTS)
        status, response_id, text = post_chat(gateway, body)
        assert status == 200, text
        assert anthropic_labels(_only_request(wire)) == _CLIENT_MARKED
    assert gateway_injected(response_id) is False


_OTHER_CALL_RESULT: Final[dict[str, JsonValue]] = {
    "web_search_results": [{"type": "web_search_tool_result", "tool_use_id": "srvtoolu_other", "content": []}]
}
_SERVER_MARK_COUNTED: Final = [
    ASK_LABEL,
    tool_use_label(CITIES[0]),
    tool_use_label(CITIES[1]),
    "assistant:tool_use:srvtoolu_web",
]


@pytest.mark.parametrize(
    "fields",
    (
        pytest.param(1, id="int"),
        pytest.param(["web_search_results"], id="list"),
        pytest.param("", id="empty-string"),
        pytest.param("x" * 5120, id="5kb-string"),
        pytest.param({"web_search_results": "srvtoolu_web"}, id="results-not-a-list"),
        pytest.param({"web_search_results": ["srvtoolu_web", 1]}, id="results-without-objects"),
        pytest.param(_OTHER_CALL_RESULT, id="result-for-another-call"),
    ),
)
def test_malformed_provider_specific_fields_count_the_server_tool_call_mark(
    gateway: Gateway, fields: JsonValue
) -> None:
    marker: Final = new_marker()
    calls: Final[list[JsonValue]] = [*marked_calls(cities=CITIES[:2]), _SERVER_CALL]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, _, text = post_chat(
            gateway, chat_body(model, conversation(marker, calls, assistant={"provider_specific_fields": fields}))
        )
        assert status == 200, text
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == _SERVER_MARK_COUNTED


@pytest.mark.parametrize(
    ("identity", "forwarded"),
    (
        pytest.param("", "tool_use_id", id="empty-string"),
        pytest.param("x" * 5120, "x" * 5120, id="5kb-string"),
    ),
)
def test_odd_tool_call_ids_still_count_their_marks(gateway: Gateway, identity: str, forwarded: str) -> None:
    marker: Final = new_marker()
    calls: Final[list[JsonValue]] = [
        *marked_calls(cities=CITIES[:2]),
        tool_call(CITIES[2], cache_control=EPHEMERAL, id=identity),
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, _, text = post_chat(gateway, chat_body(model, conversation(marker, calls)))
        assert status == 200, text
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == [
        ASK_LABEL,
        tool_use_label(CITIES[0]),
        tool_use_label(CITIES[1]),
        f"assistant:tool_use:{forwarded}",
    ]


def _without_id(call: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in call.items() if key != "id"}


@pytest.mark.parametrize(
    "odd_call",
    (
        pytest.param(tool_call(CITIES[2], cache_control=EPHEMERAL, id=123), id="int"),
        pytest.param(tool_call(CITIES[2], cache_control=EPHEMERAL, id=["call_x"]), id="list"),
        pytest.param(tool_call(CITIES[2], cache_control=EPHEMERAL, id=None), id="null"),
        pytest.param(_without_id(tool_call(CITIES[2], cache_control=EPHEMERAL)), id="missing"),
    ),
)
def test_non_string_tool_call_ids_answer_400_without_an_upstream_call(
    gateway: Gateway, odd_call: dict[str, JsonValue]
) -> None:
    marker: Final = new_marker()
    calls: Final[list[JsonValue]] = [*marked_calls(cities=CITIES[:2]), odd_call]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        status, _, text = post_chat(gateway, chat_body(model, conversation(marker, calls)))
        received: Final = wire.drain()
    assert status == 400, text
    assert '"error"' in text, text
    assert len(received) == 0, [request.target for request in received]


def _unauthorized_peer(request: Request) -> Reply:
    assert len(anthropic_labels(request)) <= 4, anthropic_labels(request)
    return Reply(
        status=401,
        body=json.dumps(
            {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        ).encode(),
    )


def test_provider_error_on_a_capped_request_reaches_the_caller(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(_unauthorized_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", chat_body(model, conversation(marker, marked_calls()))
        )
        received: Final = wire.drain()
    assert response.status_code == 401, response.text
    assert "invalid x-api-key" in response.text, response.text
    assert [anthropic_labels(request) for request in received] == [_CLIENT_MARKED]


@pytest.mark.parametrize("points", (pytest.param(None, id="null"), pytest.param([], id="empty")))
def test_points_null_or_empty_forward_only_the_client_marks(gateway: Gateway, points: JsonValue) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=points)
        status, response_id, text = post_chat(gateway, chat_body(model, conversation(marker, marked_calls())))
        assert status == 200, text
        assert anthropic_labels(_only_request(wire)) == _CLIENT_MARKED
    assert gateway_injected(response_id) is False


def _cache_hit_rows(response_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        """SELECT request_id, metadata->>'litellm_gateway_injected_cache' AS injected FROM "LiteLLM_SpendLogs" """
        "WHERE request_id LIKE %s",
        (f"{response_id}_cache_hit%",),
    )


@pytest.mark.timeout(240)
def test_response_cache_hit_records_the_injection_on_the_first_row_only(uncapped_gateway: Gateway) -> None:
    marker: Final = new_marker()
    unmarked: Final = [tool_call(city) for city in CITIES]
    with wire_server(anthropic_peer) as wire, uncapped_gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        body: Final = chat_body(model, conversation(marker, unmarked, ask_marked=False))
        first: Final = post_chat(uncapped_gateway, body)
        second: Final = post_chat(uncapped_gateway, body)
        received: Final = wire.drain()
    assert (first[0], second[0]) == (200, 200), (first[2], second[2])
    assert first[1] == second[1], (first[2], second[2])
    assert [anthropic_labels(request) for request in received] == [[SYSTEM_LABEL, final_label(marker)]]
    assert gateway_injected(first[1]) is True
    hit_rows: Final = eventually(lambda: _cache_hit_rows(first[1]), lambda rows: len(rows) == 1, seconds=70)
    assert hit_rows[0]["injected"] is None, hit_rows


@pytest.mark.timeout(240)
@pytest.mark.parametrize("path", ("/v1/messages", "/v1/responses"))
def test_response_cache_twins_on_messages_and_responses_stay_within_the_cap(
    uncapped_gateway: Gateway, path: str
) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, uncapped_gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        body: Final = (
            messages_body(model, marker, stream=False) if path == "/v1/messages" else responses_body(model, marker)
        )
        responses: Final = tuple(uncapped_gateway.request("POST", path, body) for _ in range(2))
        received: Final = wire.drain()
    assert [response.status_code for response in responses] == [200, 200], [response.text for response in responses]
    expected: Final = _CLIENT_MARKED if path == "/v1/messages" else [SYSTEM_LABEL, ASK_LABEL]
    assert [anthropic_labels(request) for request in received] == [expected] * len(received)
    assert len(received) == 1, [request.target for request in received]


def _model_id(gateway: Gateway, model: str) -> str:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    matches: Final = [
        string_value(object_value(object_value(entry)["model_info"])["id"])
        for entry in entries
        if object_value(entry)["model_name"] == model
    ]
    assert len(matches) == 1, matches
    return matches[0]


_ASSISTANT_POINT: Final[list[JsonValue]] = [{"location": "message", "role": "assistant"}]
_UPDATE_BURST: Final = 20


async def _capped_burst(gateway: Gateway, model: str) -> tuple[tuple[int, str], ...]:
    async def one(client: httpx.AsyncClient) -> tuple[int, str]:
        body: Final = chat_body(model, conversation(new_marker(), marked_calls()))
        response: Final = await client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {gateway.key}"}
        )
        return response.status_code, response.text

    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(one(client) for _ in range(_UPDATE_BURST))))


def _probe_assistant_point(gateway: Gateway, wire: Wire, model: str, marker: str) -> list[str]:
    messages: Final[list[JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ASK},
        {"role": "assistant", "content": "I will check."},
        {"role": "user", "content": final_text(marker)},
    ]
    status, _, text = post_chat(gateway, chat_body(model, messages))
    assert status == 200, text
    return anthropic_labels(wire.drain()[-1])


async def test_points_update_mid_burst_keeps_every_request_within_the_cap(gateway: Gateway) -> None:
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, cache_control_injection_points=POINTS)
        identity: Final = _model_id(gateway, model)
        burst: Final = asyncio.create_task(_capped_burst(gateway, model))
        await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= 3, 30)
        await asyncio.to_thread(
            gateway.post,
            "/model/update",
            {
                "model_name": model,
                "litellm_params": {
                    "model": f"anthropic/{ANTHROPIC_MODEL}",
                    "api_base": wire.url,
                    "api_key": PROVIDER_KEY,
                    "cache_control_injection_points": _ASSISTANT_POINT,
                },
                "model_info": {"id": identity},
            },
        )
        sent: Final = await burst
        received: Final = wire.drain()
        assert [status for status, _ in sent] == [200] * _UPDATE_BURST, [text for _, text in sent]
        assert [anthropic_labels(request) for request in received] == [_CLIENT_MARKED] * _UPDATE_BURST
        landed: Final = await asyncio.to_thread(
            eventually,
            lambda: _probe_assistant_point(gateway, wire, model, new_marker()),
            lambda labels: labels == ["assistant:text:I will check."],
            60,
        )
    assert landed == ["assistant:text:I will check."]
