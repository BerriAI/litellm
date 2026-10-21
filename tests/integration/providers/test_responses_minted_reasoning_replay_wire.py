import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import EllipsisType, MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import anthropic
import httpx
import openai
import pytest
from integration._support import claude_code as cc
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_GPT: Final = "gpt-5.6"
_CODEX: Final = "gpt-5.3-codex"
_CLAUDE: Final = cc.OPUS
_OPENAI_KEY: Final = "synthetic-openai-key"
_AZURE_KEY: Final = "synthetic-azure-key"
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])


@dataclass(frozen=True, slots=True)
class _Deployment:
    label: str
    model: str
    api_key: str
    target: str
    extra: Mapping[str, JsonValue] = MappingProxyType({})
    strips_message_status: bool = False
    types_untyped_items_as_messages: bool = False
    model_info: Mapping[str, JsonValue] | None = None

    def register(self, scenario: Scenario, wire: Wire) -> str:
        return scenario.model(
            model=self.model, api_base=wire.url, api_key=self.api_key, model_info=self.model_info, **dict(self.extra)
        )

    def on_wire(self, items: Sequence[JsonValue]) -> list[JsonValue]:
        return [self._as_sent(item) for item in items]

    def _as_sent(self, item: JsonValue) -> JsonValue:
        if not isinstance(item, dict):
            return item
        if self.strips_message_status and item.get("type") == "message":
            return {key: value for key, value in item.items() if key != "status"}
        if self.types_untyped_items_as_messages and "type" not in item:
            return {**item, "type": "message"}
        return item


_OPENAI: Final = _Deployment("openai", f"openai/{_GPT}", _OPENAI_KEY, "/responses")
_AZURE: Final = _Deployment(
    "azure",
    f"azure/{_GPT}",
    _AZURE_KEY,
    "/openai/v1/responses?api-version=preview",
    MappingProxyType({"api_version": "preview"}),
    strips_message_status=True,
)
_AZURE_AI_OPENAI_HOST: Final = _Deployment(
    "azure_ai-rewritten-to-azure",
    f"azure_ai/{_GPT}",
    _AZURE_KEY,
    "/openai/v1/responses?api-version=preview",
    strips_message_status=True,
)
_DROPPING: Final = (_OPENAI, _AZURE, _AZURE_AI_OPENAI_HOST)
_KEEPING: Final = (
    _Deployment("litellm_proxy", f"litellm_proxy/{_GPT}", "synthetic-proxy-key", "/responses"),
    _Deployment("databricks", "databricks/gpt-5.6", "synthetic-databricks-key", "/responses"),
    _Deployment("openrouter", f"openrouter/openai/{_GPT}", "synthetic-openrouter-key", "/responses"),
    _Deployment("xai", "xai/grok-4.7", "synthetic-xai-key", "/responses"),
    _Deployment("hosted_vllm", "hosted_vllm/qwen3", "synthetic-vllm-key", "/responses"),
    _Deployment("fireworks_ai", "fireworks_ai/accounts/fireworks/models/kimi", "synthetic-fireworks-key", "/responses"),
    _Deployment("volcengine", "volcengine/doubao", "synthetic-volcengine-key", "/responses"),
    _Deployment("manus", "manus/manus-1", "synthetic-manus-key", "/responses"),
    _Deployment("edenai", "edenai/openai/gpt-5.6", "synthetic-edenai-key", "/responses"),
    _Deployment(
        "perplexity",
        "perplexity/sonar-pro",
        "synthetic-perplexity-key",
        "/v1/responses",
        types_untyped_items_as_messages=True,
    ),
    _Deployment("bedrock_mantle", "bedrock_mantle/openai.gpt-oss-120b", "synthetic-mantle-key", "/v1/responses"),
    _Deployment(
        "bedrock",
        "bedrock/openai.gpt-oss-120b-1:0",
        "synthetic-bedrock-key",
        "/openai/v1/responses",
        MappingProxyType({"aws_region_name": "us-east-1"}),
        model_info=MappingProxyType({"supported_endpoints": ["/v1/responses"]}),
    ),
    *(
        _Deployment(slug, f"{slug}/{model}", f"synthetic-{slug}-key", "/responses")
        for slug, model in (
            ("sail", "sail-1"),
            ("neosantara", "nusantara-base"),
            ("tensormesh", "qwen3"),
            ("parasail", "parasail-gpt-oss-120b"),
            ("empiriolabs", "empirio-1"),
            ("meta", "llama-4-maverick"),
            ("cortecs", "gpt-oss-120b"),
            ("pinstripes", "gpt-5.6"),
            ("prism", "gpt-oss-120b"),
        )
    ),
)


def _base_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _sdk(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{_base_url(gateway)}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _async_sdk(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{_base_url(gateway)}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def _claude_sdk(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=_base_url(gateway),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _create(
    client: openai.OpenAI, model: str, history: Sequence[Mapping[str, JsonValue]], stream: bool
) -> dict[str, JsonValue]:
    if not stream:
        return client.responses.create(model=model, input=list(history), extra_body=dict(_CACHE_BUST)).model_dump()
    events: Final = list(
        client.responses.create(model=model, input=list(history), stream=True, extra_body=dict(_CACHE_BUST))
    )
    completed: Final = [event for event in events if event.type == "response.completed"]
    assert len(completed) == 1, [event.type for event in events]
    return completed[0].response.model_dump()


async def _create_async(
    client: openai.AsyncOpenAI, model: str, history: Sequence[Mapping[str, JsonValue]], stream: bool
) -> dict[str, JsonValue]:
    if not stream:
        return (
            await client.responses.create(model=model, input=list(history), extra_body=dict(_CACHE_BUST))
        ).model_dump()
    events: Final = [
        event
        async for event in await client.responses.create(
            model=model, input=list(history), stream=True, extra_body=dict(_CACHE_BUST)
        )
    ]
    completed: Final = [event for event in events if event.type == "response.completed"]
    assert len(completed) == 1, [event.type for event in events]
    return completed[0].response.model_dump()


def _raw(
    gateway: Gateway, path: str, body: Mapping[str, JsonValue], *, key: str | None | EllipsisType = ...
) -> httpx.Response:
    with httpx.Client(base_url=_base_url(gateway), trust_env=False, timeout=60) as client:
        bearer: Final = gateway.key if key is ... else key
        headers: Final = {} if bearer is None else {"Authorization": f"Bearer {bearer}"}
        with client.stream("POST", path, json={**body, **_CACHE_BUST}, headers=headers) as response:
            response.read()
            return response


def _completed_payload(response: httpx.Response) -> dict[str, JsonValue]:
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        return _JSON_OBJECT.validate_json(response.content)
    frames: Final = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    completed: Final = [frame for frame in frames if frame.get("type") == "response.completed"]
    assert len(completed) == 1, [frame.get("type") for frame in frames]
    return _JSON_OBJECT.validate_python(completed[0]["response"])


def _answer_text(payload: Mapping[str, JsonValue]) -> str:
    messages: Final = [item for item in _ITEMS.validate_python(payload["output"]) if item.get("type") == "message"]
    assert len(messages) == 1, payload
    return str(_ITEMS.validate_python(messages[0]["content"])[0]["text"])


def _only_request(wire: Wire) -> tuple[Request, dict[str, JsonValue]]:
    received: Final = wire.drain()
    assert len(received) == 1, [(request.method, request.target) for request in received]
    return received[0], _JSON_OBJECT.validate_json(received[0].body)


def _assert_spend_rows(model: str, response_ids: Sequence[str]) -> None:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= len(response_ids),
        seconds=70,
    )
    logged: Final = {str(row["request_id"]): str(row["status"]) for row in rows}
    assert len(logged) == len(rows) == len(response_ids), rows
    for response_id in response_ids:
        (match,) = [logged_id for logged_id in logged if rv.same_response(logged_id, response_id)]
        assert logged[match] == "success", rows


def _assert_vendor_body(
    body: Mapping[str, JsonValue], backend: str, forwarded: Sequence[JsonValue], stream: bool
) -> None:
    assert body["model"] == backend, body
    assert body["input"] == list(forwarded), body["input"]
    assert body.get("stream", False) is stream, body
    assert "cache" not in body and "no-cache" not in json.dumps(body), body


def _backend_of(deployment: _Deployment) -> str:
    return deployment.model.split("/", 1)[1]


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("deployment", _DROPPING, ids=[deployment.label for deployment in _DROPPING])
def test_agents_sdk_history_replays_to_openai_shaped_vendors_without_the_minted_item(
    gateway: Gateway, deployment: _Deployment, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    minted: Final = rv.minted_item(marker)
    history: Final = rv.agents_sdk_history(marker, minted)
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = deployment.register(scenario, wire)
        payload: Final = _create(_sdk(gateway), model, history, stream)
        assert _answer_text(payload) == f"answer marker-{marker}", payload
        request, body = _only_request(wire)
        assert request.target == deployment.target, request.target
        _assert_vendor_body(body, _backend_of(deployment), deployment.on_wire(rv.without(history, (minted,))), stream)
        _assert_spend_rows(model, (str(payload["id"]),))


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_async_openai_sdk_replays_without_the_minted_item(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    minted: Final = rv.minted_item(marker)
    history: Final = rv.agents_sdk_history(marker, minted)
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _OPENAI.register(scenario, wire)
        payload: Final = await _create_async(_async_sdk(gateway), model, history, stream)
        assert _answer_text(payload) == f"answer marker-{marker}", payload
        request, body = _only_request(wire)
        assert request.target == "/responses", request.target
        _assert_vendor_body(body, _GPT, rv.without(history, (minted,)), stream)


@pytest.mark.parametrize("path", ["/v1/responses", "/responses", "/openai/v1/responses"])
def test_every_responses_route_alias_drops_the_minted_item(gateway: Gateway, path: str) -> None:
    marker: Final = uuid.uuid4().hex
    minted: Final = rv.minted_item(marker)
    history: Final = rv.agents_sdk_history(marker, minted)
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _OPENAI.register(scenario, wire)
        response: Final = _raw(gateway, path, {"model": model, "input": history})
        assert response.status_code == 200, response.text
        assert _answer_text(_completed_payload(response)) == f"answer marker-{marker}"
        _, body = _only_request(wire)
        _assert_vendor_body(body, _GPT, rv.without(history, (minted,)), False)


def test_identical_replays_each_land_one_spend_row(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final = rv.agents_sdk_history(marker, rv.minted_item(marker))
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _OPENAI.register(scenario, wire)
        first: Final = _completed_payload(_raw(gateway, "/v1/responses", {"model": model, "input": history}))
        second: Final = _completed_payload(_raw(gateway, "/v1/responses", {"model": model, "input": history}))
        assert first["id"] != second["id"]
        assert len(wire.drain()) == 2
        _assert_spend_rows(model, (str(first["id"]), str(second["id"])))


def _decoded_thinking(item: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    encrypted: Final = item["encrypted_content"]
    assert isinstance(encrypted, str), item
    return _ITEMS.validate_json(encrypted)


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_claude_turn_replays_to_openai_without_its_item_and_to_claude_with_its_thinking(
    gateway: Gateway, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        claude: Final = scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        gpt: Final = _OPENAI.register(scenario, wire)
        question: Final[dict[str, JsonValue]] = {"role": "user", "content": f"Pick a city marker-{marker}"}
        produced: Final = _completed_payload(
            _raw(gateway, "/v1/responses", {"model": claude, "input": [question], "stream": stream})
        )
        reasoning, message = _ITEMS.validate_python(produced["output"])
        assert reasoning["type"] == "reasoning" and rv.MINTED_ID.match(str(reasoning["id"])), reasoning
        assert "summary" not in reasoning, reasoning
        (block,) = _decoded_thinking(reasoning)
        assert (block["type"], block["signature"]) == ("thinking", rv.signature(marker)), block
        assert message["type"] == "message", message
        producing_request, producing_body = _only_request(wire)
        assert producing_request.target == "/v1/messages"

        follow_up: Final = uuid.uuid4().hex
        history: Final[list[dict[str, JsonValue]]] = [
            question,
            reasoning,
            message,
            {"role": "user", "content": f"Name a landmark marker-{follow_up}"},
        ]
        to_openai: Final = _raw(gateway, "/v1/responses", {"model": gpt, "input": history, "stream": stream})
        assert to_openai.status_code == 200, to_openai.text
        assert _answer_text(_completed_payload(to_openai)) == f"answer marker-{follow_up}"
        openai_request, openai_body = _only_request(wire)
        assert openai_request.target == "/responses"
        _assert_vendor_body(openai_body, _GPT, [question, message, history[3]], stream)

        to_claude: Final = _raw(gateway, "/v1/responses", {"model": claude, "input": history, "stream": stream})
        assert to_claude.status_code == 200, to_claude.text
        claude_request, claude_body = _only_request(wire)
        assert claude_request.target == "/v1/messages"
        messages: Final = _ITEMS.validate_python(claude_body["messages"])
        assistant: Final = [turn for turn in messages if turn["role"] == "assistant"]
        assert len(assistant) == 1, messages
        assert assistant[0]["content"] == [
            {"type": "thinking", "thinking": block["thinking"], "signature": rv.signature(marker)},
            {"type": "text", "text": _answer_text(produced)},
        ], assistant[0]


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("deployment", _KEEPING, ids=[deployment.label for deployment in _KEEPING])
def test_other_responses_providers_forward_the_minted_item_unchanged(
    gateway: Gateway, deployment: _Deployment, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    minted: Final = rv.minted_item(marker, summary=[])
    history: Final = rv.agents_sdk_history(marker, minted)
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = deployment.register(scenario, wire)
        response: Final = _raw(gateway, "/v1/responses", {"model": model, "input": history, "stream": stream})
        request, body = _only_request(wire)
        assert urlsplit(request.target).path.endswith("/responses"), request.target
        assert body["input"] == deployment.on_wire(history), body["input"]
        assert response.status_code == 404, response.text
        assert f"Item with id '{minted['id']}' not found" in response.text, response.text


@pytest.mark.parametrize(
    ("prefix", "forwarded_blocks"),
    [
        ("litellm_proxy", ("thinking", "text", "tool_use")),
        ("openai", ("text", "tool_use")),
    ],
)
def test_chained_hop_through_this_proxy_to_claude(
    gateway: Gateway, prefix: str, forwarded_blocks: tuple[str, ...]
) -> None:
    marker: Final = uuid.uuid4().hex
    minted: Final = rv.minted_item(marker)
    history: Final = rv.agents_sdk_history(marker, minted)
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        claude: Final = scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        outer: Final = scenario.model(model=f"{prefix}/{claude}", api_base=_base_url(gateway), api_key=gateway.key)
        response: Final = _raw(gateway, "/v1/responses", {"model": outer, "input": history})
        assert response.status_code == 200, response.text
        assert _answer_text(_completed_payload(response)) == f"answer marker-{marker}"
        request, body = _only_request(wire)
        assert request.target == "/v1/messages"
        assistant: Final = [turn for turn in _ITEMS.validate_python(body["messages"]) if turn["role"] == "assistant"]
        assert len(assistant) == 1, body["messages"]
        blocks: Final = _ITEMS.validate_python(assistant[0]["content"])
        assert tuple(str(block["type"]) for block in blocks) == forwarded_blocks, blocks
        if "thinking" in forwarded_blocks:
            assert blocks[0] == {"type": "thinking", "thinking": rv.THOUGHT, "signature": rv.signature(marker)}, blocks[
                0
            ]


@dataclass(frozen=True, slots=True)
class _Hostile:
    label: str
    item: dict[str, JsonValue]
    status: int
    forwarded: bool
    detail: str = ""
    on_wire: Mapping[str, JsonValue] | None = None


def _hostile_cases() -> tuple[_Hostile, ...]:
    marker: Final = "0" * 32
    signed: Final = {"type": "thinking", "thinking": rv.THOUGHT, "signature": rv.signature(marker)}
    unsigned: Final = {"type": "thinking", "thinking": rv.THOUGHT}
    summary: Final[list[JsonValue]] = [{"type": "summary_text", "text": "thought about it"}]
    big_blob: Final = "x" * 5000
    big_blocks: Final = json.dumps([signed] * 60)
    assert len(big_blocks) > 5000
    return (
        _Hostile(
            "uppercase-uuid4-id",
            {"type": "reasoning", "id": f"rs_{str(uuid.uuid4()).upper()}", "summary": []},
            404,
            True,
            "Item with id",
        ),
        _Hostile(
            "minted-id-with-summary", {"type": "reasoning", "id": f"rs_{uuid.uuid4()}", "summary": summary}, 200, False
        ),
        _Hostile(
            "idless-opaque-blob", {"type": "reasoning", "encrypted_content": "gAAAAA-opaque", "summary": []}, 200, True
        ),
        _Hostile(
            "idless-unverifiable-blocks",
            {"type": "reasoning", "encrypted_content": json.dumps([unsigned]), "summary": []},
            200,
            True,
        ),
        _Hostile(
            "idless-mixed-blocks",
            {
                "type": "reasoning",
                "encrypted_content": json.dumps([unsigned, {"type": "text", "text": "x"}, signed]),
                "summary": [],
            },
            200,
            False,
        ),
        _Hostile("int-id", {"type": "reasoning", "id": 7, "summary": []}, 400, True, "input"),
        _Hostile("list-id", {"type": "reasoning", "id": ["rs_x"], "summary": []}, 400, True, "input"),
        _Hostile("empty-id", {"type": "reasoning", "id": "", "summary": summary}, 400, True, "empty string"),
        _Hostile("int-encrypted-content", {"type": "reasoning", "encrypted_content": 7, "summary": []}, 200, True),
        _Hostile(
            "list-encrypted-content", {"type": "reasoning", "encrypted_content": [signed], "summary": []}, 200, True
        ),
        _Hostile("empty-encrypted-content", {"type": "reasoning", "encrypted_content": "", "summary": []}, 200, True),
        _Hostile("five-kb-blob", {"type": "reasoning", "encrypted_content": big_blob, "summary": []}, 200, True),
        _Hostile(
            "five-kb-signed-blocks", {"type": "reasoning", "encrypted_content": big_blocks, "summary": []}, 200, False
        ),
        _Hostile(
            "null-id-null-encrypted",
            {"type": "reasoning", "id": None, "encrypted_content": None, "summary": []},
            200,
            True,
            on_wire={"type": "reasoning", "id": None, "summary": []},
        ),
        _Hostile(
            "message-with-minted-looking-id",
            {
                "type": "message",
                "id": f"rs_{uuid.uuid4()}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "x", "annotations": []}],
            },
            200,
            True,
        ),
    )


_HOSTILE: Final = _hostile_cases()


@pytest.mark.parametrize("case", _HOSTILE, ids=[case.label for case in _HOSTILE])
def test_hostile_reasoning_items_reach_the_vendor_or_are_dropped_as_classified(
    gateway: Gateway, case: _Hostile
) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final = rv.agents_sdk_history(marker, case.item)
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _OPENAI.register(scenario, wire)
        response: Final = _raw(gateway, "/v1/responses", {"model": model, "input": history})
        assert response.status_code == case.status, response.text
        assert case.detail in response.text, response.text
        received: Final = wire.drain()
        if response.status_code >= 400 and not received:
            return
        assert len(received) == 1, [(request.method, request.target) for request in received]
        body: Final = _JSON_OBJECT.validate_json(received[0].body)
        expected: Final = (
            [case.on_wire if item is case.item and case.on_wire is not None else item for item in history]
            if case.forwarded
            else rv.without(history, (case.item,))
        )
        assert body["input"] == expected, body["input"]
        assert response.status_code == case.status
        if case.status == 200:
            assert _answer_text(_completed_payload(response)) == f"answer marker-{marker}"
        unrelated: Final = _raw(gateway, "/v1/responses", {"model": model, "input": f"ping marker-{marker}"})
        assert unrelated.status_code == 200, unrelated.text


def test_vendor_owned_reasoning_item_from_a_producing_turn_is_kept(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _OPENAI.register(scenario, wire)
        question: Final[dict[str, JsonValue]] = {"role": "user", "content": f"Pick a city marker-{marker}"}
        produced: Final = _completed_payload(_raw(gateway, "/v1/responses", {"model": model, "input": [question]}))
        reasoning, message = _ITEMS.validate_python(produced["output"])
        assert str(reasoning["id"]).startswith("rs_") and not rv.MINTED_ID.match(str(reasoning["id"])), reasoning
        wire.drain()
        follow_up: Final = uuid.uuid4().hex
        history: Final[list[dict[str, JsonValue]]] = [
            question,
            reasoning,
            message,
            {"role": "user", "content": f"Name a landmark marker-{follow_up}"},
        ]
        response: Final = _raw(gateway, "/v1/responses", {"model": model, "input": history})
        assert response.status_code == 200, response.text
        _, body = _only_request(wire)
        assert body["input"] == history, body["input"]


def test_two_minted_items_are_both_dropped_and_a_minted_only_history_goes_out_empty(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    first: Final = rv.minted_item(marker)
    second: Final = rv.minted_item(marker)
    history: Final = rv.agents_sdk_history(marker, first, second)
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _OPENAI.register(scenario, wire)
        response: Final = _raw(gateway, "/v1/responses", {"model": model, "input": history})
        assert response.status_code == 200, response.text
        _, body = _only_request(wire)
        assert body["input"] == rv.without(history, (first, second)), body["input"]

        lonely: Final = _raw(gateway, "/v1/responses", {"model": model, "input": [rv.minted_item(marker)]})
        assert lonely.status_code == 400, lonely.text
        assert "previous_response_id" in lonely.text and "must be provided" in lonely.text, lonely.text
        _, lonely_body = _only_request(wire)
        assert lonely_body["input"] == [], lonely_body


def test_a_megabyte_of_minted_thinking_is_dropped_while_the_proxy_stays_responsive(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    block: Final = {"type": "thinking", "thinking": "t" * 4000, "signature": rv.signature(marker)}
    encrypted: Final = json.dumps([block] * 256)
    assert len(encrypted) > 1_000_000
    minted: Final[dict[str, JsonValue]] = {
        "type": "reasoning",
        "id": f"rs_{uuid.uuid4()}",
        "encrypted_content": encrypted,
    }
    history: Final = rv.agents_sdk_history(marker, minted)
    latencies: Final[deque[float]] = deque()
    done: Final = threading.Event()

    def probe() -> None:
        with httpx.Client(base_url=_base_url(gateway), trust_env=False, timeout=30) as client:
            while not done.is_set():
                started: Final = time.monotonic()
                assert client.get("/health/liveliness").status_code == 200
                latencies.append(time.monotonic() - started)

    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _OPENAI.register(scenario, wire)
        prober: Final = threading.Thread(target=probe)
        prober.start()
        started: Final = time.monotonic()
        response: Final = _raw(gateway, "/v1/responses", {"model": model, "input": history})
        elapsed: Final = time.monotonic() - started
        done.set()
        prober.join(timeout=35)
        assert response.status_code == 200, response.text[:500]
        assert elapsed < 20, elapsed
        assert latencies and max(latencies) < 5, (max(latencies), len(latencies))
        _, body = _only_request(wire)
        assert body["input"] == rv.without(history, (minted,))


def test_unauthenticated_replay_never_reaches_the_vendor_and_other_keys_keep_working(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final = rv.agents_sdk_history(marker, rv.minted_item(marker))
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _OPENAI.register(scenario, wire)
        other: Final = scenario.key(models=[model])
        anonymous: Final = _raw(gateway, "/v1/responses", {"model": model, "input": history}, key=None)
        assert anonymous.status_code == 401, anonymous.text
        forged: Final = _raw(gateway, "/v1/responses", {"model": model, "input": history}, key="sk-not-a-key")
        assert forged.status_code == 401, forged.text
        assert wire.drain() == ()
        failing: Final = _raw(
            gateway,
            "/v1/responses",
            {
                "model": model,
                "input": rv.agents_sdk_history(marker, {"type": "reasoning", "id": "rs_" + "f" * 32, "summary": []}),
            },
        )
        assert failing.status_code == 404, failing.text
        assert "rs_" + "f" * 32 in failing.text, failing.text
        healthy: Final = _raw(gateway, "/v1/responses", {"model": model, "input": history}, key=other)
        assert healthy.status_code == 200, healthy.text
        assert [request.target for request in wire.drain()] == ["/responses", "/responses"]


def _chat_history(marker: str, reasoning_items: Sequence[Mapping[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": "Pick a city."},
        {"role": "assistant", "content": "Prague", "reasoning_items": [dict(item) for item in reasoning_items]},
        {"role": "user", "content": f"Name a landmark marker-{marker}"},
    ]


def _chat_create(client: openai.OpenAI, model: str, messages: Sequence[Mapping[str, JsonValue]], stream: bool) -> str:
    if not stream:
        completion: Final = client.chat.completions.create(
            model=model, messages=list(messages), extra_body=dict(_CACHE_BUST)
        )
        return str(completion.choices[0].message.content)
    chunks: Final = list(
        client.chat.completions.create(model=model, messages=list(messages), stream=True, extra_body=dict(_CACHE_BUST))
    )
    return "".join(str(chunk.choices[0].delta.content or "") for chunk in chunks if chunk.choices)


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_chat_bridge_replays_a_stored_reasoning_item_without_inventing_an_id(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    stored: Final[dict[str, JsonValue]] = {
        "type": "reasoning",
        "encrypted_content": f"gAAAAA-stored-{marker}",
        "summary": [],
    }
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_CODEX}", api_base=wire.url, api_key=_OPENAI_KEY)
        answer: Final = _chat_create(_sdk(gateway), model, _chat_history(marker, (stored,)), stream)
        assert answer == f"answer marker-{marker}"
        request, body = _only_request(wire)
        assert request.target == "/responses"
        assert body["model"] == _CODEX
        assert rv.reasoning_items(body) == [stored], body["input"]


async def test_chat_bridge_async_client_replays_a_stored_reasoning_item_without_inventing_an_id(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    stored: Final[dict[str, JsonValue]] = {
        "type": "reasoning",
        "encrypted_content": f"gAAAAA-stored-{marker}",
        "summary": [],
    }
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_CODEX}", api_base=wire.url, api_key=_OPENAI_KEY)
        completion: Final = await _async_sdk(gateway).chat.completions.create(
            model=model, messages=_chat_history(marker, (stored,)), extra_body=dict(_CACHE_BUST)
        )
        assert completion.choices[0].message.content == f"answer marker-{marker}"
        _, body = _only_request(wire)
        assert rv.reasoning_items(body) == [stored], body["input"]


def test_chat_bridge_keeps_a_vendor_minted_id_and_sends_an_empty_item_bare(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_CODEX}", api_base=wire.url, api_key=_OPENAI_KEY)
        produced: Final = _sdk(gateway).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Pick a city marker-{marker}"}],
            extra_body=dict(_CACHE_BUST),
        )
        message: Final = produced.choices[0].message.model_dump()
        (stored,) = _ITEMS.validate_python(message["reasoning_items"])
        assert str(stored["id"]).startswith("rs_") and str(stored["encrypted_content"]).startswith("gAAAAA-vendor-"), (
            stored
        )
        wire.drain()
        follow_up: Final = uuid.uuid4().hex
        answer: Final = _chat_create(_sdk(gateway), model, _chat_history(follow_up, (stored,)), False)
        assert answer == f"answer marker-{follow_up}"
        _, body = _only_request(wire)
        assert rv.reasoning_items(body) == [
            {"type": "reasoning", "id": stored["id"], "summary": [], "encrypted_content": stored["encrypted_content"]}
        ], body["input"]

        bare: Final = uuid.uuid4().hex
        assert (
            _chat_create(_sdk(gateway), model, _chat_history(bare, ({"type": "reasoning", "summary": []},)), False)
            == f"answer marker-{bare}"
        )
        _, bare_body = _only_request(wire)
        assert rv.reasoning_items(bare_body) == [{"type": "reasoning", "summary": []}], bare_body["input"]


def test_chat_mode_model_takes_the_same_assistant_message_on_the_chat_wire(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    stored: Final[dict[str, JsonValue]] = {
        "type": "reasoning",
        "encrypted_content": f"gAAAAA-stored-{marker}",
        "summary": [],
    }
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_GPT}", api_base=wire.url, api_key=_OPENAI_KEY)
        assert _chat_create(_sdk(gateway), model, _chat_history(marker, (stored,)), False) == f"answer marker-{marker}"
        request, body = _only_request(wire)
        assert request.target == "/chat/completions"
        messages: Final = _ITEMS.validate_python(body["messages"])
        assert [turn["role"] for turn in messages] == ["user", "assistant", "user"], messages
        assert messages[1]["content"] == "Prague", messages[1]


def _thinking_turns(marker: str) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": "Pick a city."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": rv.THOUGHT, "signature": rv.signature(marker)},
                {"type": "text", "text": "Prague"},
            ],
        },
        {"role": "user", "content": f"Name a landmark marker-{marker}"},
    ]


def _messages_create(
    client: anthropic.Anthropic, model: str, messages: Sequence[Mapping[str, JsonValue]], stream: bool
) -> str:
    if not stream:
        reply: Final = client.messages.create(
            model=model, max_tokens=64, messages=list(messages), extra_body=dict(_CACHE_BUST)
        )
        return "".join(block.text for block in reply.content if block.type == "text")
    with client.messages.stream(
        model=model, max_tokens=64, messages=list(messages), extra_body=dict(_CACHE_BUST)
    ) as stream_reply:
        final: Final = stream_reply.get_final_message()
    return "".join(block.text for block in final.content if block.type == "text")


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_messages_endpoint_replays_claude_thinking_to_claude_unchanged(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    turns: Final = _thinking_turns(marker)
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        assert _messages_create(_claude_sdk(gateway), model, turns, stream) == f"answer marker-{marker}"
        request, body = _only_request(wire)
        assert request.target == "/v1/messages"
        assert body["messages"] == turns, body["messages"]
        assert body.get("stream", False) is stream, body


@pytest.mark.parametrize("backend", [_CODEX, _GPT])
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_messages_endpoint_on_an_openai_model_sends_an_idless_reasoning_item(
    gateway: Gateway, backend: str, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{backend}", api_base=wire.url, api_key=_OPENAI_KEY)
        assert (
            _messages_create(_claude_sdk(gateway), model, _thinking_turns(marker), stream) == f"answer marker-{marker}"
        )
        request, body = _only_request(wire)
        assert request.target == "/responses"
        assert body.get("stream", False) is stream, body
        (item,) = rv.reasoning_items(body)
        assert "id" not in item and "summary" in item, item
