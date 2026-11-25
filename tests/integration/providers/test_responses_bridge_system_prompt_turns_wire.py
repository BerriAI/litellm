import itertools
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.openai_wire import answering_model_discovery, responses_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])
_QUESTIONS: Final = (
    "Which file validates order quantities?",
    "Read that file and list every check it runs.",
    "Which of those checks rejects a quantity of zero?",
)
_TURNS: Final = range(1, len(_QUESTIONS) + 1)
_LEADING_ROLES: Final = frozenset({"system", "developer"})
_CHAT_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a project file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }
]
_MESSAGES_TOOLS: Final[list[JsonValue]] = [
    {
        "name": "read_file",
        "description": "Read a project file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }
]

Endpoint: TypeAlias = Literal["chat", "messages"]
SystemShape: TypeAlias = Literal["text", "cached_blocks"]

_PATHS: Final[Mapping[Endpoint, str]] = {"chat": "/v1/chat/completions", "messages": "/v1/messages"}


@dataclass(frozen=True, slots=True)
class _Deployment:
    model: str
    api_key: str
    api_version: str | None = None
    aws_region_name: str | None = None

    def register(self, scenario: Scenario, api_base: str) -> str:
        return scenario.model(
            model=self.model,
            api_base=api_base,
            api_key=self.api_key,
            api_version=self.api_version,
            aws_region_name=self.aws_region_name,
        )


_DEPLOYMENTS: Final = (
    pytest.param(_Deployment("openai/gpt-6.2-sol", "synthetic-openai-key"), id="openai"),
    pytest.param(_Deployment("azure/gpt-6.2-sol", "synthetic-azure-key", api_version="2025-04-01-preview"), id="azure"),
    pytest.param(
        _Deployment("bedrock_mantle/openai.gpt-6.2-sol", "synthetic-mantle-key", aws_region_name="us-east-1"),
        id="bedrock_mantle",
    ),
)
_ENDPOINTS: Final = (pytest.param("chat", id="chat"), pytest.param("messages", id="messages"))
_STORES: Final = (
    pytest.param(None, id="store_unset"),
    pytest.param(True, id="store_true"),
    pytest.param(False, id="store_false"),
)
_STREAMS: Final = (pytest.param(False, id="nonstream"), pytest.param(True, id="stream"))


@dataclass(frozen=True, slots=True)
class _Conversation:
    endpoint: Endpoint
    marker: str
    store: bool | None = None
    stream: bool = False
    system_shape: SystemShape = "text"
    truncation: str | None = None
    previous_response_id: str | None = None

    @property
    def system_prompt(self) -> str:
        return " ".join(
            f"Rule {index} for session {self.marker} is to answer in one short sentence." for index in range(160)
        )

    def answer(self, turn: int) -> str:
        return f"Answer {turn} for session {self.marker}"

    def body(self, model: str, turn: int) -> dict[str, JsonValue]:
        optional: Final = {
            key: value
            for key, value in (
                ("store", self.store),
                ("truncation", self.truncation),
                ("previous_response_id", self.previous_response_id),
            )
            if value is not None
        }
        fields: Final = self._chat_fields(turn) if self.endpoint == "chat" else self._messages_fields(turn)
        return {"model": model, "stream": self.stream, **optional, **fields}

    def _history(self, turn: int, assistant: Callable[[str], JsonValue]) -> list[JsonValue]:
        earlier: Final = itertools.chain.from_iterable(
            _exchange(_QUESTIONS[index - 1], assistant(self.answer(index))) for index in range(1, turn)
        )
        return [*earlier, _user(_QUESTIONS[turn - 1])]

    def _chat_fields(self, turn: int) -> dict[str, JsonValue]:
        return {
            "messages": [{"role": "system", "content": self.system_prompt}, *self._history(turn, _chat_text)],
            "tools": _CHAT_TOOLS,
            "reasoning_effort": "low",
        }

    def _messages_fields(self, turn: int) -> dict[str, JsonValue]:
        return {
            "max_tokens": 256,
            "system": self._messages_system,
            "messages": self._history(turn, _messages_text),
            "tools": _MESSAGES_TOOLS,
        }


    @property
    def _messages_system(self) -> JsonValue:
        if self.system_shape == "text":
            return self.system_prompt
        return [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}]


def _user(text: str) -> JsonValue:
    return {"role": "user", "content": text}


def _exchange(question: str, answer: JsonValue) -> tuple[JsonValue, JsonValue]:
    return _user(question), {"role": "assistant", "content": answer}


def _chat_text(text: str) -> JsonValue:
    return text


def _messages_text(text: str) -> JsonValue:
    return [{"type": "text", "text": text}]


def _upstream(conversation: _Conversation) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = _JSON_OBJECT.validate_json(request.body)
        model: Final = body["model"]
        assert isinstance(model, str), request.body
        turn: Final = sum(1 for item in _ITEMS.validate_python(body["input"]) if item.get("role") == "user")
        return responses_reply(
            f"resp_{uuid.uuid4().hex}", model, conversation.answer(turn), stream=body.get("stream") is True
        )

    return answering_model_discovery(respond)


@dataclass(frozen=True, slots=True)
class _Upstream:
    raw: bytes
    body: dict[str, JsonValue]

    @property
    def input(self) -> list[dict[str, JsonValue]]:
        return _ITEMS.validate_python(self.body["input"])


def _converse(gateway: Gateway, deployment: _Deployment, conversation: _Conversation) -> tuple[_Upstream, ...]:
    with wire_server(_upstream(conversation)) as wire, gateway.scenario() as scenario:
        model: Final = deployment.register(scenario, wire.url)
        replies: Final = tuple(
            gateway.request("POST", _PATHS[conversation.endpoint], conversation.body(model, turn)) for turn in _TURNS
        )
        posted: Final = tuple(request for request in wire.drain() if request.method == "POST")
    assert [reply.status_code for reply in replies] == [200] * len(_TURNS), [reply.text for reply in replies]
    assert all(conversation.answer(turn) in reply.text for turn, reply in zip(_TURNS, replies, strict=True)), [
        reply.text for reply in replies
    ]
    assert len(posted) == len(_TURNS), [request.target for request in posted]
    assert all("/responses" in request.target for request in posted), [request.target for request in posted]
    return tuple(_Upstream(raw=request.body, body=_JSON_OBJECT.validate_json(request.body)) for request in posted)


def _assert_system_prompt_leads_the_input(sent: _Upstream, prompt: str) -> None:
    leading: Final = sent.input[0]
    assert sent.body.get("instructions") is None, sent.body
    assert leading.get("type") == "message" and leading.get("role") in _LEADING_ROLES, sent.body
    assert leading.get("content") == [{"type": "input_text", "text": prompt}], sent.body
    assert sent.raw.count(prompt.encode()) == 1, sent.body


def _assert_each_turn_extends_the_last(upstream: tuple[_Upstream, ...]) -> None:
    for earlier, later in itertools.pairwise(upstream):
        assert later.input[: len(earlier.input)] == earlier.input, (earlier.body, later.body)
        assert len(later.input) > len(earlier.input), later.body


@pytest.mark.parametrize("stream", _STREAMS)
@pytest.mark.parametrize("store", _STORES)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("deployment", _DEPLOYMENTS)
def test_system_prompt_leads_the_input_on_every_turn_of_a_bridged_conversation(
    gateway: Gateway, deployment: _Deployment, endpoint: Endpoint, store: bool | None, stream: bool
) -> None:
    conversation: Final = _Conversation(endpoint=endpoint, marker=uuid.uuid4().hex, store=store, stream=stream)
    prompt: Final = conversation.system_prompt

    upstream: Final = _converse(gateway, deployment, conversation)

    for sent in upstream:
        _assert_system_prompt_leads_the_input(sent, prompt)
    _assert_each_turn_extends_the_last(upstream)


@pytest.mark.parametrize("stream", _STREAMS)
@pytest.mark.parametrize("deployment", _DEPLOYMENTS)
def test_system_prompt_sent_as_cached_blocks_leads_the_input_on_every_turn_of_a_bridged_messages_conversation(
    gateway: Gateway, deployment: _Deployment, stream: bool
) -> None:
    conversation: Final = _Conversation(
        endpoint="messages", marker=uuid.uuid4().hex, stream=stream, system_shape="cached_blocks"
    )
    prompt: Final = conversation.system_prompt

    upstream: Final = _converse(gateway, deployment, conversation)

    for sent in upstream:
        _assert_system_prompt_leads_the_input(sent, prompt)
    _assert_each_turn_extends_the_last(upstream)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("deployment", _DEPLOYMENTS)
def test_truncation_auto_keeps_the_system_prompt_in_instructions_on_every_turn(
    gateway: Gateway, deployment: _Deployment, endpoint: Endpoint
) -> None:
    conversation: Final = _Conversation(endpoint=endpoint, marker=uuid.uuid4().hex, truncation="auto")
    prompt: Final = conversation.system_prompt

    upstream: Final = _converse(gateway, deployment, conversation)

    for sent in upstream:
        assert sent.body.get("instructions") == prompt, sent.body
        assert sent.body.get("truncation") == "auto", sent.body
        assert sent.raw.count(prompt.encode()) == 1, sent.body
    _assert_each_turn_extends_the_last(upstream)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("deployment", _DEPLOYMENTS)
def test_previous_response_id_on_a_bridged_turn_is_rejected_before_the_provider(
    gateway: Gateway, deployment: _Deployment, endpoint: Endpoint
) -> None:
    conversation: Final = _Conversation(
        endpoint=endpoint, marker=uuid.uuid4().hex, previous_response_id=f"resp_{uuid.uuid4().hex}"
    )
    with wire_server(_upstream(conversation)) as wire, gateway.scenario() as scenario:
        model: Final = deployment.register(scenario, wire.url)
        response: Final = gateway.request("POST", _PATHS[endpoint], conversation.body(model, 2))
        posted: Final = tuple(request for request in wire.drain() if request.method == "POST")
    assert response.status_code == 400, response.text
    assert "previous_response_id" in response.text, response.text
    assert posted == (), [request.target for request in posted]
