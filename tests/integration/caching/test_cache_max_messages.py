import hashlib
import json
import math
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_VECTOR_SIZE: Final = 16
_CHAT_MODEL: Final = "capped-chat"
_CLAUDE_MODEL: Final = "capped-claude"
_EMBEDDING_MODEL: Final = "capped-embedder"
_COLLECTION: Final = "cache-max-messages"


def _vector(text: str) -> tuple[float, ...]:
    digest: Final = hashlib.sha256(text.encode()).digest()
    return tuple((byte - 127.5) / 127.5 for byte in digest[:_VECTOR_SIZE])


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot: Final = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


def _answer() -> str:
    return f"answer-{uuid.uuid4().hex[:16]}"


@dataclass(slots=True)
class _Peer:
    """Embeddings, chat, Responses, Anthropic Messages and a Qdrant collection on one owned socket"""

    lock: threading.Lock = field(default_factory=threading.Lock)
    points: list[Mapping[str, JsonValue]] = field(default_factory=list)  # mutable-ok: the Qdrant collection
    embedded: list[str] = field(default_factory=list)  # mutable-ok: every prompt the proxy embedded
    answered: list[str] = field(default_factory=list)  # mutable-ok: every completion the provider served

    def stored(self) -> int:
        with self.lock:
            return len(self.points)

    def respond(self, request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        body: Final = _JSON_OBJECT.validate_json(request.body) if request.body else {}
        collection: Final = f"/qdrant/collections/{_COLLECTION}"
        if request.method == "GET" and path == f"{collection}/exists":
            return self._json({"result": {"exists": False}, "status": "ok"})
        if request.method in {"GET", "PUT"} and path in {collection, f"{collection}/index"}:
            return self._json({"result": True, "status": "ok"})
        if request.method == "PUT" and path == f"{collection}/points":
            with self.lock:
                self.points.extend(_JSON_OBJECT.validate_python(point) for point in _list(body["points"]))
            return self._json({"result": {"status": "completed"}, "status": "ok"})
        if request.method == "POST" and path == f"{collection}/points/search":
            return self._json({"result": self._search(body), "status": "ok"})
        if request.method == "GET" and path == "/v1/models":
            return self._json({"object": "list", "data": []})
        if request.method == "POST" and path == "/v1/embeddings":
            text: Final = str(body["input"])
            with self.lock:
                self.embedded.append(text)
            return self._json(
                {
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [{"object": "embedding", "index": 0, "embedding": list(_vector(text))}],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                }
            )
        if request.method == "POST" and path == "/v1/chat/completions":
            return self._json(self._chat_reply(_answer()))
        if request.method == "POST" and path == "/v1/responses":
            return self._json(self._responses_reply(_answer()))
        if request.method == "POST" and path == "/v1/messages":
            return self._json(self._messages_reply(_answer()))
        raise AssertionError(f"unexpected peer request {request.method} {request.target}")

    def _search(self, body: Mapping[str, JsonValue]) -> list[JsonValue]:
        query: Final = [float(str(value)) for value in _list(body["vector"])]
        key: Final = _JSON_OBJECT.validate_python(
            _JSON_OBJECT.validate_python(_list(_JSON_OBJECT.validate_python(body["filter"])["must"])[0])["match"]
        )["value"]
        with self.lock:
            scoped: Final = [
                point
                for point in self.points
                if _JSON_OBJECT.validate_python(point["payload"])["litellm_cache_key"] == key
            ]
        ranked: Final = sorted(
            (
                {
                    "id": point["id"],
                    "score": _cosine(query, [float(str(value)) for value in _list(point["vector"])]),
                    "payload": point["payload"],
                }
                for point in scoped
            ),
            key=lambda hit: -float(str(hit["score"])),
        )
        return list(ranked[:1])

    def _chat_reply(self, answer: str) -> Mapping[str, JsonValue]:
        with self.lock:
            self.answered.append(answer)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1789788253,
            "model": "gpt-5.4-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

    def _responses_reply(self, answer: str) -> Mapping[str, JsonValue]:
        with self.lock:
            self.answered.append(answer)
        identity: Final = uuid.uuid4().hex
        return {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": "gpt-5.4-mini",
            "output": [
                {
                    "id": f"msg_{identity}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": answer, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        }

    def _messages_reply(self, answer: str) -> Mapping[str, JsonValue]:
        with self.lock:
            self.answered.append(answer)
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": answer}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }

    @staticmethod
    def _json(value: Mapping[str, JsonValue]) -> Reply:
        return Reply(body=json.dumps(value).encode())


def _list(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), value
    return value


@dataclass(frozen=True, slots=True)
class _Proxy:
    gateway: Gateway
    peer: _Peer

    def reply(self, route: str, conversation: list[JsonValue]) -> str:
        """The answer text for one request, as the client sees it"""
        model: Final = _CLAUDE_MODEL if route == "/v1/messages" else _CHAT_MODEL
        body: Final[dict[str, JsonValue]] = (
            {"model": model, "input": conversation}
            if route == "/v1/responses"
            else {"model": model, "max_tokens": 16, "messages": conversation}
        )
        response: Final = self.gateway.request("POST", route, body)
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        if route == "/v1/messages":
            return str(_JSON_OBJECT.validate_python(_list(payload["content"])[0])["text"])
        if route == "/v1/responses":
            message: Final = _JSON_OBJECT.validate_python(_list(payload["output"])[0])
            return str(_JSON_OBJECT.validate_python(_list(message["content"])[0])["text"])
        choice: Final = _JSON_OBJECT.validate_python(_list(payload["choices"])[0])
        return str(_JSON_OBJECT.validate_python(choice["message"])["content"])

    def miss(self, route: str, conversation: list[JsonValue]) -> str:
        answered_before: Final = len(self.peer.answered)
        answer: Final = self.reply(route, conversation)
        assert len(self.peer.answered) == answered_before + 1, f"{answer} was served from the cache"
        return answer

    def hit(self, route: str, conversation: list[JsonValue]) -> str:
        """Repeats the request until the cache serves it, since the store after a miss is asynchronous"""

        def attempt() -> tuple[str, bool]:
            answered_before: Final = len(self.peer.answered)
            answer: Final = self.reply(route, conversation)
            return answer, len(self.peer.answered) == answered_before

        return eventually(attempt, lambda outcome: outcome[1])[0]


def _proxy(tmp_path_factory: pytest.TempPathFactory, cache_params: Mapping[str, JsonValue]) -> Iterator[_Proxy]:
    peer: Final = _Peer()
    directory: Final = tmp_path_factory.mktemp("cache-max-messages")
    with gateway_from_environment() as gateway, wire_server(peer.respond) as wire:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            {
                "model_name": _CHAT_MODEL,
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_base": f"{wire.url}/v1", "api_key": "k"},
            },
            {
                "model_name": _CLAUDE_MODEL,
                "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_base": wire.url, "api_key": "k"},
            },
            {
                "model_name": _EMBEDDING_MODEL,
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_base": f"{wire.url}/v1",
                    "api_key": "k",
                },
            },
        ]
        config["litellm_settings"]["cache_params"] = {
            **cache_params,
            **({"qdrant_api_base": f"{wire.url}/qdrant"} if cache_params["type"] == "qdrant-semantic" else {}),
        }
        path: Final = directory / "cache_max_messages.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, directory, {}, config=path) as candidate:
            yield _Proxy(candidate, peer)


@pytest.fixture(scope="module")
def qdrant_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Proxy]:
    """Qdrant semantic cache with the default max_messages of 4"""
    yield from _proxy(
        tmp_path_factory,
        {
            "type": "qdrant-semantic",
            "qdrant_collection_name": _COLLECTION,
            "qdrant_semantic_cache_embedding_model": _EMBEDDING_MODEL,
            "qdrant_semantic_cache_vector_size": _VECTOR_SIZE,
            "qdrant_quantization_config": "binary",
            "similarity_threshold": 0.99,
        },
    )


@pytest.fixture(scope="module")
def exact_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Proxy]:
    """Exact cache with max_messages lowered to 2 in cache_params"""
    yield from _proxy(tmp_path_factory, {"type": "local", "max_messages": 2})


def _claude_code_turns(task: str) -> tuple[list[JsonValue], list[JsonValue], list[JsonValue]]:
    """Turns 1 to 3 of a Claude Code session on /v1/messages: 1, 3 and 5 messages"""
    first: Final[list[JsonValue]] = [{"role": "user", "content": task}]
    second: Final[list[JsonValue]] = [
        *first,
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "calc.py test_calc.py"}],
        },
    ]
    third: Final[list[JsonValue]] = [
        *second,
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {"file_path": "calc.py"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": [{"type": "text", "text": "def add(a, b): return a - b"}],
                }
            ],
        },
    ]
    return first, second, third


def _agent_turns(task: str) -> tuple[list[JsonValue], list[JsonValue], list[JsonValue]]:
    """Turns 1 to 3 of an OpenAI tool loop on /v1/chat/completions: 2, 4 and 6 messages"""

    def call(call_id: str, path: str) -> list[JsonValue]:
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "write_file", "arguments": json.dumps({"path": path})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": f"wrote {path}"},
        ]

    first: Final[list[JsonValue]] = [
        {"role": "system", "content": "You are a coding agent"},
        {"role": "user", "content": task},
    ]
    second: Final[list[JsonValue]] = [*first, *call("call_1", "a.yaml")]
    third: Final[list[JsonValue]] = [*second, *call("call_2", "b.yaml")]
    return first, second, third


def _responses_turns(task: str) -> tuple[list[JsonValue], list[JsonValue], list[JsonValue]]:
    """Turns 1 to 3 of an agent on /v1/responses: 1, 3 and 5 input items"""

    def call(call_id: str, path: str) -> list[JsonValue]:
        return [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "write_file",
                "arguments": json.dumps({"path": path}),
            },
            {"type": "function_call_output", "call_id": call_id, "output": f"wrote {path}"},
        ]

    first: Final[list[JsonValue]] = [{"role": "user", "content": task}]
    second: Final[list[JsonValue]] = [*first, *call("call_1", "a.yaml")]
    third: Final[list[JsonValue]] = [*second, *call("call_2", "b.yaml")]
    return first, second, third


def _assert_turns_are_cached_up_to_four_messages(
    proxy: _Proxy, route: str, turns: Callable[[str], tuple[list[JsonValue], list[JsonValue], list[JsonValue]]]
) -> None:
    first, second, third = turns(f"update the config {uuid.uuid4().hex}")
    stored_before: Final = proxy.peer.stored()

    short_answers: Final = (proxy.miss(route, first), proxy.miss(route, second))
    repeated: Final = (proxy.hit(route, first), proxy.hit(route, second))
    long_answers: Final = (proxy.miss(route, third), proxy.miss(route, third))
    proxy.hit(route, first)

    assert repeated == short_answers, f"a repeated short turn got another turn's answer: {short_answers} {repeated}"
    assert long_answers[0] != long_answers[1], "a turn past max_messages was served from the cache"
    assert all("b.yaml" not in text and "def add" not in text for text in proxy.peer.embedded), (
        "a turn past max_messages was embedded"
    )
    assert proxy.peer.stored() == stored_before + 2, "a turn past max_messages was written to the cache"


@pytest.mark.parametrize(
    ("route", "turns"),
    [
        pytest.param("/v1/messages", _claude_code_turns, id="messages"),
        pytest.param("/v1/chat/completions", _agent_turns, id="chat-completions"),
    ],
)
def test_agent_turns_are_cached_up_to_four_messages_and_bypass_the_cache_past_it(
    qdrant_proxy: _Proxy,
    route: str,
    turns: Callable[[str], tuple[list[JsonValue], list[JsonValue], list[JsonValue]]],
) -> None:
    _assert_turns_are_cached_up_to_four_messages(qdrant_proxy, route, turns)


def test_tool_result_text_tells_a_tool_turn_apart_from_the_turn_before_it(qdrant_proxy: _Proxy) -> None:
    task: Final = f"list the files {uuid.uuid4().hex}"
    first, second, _ = _claude_code_turns(task)

    qdrant_proxy.miss("/v1/messages", first)
    qdrant_proxy.miss("/v1/messages", second)

    assert task in qdrant_proxy.peer.embedded[-1], qdrant_proxy.peer.embedded[-1]
    assert "calc.py test_calc.py" in qdrant_proxy.peer.embedded[-1], qdrant_proxy.peer.embedded[-1]


@pytest.mark.parametrize(
    ("route", "turns"),
    [
        pytest.param("/v1/messages", _claude_code_turns, id="messages"),
        pytest.param("/v1/chat/completions", _agent_turns, id="chat-completions"),
        pytest.param("/v1/responses", _responses_turns, id="responses"),
    ],
)
def test_exact_cache_honours_max_messages_from_cache_params(
    exact_proxy: _Proxy,
    route: str,
    turns: Callable[[str], tuple[list[JsonValue], list[JsonValue], list[JsonValue]]],
) -> None:
    first, second, _ = turns(f"what is {uuid.uuid4().hex}")

    answer: Final = exact_proxy.miss(route, first)
    repeated: Final = exact_proxy.hit(route, first)
    long_answers: Final = (exact_proxy.miss(route, second), exact_proxy.miss(route, second))

    assert repeated == answer
    assert long_answers[0] != long_answers[1], "a turn past max_messages was served from a cache capped at 2"
