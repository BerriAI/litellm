import asyncio
import hashlib
import json
import math
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_VECTOR_SIZE: Final = 16
_CHAT_MODEL: Final = "semantic-chat"
_CLAUDE_MODEL: Final = "semantic-claude"
_EMBEDDING_MODEL: Final = "semantic-embedder"
_COLLECTION: Final = "semantic-tool-turns"


def _vector(text: str) -> tuple[float, ...]:
    digest: Final = hashlib.sha256(text.encode()).digest()
    return tuple((byte - 127.5) / 127.5 for byte in digest[:_VECTOR_SIZE])


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot: Final = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


def _answer(body: Mapping[str, JsonValue]) -> str:
    return "answer-" + hashlib.sha256(json.dumps(body["messages"], sort_keys=True).encode()).hexdigest()[:16]


@dataclass(slots=True)
class _Peer:
    """Embeddings, chat, Anthropic Messages and a Qdrant collection on one owned socket"""

    lock: threading.Lock = field(default_factory=threading.Lock)
    points: list[Mapping[str, JsonValue]] = field(default_factory=list)  # mutable-ok: the Qdrant collection
    embedded: list[str] = field(default_factory=list)  # mutable-ok: every prompt the proxy embedded
    answered: list[str] = field(default_factory=list)  # mutable-ok: every completion the provider served
    qdrant_down: threading.Event = field(default_factory=threading.Event)
    refused_writes: list[str] = field(default_factory=list)  # mutable-ok: upserts refused while Qdrant is down

    def stored(self) -> int:
        with self.lock:
            return len(self.points)

    def respond(self, request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        body: Final = _JSON_OBJECT.validate_json(request.body) if request.body else {}
        collection: Final = f"/qdrant/collections/{_COLLECTION}"
        if path.startswith("/qdrant/") and self.qdrant_down.is_set():
            if path == f"{collection}/points":
                with self.lock:
                    self.refused_writes.append(path)
            return Reply(status=503, body=b'{"status":{"error":"qdrant unavailable"}}')
        if request.method == "GET" and path == f"{collection}/exists":
            return self._json({"result": {"exists": False}, "status": "ok"})
        if request.method in {"GET", "PUT"} and path in {collection, f"{collection}/index"}:
            return self._json({"result": True, "status": "ok"})
        if request.method == "PUT" and path == f"{collection}/points":
            with self.lock:
                self.points.extend(_JSON_OBJECT.validate_python(point) for point in body["points"])
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
            return self._json(self._chat_reply(_answer(body)))
        if request.method == "POST" and path == "/v1/messages":
            return self._json(self._messages_reply(_answer(body)))
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
class _SemanticProxy:
    gateway: Gateway
    peer: _Peer


@pytest.fixture(scope="module")
def semantic_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_SemanticProxy]:
    peer: Final = _Peer()
    directory: Final = tmp_path_factory.mktemp("semantic-tool-turns")
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
            "type": "qdrant-semantic",
            "qdrant_api_base": f"{wire.url}/qdrant",
            "qdrant_collection_name": _COLLECTION,
            "qdrant_semantic_cache_embedding_model": _EMBEDDING_MODEL,
            "qdrant_semantic_cache_vector_size": _VECTOR_SIZE,
            "qdrant_quantization_config": "binary",
            "similarity_threshold": 0.99,
        }
        path: Final = directory / "semantic_tool_turns.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, directory, {}, config=path) as candidate:
            yield _SemanticProxy(candidate, peer)


def _send_turns(proxy: _SemanticProxy, route: str, model: str, turns: Sequence[list[JsonValue]]) -> list[str]:
    def reply_text(turn: list[JsonValue]) -> str:
        stored_before: Final = proxy.peer.stored()
        answered_before: Final = len(proxy.peer.answered)
        body: Final[dict[str, JsonValue]] = {"model": model, "max_tokens": 16, "messages": turn}
        response: Final = proxy.gateway.request("POST", route, body)
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        if len(proxy.peer.answered) > answered_before:
            eventually(proxy.peer.stored, lambda count: count > stored_before)
        if route == "/v1/messages":
            return str(_JSON_OBJECT.validate_python(_list(payload["content"])[0])["text"])
        choice: Final = _JSON_OBJECT.validate_python(_list(payload["choices"])[0])
        return str(_JSON_OBJECT.validate_python(choice["message"])["content"])

    return [reply_text(turn) for turn in turns]


def test_claude_code_tool_turns_on_messages_are_not_served_the_first_turn_answer(
    semantic_proxy: _SemanticProxy,
) -> None:
    task: Final[JsonValue] = {"role": "user", "content": f"fix the failing test {uuid.uuid4().hex}"}
    list_files: Final[list[JsonValue]] = [
        task,
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "calc.py test_calc.py"}],
        },
    ]
    read_file: Final[list[JsonValue]] = [
        *list_files,
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

    answers: Final = _send_turns(
        semantic_proxy, "/v1/messages", _CLAUDE_MODEL, ([task], list_files, read_file, list_files)
    )

    assert len(set(answers[:3])) == 3, f"a later tool turn replayed an earlier cached answer: {answers}"
    assert answers[:3] == semantic_proxy.peer.answered[-3:], semantic_proxy.peer.answered
    assert answers[3] == answers[1], f"a repeated tool turn missed the cache: {answers}"


def test_openai_agent_loop_tool_calls_on_chat_completions_are_not_served_a_cached_answer(
    semantic_proxy: _SemanticProxy,
) -> None:
    task: Final[JsonValue] = {"role": "user", "content": f"update both config files {uuid.uuid4().hex}"}

    def wrote(path: str) -> list[JsonValue]:
        return [
            task,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": json.dumps({"path": path})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]

    answers: Final = _send_turns(
        semantic_proxy, "/v1/chat/completions", _CHAT_MODEL, (wrote("a.yaml"), wrote("b.yaml"), wrote("a.yaml"))
    )

    assert len(set(answers[:2])) == 2, f"a different tool call replayed an earlier cached answer: {answers}"
    assert answers[:2] == semantic_proxy.peer.answered[-2:], semantic_proxy.peer.answered
    assert answers[2] == answers[0], f"a repeated tool call missed the cache: {answers}"


def _tool_turn(task: str, calls: Sequence[tuple[str, str]], results: Sequence[tuple[str, str]]) -> list[JsonValue]:
    return [
        {"role": "user", "content": task},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": arguments}}
                for call_id, arguments in calls
            ],
        },
        *({"role": "tool", "tool_call_id": call_id, "content": output} for call_id, output in results),
    ]


def _claude_turn(task: str, calls: Sequence[tuple[str, str]], results: Sequence[tuple[str, str]]) -> list[JsonValue]:
    return [
        {"role": "user", "content": task},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": "Read", "input": {"file_path": file_path}}
                for call_id, file_path in calls
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": output} for call_id, output in results
            ],
        },
    ]


_TWO_READS: Final = (("c1", '{"path":"a.py"}'), ("c2", '{"path":"b.py"}'))
_TWO_CLAUDE_READS: Final = (("toolu_1", "a.py"), ("toolu_2", "b.py"))


@pytest.mark.parametrize(
    ("route", "model", "build", "calls", "first", "second"),
    [
        pytest.param(
            "/v1/chat/completions",
            _CHAT_MODEL,
            _tool_turn,
            _TWO_READS,
            (("c1", "x = 1"), ("c2", "y = 2")),
            (("c2", "x = 1"), ("c1", "y = 2")),
            id="chat-parallel-results-answer-the-other-call",
        ),
        pytest.param(
            "/v1/messages",
            _CLAUDE_MODEL,
            _claude_turn,
            _TWO_CLAUDE_READS,
            (("toolu_1", "x = 1"), ("toolu_2", "y = 2")),
            (("toolu_2", "x = 1"), ("toolu_1", "y = 2")),
            id="messages-parallel-results-answer-the-other-call",
        ),
        pytest.param(
            "/v1/chat/completions",
            _CHAT_MODEL,
            _tool_turn,
            _TWO_READS[:1],
            (("c1", "x = 1"),),
            (("c9", "x = 1"),),
            id="chat-result-for-an-unknown-call",
        ),
        pytest.param(
            "/v1/chat/completions",
            _CHAT_MODEL,
            _tool_turn,
            _TWO_READS,
            (("c1", '"},{"result_of_call":2,"output":"y = 2'),),
            (("c1", ""), ("c2", "y = 2")),
            id="chat-output-shaped-like-a-second-result",
        ),
        pytest.param(
            "/v1/chat/completions",
            _CHAT_MODEL,
            _tool_turn,
            _TWO_READS[:1],
            (("c1", ""),),
            (("c1", "x" * 5000),),
            id="chat-empty-and-5kb-output",
        ),
    ],
)
def test_tool_turns_that_differ_only_in_their_results_do_not_share_a_cached_answer(
    semantic_proxy: _SemanticProxy,
    route: str,
    model: str,
    build: Callable[[str, Sequence[tuple[str, str]], Sequence[tuple[str, str]]], list[JsonValue]],
    calls: Sequence[tuple[str, str]],
    first: Sequence[tuple[str, str]],
    second: Sequence[tuple[str, str]],
) -> None:
    task: Final = f"read both files {uuid.uuid4().hex}"

    answers: Final = _send_turns(
        semantic_proxy, route, model, (build(task, calls, first), build(task, calls, second), build(task, calls, first))
    )

    assert answers[0] != answers[1], f"a different tool result replayed an earlier cached answer: {answers}"
    assert answers[:2] == semantic_proxy.peer.answered[-2:], semantic_proxy.peer.answered
    assert answers[2] == answers[0], f"a repeated tool turn missed the cache: {answers}"


def _openai_sdk(proxy: _SemanticProxy, turn: list[JsonValue]) -> str:
    sdk: Final = openai.OpenAI(
        base_url=str(proxy.gateway.client.base_url) + "/v1",
        api_key=proxy.gateway.key,
        http_client=httpx.Client(trust_env=False, timeout=15),
    )
    completion: Final = sdk.chat.completions.create(model=_CHAT_MODEL, messages=turn)  # pyright: ignore[reportArgumentType, reportCallIssue]  # JSON turns are validated by the proxy, not the SDK types
    return str(completion.choices[0].message.content)


def _async_openai_sdk(proxy: _SemanticProxy, turn: list[JsonValue]) -> str:
    async def create() -> str:
        async with httpx.AsyncClient(trust_env=False, timeout=15) as http_client:
            sdk: Final = openai.AsyncOpenAI(
                base_url=str(proxy.gateway.client.base_url) + "/v1", api_key=proxy.gateway.key, http_client=http_client
            )
            completion: Final = await sdk.chat.completions.create(model=_CHAT_MODEL, messages=turn)  # pyright: ignore[reportArgumentType, reportCallIssue]  # JSON turns are validated by the proxy, not the SDK types
            return str(completion.choices[0].message.content)

    return asyncio.run(create())


def _anthropic_text(message: anthropic.types.Message) -> str:
    block: Final = message.content[0]
    assert isinstance(block, anthropic.types.TextBlock), message
    return block.text


def _anthropic_sdk(proxy: _SemanticProxy, turn: list[JsonValue]) -> str:
    sdk: Final = anthropic.Anthropic(
        base_url=str(proxy.gateway.client.base_url),
        api_key=proxy.gateway.key,
        http_client=httpx.Client(trust_env=False, timeout=15),
    )
    return _anthropic_text(sdk.messages.create(model=_CLAUDE_MODEL, max_tokens=16, messages=turn))  # pyright: ignore[reportArgumentType]  # JSON turns are validated by the proxy, not the SDK types


def _async_anthropic_sdk(proxy: _SemanticProxy, turn: list[JsonValue]) -> str:
    async def create() -> str:
        async with httpx.AsyncClient(trust_env=False, timeout=15) as http_client:
            sdk: Final = anthropic.AsyncAnthropic(
                base_url=str(proxy.gateway.client.base_url), api_key=proxy.gateway.key, http_client=http_client
            )
            message: Final = await sdk.messages.create(model=_CLAUDE_MODEL, max_tokens=16, messages=turn)  # pyright: ignore[reportArgumentType]  # JSON turns are validated by the proxy, not the SDK types
            return _anthropic_text(message)

    return asyncio.run(create())


@pytest.mark.parametrize(
    ("send", "build"),
    [
        pytest.param(_openai_sdk, _tool_turn, id="openai-sdk"),
        pytest.param(_async_openai_sdk, _tool_turn, id="async-openai-sdk"),
        pytest.param(_anthropic_sdk, _claude_turn, id="anthropic-sdk"),
        pytest.param(_async_anthropic_sdk, _claude_turn, id="async-anthropic-sdk"),
    ],
)
def test_sdk_clients_get_a_fresh_answer_for_each_tool_turn_and_a_hit_for_a_repeat(
    semantic_proxy: _SemanticProxy,
    send: Callable[[_SemanticProxy, list[JsonValue]], str],
    build: Callable[[str, Sequence[tuple[str, str]], Sequence[tuple[str, str]]], list[JsonValue]],
) -> None:
    task: Final = f"read the file {uuid.uuid4().hex}"
    calls: Final = _TWO_READS[:1] if build is _tool_turn else _TWO_CLAUDE_READS[:1]
    call_id: Final = calls[0][0]
    alpha: Final = build(task, calls, ((call_id, "x = 1"),))
    beta: Final = build(task, calls, ((call_id, "PermissionError"),))

    def answer(turn: list[JsonValue]) -> str:
        stored_before: Final = semantic_proxy.peer.stored()
        answered_before: Final = len(semantic_proxy.peer.answered)
        text: Final = send(semantic_proxy, turn)
        if len(semantic_proxy.peer.answered) > answered_before:
            eventually(semantic_proxy.peer.stored, lambda count: count > stored_before)
        return text

    answers: Final = [answer(turn) for turn in (alpha, beta, alpha)]

    assert answers[0] != answers[1], f"a different tool result replayed an earlier cached answer: {answers}"
    assert answers[:2] == semantic_proxy.peer.answered[-2:], semantic_proxy.peer.answered
    assert answers[2] == answers[0], f"a repeated tool turn missed the cache: {answers}"


def test_concurrent_agent_loops_each_get_their_own_answer_and_their_own_hit(semantic_proxy: _SemanticProxy) -> None:
    task: Final = f"read the file {uuid.uuid4().hex}"
    turns: Final = tuple(_tool_turn(task, _TWO_READS[:1], (("c1", f"line {index}"),)) for index in range(8))

    def send(turn: list[JsonValue]) -> str:
        response: Final = semantic_proxy.gateway.request(
            "POST", "/v1/chat/completions", {"model": _CHAT_MODEL, "messages": turn}
        )
        assert response.status_code == 200, response.text
        choice: Final = _JSON_OBJECT.validate_python(_list(_JSON_OBJECT.validate_json(response.content)["choices"])[0])
        return str(_JSON_OBJECT.validate_python(choice["message"])["content"])

    stored_before: Final = semantic_proxy.peer.stored()
    with ThreadPoolExecutor(max_workers=len(turns)) as pool:
        first: Final = tuple(pool.map(send, turns))
    eventually(semantic_proxy.peer.stored, lambda count: count >= stored_before + len(turns))
    answered_before: Final = len(semantic_proxy.peer.answered)
    with ThreadPoolExecutor(max_workers=len(turns)) as pool:
        second: Final = tuple(pool.map(send, turns))

    assert len(set(first)) == len(turns), f"concurrent tool turns shared a cached answer: {first}"
    assert second == first, f"a repeated tool turn got another turn's answer: {first} {second}"
    assert len(semantic_proxy.peer.answered) == answered_before, "a repeated tool turn missed the cache"


def test_tool_turns_are_answered_while_qdrant_is_down_and_cached_once_it_recovers(
    semantic_proxy: _SemanticProxy,
) -> None:
    task: Final = f"read the file {uuid.uuid4().hex}"
    turn: Final = _tool_turn(task, _TWO_READS[:1], (("c1", "x = 1"),))
    body: Final[dict[str, JsonValue]] = {"model": _CHAT_MODEL, "messages": turn}

    def send() -> httpx.Response:
        return semantic_proxy.gateway.request("POST", "/v1/chat/completions", body)

    refused_before: Final = len(semantic_proxy.peer.refused_writes)
    answered_before: Final = len(semantic_proxy.peer.answered)
    semantic_proxy.peer.qdrant_down.set()
    try:
        outage: Final = (send(), send())
        eventually(lambda: len(semantic_proxy.peer.refused_writes), lambda count: count >= refused_before + 2)
    finally:
        semantic_proxy.peer.qdrant_down.clear()
    answered_during_outage: Final = len(semantic_proxy.peer.answered) - answered_before
    stored_before: Final = semantic_proxy.peer.stored()
    recovered: Final = send()
    eventually(semantic_proxy.peer.stored, lambda count: count > stored_before)
    answered_after_recovery: Final = len(semantic_proxy.peer.answered)
    repeat: Final = send()

    assert [response.status_code for response in outage] == [200, 200], [response.text for response in outage]
    assert answered_during_outage == 2, "an outage request was not sent to the provider"
    assert recovered.status_code == 200, recovered.text
    assert answered_after_recovery - answered_before == 3, "the first request after recovery hit a stale entry"
    assert repeat.status_code == 200, repeat.text
    assert repeat.json()["choices"] == recovered.json()["choices"]
    assert len(semantic_proxy.peer.answered) == answered_after_recovery, "the repeat after recovery missed the cache"
