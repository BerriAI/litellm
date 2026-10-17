import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, list_value, object_value
from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

KEY: Final = "synthetic-trailing-slash-key"
VECTOR: Final = (0.25, 0.5)
TRAILING: Final = ("", "/")
TRAILING_IDS: Final = ("no-slash", "slash")

_MARKER: Final = re.compile(rb"slash-[0-9a-f]{32}")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CREATED: Final = 1700000000


@dataclass(frozen=True, slots=True)
class Row:
    model: str
    base: str
    route: str
    keyed: bool = True
    token: bool = False
    drops_params: bool = False

    @property
    def target(self) -> str:
        return f"{self.base}{self.route}"


def new_marker() -> str:
    return f"slash-{uuid.uuid4().hex}"


def reply_text(marker: str) -> str:
    return f"reply {marker}"


def result_url(marker: str) -> str:
    return f"https://result.invalid/{marker}"


def file_id(upload: Request) -> str:
    return f"file-{hashlib.sha256(upload.body).hexdigest()[:24]}"


def marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body)
    return found.group().decode() if found else "unmarked"


def request_json(request: Request) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def _json(payload: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(payload).encode())


def _wants_stream(request: Request) -> bool:
    return request_json(request).get("stream") is True


def _sse(events: tuple[Mapping[str, JsonValue], ...]) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(*(f"data: {json.dumps(event)}\n\n".encode() for event in events), b"data: [DONE]\n\n"),
    )


def _chunk(marker: str, delta: Mapping[str, JsonValue], finish: str | None) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion.chunk",
        "created": _CREATED,
        "model": "scripted",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
    }


def _openai_chat(request: Request) -> Reply:
    marker: Final = marker_of(request)
    if _wants_stream(request):
        return _sse(
            (
                _chunk(marker, {"role": "assistant", "content": "reply "}, None),
                _chunk(marker, {"content": marker}, None),
                {
                    **_chunk(marker, {}, "stop"),
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )
        )
    return _json(
        {
            "id": f"chatcmpl-{marker}",
            "object": "chat.completion",
            "created": _CREATED,
            "model": "scripted",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": reply_text(marker)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )


def _anthropic_message(request: Request) -> Reply:
    marker: Final = marker_of(request)
    return _json(
        {
            "id": f"msg_{marker}",
            "type": "message",
            "role": "assistant",
            "model": "scripted",
            "content": [{"type": "text", "text": reply_text(marker)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    )


def _deepseek(request: Request) -> Reply:
    if request.target.endswith("/v1/messages"):
        return _anthropic_message(request)
    return _openai_chat(request)


def _ollama_line(content: str, done: bool) -> bytes:
    final: Final = {"done_reason": "stop", "prompt_eval_count": 5, "eval_count": 2} if done else {}
    line: Final = {
        "model": "llama3.1",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": done,
        **final,
    }
    return json.dumps(line).encode() + b"\n"


def _ollama_chat(request: Request) -> Reply:
    marker: Final = marker_of(request)
    if _wants_stream(request):
        return Reply(
            content_type="application/x-ndjson",
            chunks=(_ollama_line("reply ", False), _ollama_line(marker, False), _ollama_line("", True)),
        )
    return _json(
        {
            "model": "llama3.1",
            "message": {"role": "assistant", "content": reply_text(marker)},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 2,
        }
    )


def _ollama(request: Request) -> Reply:
    if request.target.endswith("/api/chat"):
        return _ollama_chat(request)
    if request.target.endswith("/api/generate"):
        return _json(
            {
                "model": "llama3.1",
                "response": reply_text(marker_of(request)),
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 2,
            }
        )
    return _json({"embeddings": [list(VECTOR)], "prompt_eval_count": 3})


def _embedding_list(model: str) -> Reply:
    return _json(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": list(VECTOR)}],
            "model": model,
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
    )


def _voyage(request: Request) -> Reply:
    if request.target.endswith("/contextualizedembeddings"):
        return _json(
            {
                "object": "list",
                "data": [
                    {
                        "object": "list",
                        "index": 0,
                        "data": [{"object": "embedding", "index": 0, "embedding": list(VECTOR)}],
                    }
                ],
                "model": "voyage-context-3",
                "usage": {"total_tokens": 4},
            }
        )
    return _embedding_list("voyage-multimodal-3" if request.target.endswith("/multimodalembeddings") else "voyage-3")


def _gigachat(request: Request) -> Reply:
    if request.target.endswith("/files"):
        return _json({"id": file_id(request), "object": "file", "purpose": "general"})
    if request.target.endswith("/models"):
        return _json({"object": "list", "data": [{"id": marker_of(request), "object": "model"}]})
    if request.target.endswith("/embeddings"):
        return _json(
            {
                "object": "list",
                "data": [{"object": "embedding", "embedding": list(VECTOR), "index": 0, "usage": {"prompt_tokens": 3}}],
                "model": "Embeddings",
            }
        )
    return _openai_chat(request)


def _search(results: Callable[[str], Mapping[str, JsonValue]]) -> Callable[[Request], Reply]:
    return lambda request: _json(results(result_url(marker_of(request))))


_PERPLEXITY_SEARCH: Final = _search(lambda url: {"results": [{"title": "T", "url": url, "snippet": "S"}]})


def _perplexity(request: Request) -> Reply:
    if request.target.endswith("/search"):
        return _PERPLEXITY_SEARCH(request)
    return _embedding_list("pplx-embed-v1-0.6b")


def _topaz(request: Request) -> Reply:
    return Reply(body=f"enhanced {marker_of(request)}".encode(), content_type="image/jpeg")


_BY_PREFIX: Final[Mapping[str, Callable[[Request], Reply]]] = MappingProxyType(
    {
        "ds": _deepseek,
        "hk": _openai_chat,
        "gr": _openai_chat,
        "gragent": _openai_chat,
        "ms": _openai_chat,
        "dc": _openai_chat,
        "qc": _openai_chat,
        "qp": _openai_chat,
        "mo": _openai_chat,
        "gc": _gigachat,
        "gcenv": _gigachat,
        "ol": _ollama,
        "jn": lambda _: _embedding_list("jina-embeddings-v3"),
        "vy": _voyage,
        "px": _perplexity,
        "pxenv": _PERPLEXITY_SEARCH,
        "ve": lambda _: _embedding_list("doubao-embedding"),
        "tv": _search(lambda url: {"results": [{"title": "T", "url": url, "content": "S"}]}),
        "ex": _search(lambda url: {"results": [{"title": "T", "url": url, "text": "S"}]}),
        "fc": _search(lambda url: {"data": {"web": [{"title": "T", "url": url, "description": "S"}]}}),
        "lk": _search(lambda url: {"results": [{"type": "text", "name": "T", "url": url, "content": "S"}]}),
        "cr": _search(lambda url: {"success": True, "data": [{"title": "T", "url": url, "description": "S"}]}),
        "tz": _topaz,
    }
)


def provider_peer(request: Request) -> Reply:
    return _BY_PREFIX[request.target.split("/")[1]](request)


def assert_one_request(received: tuple[Request, ...], target: str, marker: str, keyed: bool = True) -> Request:
    assert [(request.method, request.target) for request in received] == [("POST", target)], [
        (request.method, request.target[:200]) for request in received
    ]
    (request,) = received
    assert marker.encode() in request.body, request.body[:500]
    assert not keyed or any(KEY in value for value in request.headers.values()), sorted(request.headers)
    return request


def of_marker(received: tuple[Request, ...], marker: str) -> tuple[Request, ...]:
    return tuple(request for request in received if marker_of(request) == marker)


def body_of(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def first_item(value: JsonValue) -> dict[str, JsonValue]:
    return object_value(list_value(value)[0])


def deployment(scenario: Scenario, row: Row, api_base: str | None) -> str:
    if row.token:
        return scenario.model(model=row.model, api_base=api_base, api_key=KEY, gigachat_access_token=KEY)
    if row.drops_params:
        return scenario.model(model=row.model, api_base=api_base, api_key=KEY, drop_params=True)
    return scenario.model(model=row.model, api_base=api_base, api_key=KEY)


def assert_control_answers(gateway: Gateway, scenario: Scenario) -> None:
    assert first_item(gateway.chat(scenario.model())["choices"])["finish_reason"] == "stop"
