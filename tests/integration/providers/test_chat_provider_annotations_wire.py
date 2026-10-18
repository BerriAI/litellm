import json
import uuid
from collections.abc import Callable, Iterator
from itertools import chain
from typing import Final

import httpx
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.caching.response_cache_case import await_entry, redis_store
from pydantic import JsonValue, TypeAdapter
from redis import Redis

_BACKEND: Final = "z-ai/glm-5.2"
_API_KEY: Final = "synthetic-openrouter-key"
_ANSWER: Final = "The report covers revenue."
_URL_CITATION: Final[dict[str, JsonValue]] = {
    "type": "url_citation",
    "url_citation": {
        "url": "https://example.com/report",
        "title": "Report",
        "start_index": 0,
        "end_index": 10,
        "content": "cited passage",
    },
}
_FILE_ANNOTATION: Final[dict[str, JsonValue]] = {
    "type": "file",
    "file": {"hash": "sha-report", "name": "report.pdf", "content": [{"type": "text", "text": "parsed page one"}]},
}
_USAGE: Final[dict[str, JsonValue]] = {"prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CHOICES: Final = TypeAdapter(list[dict[str, JsonValue]])


def _completion(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": _ANSWER,
                        "annotations": [_URL_CITATION, _FILE_ANNOTATION],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": _USAGE,
        }
    ).encode()


def _frame(frame: dict[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def _stream(identity: str) -> Reply:
    chunk: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _BACKEND,
    }
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _frame({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": _ANSWER}}]}),
            _frame({**chunk, "choices": [{"index": 0, "delta": {"annotations": [_URL_CITATION, _FILE_ANNOTATION]}}]}),
            _frame({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": _USAGE}),
            b"data: [DONE]\n\n",
        ),
    )


def _responder(identity: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return _stream(identity)
        return Reply(body=_completion(identity))

    return respond


def _chat_body(model: str, marker: str, stream: bool) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": f"Summarize the report {marker}"}],
        "stream": stream,
    }


def _posted(wire: Wire) -> int:
    return sum(1 for request in wire.drain() if request.method == "POST")


def _message(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    choices: Final = _CHOICES.validate_python(_JSON_OBJECT.validate_json(response.content)["choices"])
    return _JSON_OBJECT.validate_python(choices[0]["message"])


def _assert_split(message: dict[str, JsonValue]) -> None:
    assert message["content"] == _ANSWER, message
    assert message["annotations"] == [_URL_CITATION], message
    assert _JSON_OBJECT.validate_python(message["provider_specific_fields"])["annotations"] == [_FILE_ANNOTATION]


def test_file_annotation_reaches_the_caller_in_provider_specific_fields_and_survives_a_cache_hit(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    identity: Final = f"gen-{marker}"
    with wire_server(_responder(identity)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openrouter/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        first: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, stream=False))
        _assert_split(_message(first))
        assert _posted(wire) == 1
        await_entry(redis_store(), identity)
        cached: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, stream=False))
        assert cached.headers.get("x-litellm-cache-key"), dict(cached.headers)
        _assert_split(_message(cached))
        assert _posted(wire) == 0


def _deltas(response: httpx.Response) -> Iterator[dict[str, JsonValue]]:
    for line in response.iter_lines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        choices = _CHOICES.validate_python(_JSON_OBJECT.validate_json(line.removeprefix("data: "))["choices"])
        yield from (_JSON_OBJECT.validate_python(choice["delta"]) for choice in choices)


def _streamed(gateway: Gateway, body: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    with gateway.client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        assert response.status_code == 200, response.read()
        return tuple(_deltas(response))


def _listed(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _provider_fields(delta: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_python(delta.get("provider_specific_fields") or {})


def _annotations(deltas: tuple[dict[str, JsonValue], ...]) -> list[JsonValue]:
    return list(chain.from_iterable(_listed(delta.get("annotations")) for delta in deltas))


def _provider_annotations(deltas: tuple[dict[str, JsonValue], ...]) -> list[JsonValue]:
    return list(chain.from_iterable(_listed(_provider_fields(delta).get("annotations")) for delta in deltas))


def _text(deltas: tuple[dict[str, JsonValue], ...]) -> str:
    return "".join(str(delta.get("content") or "") for delta in deltas)


def _cached_message(store: Redis, key: str) -> dict[str, JsonValue]:
    raw: Final = store.get(key)
    assert isinstance(raw, bytes), key
    response: Final = _JSON_OBJECT.validate_json(raw)["response"]
    stored: Final = _JSON_OBJECT.validate_json(response) if isinstance(response, str) else response
    choices: Final = _CHOICES.validate_python(_JSON_OBJECT.validate_python(stored)["choices"])
    return _JSON_OBJECT.validate_python(choices[0]["message"])


def test_streamed_annotations_reach_the_caller_as_sent_and_the_cached_stream_splits_them(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    identity: Final = f"gen-{marker}"
    with wire_server(_responder(identity)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openrouter/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        live: Final = _streamed(gateway, _chat_body(model, marker, stream=True))
        assert _text(live) == _ANSWER, live
        assert _annotations(live) == [_URL_CITATION, _FILE_ANNOTATION], live
        assert _posted(wire) == 1
        store: Final = redis_store()
        _assert_split(_cached_message(store, await_entry(store, identity)))
        replayed: Final = _streamed(gateway, _chat_body(model, marker, stream=True))
        assert _text(replayed) == _ANSWER, replayed
        assert _provider_annotations(replayed) == [_FILE_ANNOTATION], replayed
        assert _posted(wire) == 0
