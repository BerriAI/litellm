from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_CHAT_PATH: Final = "/v1/chat/completions"


@dataclass(frozen=True, slots=True)
class Upstream:
    model: str
    text: str
    wire: Wire


def _answering(text: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        streamed: Final = json.loads(request.body).get("stream") is True
        return chat_reply(f"chatcmpl-{text}", "gpt-4o-mini", text, stream=streamed)

    return answering_model_discovery(respond)


def _chat_posts(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        json.loads(request.body)
        for request in wire.drain()
        if request.method == "POST" and request.target.endswith("/chat/completions")
    )


def _body(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"messages": [{"role": "user", "content": "comma separated models"}], **fields}


def _post(gateway: Gateway, fields: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", _CHAT_PATH, _body(fields))


def _streamed_text(gateway: Gateway, fields: dict[str, JsonValue]) -> str:
    with gateway.client.stream(
        "POST", _CHAT_PATH, json=_body(fields), headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        assert response.status_code == 200, response.read().decode()
        assert response.headers["content-type"].startswith("text/event-stream"), response.headers
        frames: Final = tuple(line.removeprefix("data: ") for line in response.iter_lines() if line.startswith("data:"))
    assert frames[-1] == "[DONE]", frames
    chunks: Final = tuple(json.loads(frame) for frame in frames[:-1])
    choices: Final = tuple(chain.from_iterable(chunk["choices"] for chunk in chunks))
    return "".join(choice["delta"].get("content") or "" for choice in choices)


@pytest.fixture
def upstreams(scenario: Scenario) -> Iterator[tuple[Upstream, Upstream]]:
    with ExitStack() as stack:

        def upstream(label: str) -> Upstream:
            text: Final = f"{label}-{uuid.uuid4().hex}"
            wire: Final = stack.enter_context(wire_server(_answering(text)))
            return Upstream(scenario.model(api_base=f"{wire.url}/v1"), text, wire)

        yield upstream("first"), upstream("second")


def test_comma_separated_models_with_stream_true_are_rejected_as_an_invalid_stream_request(
    gateway: Gateway, upstreams: tuple[Upstream, Upstream]
) -> None:
    first, second = upstreams

    response: Final = _post(gateway, {"model": f"{first.model},{second.model}", "stream": True})

    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert error["type"] == "invalid_request_error", error
    assert error["param"] == "stream", error
    assert "comma" in error["message"].lower() and "stream" in error["message"].lower(), error


def test_comma_separated_models_with_stream_true_never_reach_any_listed_upstream(
    gateway: Gateway, upstreams: tuple[Upstream, Upstream]
) -> None:
    first, second = upstreams

    _post(gateway, {"model": f"{first.model},{second.model}", "stream": True})
    controls: Final = tuple(_post(gateway, {"model": upstream.model}) for upstream in upstreams)

    assert [control.status_code for control in controls] == [200, 200], [control.text for control in controls]
    assert [control.json()["choices"][0]["message"]["content"] for control in controls] == [first.text, second.text]
    observed: Final = tuple(_chat_posts(upstream.wire) for upstream in upstreams)
    assert [len(posts) for posts in observed] == [1, 1], observed
    assert [posts[0].get("stream") is True for posts in observed] == [False, False], observed


@pytest.mark.parametrize("stream", ({}, {"stream": False}), ids=("stream_absent", "stream_false"))
def test_comma_separated_models_without_streaming_answer_one_response_per_model(
    gateway: Gateway, upstreams: tuple[Upstream, Upstream], stream: dict[str, JsonValue]
) -> None:
    first, second = upstreams

    response: Final = _post(gateway, {"model": f"{first.model},{second.model}", **stream})

    assert response.status_code == 200, response.text
    answers: Final = response.json()
    assert isinstance(answers, list), answers
    assert [answer["choices"][0]["message"]["content"] for answer in answers] == [first.text, second.text], answers
    observed: Final = tuple(_chat_posts(upstream.wire) for upstream in upstreams)
    assert [len(posts) for posts in observed] == [1, 1], observed
    assert [posts[0].get("stream") is True for posts in observed] == [False, False], observed


def test_comma_separated_models_with_stream_and_fastest_response_stream_the_fastest_answer(
    gateway: Gateway, upstreams: tuple[Upstream, Upstream]
) -> None:
    first, second = upstreams

    text: Final = _streamed_text(
        gateway, {"model": f"{first.model},{second.model}", "stream": True, "fastest_response": True}
    )

    assert text in (first.text, second.text), text
    served: Final = next(upstream for upstream in upstreams if upstream.text == text)
    posts: Final = _chat_posts(served.wire)
    assert len(posts) == 1, posts
    assert posts[0]["stream"] is True, posts


def test_single_model_with_stream_true_still_streams(gateway: Gateway, upstreams: tuple[Upstream, Upstream]) -> None:
    first, _second = upstreams

    text: Final = _streamed_text(gateway, {"model": first.model, "stream": True})

    assert text == first.text, text
    posts: Final = _chat_posts(first.wire)
    assert len(posts) == 1, posts
    assert posts[0]["stream"] is True, posts
