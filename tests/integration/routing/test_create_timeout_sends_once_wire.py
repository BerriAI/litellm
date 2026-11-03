import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.openai_wire import openai_error
from integration._support.wire import Reply, Request, Wire, wire_server

CALLER_TIMEOUT_SECONDS: Final = 0.5
BATCH_INPUT: Final = json.dumps(
    {
        "custom_id": "1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    }
).encode()


Failure = Callable[[threading.Event], Callable[[Request], Reply]]


def _no_answer_in_time(released: threading.Event) -> Callable[[Request], Reply]:
    def respond(_request: Request) -> Reply:
        released.wait(timeout=30)
        return Reply(status=500)

    return respond


def _gateway_timeout(_released: threading.Event) -> Callable[[Request], Reply]:
    return lambda _request: openai_error(504)


def _connection_dropped_after_the_send(_released: threading.Event) -> Callable[[Request], Reply]:
    return lambda _request: Reply(drop_connection=True)


FAILURES_AFTER_THE_SEND: Final = (
    pytest.param(_no_answer_in_time, id="no-answer-in-time"),
    pytest.param(_gateway_timeout, id="504"),
    pytest.param(_connection_dropped_after_the_send, id="connection-dropped-after-the-send"),
)


@contextmanager
def _failing_upstream(failure: Failure) -> Iterator[Wire]:
    released: Final = threading.Event()
    with wire_server(failure(released)) as wire:
        try:
            yield wire
        finally:
            released.set()


def _posts_to(wire: Wire, path: str) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST" and request.target == path)


def _assert_sent_once_with_the_notice(response: httpx.Response, posts: tuple[Request, ...]) -> None:
    assert len(posts) == 1, [post.headers.get("x-stainless-retry-count") for post in posts]
    assert response.status_code >= 400, response.text
    assert response.headers.get("x-should-retry") == "false", response.headers
    assert "may already have created" in response.text, response.text


@pytest.mark.parametrize("failure", FAILURES_AFTER_THE_SEND)
def test_a_file_upload_that_fails_after_the_send_is_sent_upstream_once(gateway: Gateway, failure: Failure) -> None:
    with gateway.scenario() as scenario, _failing_upstream(failure) as wire:
        model: Final = scenario.model(api_base=wire.url + "/v1")
        response: Final = gateway.client.post(
            "/v1/files",
            headers={"Authorization": f"Bearer {gateway.key}", "x-litellm-timeout": str(CALLER_TIMEOUT_SECONDS)},
            data={"purpose": "batch", "model": model},
            files={"file": ("input.jsonl", BATCH_INPUT, "application/jsonl")},
        )
        _assert_sent_once_with_the_notice(response, _posts_to(wire, "/v1/files"))


@pytest.mark.parametrize("failure", FAILURES_AFTER_THE_SEND)
def test_a_batch_create_that_fails_after_the_send_is_sent_upstream_once(gateway: Gateway, failure: Failure) -> None:
    with gateway.scenario() as scenario, _failing_upstream(failure) as wire:
        model: Final = scenario.model(api_base=wire.url + "/v1")
        response: Final = gateway.client.post(
            "/v1/batches",
            headers={"Authorization": f"Bearer {gateway.key}"},
            json={
                "model": model,
                "timeout": CALLER_TIMEOUT_SECONDS,
                "input_file_id": "file-abc",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        _assert_sent_once_with_the_notice(response, _posts_to(wire, "/v1/batches"))
