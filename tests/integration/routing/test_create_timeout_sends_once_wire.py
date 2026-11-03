import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

import httpx
from integration._support.client import Gateway
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


def _answer_after(released: threading.Event) -> Callable[[Request], Reply]:
    def respond(_request: Request) -> Reply:
        released.wait(timeout=30)
        return Reply(status=500)

    return respond


@contextmanager
def _upstream_that_never_answers_in_time() -> Iterator[Wire]:
    released: Final = threading.Event()
    with wire_server(_answer_after(released)) as wire:
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


def test_a_file_upload_that_times_out_is_sent_upstream_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, _upstream_that_never_answers_in_time() as wire:
        model: Final = scenario.model(api_base=wire.url + "/v1")
        response: Final = gateway.client.post(
            "/v1/files",
            headers={"Authorization": f"Bearer {gateway.key}", "x-litellm-timeout": str(CALLER_TIMEOUT_SECONDS)},
            data={"purpose": "batch", "model": model},
            files={"file": ("input.jsonl", BATCH_INPUT, "application/jsonl")},
        )
        _assert_sent_once_with_the_notice(response, _posts_to(wire, "/v1/files"))


def test_a_batch_create_that_times_out_is_sent_upstream_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, _upstream_that_never_answers_in_time() as wire:
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
