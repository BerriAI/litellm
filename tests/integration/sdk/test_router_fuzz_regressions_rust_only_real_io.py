"""
Router behavior the router-code differential fuzz run found changed once the Rust core became the
only core: count settings below zero, an allowed_fails_policy write that is no policy, a model
response object as a deployment's mock_response, and endpoints read or patched on the Router class.
"""

import asyncio
import json
import socket
import threading
from collections.abc import Iterator, Mapping
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest

import litellm
from litellm import Router

MESSAGES: Final = [{"role": "user", "content": "hi"}]


class _Upstream(BaseHTTPRequestHandler):
    """An OpenAI upstream whose path's first segment names how it answers: ``cut`` sends two
    Responses stream events then cuts the chunked body, ``filtered`` answers a chat completion the
    provider blocked, ``down`` answers 502, and anything else a plain chat completion."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002  # the base class's name
        del format, args

    def _json(self, status: int, body: Mapping[str, object]) -> None:
        data: Final = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        _ = self.wfile.write(data)

    def _chunk(self, event: Mapping[str, object]) -> None:
        data: Final = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        _ = self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _cut(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        response: Final = {"id": "r1", "object": "response", "created_at": 1, "status": "in_progress", "output": []}
        self._chunk({"type": "response.created", "sequence_number": 0, "response": {**response, "model": "gpt-4o"}})
        self._chunk(
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "m1",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello ",
            }
        )
        self.connection.shutdown(socket.SHUT_RDWR)
        self.close_connection = True

    def do_POST(self) -> None:
        _ = self.rfile.read(int(self.headers.get("content-length") or 0))
        mode: Final = self.path.split("/")[1]
        if mode == "cut":
            self._cut()
            return
        if mode == "down":
            self._json(502, {"error": {"message": "provider error", "type": "api_error"}})
            return
        finish: Final = "content_filter" if mode == "filtered" else "stop"
        message: Final = {"role": "assistant", "content": mode}
        choice: Final = {"index": 0, "finish_reason": finish, "message": message}
        usage: Final = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self._json(200, {"id": "c1", "object": "chat.completion", "created": 1, "choices": [choice], "usage": usage})


@pytest.fixture
def upstream() -> Iterator[str]:
    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join()


def _served(name: str, deployment_id: str, model: str, api_base: str) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_key": "k", "api_base": api_base},
        "model_info": {"id": deployment_id},
    }


def _request_logging(call_type: str, *, stream: bool) -> object:
    from litellm.litellm_core_utils.litellm_logging import Logging

    return Logging(
        model="g",
        messages=MESSAGES,
        stream=stream,
        call_type=call_type,
        start_time=datetime.now(),  # noqa: DTZ005  # litellm's own naive start time
        litellm_call_id="call-1",
        function_id="f",
    )


async def _recorded() -> None:
    """Let the background recordings of a stream's outcome run."""
    for _ in range(50):
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_a_responses_stream_cut_after_it_opened_counts_one_fail_with_a_shared_logging_object(
    upstream: str,
) -> None:
    router: Final = Router(
        model_list=[_served("g", "d1", "openai/gpt-4o", f"{upstream}/cut/v1")], num_retries=0, disable_cooldowns=True
    )

    stream: Final = await router.aresponses(
        model="g", input="hi", stream=True, litellm_logging_obj=_request_logging("aresponses", stream=True)
    )
    with pytest.raises(Exception):  # noqa: B017, PT011  # the transport's read error, whatever class maps it
        async for _ in stream:  # pyright: ignore[reportGeneralTypeIssues]  # a stream when stream=True
            pass
    await _recorded()

    assert router.health().store.get_cache(key="d1:fails") == 1


@pytest.mark.asyncio
async def test_a_blocked_response_raised_for_a_fallback_counts_no_success_with_a_shared_logging_object(
    upstream: str,
) -> None:
    router: Final = Router(
        model_list=[
            _served("a", "a1", "openai/gpt-4o", f"{upstream}/filtered/v1"),
            _served("safe", "s1", "openai/gpt-4o", f"{upstream}/ok/v1"),
        ],
        content_policy_fallbacks=[{"a": ["safe"]}],
        num_retries=0,
        disable_cooldowns=True,
    )

    response: Final = await router.acompletion(
        model="a", messages=MESSAGES, litellm_logging_obj=_request_logging("acompletion", stream=False)
    )
    await _recorded()

    assert response.choices[0].message.content == "ok"
    assert router.health().store.get_cache(key="a1:successes") is None
    assert router.health().store.get_cache(key="s1:successes") == 1


@pytest.mark.asyncio
async def test_a_text_completion_stream_that_fails_before_its_first_chunk_counts_nothing(upstream: str) -> None:
    router: Final = Router(
        model_list=[_served("g", "d1", "openai/gpt-3.5-turbo-instruct", f"{upstream}/down/v1")],
        num_retries=0,
        disable_cooldowns=True,
    )

    stream: Final = await router.atext_completion(model="g", prompt="hi", stream=True)
    with pytest.raises(litellm.BadGatewayError):
        async for _ in stream:  # pyright: ignore[reportGeneralTypeIssues]  # a stream when stream=True
            pass
    await _recorded()

    assert router.health().store.get_cache(key="d1:successes") is None
    assert router.health().store.get_cache(key="d1:fails") is None


@pytest.mark.asyncio
async def test_a_request_refused_before_the_provider_counts_against_no_deployment(upstream: str) -> None:
    router: Final = Router(
        model_list=[_served("g", "d1", "openai/gpt-4o", f"{upstream}/ok/v1")], num_retries=0, disable_cooldowns=True
    )

    with pytest.raises(litellm.UnsupportedParamsError):
        await router.acompletion(model="g", messages=MESSAGES, reasoning_effort="low")
    await _recorded()

    assert router.health().store.get_cache(key="d1:fails") is None
    assert not [key for key in router.health().store.in_memory_cache.cache_dict if ":rpm" in key]
