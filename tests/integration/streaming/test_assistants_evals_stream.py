from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from openai import APITimeoutError, OpenAI
from pydantic import JsonValue

_QUEUED_RUN: Final[dict[str, JsonValue]] = {
    "id": "run_abc",
    "assistant_id": "asst_abc",
    "cancelled_at": None,
    "completed_at": None,
    "created_at": 1700000002,
    "expires_at": None,
    "failed_at": None,
    "incomplete_details": None,
    "instructions": "Be terse",
    "last_error": None,
    "max_completion_tokens": 50,
    "max_prompt_tokens": 500,
    "metadata": {"run": "r1"},
    "model": "gpt-4o",
    "object": "thread.run",
    "parallel_tool_calls": False,
    "required_action": None,
    "response_format": {"type": "json_object"},
    "started_at": None,
    "status": "queued",
    "thread_id": "thread_abc",
    "tool_choice": "auto",
    "tools": [{"type": "code_interpreter"}],
    "truncation_strategy": {"type": "last_messages", "last_messages": 4},
    "usage": None,
    "temperature": 0.3,
    "top_p": 0.9,
}
_MESSAGE_CREATED: Final[dict[str, JsonValue]] = {
    "id": "msg_abc",
    "assistant_id": "asst_abc",
    "attachments": None,
    "completed_at": None,
    "content": [],
    "created_at": 1700000003,
    "incomplete_at": None,
    "incomplete_details": None,
    "metadata": {},
    "object": "thread.message",
    "role": "assistant",
    "run_id": "run_abc",
    "status": "in_progress",
    "thread_id": "thread_abc",
}
_MESSAGE_DELTA: Final[dict[str, JsonValue]] = {
    "id": "msg_abc",
    "object": "thread.message.delta",
    "delta": {"content": [{"index": 0, "type": "text", "text": {"value": "Hi"}}]},
}
_COMPLETED_RUN: Final[dict[str, JsonValue]] = {
    **_QUEUED_RUN,
    "completed_at": 1700000004,
    "status": "completed",
    "usage": {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
}
_RUN_REQUEST: Final[dict[str, JsonValue]] = {
    "assistant_id": "asst_abc",
    "instructions": "Be terse",
    "additional_instructions": "Cite sources",
    "additional_messages": [{"role": "user", "content": "Also check page 2"}],
    "metadata": {"run": "r1"},
    "model": "gpt-4o",
    "tools": [{"type": "code_interpreter"}],
    "temperature": 0.3,
    "top_p": 0.9,
    "max_prompt_tokens": 500,
    "max_completion_tokens": 50,
    "truncation_strategy": {"type": "last_messages", "last_messages": 4},
    "tool_choice": "auto",
    "response_format": {"type": "json_object"},
    "parallel_tool_calls": False,
    "reasoning_effort": "low",
}


def _assistant_config(tmp_path: Path, api_base: str) -> Path:
    config: Final[dict[str, JsonValue]] = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["assistant_settings"] = {
        "custom_llm_provider": "openai",
        "litellm_params": {"api_base": api_base, "api_key": "synthetic-openai-key"},
    }
    config_path: Final = tmp_path / "assistants_evals_stream.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _frame(event: str, data: dict[str, JsonValue] | str) -> bytes:
    encoded_data: Final = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {encoded_data}\n\n".encode()


def _respond(request: Request) -> Reply:
    if request.method != "POST" or request.target != "/v1/threads/thread_abc/runs":
        return Reply(status=404, body=b'{"error":"unexpected upstream request"}')
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _frame("thread.run.created", _QUEUED_RUN),
            _frame("thread.message.created", _MESSAGE_CREATED),
            _frame("thread.message.delta", _MESSAGE_DELTA),
            _frame("thread.run.completed", _COMPLETED_RUN),
            _frame("done", "[DONE]"),
        ),
    )


def _parse_frame(frame: str) -> tuple[str, JsonValue]:
    lines: Final = frame.splitlines()
    event: Final = next((line.removeprefix("event: ") for line in lines if line.startswith("event: ")), "")
    data: Final = next((line.removeprefix("data: ") for line in lines if line.startswith("data: ")), "")
    return event, data if data == "[DONE]" else json.loads(data)


def _parse_frames(body: str) -> tuple[tuple[str, JsonValue], ...]:
    return tuple(_parse_frame(frame) for frame in body.strip().split("\n\n"))


_EXPECTED_FRAMES: Final = (
    ("thread.run.created", _QUEUED_RUN),
    ("thread.message.created", _MESSAGE_CREATED),
    ("thread.message.delta", _MESSAGE_DELTA),
    ("thread.run.completed", _COMPLETED_RUN),
    ("done", "[DONE]"),
)


class _RecordingStream(httpx.SyncByteStream):
    def __init__(self, inner: httpx.SyncByteStream, sink: list[bytes]) -> None:
        self._inner: Final = inner
        self._sink: Final = sink

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._inner:
            self._sink.append(chunk)
            yield chunk

    def close(self) -> None:
        self._inner.close()


class _RecordingTransport(httpx.BaseTransport):
    """Records the bytes the SDK client read off the proxy socket, so the done frame the SDK swallows is visible."""

    def __init__(self) -> None:
        self.received: Final[list[bytes]] = []
        self._inner: Final = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response: Final = self._inner.handle_request(request)
        assert isinstance(response.stream, httpx.SyncByteStream)
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_RecordingStream(response.stream, self.received),
            extensions=response.extensions,
            request=request,
        )

    def close(self) -> None:
        self._inner.close()

    def text(self) -> str:
        return b"".join(self.received).decode()


@pytest.mark.parametrize("prefix", ("", "/v1"), ids=("root-alias", "v1-alias"))
def test_run_stream_relays_upstream_events(gateway: Gateway, tmp_path: Path, prefix: str) -> None:
    pytest.skip(
        "BUG: streamed run frames carry the event name inside data with no event: line and drop run options upstream"
    )
    with wire_server(_respond) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            body: Final[dict[str, JsonValue]] = {**_RUN_REQUEST, "stream": True}
            response: Final = candidate.request("POST", f"{prefix}/threads/thread_abc/runs", body)
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("text/event-stream"), response.text
            assert _parse_frames(response.text) == _EXPECTED_FRAMES, response.text
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs")
            ], response.text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", response.text
            assert json.loads(requests[0].body) == body, response.text


@pytest.mark.parametrize("prefix", ("", "/v1"), ids=("root-alias", "v1-alias"))
def test_sdk_run_stream_event_handler_sees_events(gateway: Gateway, tmp_path: Path, prefix: str) -> None:
    pytest.skip("BUG: runs.stream drops additional_messages, sampling, token caps and other run options upstream")
    with wire_server(_respond) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            transport: Final = _RecordingTransport()
            with OpenAI(
                base_url=f"{str(candidate.client.base_url).rstrip('/')}{prefix}",
                api_key=candidate.key,
                max_retries=0,
                http_client=httpx.Client(transport=transport),
            ) as client:
                with client.beta.threads.runs.stream(
                    thread_id="thread_abc",
                    assistant_id="asst_abc",
                    instructions="Be terse",
                    additional_instructions="Cite sources",
                    additional_messages=[{"role": "user", "content": "Also check page 2"}],
                    metadata={"run": "r1"},
                    model="gpt-4o",
                    tools=[{"type": "code_interpreter"}],
                    temperature=0.3,
                    top_p=0.9,
                    max_prompt_tokens=500,
                    max_completion_tokens=50,
                    truncation_strategy={"type": "last_messages", "last_messages": 4},
                    tool_choice="auto",
                    response_format={"type": "json_object"},
                    parallel_tool_calls=False,
                    reasoning_effort="low",
                ) as stream:
                    events: Final = tuple((event.event, event.data.to_dict(mode="json")) for event in stream)
            response_text: Final = transport.text()
            assert events == _EXPECTED_FRAMES[:-1], response_text
            assert _parse_frames(response_text) == _EXPECTED_FRAMES, response_text
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs")
            ], response_text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", response_text
            assert json.loads(requests[0].body) == {**_RUN_REQUEST, "stream": True}, response_text


@pytest.mark.parametrize("prefix", ("", "/v1"), ids=("root-alias", "v1-alias"))
def test_sdk_run_create_stream_yields_events(gateway: Gateway, tmp_path: Path, prefix: str) -> None:
    pytest.skip("BUG: runs.create(stream=True) yields null event names and drops run options upstream")
    with wire_server(_respond) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            transport: Final = _RecordingTransport()
            with OpenAI(
                base_url=f"{str(candidate.client.base_url).rstrip('/')}{prefix}",
                api_key=candidate.key,
                max_retries=0,
                http_client=httpx.Client(transport=transport),
            ) as client:
                with client.beta.threads.runs.create(
                    "thread_abc",
                    assistant_id="asst_abc",
                    instructions="Be terse",
                    additional_instructions="Cite sources",
                    additional_messages=[{"role": "user", "content": "Also check page 2"}],
                    metadata={"run": "r1"},
                    model="gpt-4o",
                    tools=[{"type": "code_interpreter"}],
                    temperature=0.3,
                    top_p=0.9,
                    max_prompt_tokens=500,
                    max_completion_tokens=50,
                    truncation_strategy={"type": "last_messages", "last_messages": 4},
                    tool_choice="auto",
                    response_format={"type": "json_object"},
                    parallel_tool_calls=False,
                    reasoning_effort="low",
                    stream=True,
                ) as stream:
                    events: Final = tuple((event.event, event.data.to_dict(mode="json")) for event in stream)
            response_text: Final = transport.text()
            assert events == _EXPECTED_FRAMES[:-1], response_text
            assert _parse_frames(response_text) == _EXPECTED_FRAMES, response_text
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs")
            ], response_text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", response_text
            assert json.loads(requests[0].body) == {**_RUN_REQUEST, "stream": True}, response_text


_MARKER_MESSAGE: Final[dict[str, JsonValue]] = {
    "id": "msg_marker",
    "assistant_id": None,
    "attachments": [],
    "completed_at": None,
    "object": "thread.message",
    "content": [{"type": "text", "text": {"value": "marker", "annotations": []}}],
    "created_at": 1700000005,
    "incomplete_at": None,
    "incomplete_details": None,
    "metadata": {},
    "role": "user",
    "run_id": None,
    "status": "completed",
    "thread_id": "thread_abc",
}
_HELD_FRAME: Final = b":" + b"x" * 4_000_000 + b"\n\n"


def _held_respond(gate: threading.Event, run_reply: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "POST" and request.target == "/v1/threads/thread_abc/runs":
            return run_reply
        if request.method == "GET" and request.target == "/v1/threads/thread_abc/runs/run_abc":
            return Reply(body=json.dumps(_QUEUED_RUN).encode())
        if request.method == "POST" and request.target == "/v1/threads/thread_abc/messages":
            return Reply(body=json.dumps(_MARKER_MESSAGE).encode())
        return Reply(status=404, body=b'{"error":"unexpected upstream request"}')

    assert run_reply.gate_after_first is gate
    return respond


def _marker_message(base_url: str, key: str) -> str:
    with OpenAI(base_url=base_url, api_key=key, max_retries=0) as client:
        raw: Final = client.beta.threads.messages.with_raw_response.create("thread_abc", role="user", content="marker")
    assert json.loads(raw.http_response.text) == _MARKER_MESSAGE, raw.http_response.text
    return raw.http_response.text


def test_run_stream_client_disconnect_releases_upstream_without_polling(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip(
        "BUG: runs.create(stream=True) yields null event names, so the first SDK event is not thread.run.created"
    )
    gate: Final = threading.Event()
    run_reply: Final = Reply(
        content_type="text/event-stream",
        chunks=(
            _frame("thread.run.created", _QUEUED_RUN),
            _HELD_FRAME,
            _frame("thread.message.created", _MESSAGE_CREATED),
            _frame("thread.run.completed", _COMPLETED_RUN),
            _frame("done", "[DONE]"),
        ),
        gate_after_first=gate,
    )
    with wire_server(_held_respond(gate, run_reply)) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            base_url: Final = f"{str(candidate.client.base_url).rstrip('/')}/v1"
            try:
                with OpenAI(base_url=base_url, api_key=candidate.key, max_retries=0) as client:
                    with client.beta.threads.runs.create("thread_abc", assistant_id="asst_abc", stream=True) as stream:
                        first: Final = next(iter(stream))
            finally:
                gate.set()
            response_text: Final = json.dumps({"event": first.event, "data": first.data.to_dict(mode="json")})
            assert (first.event, first.data.to_dict(mode="json")) == ("thread.run.created", _QUEUED_RUN), response_text
            assert wire.disconnected.get(timeout=10) == "/v1/threads/thread_abc/runs", response_text
            marker_text: Final = _marker_message(base_url, candidate.key)
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs"),
                ("POST", "/v1/threads/thread_abc/messages"),
            ], f"{response_text}\n{marker_text}"
            assert json.loads(requests[0].body) == {"assistant_id": "asst_abc", "stream": True}, response_text


def test_run_client_disconnect_does_not_leave_the_proxy_polling(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip("BUG: after the client of a non-stream runs.create disconnects, the proxy keeps polling GET /runs/{id}")
    gate: Final = threading.Event()
    queued_body: Final = json.dumps(_QUEUED_RUN).encode()
    run_reply: Final = Reply(chunks=(queued_body[:20], queued_body[20:]), gate_after_first=gate)
    with wire_server(_held_respond(gate, run_reply)) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            base_url: Final = f"{str(candidate.client.base_url).rstrip('/')}/v1"
            try:
                with OpenAI(
                    base_url=base_url, api_key=candidate.key, max_retries=0, timeout=httpx.Timeout(30, read=2)
                ) as client:
                    with pytest.raises(APITimeoutError):
                        client.beta.threads.runs.create("thread_abc", assistant_id="asst_abc")
            finally:
                gate.set()
            marker_text: Final = _marker_message(base_url, candidate.key)
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs"),
                ("POST", "/v1/threads/thread_abc/messages"),
            ], marker_text
            assert json.loads(requests[0].body) == {"assistant_id": "asst_abc"}, marker_text
