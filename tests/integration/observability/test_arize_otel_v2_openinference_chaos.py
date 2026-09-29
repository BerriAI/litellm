from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
from _openinference_support import (
    CHAT_TOOLS,
    RESPONSES_TOOLS,
    _anthropic_response,
    _anthropic_stream_response,
    _assert_chat_request,
    _assert_messages_request,
    _assert_responses_request,
    _chat_caller_response,
    _chat_caller_stream,
    _chat_request_marker,
    _chat_response,
    _chat_stream_response,
    _chat_tool_call,
    _collect_marker_spans,
    _json_object,
    _messages_caller_raw_stream,
    _messages_caller_response,
    _normalize_chat_caller_stream,
    _normalize_responses_caller_body,
    _normalize_responses_caller_stream,
    _owned_sink_handler,
    _responses_caller_response,
    _responses_caller_stream,
    _responses_response,
    _responses_stream_response,
    _rig,
    _spans,
    _sse_json_values,
)
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, wire_server


def _call(
    proxy: Gateway,
    model: str,
    marker: str,
    *,
    surface: str = "chat",
    stream: bool = False,
    prompt: str | None = None,
) -> httpx.Response:
    match surface:
        case "chat":
            return proxy.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt or "weather in Paris?"}],
                    "tools": CHAT_TOOLS,
                    "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
                    "metadata": {"trace_marker": marker},
                    **({"stream": True} if stream else {}),
                    **({"stream_options": {"include_usage": True}} if stream and surface == "chat" else {}),
                    "cache": {"no-cache": True},
                },
            )
        case "responses":
            return proxy.request(
                "POST",
                "/v1/responses",
                {
                    "model": model,
                    "input": prompt or "weather in Paris?",
                    "tools": RESPONSES_TOOLS,
                    "tool_choice": {"type": "function", "name": "lookup_weather"},
                    "metadata": {"trace_marker": marker},
                    **({"stream": True} if stream else {}),
                    "cache": {"no-cache": True},
                },
            )
        case "messages":
            return proxy.request(
                "POST",
                "/v1/messages",
                {
                    "model": model,
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": prompt or "weather in Paris?"}],
                    "tools": [
                        {
                            "name": "lookup_weather",
                            "description": "Get weather",
                            "input_schema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                    "tool_choice": {"type": "auto"},
                    "metadata": {"trace_marker": marker},
                    **({"stream": True} if stream else {}),
                    "cache": {"no-cache": True},
                },
            )
        case _:
            raise AssertionError(f"Unknown endpoint: {surface}")


def _assert_response(response: httpx.Response, marker: str, surface: str, stream: bool, model: str) -> None:
    assert response.status_code == 200, response.text
    if stream:
        observed_stream: Final = _sse_json_values(response.content)
        if surface == "chat":
            chat_reply: Final = _chat_stream_response(marker, (_chat_tool_call(marker),))
            assert _normalize_chat_caller_stream(observed_stream) == _chat_caller_stream(chat_reply, model), (
                response.text
            )
            return
        if surface == "responses":
            responses_reply: Final = _responses_stream_response(marker, (_chat_tool_call(marker),))
            assert _normalize_responses_caller_stream(observed_stream) == _responses_caller_stream(
                responses_reply, model
            ), response.text
            return
        if surface == "messages":
            messages_reply: Final = _anthropic_stream_response(marker)
            assert observed_stream == _messages_caller_raw_stream(messages_reply, model), response.text
            return
        raise AssertionError(f"Unknown endpoint: {surface}")

    observed: Final = _json_object(response.content)
    if surface == "chat":
        chat_reply: Final = _chat_response(marker)
        assert observed == _chat_caller_response(chat_reply, model), response.text
        return
    if surface == "responses":
        responses_reply: Final = _responses_response(marker)
        assert _normalize_responses_caller_body(observed) == _responses_caller_response(responses_reply, model), (
            response.text
        )
        return
    if surface == "messages":
        messages_reply: Final = _anthropic_response(marker)
        assert observed == _messages_caller_response(messages_reply, model), response.text
        return
    raise AssertionError(f"Unknown endpoint: {surface}")


def _call_without_worker_error(
    proxy: Gateway, model: str, marker: str, *, prompt: str | None = None
) -> httpx.Response | None:
    try:
        return _call(proxy, model, marker, prompt=prompt)
    except httpx.HTTPError:
        return None


def test_arize_otel_v2_f1_sink_outage_and_recovery(gateway: Gateway, tmp_path: Path) -> None:
    markers: Final = tuple("f1-" + uuid.uuid4().hex for _ in range(30))
    surfaces: Final = ("chat", "responses", "messages")
    calls: Final = tuple(
        (marker, surfaces[index % len(surfaces)], index % 2 == 0) for index, marker in enumerate(markers)
    )

    def upstream(request: Request) -> Reply:
        body: Final = _json_object(request.body)
        if request.target.endswith("/messages"):
            messages: Final = body.get("messages")
            assert isinstance(messages, list) and isinstance(messages[0], dict), body
            marker: Final = messages[0].get("content")
            assert isinstance(marker, str), body
            _assert_messages_request(
                request,
                marker=marker,
                prompt=marker,
                stream=True if body.get("stream") is True else False,
            )
            return _anthropic_stream_response(marker) if body.get("stream") is True else _anthropic_response(marker)
        if request.target.endswith("/responses"):
            marker: Final = body.get("input")
            assert isinstance(marker, str), body
            _assert_responses_request(
                request,
                marker=marker,
                input_value=marker,
                stream=body.get("stream") is True,
            )
            return (
                _responses_stream_response(marker, (_chat_tool_call(marker),))
                if body.get("stream") is True
                else _responses_response(marker)
            )
        marker: Final = _chat_request_marker(request)
        _assert_chat_request(
            request,
            messages=[{"role": "user", "content": marker}],
            stream=True if body.get("stream") is True else None,
            stream_options={"include_usage": True} if body.get("stream") is True else None,
        )
        return (
            _chat_stream_response(marker, (_chat_tool_call(marker),))
            if body.get("stream") is True
            else _chat_response(marker)
        )

    def sink(_request: Request) -> Reply:
        return Reply(body=b"", content_type="application/x-protobuf")

    with ExitStack() as servers:
        initial_stack: Final = servers.enter_context(ExitStack())
        stopped_destination: Final = initial_stack.enter_context(wire_server(_owned_sink_handler(sink)))
        sink_port: Final = urlsplit(stopped_destination.url).port
        assert sink_port is not None, stopped_destination.url
        initial_stack.close()
        with _rig(
            gateway,
            tmp_path,
            upstream,
            environment={"OTEL_BSP_SCHEDULE_DELAY": "20000"},
            destination_wire=stopped_destination,
        ) as rig:
            with httpx.Client(trust_env=False) as client, pytest.raises(httpx.ConnectError):
                client.get(stopped_destination.url + "/health", timeout=2)
            with rig.proxy.scenario() as scenario:
                messages_model: Final = scenario.model(
                    model="anthropic/claude-opus-5-5",
                    api_base=rig.provider.url,
                )
                with ThreadPoolExecutor(max_workers=len(calls)) as executor:
                    futures: Final = tuple(
                        executor.submit(
                            _call,
                            rig.proxy,
                            messages_model if surface == "messages" else rig.model,
                            marker,
                            surface=surface,
                            stream=stream,
                            prompt=marker,
                        )
                        for marker, surface, stream in calls
                    )
                    responses: Final = tuple(
                        (marker, surface, stream, future.result(timeout=60))
                        for (marker, surface, stream), future in zip(calls, futures, strict=True)
                    )
                for marker, surface, stream, response in responses:
                    model: Final = messages_model if surface == "messages" else rig.model
                    _assert_response(response, marker, surface, stream, model)
                with httpx.Client(trust_env=False) as client, pytest.raises(httpx.ConnectError):
                    client.get(stopped_destination.url + "/health", timeout=2)
                recovered_stack: Final = servers.enter_context(ExitStack())
                recovered_destination: Final = recovered_stack.enter_context(
                    wire_server(_owned_sink_handler(sink), port=sink_port)
                )
                spans: Final = _collect_marker_spans(recovered_destination, markers, timeout_seconds=90)
                assert len(spans) == len(markers), spans
                recorded: Final = tuple(span["litellm.metadata.trace_marker"] for span in spans)
                assert len(recorded) == len(markers), recorded
                assert frozenset(recorded) == frozenset(markers), recorded


def test_arize_otel_v2_f2_slow_sink_does_not_deadlock(gateway: Gateway, tmp_path: Path) -> None:
    release: Final = threading.Event()
    blocked: Final = threading.Event()
    completed: Final = threading.Event()
    marker: Final = "f2-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _chat_response(marker)

    def sink(request: Request) -> Reply:
        if any(attributes.get("litellm.metadata.trace_marker") == marker for attributes in _spans((request,))):
            blocked.set()
            assert release.wait(timeout=5), "slow sink was not released"
            completed.set()
        return Reply(body=b"", content_type="application/x-protobuf")

    with _rig(gateway, tmp_path, upstream, destination_handler=sink) as rig:
        timer: Final = threading.Timer(2, release.set)
        try:
            response: Final = _call(rig.proxy, rig.model, marker)
            _assert_response(response, marker, "chat", False, rig.model)
            assert eventually(lambda: blocked.is_set(), bool, seconds=10)
            timer.start()
            assert eventually(lambda: completed.is_set(), bool, seconds=10)
            timer.join(timeout=5)
            assert not timer.is_alive(), "Slow sink timer did not finish"
        finally:
            release.set()
            timer.cancel()
            if timer.ident is not None:
                timer.join(timeout=5)
    requests: Final = rig.destination.drain()
    spans: Final = tuple(
        attributes
        for attributes in _spans(requests)
        if attributes.get("openinference.span.kind") == "LLM"
        and attributes.get("litellm.metadata.trace_marker") == marker
    )
    assert len(spans) == 1, spans


def test_arize_otel_v2_f3_one_proxy_worker_can_die(gateway: Gateway, tmp_path: Path) -> None:
    markers: Final = tuple("f3-" + uuid.uuid4().hex for _ in range(8))
    release: Final = threading.Event()

    def upstream(request: Request) -> Reply:
        request_marker: Final = _chat_request_marker(request)
        _assert_chat_request(
            request,
            messages=[{"role": "user", "content": request_marker}],
        )
        assert release.wait(timeout=20), "F3 upstream barrier was not released"
        return _chat_response(request_marker)

    def sink(_request: Request) -> Reply:
        return Reply(body=b"", content_type="application/x-protobuf")

    with _rig(
        gateway,
        tmp_path,
        upstream,
        destination_handler=sink,
        fresh_client_connections=True,
    ) as rig:
        children: Final = psutil.Process(rig.owned.process.pid).children(recursive=True)
        workers: Final = tuple(
            child for child in children if child.is_running() and "resource_tracker" not in " ".join(child.cmdline())
        )
        assert len(workers) >= 2, tuple((worker.pid, worker.name()) for worker in workers)
        with ThreadPoolExecutor(max_workers=len(markers)) as executor:
            try:
                futures: Final = tuple(
                    executor.submit(
                        _call_without_worker_error,
                        rig.proxy,
                        rig.model,
                        marker,
                        prompt=marker,
                    )
                    for marker in markers
                )
                observed: Final = eventually(
                    lambda: rig.provider.received.qsize(),
                    lambda count: count >= 2,
                    seconds=10,
                )
                assert observed >= 2, observed
                workers[0].kill()
                assert eventually(lambda: not workers[0].is_running(), bool, seconds=10), workers[0]
                release.set()
                in_flight: Final = tuple(
                    (marker, future.result(timeout=60)) for marker, future in zip(markers, futures, strict=True)
                )
            finally:
                release.set()
        survivor: Final = "f3-survivor-" + uuid.uuid4().hex
        survivor_response: Final = _call(rig.proxy, rig.model, survivor, prompt=survivor)
        _assert_response(survivor_response, survivor, "chat", False, rig.model)
        served_responses: Final = tuple(
            (marker, response) for marker, response in in_flight if response is not None and response.status_code == 200
        )
        for marker, response in served_responses:
            _assert_response(response, marker, "chat", False, rig.model)
        served: Final = tuple(marker for marker, _response in served_responses) + (survivor,)
        collected: Final = _collect_marker_spans(rig.destination, served)
        assert len(collected) == len(served), collected
        exported: Final = tuple(span["litellm.metadata.trace_marker"] for span in collected)
        assert len(exported) == len(served), exported
        assert frozenset(exported) == frozenset(served), exported
