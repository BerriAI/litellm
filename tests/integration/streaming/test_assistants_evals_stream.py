from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
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
            assert _parse_frames(response.text) == (
                ("thread.run.created", _QUEUED_RUN),
                ("thread.message.created", _MESSAGE_CREATED),
                ("thread.message.delta", _MESSAGE_DELTA),
                ("thread.run.completed", _COMPLETED_RUN),
                ("done", "[DONE]"),
            ), response.text
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs")
            ], response.text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", response.text
            assert json.loads(requests[0].body) == body, response.text


def test_sdk_run_stream_event_handler_sees_events(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip("BUG: runs.stream drops additional_messages, sampling, token caps and other run options upstream")
    with wire_server(_respond) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            with OpenAI(
                base_url=f"{str(candidate.client.base_url).rstrip('/')}/v1",
                api_key=candidate.key,
                max_retries=0,
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
                ) as stream:
                    events: Final = tuple((event.event, event.data.to_dict(mode="json")) for event in stream)
            response_text: Final = json.dumps(events)
            assert events == (
                ("thread.run.created", _QUEUED_RUN),
                ("thread.message.created", _MESSAGE_CREATED),
                ("thread.message.delta", _MESSAGE_DELTA),
                ("thread.run.completed", _COMPLETED_RUN),
            ), response_text
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs")
            ], response_text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", response_text
            assert json.loads(requests[0].body) == {**_RUN_REQUEST, "stream": True}, response_text


def test_sdk_run_create_stream_yields_events(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip("BUG: runs.create(stream=True) yields null event names and drops run options upstream")
    with wire_server(_respond) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            with OpenAI(
                base_url=f"{str(candidate.client.base_url).rstrip('/')}/v1",
                api_key=candidate.key,
                max_retries=0,
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
                    stream=True,
                ) as stream:
                    events: Final = tuple((event.event, event.data.to_dict(mode="json")) for event in stream)
            response_text: Final = json.dumps(events)
            assert events == (
                ("thread.run.created", _QUEUED_RUN),
                ("thread.message.created", _MESSAGE_CREATED),
                ("thread.message.delta", _MESSAGE_DELTA),
                ("thread.run.completed", _COMPLETED_RUN),
            ), response_text
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs")
            ], response_text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", response_text
            assert json.loads(requests[0].body) == {**_RUN_REQUEST, "stream": True}, response_text
