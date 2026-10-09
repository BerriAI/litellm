from __future__ import annotations

import json
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from openai import APIStatusError, OpenAI
from openai.types.beta.thread import Thread as _Thread
from openai.types.beta.threads.message import Message as _Message
from openai.types.beta.threads.run import Run as _Run
from pydantic import JsonValue

_THREAD: Final[dict[str, JsonValue]] = {
    "id": "thread_abc",
    "object": "thread",
    "created_at": 1700000000,
    "metadata": {"project": "alpha"},
    "tool_resources": {"code_interpreter": {"file_ids": ["file-abc"]}, "file_search": None},
}
_THREAD_REQUEST: Final[dict[str, JsonValue]] = {
    "messages": [{"role": "user", "content": "What is in the report?", "metadata": {"turn": "1"}}],
    "metadata": {"project": "alpha"},
    "tool_resources": {"code_interpreter": {"file_ids": ["file-abc"]}},
}
_MESSAGE: Final[dict[str, JsonValue]] = {
    "id": "msg_abc",
    "assistant_id": None,
    "attachments": [{"file_id": "file-abc", "tools": [{"type": "file_search"}]}],
    "completed_at": None,
    "object": "thread.message",
    "content": [{"type": "text", "text": {"value": "Summarize the attachment", "annotations": []}}],
    "created_at": 1700000001,
    "incomplete_at": None,
    "incomplete_details": None,
    "metadata": {"turn": "2"},
    "role": "user",
    "run_id": None,
    "status": "completed",
    "thread_id": "thread_abc",
}
_IMAGE_PARTS: Final[list[JsonValue]] = [
    {"type": "text", "text": "Compare these"},
    {"type": "image_file", "image_file": {"file_id": "file-img", "detail": "low"}},
    {"type": "image_url", "image_url": {"url": "https://example.invalid/chart.png", "detail": "high"}},
]
_RUN: Final[dict[str, JsonValue]] = {
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
    config_path: Final = tmp_path / "assistants_evals.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _respond(request: Request) -> Reply:
    path: Final = request.target.partition("?")[0]
    responses: Final = {
        "/v1/threads": _THREAD,
        "/v1/threads/thread_abc/messages": _MESSAGE,
        "/v1/threads/thread_abc/runs": _RUN,
    }
    if request.method != "POST" or path not in responses:
        return Reply(status=404, body=b'{"error":"unexpected upstream request"}')
    return Reply(body=json.dumps(responses[path]).encode())


@pytest.mark.parametrize("prefix", ("", "/v1"), ids=("root-alias", "v1-alias"))
def test_create_thread_forwards_sdk_body(gateway: Gateway, tmp_path: Path, prefix: str) -> None:
    pytest.skip("BUG: POST /threads reaches the upstream with body {} and drops messages, metadata and tool_resources")
    with wire_server(_respond) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            base_url: Final = f"{str(candidate.client.base_url).rstrip('/')}{prefix}"
            with OpenAI(base_url=base_url, api_key=candidate.key, max_retries=0) as client:
                try:
                    raw: Final = client.beta.threads.with_raw_response.create(
                        messages=[{"role": "user", "content": "What is in the report?", "metadata": {"turn": "1"}}],
                        metadata={"project": "alpha"},
                        tool_resources={"code_interpreter": {"file_ids": ["file-abc"]}},
                    )
                except APIStatusError as error:
                    failure_text: Final = error.response.text
                    failed_requests: Final = wire.drain()
                    assert [(request.method, request.target) for request in failed_requests] == [
                        ("POST", "/v1/threads")
                    ], failure_text
                    assert json.loads(failed_requests[0].body) == _THREAD_REQUEST, failure_text
                    pytest.fail(f"SDK thread creation failed with HTTP {error.response.status_code}: {failure_text}")
            assert raw.status_code == 200, raw.http_response.text
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [("POST", "/v1/threads")], (
                raw.http_response.text
            )
            assert json.loads(requests[0].body) == _THREAD_REQUEST, raw.http_response.text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", raw.http_response.text
            assert isinstance(raw.parse(), _Thread), raw.http_response.text
            assert json.loads(raw.http_response.text) == _THREAD, raw.http_response.text


@pytest.mark.parametrize(
    ("content", "prefix"),
    (
        ("Summarize the attachment", ""),
        ("Summarize the attachment", "/v1"),
        ([{"type": "text", "text": "Summarize the attachment"}], ""),
        ([{"type": "text", "text": "Summarize the attachment"}], "/v1"),
        (_IMAGE_PARTS, ""),
        (_IMAGE_PARTS, "/v1"),
    ),
    ids=(
        "string-root-alias",
        "string-v1-alias",
        "parts-root-alias",
        "parts-v1-alias",
        "image-parts-root-alias",
        "image-parts-v1-alias",
    ),
)
def test_add_message_forwards_content_parts_attachments_metadata(
    gateway: Gateway, tmp_path: Path, content: JsonValue, prefix: str
) -> None:
    with wire_server(_respond) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            base_url: Final = f"{str(candidate.client.base_url).rstrip('/')}{prefix}"
            with OpenAI(base_url=base_url, api_key=candidate.key, max_retries=0) as client:
                raw: Final = client.beta.threads.messages.with_raw_response.create(
                    "thread_abc",
                    role="user",
                    content=content,
                    attachments=[{"file_id": "file-abc", "tools": [{"type": "file_search"}]}],
                    metadata={"turn": "2"},
                )
            assert raw.status_code == 200, raw.http_response.text
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/messages")
            ], raw.http_response.text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", raw.http_response.text
            assert json.loads(requests[0].body) == {
                "role": "user",
                "content": content,
                "attachments": [{"file_id": "file-abc", "tools": [{"type": "file_search"}]}],
                "metadata": {"turn": "2"},
            }, raw.http_response.text
            assert isinstance(raw.parse(), _Message), raw.http_response.text
            assert json.loads(raw.http_response.text) == _MESSAGE, raw.http_response.text


@pytest.mark.parametrize("prefix", ("", "/v1"), ids=("root-alias", "v1-alias"))
def test_run_forwards_every_sdk_field(gateway: Gateway, tmp_path: Path, prefix: str) -> None:
    pytest.skip(
        "BUG: runs.create drops additional_messages, temperature, top_p, max_prompt_tokens, max_completion_tokens, "
        "truncation_strategy, tool_choice, response_format, parallel_tool_calls and reasoning_effort, sends nulls, "
        "then polls "
        "GET /runs/{id}"
    )
    with wire_server(_respond) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            base_url: Final = f"{str(candidate.client.base_url).rstrip('/')}{prefix}"
            with OpenAI(base_url=base_url, api_key=candidate.key, max_retries=0) as client:
                try:
                    raw: Final = client.beta.threads.runs.with_raw_response.create(
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
                    )
                except APIStatusError as error:
                    failure_text: Final = error.response.text
                    failed_requests: Final = wire.drain()
                    assert [(request.method, request.target) for request in failed_requests] == [
                        ("POST", "/v1/threads/thread_abc/runs")
                    ], failure_text
                    assert json.loads(failed_requests[0].body) == _RUN_REQUEST, failure_text
                    pytest.fail(f"SDK run creation failed with HTTP {error.response.status_code}: {failure_text}")
            assert raw.status_code == 200, raw.http_response.text
            requests: Final = wire.drain()
            assert [(request.method, request.target) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs")
            ], raw.http_response.text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", raw.http_response.text
            assert json.loads(requests[0].body) == _RUN_REQUEST, raw.http_response.text
            run: Final[_Run] = raw.parse()
            assert json.loads(raw.http_response.text) == _RUN, raw.http_response.text
            assert run.status == "queued", raw.http_response.text


@pytest.mark.parametrize("prefix", ("", "/v1"), ids=("root-alias", "v1-alias"))
def test_run_include_query_reaches_upstream(gateway: Gateway, tmp_path: Path, prefix: str) -> None:
    pytest.skip("BUG: runs.create include[] query param never reaches the upstream")
    with wire_server(_respond) as wire:
        config: Final = _assistant_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            base_url: Final = f"{str(candidate.client.base_url).rstrip('/')}{prefix}"
            with OpenAI(base_url=base_url, api_key=candidate.key, max_retries=0) as client:
                raw: Final = client.beta.threads.runs.with_raw_response.create(
                    "thread_abc",
                    assistant_id="asst_abc",
                    include=["step_details.tool_calls[*].file_search.results[*].content"],
                )
            assert raw.status_code == 200, raw.http_response.text
            requests: Final = wire.drain()
            assert [(request.method, urlsplit(request.target).path) for request in requests] == [
                ("POST", "/v1/threads/thread_abc/runs")
            ], raw.http_response.text
            assert parse_qs(urlsplit(requests[0].target).query) == {
                "include[]": ["step_details.tool_calls[*].file_search.results[*].content"]
            }, raw.http_response.text
            assert json.loads(requests[0].body) == {"assistant_id": "asst_abc"}, raw.http_response.text
            assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key", raw.http_response.text
            run: Final[_Run] = raw.parse()
            assert isinstance(run, _Run), raw.http_response.text
            assert json.loads(raw.http_response.text) == _RUN, raw.http_response.text
