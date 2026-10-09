"""Akto guardrail in `mode: logging_only`, driven through a real proxy.

The Akto service is the only guardrail double: an owned wire peer that answers the `/api/http-proxy`
verdict protocol. The provider is a second owned peer. The proxy, its guardrail registry, Postgres and
Redis run for real with two workers. Every test waits for the spend row, which is written after the
logging-only scans finish, before it reads what Akto received.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import anthropic
import openai
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

LOG_KEY: Final = "synthetic-akto-log-key"
INPUT_KEY: Final = "synthetic-akto-input-key"
OUTPUT_KEY: Final = "synthetic-akto-output-key"
DOWN_KEY: Final = "synthetic-akto-down-key"
DOWN_OPEN_KEY: Final = "synthetic-akto-down-open-key"
MIXED_KEY: Final = "synthetic-akto-mixed-key"
BLOCK_MARK: Final = "SYNTHETIC-AKTO-BLOCK"
BLOCK_REASON: Final = "Synthetic Akto policy block"
AKTO_DROP_MARK: Final = "SYNTHETIC-AKTO-DROP"
AKTO_ERROR_MARK: Final = "SYNTHETIC-AKTO-500"
AKTO_GARBAGE_MARK: Final = "SYNTHETIC-AKTO-GARBAGE"
PROVIDER_FAIL_MARK: Final = "SYNTHETIC-PROVIDER-FAIL"
REQUEST_CHECK: Final = {"akto_connector": "litellm", "guardrails": "true", "ingest_data": "true"}
RESPONSE_CHECK: Final = {"akto_connector": "litellm", "response_guardrails": "true", "ingest_data": "true"}
FILE_CHECK: Final = {"akto_connector": "litellm", "file_guardrails": "true"}


@dataclass(frozen=True, slots=True)
class AktoCall:
    path: str
    flags: dict[str, str]
    authorization: str
    payload: dict[str, object]

    def request_text(self) -> str:
        return str(self.payload.get("requestPayload", ""))

    def response_text(self) -> str:
        return str(self.payload.get("responsePayload", ""))


def _akto_call(request: Request) -> AktoCall:
    target: Final = urlsplit(request.target)
    return AktoCall(
        path=target.path,
        flags={name: values[0] for name, values in parse_qs(target.query).items()},
        authorization=request.headers.get("authorization", ""),
        payload=json.loads(request.body),
    )


def _akto_verdict(request: Request) -> Reply:
    if AKTO_DROP_MARK.encode() in request.body:
        return Reply(drop_connection=True)
    if AKTO_ERROR_MARK.encode() in request.body:
        return Reply(status=500, body=b'{"error": "synthetic akto failure"}')
    if AKTO_GARBAGE_MARK.encode() in request.body:
        return Reply(body=b"synthetic akto garbage", content_type="text/plain")
    verdict: Final = (
        {"Allowed": False, "Behaviour": "block", "Reason": BLOCK_REASON}
        if BLOCK_MARK.encode() in request.body
        else {"Allowed": True}
    )
    return Reply(body=json.dumps({"data": {"guardrailsResult": verdict}}).encode())


def _answer(marker: str) -> str:
    return "synthetic answer " + marker


def _marker_in(body: bytes) -> str:
    text: Final = body.decode()
    start: Final = text.find("mark-")
    assert start >= 0, text
    return text[start : start + 37]


def _sse(events: tuple[dict[str, object], ...]) -> tuple[bytes, ...]:
    return tuple(("data: " + json.dumps(event) + "\n\n").encode() for event in events)


def _chat_reply(marker: str, streaming: bool) -> Reply:
    answer: Final = _answer(marker)
    if not streaming:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-" + marker,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-5.4-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    def chunk(delta: dict[str, object], finish: str | None) -> dict[str, object]:
        return {
            "id": "chatcmpl-" + marker,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5.4-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    events: Final = (chunk({"role": "assistant", "content": answer[:9]}, None), chunk({"content": answer[9:]}, "stop"))
    return Reply(chunks=(*_sse(events), b"data: [DONE]\n\n"), content_type="text/event-stream")


def _messages_reply(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "msg_" + marker,
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": _answer(marker)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode()
    )


def _responses_reply(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "resp_" + marker,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-5.4-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msgo_" + marker,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _answer(marker), "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    if PROVIDER_FAIL_MARK.encode() in request.body:
        return Reply(status=500, body=b'{"error": {"message": "synthetic provider failure", "type": "server_error"}}')
    marker: Final = _marker_in(request.body)
    if request.target.endswith("/v1/messages"):
        return _messages_reply(marker)
    if request.target.endswith("/v1/responses"):
        return _responses_reply(marker)
    assert request.target.endswith("/v1/chat/completions"), request.target
    return _chat_reply(marker, bool(json.loads(request.body).get("stream")))


def _guardrail(name: str, key: str, url: str, **params: object) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "akto",
            "mode": "logging_only",
            "default_on": True,
            "akto_base_url": url,
            "akto_api_key": key,
            **params,
        },
    }


def _rig_config(akto_url: str, down_url: str, root: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["guardrails"] = [
        _guardrail("akto-log", LOG_KEY, akto_url),
        _guardrail("akto-log-input", INPUT_KEY, akto_url, logging_only_scope="input"),
        _guardrail("akto-log-output", OUTPUT_KEY, akto_url, logging_only_scope="output"),
        _guardrail("akto-log-down", DOWN_KEY, down_url),
        _guardrail("akto-log-down-open", DOWN_OPEN_KEY, down_url, unreachable_fallback="fail_open"),
        _guardrail("akto-mixed", MIXED_KEY, akto_url, mode=["pre_call", "logging_only"], default_on=False),
    ]
    path: Final = root / "akto-logging-only.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    akto: Wire
    akto_down: Wire
    provider: Wire
    chat_model: str
    claude_model: str
    responses_model: str

    def base(self) -> str:
        return str(self.proxy.client.base_url).rstrip("/")

    def auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.proxy.key}"}

    def provider_calls(self, marker: str) -> tuple[Request, ...]:
        return tuple(request for request in self.provider.drain() if marker.encode() in request.body)

    def guardrail_entries(self, response_id: str, name: str) -> tuple[dict[str, object], ...]:
        rows: Final = spend_rows(response_id)
        assert len(rows) == 1, rows
        entries: Final = object_value(rows[0]["metadata"]).get("guardrail_information")
        assert isinstance(entries, list), rows[0]
        return tuple(entry for entry in (object_value(item) for item in entries) if entry.get("guardrail_name") == name)

    def settled_akto_calls(self, response_id: str, marker: str, key: str) -> tuple[AktoCall, ...]:
        self.guardrail_entries(response_id, "akto-log")
        calls: Final = tuple(_akto_call(request) for request in self.akto.drain() if marker.encode() in request.body)
        return tuple(call for call in calls if call.authorization == key)


def spend_rows(request_id: str) -> tuple[dict[str, object], ...]:
    return tuple(
        eventually(
            lambda: read_rows(
                'SELECT request_id, status, metadata FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)
            ),
            lambda values: len(values) >= 1,
            seconds=70,
        )
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("akto-logging-only")
    with (
        gateway_from_environment() as gateway,
        wire_server(_provider) as provider,
        wire_server(_akto_verdict) as akto,
        wire_server(lambda _: Reply(drop_connection=True)) as akto_down,
        owned_proxy_process(gateway, root, {}, config=_rig_config(akto.url, akto_down.url, root), workers=2) as owned,
        owned.gateway.scenario() as scenario,
    ):
        chat: Final = scenario.model(
            model="openai/gpt-5.4-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
        )
        claude: Final = scenario.model(
            model="anthropic/claude-haiku-5-5", api_base=provider.url, api_key="synthetic-anthropic-key"
        )
        responses: Final = scenario.model(
            model="openai/gpt-5.4-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
        )
        yield Rig(owned.gateway, akto, akto_down, provider, chat, claude, responses)


def _marker() -> str:
    return "mark-" + uuid.uuid4().hex


def _assert_checked_both_ways(calls: tuple[AktoCall, ...], marker: str) -> None:
    assert [call.flags for call in calls] == [REQUEST_CHECK, RESPONSE_CHECK], calls
    assert all(call.path == "/api/http-proxy" for call in calls), calls
    assert marker in calls[0].request_text(), calls[0].payload
    assert _answer(marker) in calls[1].response_text(), calls[1].payload


def _chat(rig: Rig, content: object, **extra: object) -> dict[str, object]:
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": rig.chat_model, "messages": [{"role": "user", "content": content}], **extra},
        headers=rig.auth(),
    )
    assert response.status_code == 200, response.text
    return response.json()


def _content(body: dict[str, object]) -> object:
    choices: Final = body["choices"]
    assert isinstance(choices, list), body
    return object_value(object_value(choices[0])["message"])["content"]


def test_logging_only_checks_a_sync_openai_chat_request_and_response(rig: Rig) -> None:
    marker: Final = _marker()
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    completion: Final = client.chat.completions.create(
        model=rig.chat_model, messages=[{"role": "user", "content": "hello " + marker}]
    )
    assert completion.choices[0].message.content == _answer(marker)
    assert len(rig.provider_calls(marker)) == 1

    _assert_checked_both_ways(rig.settled_akto_calls(completion.id, marker, LOG_KEY), marker)


@pytest.mark.asyncio
async def test_logging_only_checks_an_async_openai_chat_request_and_response(rig: Rig) -> None:
    marker: Final = _marker()
    client: Final = openai.AsyncOpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    completion: Final = await client.chat.completions.create(
        model=rig.chat_model, messages=[{"role": "user", "content": "hello " + marker}]
    )
    assert completion.choices[0].message.content == _answer(marker)

    _assert_checked_both_ways(rig.settled_akto_calls(completion.id, marker, LOG_KEY), marker)


def test_logging_only_block_verdict_never_blocks_the_caller(rig: Rig) -> None:
    marker: Final = _marker()
    body: Final = _chat(rig, f"{BLOCK_MARK} {marker}")
    assert _content(body) == _answer(marker)
    assert len(rig.provider_calls(marker)) == 1

    calls: Final = rig.settled_akto_calls(str(body["id"]), marker, LOG_KEY)
    assert [call.flags for call in calls] == [REQUEST_CHECK], calls
    entries: Final = rig.guardrail_entries(str(body["id"]), "akto-log")
    assert [(entry["guardrail_mode"], entry["guardrail_status"]) for entry in entries] == [
        ("logging_only", "guardrail_intervened")
    ], entries


def test_logging_only_scope_input_and_output_each_check_one_direction(rig: Rig) -> None:
    marker: Final = _marker()
    body: Final = _chat(rig, "scoped " + marker)

    input_calls: Final = rig.settled_akto_calls(str(body["id"]), marker, INPUT_KEY)
    assert [call.flags for call in input_calls] == [REQUEST_CHECK], input_calls
    assert marker in input_calls[0].request_text(), input_calls[0].payload
    entries: Final = rig.guardrail_entries(str(body["id"]), "akto-log-output")
    assert [entry["guardrail_status"] for entry in entries] == ["success"], entries


def test_logging_only_scope_output_checks_only_the_response(rig: Rig) -> None:
    marker: Final = _marker()
    body: Final = _chat(rig, "scoped output " + marker)

    output_calls: Final = rig.settled_akto_calls(str(body["id"]), marker, OUTPUT_KEY)
    assert [call.flags for call in output_calls] == [RESPONSE_CHECK], output_calls
    assert _answer(marker) in output_calls[0].response_text(), output_calls[0].payload


def test_unreachable_akto_under_logging_only_never_fails_the_caller(rig: Rig) -> None:
    marker: Final = _marker()
    body: Final = _chat(rig, "outage " + marker)
    assert _content(body) == _answer(marker)

    entries: Final = rig.guardrail_entries(str(body["id"]), "akto-log-down")
    assert [entry["guardrail_mode"] for entry in entries] == ["logging_only"], entries
    dropped: Final = [_akto_call(request) for request in rig.akto_down.drain() if marker.encode() in request.body]
    assert [call.flags for call in dropped if call.authorization == DOWN_KEY] == [REQUEST_CHECK], dropped


def test_unreachable_akto_with_fail_open_still_attempts_the_response_check(rig: Rig) -> None:
    marker: Final = _marker()
    body: Final = _chat(rig, "outage open " + marker)
    assert _content(body) == _answer(marker)

    entries: Final = rig.guardrail_entries(str(body["id"]), "akto-log-down-open")
    assert [entry["guardrail_mode"] for entry in entries] == ["logging_only", "logging_only"], entries
    dropped: Final = [_akto_call(request) for request in rig.akto_down.drain() if marker.encode() in request.body]
    assert [call.flags for call in dropped if call.authorization == DOWN_OPEN_KEY] == [REQUEST_CHECK, RESPONSE_CHECK]


def test_logging_only_streaming_chat_sends_the_assembled_answer(rig: Rig) -> None:
    marker: Final = _marker()
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    chunks: Final = tuple(
        client.chat.completions.create(
            model=rig.chat_model, messages=[{"role": "user", "content": "stream " + marker}], stream=True
        )
    )
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == _answer(marker)

    _assert_checked_both_ways(rig.settled_akto_calls(chunks[0].id, marker, LOG_KEY), marker)


def test_logging_only_anthropic_messages_checks_both_directions(rig: Rig) -> None:
    marker: Final = _marker()
    client: Final = anthropic.Anthropic(base_url=rig.base(), api_key=rig.proxy.key, max_retries=0)
    message: Final = client.messages.create(
        model=rig.claude_model, max_tokens=64, messages=[{"role": "user", "content": "claude " + marker}]
    )
    assert message.content[0].type == "text" and message.content[0].text == _answer(marker)

    _assert_checked_both_ways(rig.settled_akto_calls(message.id, marker, LOG_KEY), marker)


def test_logging_only_responses_api_checks_both_directions(rig: Rig) -> None:
    marker: Final = _marker()
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    result: Final = client.responses.create(model=rig.responses_model, input="responses " + marker)
    assert result.output_text == _answer(marker)

    _assert_checked_both_ways(rig.settled_akto_calls(result.id, marker, LOG_KEY), marker)


def test_logging_only_sends_each_chat_attachment_to_akto_once(rig: Rig) -> None:
    marker: Final = _marker()
    image_url: Final = f"https://example.com/{marker}.png"
    body: Final = _chat(
        rig, [{"type": "text", "text": "describe " + marker}, {"type": "image_url", "image_url": {"url": image_url}}]
    )

    calls: Final = rig.settled_akto_calls(str(body["id"]), marker, LOG_KEY)
    file_checks: Final = [call.payload["files"] for call in calls if call.flags == FILE_CHECK]
    assert file_checks == [[{"filename": f"{marker}.png", "type": "image", "url": image_url}]], calls


def test_logging_only_sends_each_responses_attachment_to_akto_once(rig: Rig) -> None:
    marker: Final = _marker()
    image_url: Final = f"https://example.com/{marker}.png"
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    result: Final = client.responses.create(
        model=rig.responses_model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe " + marker},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ],
    )

    calls: Final = rig.settled_akto_calls(result.id, marker, LOG_KEY)
    file_checks: Final = [call.payload["files"] for call in calls if call.flags == FILE_CHECK]
    assert file_checks == [[{"filename": f"{marker}.png", "type": "image", "url": image_url}]], calls


def test_pre_call_plus_logging_only_checks_the_request_inline_and_both_directions_after(rig: Rig) -> None:
    marker: Final = _marker()
    body: Final = _chat(rig, "mixed " + marker, guardrails=["akto-mixed"])

    flags: Final = [call.flags for call in rig.settled_akto_calls(str(body["id"]), marker, MIXED_KEY)]
    assert (flags.count(REQUEST_CHECK), flags.count(RESPONSE_CHECK), len(flags)) == (2, 1, 3), flags


def test_dashboard_offers_logging_only_for_akto(rig: Rig) -> None:
    response: Final = rig.proxy.client.get("/guardrails/ui/add_guardrail_settings", headers=rig.auth())
    assert response.status_code == 200, response.text
    settings: Final = response.json()
    assert "logging_only" in settings["supported_modes_by_provider"]["akto"], settings["supported_modes_by_provider"]
    assert "akto" not in settings["providers_without_directional_logging_only_scope"]


@pytest.mark.parametrize("failure", [AKTO_ERROR_MARK, AKTO_GARBAGE_MARK, AKTO_DROP_MARK])
def test_failing_akto_under_logging_only_never_fails_the_caller(rig: Rig, failure: str) -> None:
    marker: Final = _marker()
    body: Final = _chat(rig, f"{failure} {marker}")
    assert _content(body) == _answer(marker)

    calls: Final = rig.settled_akto_calls(str(body["id"]), marker, LOG_KEY)
    assert [call.flags for call in calls] == [REQUEST_CHECK], calls
    entries: Final = rig.guardrail_entries(str(body["id"]), "akto-log")
    assert [entry["guardrail_mode"] for entry in entries] == ["logging_only"], entries


def test_provider_failure_under_logging_only_reaches_the_caller_and_checks_nothing(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": rig.chat_model, "messages": [{"role": "user", "content": f"{PROVIDER_FAIL_MARK} {marker}"}]},
        headers=rig.auth(),
    )
    assert response.status_code == 500, response.text
    assert "synthetic provider failure" in response.text

    rows: Final = spend_rows(response.headers["x-litellm-call-id"])
    assert [row["status"] for row in rows] == ["failure"], rows
    assert [request for request in rig.akto.drain() if marker.encode() in request.body] == []


async def _burst_call(rig: Rig, kind: str, content: str) -> tuple[str, str]:
    if kind == "messages":
        claude: Final = anthropic.AsyncAnthropic(base_url=rig.base(), api_key=rig.proxy.key, max_retries=0)
        message: Final = await claude.messages.create(
            model=rig.claude_model, max_tokens=64, messages=[{"role": "user", "content": content}]
        )
        assert message.content[0].type == "text"
        return message.id, message.content[0].text
    client: Final = openai.AsyncOpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    if kind == "responses":
        result: Final = await client.responses.create(model=rig.responses_model, input=content)
        return result.id, result.output_text
    if kind == "stream":
        stream: Final = await client.chat.completions.create(
            model=rig.chat_model, messages=[{"role": "user", "content": content}], stream=True
        )
        chunks: Final = [chunk async for chunk in stream]
        return chunks[0].id, "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    completion: Final = await client.chat.completions.create(
        model=rig.chat_model, messages=[{"role": "user", "content": content}]
    )
    return completion.id, completion.choices[0].message.content or ""


@pytest.mark.asyncio
async def test_burst_with_akto_failing_for_half_the_calls_answers_and_logs_every_call_once(rig: Rig) -> None:
    kinds: Final = ("chat", "stream", "messages", "responses") * 6
    markers: Final = tuple(_marker() for _ in kinds)
    failing: Final = frozenset(markers[::2])
    results: Final = await asyncio.gather(
        *(
            _burst_call(rig, kind, (AKTO_DROP_MARK + " " if marker in failing else "") + "burst " + marker)
            for kind, marker in zip(kinds, markers, strict=True)
        )
    )
    assert [answer for _, answer in results] == [_answer(marker) for marker in markers]

    for response_id, _ in results:
        assert [row["status"] for row in spend_rows(response_id)] == ["success"], response_id
    calls: Final = tuple(
        call for call in (_akto_call(request) for request in rig.akto.drain()) if call.authorization == LOG_KEY
    )
    observed: Final = {
        marker: [call.flags for call in calls if marker in json.dumps(call.payload)] for marker in markers
    }
    expected: Final = {
        marker: [REQUEST_CHECK] if marker in failing else [REQUEST_CHECK, RESPONSE_CHECK] for marker in markers
    }
    assert observed == expected
    assert rig.proxy.client.get("/health/liveliness").status_code == 200
