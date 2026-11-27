"""What a guardrail run leaves in the spend log when `store_prompts_in_spend_logs` is off.

The provider is the only double, an owned wire peer. The proxy, its guardrail registry, Postgres and Redis run
for real with two workers. A run that matched nothing has nothing to redact, so its record carries no redaction
marker. A run that matched keeps the marker in place of the match text
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import anthropic
import openai
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.shares_worker_state

MARKER: Final = "REDACTED_BY_LITELM"
AWS_KEY: Final = "AKIA" + "SYNTHETIC0000000"
PASSING: Final = "cf-default-on"
MASKING: Final = "cf-mask"
BLOCKING: Final = "cf-block"
PROMPT_FIELDS: Final = ("guardrail_request", "guardrail_response", "match_details", "classification")


def _answer(mark: str) -> str:
    return "synthetic answer " + mark


def _provider_answer(body: bytes) -> str:
    text: Final = body.decode()
    start: Final = text.find("mark-")
    assert start >= 0, text
    return _answer(text[start : start + 37])


def _sse(events: tuple[dict[str, object], ...]) -> tuple[bytes, ...]:
    return tuple(("data: " + json.dumps(event) + "\n\n").encode() for event in events)


def _chat_reply(answer: str, streaming: bool) -> Reply:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex
    if not streaming:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
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
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5.4-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    events: Final = (chunk({"role": "assistant", "content": answer[:9]}, None), chunk({"content": answer[9:]}, "stop"))
    return Reply(chunks=(*_sse(events), b"data: [DONE]\n\n"), content_type="text/event-stream")


def _messages_reply(answer: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "msg_" + uuid.uuid4().hex,
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": answer}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode()
    )


def _responses_reply(answer: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "resp_" + uuid.uuid4().hex,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-5.4-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msgo_" + uuid.uuid4().hex,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": answer, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    answer: Final = _provider_answer(request.body)
    if request.target.endswith("/v1/messages"):
        return _messages_reply(answer)
    if request.target.endswith("/v1/responses"):
        return _responses_reply(answer)
    assert request.target.endswith("/v1/chat/completions"), request.target
    return _chat_reply(answer, bool(json.loads(request.body).get("stream")))


def _content_filter(
    name: str, action: str, *, default_on: bool, pattern: str = "AKIA[0-9A-Z]{16}"
) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": "pre_call",
            "default_on": default_on,
            "patterns": [{"pattern_type": "regex", "pattern": pattern, "name": "aws_access_key", "action": action}],
        },
    }


def _rig_config(root: Path) -> Path:
    shared: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **shared,
        "litellm_settings": {**object_value(shared["litellm_settings"]), "cache": False},
        "general_settings": {**object_value(shared["general_settings"]), "store_prompts_in_spend_logs": False},
        "guardrails": [
            _content_filter(PASSING, "BLOCK", default_on=True, pattern="ghp_[A-Za-z0-9]{36}"),
            _content_filter(MASKING, "MASK", default_on=False),
            _content_filter(BLOCKING, "BLOCK", default_on=False),
        ],
    }
    path: Final = root / "guardrail-spend-log-redaction-marker.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    provider: Wire
    chat_model: str
    claude_model: str
    scenario: Scenario

    def base(self) -> str:
        return str(self.proxy.client.base_url).rstrip("/")

    def provider_bodies(self, mark: str) -> tuple[str, ...]:
        return tuple(request.body.decode() for request in self.provider.drain() if mark.encode() in request.body)


@dataclass(frozen=True, slots=True)
class Recorded:
    request_id: str
    row: str
    entries: tuple[dict[str, object], ...]


def _recorded(query: str, parameters: tuple[str, ...], name: str) -> Recorded:
    rows: Final = eventually(lambda: read_rows(query, parameters), lambda values: len(values) >= 1, seconds=70)
    assert len(rows) == 1, rows
    recorded: Final = object_value(rows[0]["metadata"]).get("guardrail_information")
    assert isinstance(recorded, list), rows[0]
    entries: Final = tuple(
        entry for entry in (object_value(item) for item in recorded) if entry.get("guardrail_name") == name
    )
    assert entries, recorded
    return Recorded(str(rows[0]["request_id"]), json.dumps(rows[0], default=str), entries)


def _by_response(response_id: str, name: str) -> Recorded:
    return _recorded(
        'SELECT request_id, status, metadata, messages, response, proxy_server_request FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
        (response_id,),
        name,
    )


def _by_key(key: str, name: str) -> Recorded:
    return _recorded(
        'SELECT request_id, status, metadata, messages, response, proxy_server_request FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
        (sha256(key.encode()).hexdigest(),),
        name,
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("guardrail-spend-log-redaction-marker")
    with (
        gateway_from_environment() as gateway,
        wire_server(_provider) as provider,
        owned_proxy_process(gateway, root, {}, config=_rig_config(root), workers=2) as owned,
        owned.gateway.scenario() as scenario,
    ):
        chat: Final = scenario.model(
            model="openai/gpt-5.4-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
        )
        claude: Final = scenario.model(
            model="anthropic/claude-haiku-5-5", api_base=provider.url, api_key="synthetic-anthropic-key"
        )
        yield Rig(owned.gateway, provider, chat, claude, scenario)


def _mark() -> str:
    return "mark-" + uuid.uuid4().hex


def _sync_chat(rig: Rig, text: str) -> tuple[str, str]:
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    completion: Final = client.chat.completions.create(
        model=rig.chat_model, messages=[{"role": "user", "content": text}]
    )
    return completion.id, completion.choices[0].message.content or ""


async def _async_chat(rig: Rig, text: str) -> tuple[str, str]:
    client: Final = openai.AsyncOpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    completion: Final = await client.chat.completions.create(
        model=rig.chat_model, messages=[{"role": "user", "content": text}]
    )
    return completion.id, completion.choices[0].message.content or ""


def _streamed_chat(rig: Rig, text: str) -> tuple[str, str]:
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    chunks: Final = tuple(
        client.chat.completions.create(model=rig.chat_model, messages=[{"role": "user", "content": text}], stream=True)
    )
    return chunks[0].id, "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)


def _messages(rig: Rig, text: str) -> tuple[str, str]:
    client: Final = anthropic.Anthropic(base_url=rig.base(), api_key=rig.proxy.key, max_retries=0)
    message: Final = client.messages.create(
        model=rig.claude_model, max_tokens=64, messages=[{"role": "user", "content": text}]
    )
    assert message.content[0].type == "text"
    return message.id, message.content[0].text


def _responses(rig: Rig, text: str) -> tuple[str, str]:
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    result: Final = client.responses.create(model=rig.chat_model, input=text)
    return result.id, result.output_text


SYNC_CALLS: Final[dict[str, Callable[[Rig, str], tuple[str, str]]]] = {
    "chat": _sync_chat,
    "chat_stream": _streamed_chat,
    "messages": _messages,
    "responses": _responses,
}


def _assert_passed_without_marker(recorded: Recorded, mark: str) -> None:
    assert [entry["guardrail_status"] for entry in recorded.entries] == ["success"], recorded.entries
    assert mark not in recorded.row, recorded.row
    assert MARKER not in json.dumps(recorded.entries), recorded.entries


@pytest.mark.parametrize("endpoint", list(SYNC_CALLS))
def test_a_passed_guardrail_run_stores_no_redaction_marker(rig: Rig, endpoint: str) -> None:
    mark: Final = _mark()
    response_id, answer = SYNC_CALLS[endpoint](rig, f"{mark} what is the capital of France?")
    assert answer == _answer(mark)

    _assert_passed_without_marker(_by_response(response_id, PASSING), mark)


@pytest.mark.asyncio
async def test_a_passed_guardrail_run_stores_no_redaction_marker_for_an_async_client(rig: Rig) -> None:
    mark: Final = _mark()
    response_id, answer = await _async_chat(rig, f"{mark} what is the capital of France?")
    assert answer == _answer(mark)

    _assert_passed_without_marker(_by_response(response_id, PASSING), mark)


def test_a_masked_guardrail_run_stores_neither_the_match_nor_a_marker(rig: Rig) -> None:
    mark: Final = _mark()
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={
            "model": rig.chat_model,
            "messages": [{"role": "user", "content": f"{mark} key {AWS_KEY}"}],
            "guardrails": [MASKING],
        },
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 200, response.text
    sent: Final = rig.provider_bodies(mark)
    assert len(sent) == 1 and AWS_KEY not in sent[0], sent

    recorded: Final = _by_response(str(object_value(response.json())["id"]), MASKING)
    assert [entry["masked_entity_count"] for entry in recorded.entries] == [{"aws_access_key": 1}], recorded.entries
    assert [recorded.entries[0][field] for field in ("guardrail_response", "match_details")] == [None, None]
    assert MARKER not in json.dumps(recorded.entries), recorded.entries
    assert AWS_KEY not in recorded.row and mark not in recorded.row, recorded.row


def test_a_blocked_guardrail_run_stores_neither_the_match_nor_a_marker(rig: Rig) -> None:
    mark: Final = _mark()
    key: Final = rig.scenario.key()
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={
            "model": rig.chat_model,
            "messages": [{"role": "user", "content": f"{mark} key {AWS_KEY}"}],
            "guardrails": [BLOCKING],
        },
        headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 400, response.text
    assert rig.provider_bodies(mark) == ()

    recorded: Final = _by_key(key, BLOCKING)
    assert [entry["guardrail_status"] for entry in recorded.entries] == ["guardrail_intervened"], recorded.entries
    assert all(
        recorded.entries[0][field] in (None, "", [], {}) for field in PROMPT_FIELDS if field in recorded.entries[0]
    ), recorded.entries
    assert MARKER not in json.dumps(recorded.entries), recorded.entries
    assert AWS_KEY not in recorded.row and mark not in recorded.row, recorded.row


def _monitor_log(rig: Rig, recorded: Recorded) -> dict[str, object]:
    guardrail_id: Final = str(recorded.entries[0]["guardrail_id"])

    def logged() -> list[dict[str, object]]:
        page: Final = rig.proxy.get("/guardrails/usage/logs", {"guardrail_id": guardrail_id, "page_size": "100"})
        logs: Final = page["logs"]
        assert isinstance(logs, list), page
        return [log for log in (object_value(item) for item in logs) if log["id"] == recorded.request_id]

    return eventually(logged, lambda logs: len(logs) == 1, seconds=70)[0]


def test_the_guardrail_monitor_shows_no_reason_for_a_passed_run(rig: Rig) -> None:
    mark: Final = _mark()
    response_id, answer = _sync_chat(rig, f"{mark} what is the capital of France?")
    assert answer == _answer(mark)

    log: Final = _monitor_log(rig, _by_response(response_id, PASSING))
    assert (log["action"], log["reason"]) == ("passed", None), log


def test_the_guardrail_monitor_shows_no_reason_for_a_blocked_run(rig: Rig) -> None:
    mark: Final = _mark()
    key: Final = rig.scenario.key()
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": rig.chat_model,
            "messages": [{"role": "user", "content": f"{mark} key {AWS_KEY}"}],
            "guardrails": [BLOCKING],
        },
        key=key,
    )
    assert response.status_code == 400, response.text

    log: Final = _monitor_log(rig, _by_key(key, BLOCKING))
    assert (log["action"], log["reason"]) == ("blocked", None), log
