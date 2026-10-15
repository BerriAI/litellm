"""`litellm_content_filter` in `mode: logging_only`, driven through a real proxy.

The provider is the only double, an owned wire peer. The proxy, its guardrail registry, Postgres and Redis run
for real with two workers. Every check reads the spend row, which is written after the logging-only scans finish
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import openai
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

AWS_KEY: Final = "AKIA" + "SYNTHETIC0000000"
GITHUB_TOKEN: Final = "ghp_" + "a1" * 18
FORBIDDEN_WORD: Final = "synthetic-forbidden-word"
LEAK_MARK: Final = "SYNTHETIC-LEAK-TOKEN"
OBSERVE_BLOCK: Final = "cf-observe-block"
OBSERVE_MASK: Final = "cf-observe-mask"
ENFORCE_BLOCK: Final = "cf-enforce-block"
SECRETS: Final = f"AWS_ACCESS_KEY_ID={AWS_KEY}\nGITHUB_TOKEN={GITHUB_TOKEN}"


def _answer(marker: str) -> str:
    return "synthetic answer " + marker


def _provider_answer(body: bytes) -> str:
    text: Final = body.decode()
    start: Final = text.find("mark-")
    assert start >= 0, text
    answer: Final = _answer(text[start : start + 37])
    return answer + " " + GITHUB_TOKEN if LEAK_MARK in text else answer


def _sse(events: tuple[dict[str, object], ...]) -> tuple[bytes, ...]:
    return tuple(("data: " + json.dumps(event) + "\n\n").encode() for event in events)


def _chat_reply(answer: str, streaming: bool) -> Reply:
    if not streaming:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-" + uuid.uuid4().hex,
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
    identity: Final = "chatcmpl-" + uuid.uuid4().hex

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


def _content_filter(name: str, mode: str, action: str, *, default_on: bool) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": mode,
            "default_on": default_on,
            "patterns": [
                {"pattern_type": "regex", "pattern": "AKIA[0-9A-Z]{16}", "name": "aws_access_key", "action": action},
                {"pattern_type": "regex", "pattern": "ghp_[A-Za-z0-9]{36}", "name": "github_pat", "action": action},
            ],
            "blocked_words": [{"keyword": FORBIDDEN_WORD, "action": action}],
        },
    }


def _rig_config(root: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["general_settings"]["store_prompts_in_spend_logs"] = True
    config["guardrails"] = [
        _content_filter(OBSERVE_BLOCK, "logging_only", "BLOCK", default_on=True),
        _content_filter(OBSERVE_MASK, "logging_only", "MASK", default_on=True),
        _content_filter(ENFORCE_BLOCK, "pre_call", "BLOCK", default_on=False),
    ]
    path: Final = root / "content-filter-logging-only.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    provider: Wire
    chat_model: str
    claude_model: str
    responses_model: str

    def base(self) -> str:
        return str(self.proxy.client.base_url).rstrip("/")

    def provider_bodies(self, marker: str) -> tuple[str, ...]:
        return tuple(request.body.decode() for request in self.provider.drain() if marker.encode() in request.body)

    def entries(self, response_id: str, name: str) -> tuple[dict[str, object], ...]:
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, status, metadata FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (response_id,)
            ),
            lambda values: len(values) >= 1,
            seconds=70,
        )
        assert len(rows) == 1, rows
        recorded: Final = object_value(rows[0]["metadata"]).get("guardrail_information")
        assert isinstance(recorded, list), rows[0]
        return tuple(
            entry for entry in (object_value(item) for item in recorded) if entry.get("guardrail_name") == name
        )


def _findings(entry: dict[str, object]) -> tuple[str, ...]:
    response: Final = entry.get("guardrail_response")
    assert isinstance(response, list), entry
    return tuple(
        str(detection.get("pattern_name") or detection.get("keyword"))
        for detection in (object_value(item) for item in response)
    )


def _verdicts(entries: tuple[dict[str, object], ...]) -> list[tuple[object, object, tuple[str, ...]]]:
    return [(entry["guardrail_mode"], entry["guardrail_status"], _findings(entry)) for entry in entries]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("content-filter-logging-only")
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
        responses: Final = scenario.model(
            model="openai/gpt-5.4-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
        )
        yield Rig(owned.gateway, provider, chat, claude, responses)


def _marker() -> str:
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
        client.chat.completions.create(
            model=rig.chat_model, messages=[{"role": "user", "content": text}], stream=True
        )
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
    result: Final = client.responses.create(model=rig.responses_model, input=text)
    return result.id, result.output_text


SYNC_CALLS: Final = {"chat": _sync_chat, "chat_stream": _streamed_chat, "messages": _messages, "responses": _responses}
EVERY_FINDING: Final = ("aws_access_key", "github_pat", FORBIDDEN_WORD)


@pytest.mark.parametrize("endpoint", list(SYNC_CALLS))
def test_logging_only_block_records_every_match_without_touching_the_call(rig: Rig, endpoint: str) -> None:
    marker: Final = _marker()
    prompt: Final = f"{marker} what did the file contain?\n{SECRETS}\n{FORBIDDEN_WORD}"
    response_id, answer = SYNC_CALLS[endpoint](rig, prompt)
    assert answer == _answer(marker)
    sent: Final = rig.provider_bodies(marker)
    assert len(sent) == 1 and AWS_KEY in sent[0] and GITHUB_TOKEN in sent[0] and FORBIDDEN_WORD in sent[0], sent

    assert _verdicts(rig.entries(response_id, OBSERVE_BLOCK)) == [
        ("logging_only", "guardrail_flagged", EVERY_FINDING),
        ("logging_only", "success", ()),
    ]


@pytest.mark.asyncio
async def test_logging_only_block_records_every_match_for_an_async_client(rig: Rig) -> None:
    marker: Final = _marker()
    response_id, answer = await _async_chat(rig, f"{marker} {SECRETS}")
    assert answer == _answer(marker)

    assert _verdicts(rig.entries(response_id, OBSERVE_BLOCK)) == [
        ("logging_only", "guardrail_flagged", ("aws_access_key", "github_pat")),
        ("logging_only", "success", ()),
    ]


def test_logging_only_block_still_scans_the_response_after_a_request_match(rig: Rig) -> None:
    marker: Final = _marker()
    response_id, answer = _sync_chat(rig, f"{marker} {LEAK_MARK} key {AWS_KEY}")
    assert answer == _answer(marker) + " " + GITHUB_TOKEN

    assert _verdicts(rig.entries(response_id, OBSERVE_BLOCK)) == [
        ("logging_only", "guardrail_flagged", ("aws_access_key",)),
        ("logging_only", "guardrail_flagged", ("github_pat",)),
    ]


def test_logging_only_mask_records_every_match(rig: Rig) -> None:
    marker: Final = _marker()
    response_id, answer = _sync_chat(rig, f"{marker} {SECRETS}\n{FORBIDDEN_WORD}")
    assert answer == _answer(marker)

    entries: Final = rig.entries(response_id, OBSERVE_MASK)
    assert _verdicts(entries) == [
        ("logging_only", "success", EVERY_FINDING),
        ("logging_only", "success", ()),
    ]
    assert entries[0]["masked_entity_count"] == {"aws_access_key": 1, "github_pat": 1, "blocked_word": 1}, entries


def test_logging_only_records_nothing_for_a_clean_call(rig: Rig) -> None:
    marker: Final = _marker()
    response_id, answer = _sync_chat(rig, f"{marker} nothing secret here")
    assert answer == _answer(marker)

    assert _verdicts(rig.entries(response_id, OBSERVE_BLOCK)) == [
        ("logging_only", "success", ()),
        ("logging_only", "success", ()),
    ]


def test_pre_call_block_still_rejects_the_request(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={
            "model": rig.chat_model,
            "messages": [{"role": "user", "content": f"{marker} {SECRETS}"}],
            "guardrails": [ENFORCE_BLOCK],
        },
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 400, response.text
    assert "aws_access_key" in response.text
    assert rig.provider_bodies(marker) == ()


def test_apply_guardrail_endpoint_still_blocks_with_a_logging_only_content_filter(rig: Rig) -> None:
    response: Final = rig.proxy.client.post(
        "/guardrails/apply_guardrail",
        json={"guardrail_name": OBSERVE_BLOCK, "text": f"key {AWS_KEY}"},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 400, response.text
    assert "aws_access_key" in response.text


def _advertised_modes(rig: Rig) -> dict[str, object]:
    response: Final = rig.proxy.client.get(
        "/guardrails/ui/add_guardrail_settings", headers={"Authorization": f"Bearer {rig.proxy.key}"}
    )
    assert response.status_code == 200, response.text
    return object_value(response.json()["supported_modes_by_provider"])


@pytest.mark.parametrize("provider", ["litellm_content_filter", "generic_guardrail_api", "bedrock", "akto"])
def test_dashboard_offers_logging_only_where_the_config_accepts_it(rig: Rig, provider: str) -> None:
    modes: Final = _advertised_modes(rig)
    offered: Final = modes[provider]
    assert isinstance(offered, list) and "logging_only" in offered, modes
    assert offered.count("logging_only") == 1, offered


@pytest.mark.parametrize("provider", ["azure/prompt_shield", "lakera_v2"])
def test_dashboard_does_not_offer_logging_only_where_the_config_rejects_it(rig: Rig, provider: str) -> None:
    offered: Final = _advertised_modes(rig)[provider]
    assert isinstance(offered, list) and "logging_only" not in offered, offered


def test_a_logging_only_content_filter_created_through_the_api_records_every_match(rig: Rig) -> None:
    name: Final = "cf-observe-api-" + uuid.uuid4().hex[:8]
    created: Final = rig.proxy.client.post(
        "/guardrails",
        json={"guardrail": {"guardrail_name": name, **_content_filter(name, "logging_only", "BLOCK", default_on=True)}},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert created.status_code == 200, created.text
    guardrail_id: Final = str(object_value(created.json())["guardrail_id"])
    try:
        marker: Final = _marker()
        response_id: Final = eventually(
            lambda: _sync_chat(rig, f"{marker} {SECRETS}")[0],
            lambda identity: bool(rig.entries(identity, name)),
            seconds=70,
        )
        assert _verdicts(rig.entries(response_id, name))[0] == (
            "logging_only",
            "guardrail_flagged",
            ("aws_access_key", "github_pat"),
        )
    finally:
        rig.proxy.client.delete(f"/guardrails/{guardrail_id}", headers={"Authorization": f"Bearer {rig.proxy.key}"})
