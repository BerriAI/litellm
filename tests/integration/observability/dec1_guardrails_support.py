"""Scripted guardrail peer, scripted provider, config writer and readers for the dec1 guardrail contract tests.

One rig is one owned single-worker proxy plus two owned wire servers: a policy peer that answers the Generic
Guardrail API (`POST /beta/litellm_basic_guardrail_api`) and a provider that answers OpenAI chat, OpenAI
Responses and Anthropic Messages. Each test module boots its own rig with the guardrails it needs, because the
guardrail registry is read from the config at boot and a `default_on` guardrail changes every request.

The policy peer decides from markers in the texts it is shown, so one peer serves every generic guardrail of a
rig. Each generic guardrail sends its own name as `additional_provider_specific_params.rail`, so the peer and the
tests can tell which guardrail made a call:

- `BLOCK-IN` in a request text: `{"action": "BLOCKED", "blocked_reason": "dec1 refused input by <rail>"}`
- `BLOCK-OUT` in a response text: `{"action": "BLOCKED", "blocked_reason": "dec1 refused output by <rail>"}`
- `SECRET-1234` in any text: `GUARDRAIL_INTERVENED` with every text rewritten to `[MASKED]` in its place
- `STATUS-500`, `STATUS-503`, `STATUS-400`: that HTTP status with a JSON error body
- `BAD-JSON`: status 200 with a body that is not JSON
- anything else: `{"action": "NONE"}`

The provider answers `plain answer` unless the prompt carries `REPLY-BLOCK-OUT` (answer `this reply has BLOCK-OUT
inside`) or `REPLY-SECRET` (answer `the code is SECRET-1234`). Every deployment carries custom prices of 0.001 per
input token and 0.002 per output token and the provider reports 5 prompt and 3 completion tokens, so a served chat
costs exactly 0.011.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import yaml
from integration._support.anthropic_sse import message_json, message_start, sse
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.ports import allocate_port
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

POLICY_PATH: Final = "/beta/litellm_basic_guardrail_api"
POLICY_KEY: Final = "synthetic-guardrail-key"
POLICY_KEY_MASKED: Final = "sy****ey"
PROVIDER_KEY: Final = "synthetic-dec1-guardrails-provider-key"
APPLIED_HEADER: Final = "x-litellm-applied-guardrails"
CALL_ID_HEADER: Final = "x-litellm-call-id"

INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
PROMPT_TOKENS: Final = 5
COMPLETION_TOKENS: Final = 3
CHAT_COST: Final = PROMPT_TOKENS * INPUT_RATE + COMPLETION_TOKENS * OUTPUT_RATE

PLAIN_ANSWER: Final = "plain answer"
BLOCKING_ANSWER: Final = "this reply has BLOCK-OUT inside"
SECRET_ANSWER: Final = "the code is SECRET-1234"
MASKED_SECRET_ANSWER: Final = "the code is [MASKED]"
SECRET: Final = "SECRET-1234"
MASK: Final = "[MASKED]"
STREAM_WORDS: Final = ("one ", "two ", "three ", "four ", "five ", "six ", "seven")

UPSTREAM_CHAT_MODEL: Final = "gpt-5.4-mini"
UPSTREAM_CLAUDE_MODEL: Final = "claude-haiku-5-5"

_JSON: Final = TypeAdapter(dict[str, JsonValue])


def refused_input(rail: str) -> str:
    return f"dec1 refused input by {rail}"


def refused_output(rail: str) -> str:
    return f"dec1 refused output by {rail}"


def marker() -> str:
    return "dec1-" + uuid.uuid4().hex


def _status(code: int) -> Reply:
    return Reply(status=code, body=json.dumps({"error": f"scripted guardrail status {code}"}).encode())


def policy_verdict(request: Request) -> Reply:
    assert request.target == POLICY_PATH, request.target
    body: Final = _JSON.validate_json(request.body)
    extra: Final = body.get("additional_provider_specific_params")
    rail: Final = str(extra.get("rail")) if isinstance(extra, dict) else "unknown"
    raw_texts: Final = body.get("texts")
    texts: Final = [str(text) for text in raw_texts] if isinstance(raw_texts, list) else []
    joined: Final = " ".join(texts)
    for code in (500, 503, 400):
        if f"STATUS-{code}" in joined:
            return _status(code)
    if "BAD-JSON" in joined:
        return Reply(body=b"this is not json", content_type="text/plain")
    if body.get("input_type") == "request" and "BLOCK-IN" in joined:
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": refused_input(rail)}).encode())
    if body.get("input_type") == "response" and "BLOCK-OUT" in joined:
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": refused_output(rail)}).encode())
    if SECRET in joined:
        masked: Final = [text.replace(SECRET, MASK) for text in texts]
        return Reply(body=json.dumps({"action": "GUARDRAIL_INTERVENED", "texts": masked}).encode())
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def answer_for(prompt: str) -> str:
    if "REPLY-BLOCK-OUT" in prompt:
        return BLOCKING_ANSWER
    if "REPLY-SECRET" in prompt:
        return SECRET_ANSWER
    return PLAIN_ANSWER


def _usage() -> dict[str, JsonValue]:
    return {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
    }


def _chat_frame(identity: str, delta: dict[str, JsonValue], finish: str | None, usage: bool = False) -> bytes:
    frame: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": UPSTREAM_CHAT_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        frame["usage"] = _usage()
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def chat_reply(text: str, *, stream: bool) -> Reply:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": UPSTREAM_CHAT_MODEL,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": _usage(),
                }
            ).encode()
        )
    words: Final = STREAM_WORDS if text == PLAIN_ANSWER else tuple(word + " " for word in text.split(" "))
    frames: Final = (
        _chat_frame(identity, {"role": "assistant", "content": ""}, None),
        *(_chat_frame(identity, {"content": word}, None) for word in words),
        _chat_frame(identity, {}, "stop", usage=True) + b"data: [DONE]\n\n",
    )
    return Reply(content_type="text/event-stream", chunks=frames)


def messages_reply(text: str, *, stream: bool) -> Reply:
    identity: Final = "msg_" + uuid.uuid4().hex
    if not stream:
        return Reply(body=message_json(identity, UPSTREAM_CLAUDE_MODEL, text))
    frames: Final = (
        message_start(identity, UPSTREAM_CLAUDE_MODEL),
        sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        + sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": COMPLETION_TOKENS},
            },
        )
        + sse("message_stop", {"type": "message_stop"}),
    )
    return Reply(content_type="text/event-stream", chunks=frames)


def responses_reply(text: str) -> Reply:
    identity: Final = "resp_" + uuid.uuid4().hex
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": UPSTREAM_CHAT_MODEL,
                "output": [
                    {
                        "id": "msg_" + identity,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS, "total_tokens": 8},
            }
        ).encode()
    )


def provider_answer(request: Request) -> Reply:
    body: Final = _JSON.validate_json(request.body)
    prompt: Final = request.body.decode()
    text: Final = answer_for(prompt)
    streaming: Final = body.get("stream") is True
    if request.target.endswith("/v1/messages"):
        return messages_reply(text, stream=streaming)
    if request.target.endswith("/v1/responses"):
        return responses_reply(text)
    assert request.target.endswith("/v1/chat/completions"), request.target
    return chat_reply(text, stream=streaming)


def generic(name: str, mode: str | list[str], api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    """A Generic Guardrail API guardrail that tells the peer its own name through `rail`."""
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": mode,
            "default_on": False,
            "api_base": api_base,
            "api_key": POLICY_KEY,
            "additional_provider_specific_params": {"rail": name},
            **params,
        },
    }


def dead_base() -> str:
    """An address nothing listens on, so a connect to it is refused."""
    return f"http://127.0.0.1:{allocate_port()}"


def write_config(
    directory: Path,
    guardrails: Sequence[Mapping[str, JsonValue]],
    *,
    store_prompts: bool = True,
    modules: Mapping[str, str] | None = None,
) -> Path:
    for module, source in (modules or {}).items():
        (directory / f"{module}.py").write_text(source)
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"]["cache"] = False
    configuration["general_settings"]["store_prompts_in_spend_logs"] = store_prompts
    configuration["guardrails"] = [dict(guardrail) for guardrail in guardrails]
    path: Final = directory / "dec1-guardrails.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    policy: Wire
    provider: Wire
    scenario: Scenario
    chat_model: str
    claude_model: str

    @property
    def gateway(self) -> Gateway:
        return self.owned.gateway

    def base(self) -> str:
        return str(self.gateway.client.base_url).rstrip("/")

    def policy_calls(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(_JSON.validate_json(request.body) for request in self.policy.drain())

    def provider_calls(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(_JSON.validate_json(request.body) for request in self.provider.drain())

    def drain(self) -> None:
        self.policy.drain()
        self.provider.drain()


@contextmanager
def boot_rig(
    directory: Path,
    guardrails_for: Callable[[str], Sequence[Mapping[str, JsonValue]]],
    *,
    store_prompts: bool = True,
    modules: Mapping[str, str] | None = None,
) -> Iterator[Rig]:
    with (
        wire_server(policy_verdict) as policy,
        wire_server(provider_answer) as provider,
        gateway_from_environment() as shared,
    ):
        config: Final = write_config(
            directory, guardrails_for(policy.url), store_prompts=store_prompts, modules=modules
        )
        with owned_proxy_process(shared, directory, {}, config=config) as owned, owned.gateway.scenario() as scenario:
            chat: Final = scenario.model(
                model=f"openai/{UPSTREAM_CHAT_MODEL}",
                api_base=f"{provider.url}/v1",
                api_key=PROVIDER_KEY,
                input_cost_per_token=INPUT_RATE,
                output_cost_per_token=OUTPUT_RATE,
            )
            claude: Final = scenario.model(
                model=f"anthropic/{UPSTREAM_CLAUDE_MODEL}",
                api_base=provider.url,
                api_key=PROVIDER_KEY,
                input_cost_per_token=INPUT_RATE,
                output_cost_per_token=OUTPUT_RATE,
            )
            yield Rig(owned, policy, provider, scenario, chat, claude)


def chat(
    rig: Rig,
    prompt: str,
    *,
    key: str | None = None,
    stream: bool = False,
    extra: Mapping[str, JsonValue] | None = None,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": rig.chat_model,
        "messages": [{"role": "user", "content": prompt}],
        **({"stream": True} if stream else {}),
        **(extra or {}),
    }
    return rig.gateway.request("POST", "/v1/chat/completions", body, key=key)


def messages(
    rig: Rig,
    prompt: str,
    *,
    key: str | None = None,
    stream: bool = False,
    extra: Mapping[str, JsonValue] | None = None,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": rig.claude_model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
        **({"stream": True} if stream else {}),
        **(extra or {}),
    }
    return rig.gateway.request("POST", "/v1/messages", body, key=key)


def responses(
    rig: Rig,
    prompt: str,
    *,
    key: str | None = None,
    extra: Mapping[str, JsonValue] | None = None,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": rig.chat_model, "input": prompt, **(extra or {})}
    return rig.gateway.request("POST", "/v1/responses", body, key=key)


def chat_text(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    choices: Final = response.json()["choices"]
    return str(choices[0]["message"]["content"])


def sse_payloads(text: str) -> tuple[JsonValue, ...]:
    """The decoded `data:` payloads of an OpenAI-style event stream, without the closing `[DONE]`."""
    lines: Final = tuple(line.removeprefix("data: ") for line in text.split("\n") if line.startswith("data: "))
    return tuple(json.loads(line) for line in lines if line != "[DONE]")


def streamed_content(text: str) -> str:
    parts: list[str] = []
    for payload in sse_payloads(text):
        if not isinstance(payload, dict):
            continue
        for choice in payload.get("choices") or []:
            if isinstance(choice, dict) and isinstance(choice.get("delta"), dict):
                parts.append(str(choice["delta"].get("content") or ""))
    return "".join(parts)


def rails_called(calls: Sequence[Mapping[str, JsonValue]]) -> tuple[tuple[str, str], ...]:
    """(rail, input_type) for every peer call, in arrival order."""
    return tuple(
        (str(object_value(call["additional_provider_specific_params"]).get("rail")), str(call.get("input_type")))
        for call in calls
    )


def prompts_seen_by_provider(calls: Sequence[Mapping[str, JsonValue]]) -> tuple[str, ...]:
    seen: list[str] = []
    for call in calls:
        for message in call.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                seen.append(content if isinstance(content, str) else json.dumps(content))
        if isinstance(call.get("input"), str):
            seen.append(str(call["input"]))
    return tuple(seen)


def spend_row(call_id: str, seconds: float = 70) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status, spend, metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s',
            (call_id,),
        ),
        lambda values: len(values) >= 1,
        seconds=seconds,
    )
    assert len(rows) == 1, rows
    return rows[0]


def guardrail_entries(row: Mapping[str, JsonValue], name: str) -> tuple[dict[str, JsonValue], ...]:
    recorded: Final = object_value(row["metadata"]).get("guardrail_information")
    assert isinstance(recorded, list), row
    return tuple(entry for entry in (object_value(item) for item in recorded) if entry.get("guardrail_name") == name)


def error_message(response: httpx.Response) -> str:
    return str(object_value(response.json()["error"])["message"])
