"""Scripted edges for the Claude Code auto-mode safeguards cells: the backends behind /v1/messages and the classifier.

Every cell talks to one owned proxy whose deployments all point at one ``wire_server``. The server answers each POST
from the script a cell registered under the ``nonce-<hex>`` token the cell put in its prompt, so cells share the proxy
while each one controls its own backend reply and its own classifier verdicts, and reads back exactly the requests the
proxy sent for it.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import yaml
from integration._support.client import (
    Gateway,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, OwnedState, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

CHAT_MODEL: Final = "safeguards-chat"
RESPONSES_MODEL: Final = "safeguards-responses"
NATIVE_MODEL: Final = "safeguards-native"
CLAUDE_MODEL: Final = "safeguards-claude"
CLASSIFIER_MODEL: Final = "safeguards-classifier"
CLASSIFIER_TARGET: Final = "/classifier/v1/chat/completions"
CHAT_TARGET: Final = "/fireworks/inference/v1/chat/completions"
RESPONSES_TARGET: Final = "/openai/v1/responses"
NATIVE_TARGET: Final = "/deepseek/anthropic/v1/messages"
CLAUDE_TARGET: Final = "/anthropic/v1/messages"
TAVILY_TARGET: Final = "/tavily/search"
DANGEROUS_TOOL_USE_BETA: Final = "dangerous-tool-use-2026-09-03"
BASH_TOOL: Final[Mapping[str, JsonValue]] = {
    "name": "Bash",
    "description": "Run a shell command",
    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
}
CLASSIFIER_CONTEXT: Final[Mapping[str, JsonValue]] = {
    "v": 1,
    "permission_mode": "auto",
    "platform": "linux",
    "live_cwd": "/work/repo",
    "git_state": {"cwd": "/work/repo", "branch": "main", "remote_url": "git@github.com:example-org/repo.git"},
    "session_token": "never-forwarded",
}
SAFEGUARDS: Final[Sequence[JsonValue]] = (
    {"type": "dangerous_tool_use", "classifier_context": dict(CLASSIFIER_CONTEXT)},
)
DANGEROUS_COMMAND: Final = "rm -rf /"
CLASSIFIER_USAGE: Final[Mapping[str, JsonValue]] = {
    "prompt_tokens": 2000,
    "completion_tokens": 200,
    "total_tokens": 2200,
}
UNSUPPORTED: Final[Sequence[JsonValue]] = ({"type": "dangerous_tool_use", "status": {"type": "unsupported"}},)

OTHER_NATIVE_MODELS: Final[Mapping[str, str]] = {
    "safeguards-tencent": "/tencent/v1/messages",
    "safeguards-minimax": "/minimax/v1/messages",
    "safeguards-mantle": "/mantle/anthropic/v1/messages",
    "safeguards-openai-like": "/openailike/v1/messages",
    "safeguards-edenai": "/edenai/v3/v1/messages",
}

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_NONCE: Final = re.compile(rb"nonce-[0-9a-f]{32}")
_MODEL_DISCOVERY_BODY: Final = json.dumps({"object": "list", "data": []}).encode()


class Omitted:
    pass


OMITTED: Final = Omitted()


def no_environment(wire_url: str) -> Mapping[str, str]:
    return {}


TopLevel = Callable[[str], Mapping[str, JsonValue]]


def no_top_level(wire_url: str) -> Mapping[str, JsonValue]:
    return {}


ExtraModels = Callable[[str], Sequence[Mapping[str, JsonValue]]]


def no_extra_models(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return ()


def nonce() -> str:
    return f"nonce-{uuid.uuid4().hex}"


def tool_id(request_nonce: str) -> str:
    return f"toolu_{request_nonce.removeprefix('nonce-')[:12]}"


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


Responder = Callable[[Request], Reply]


@dataclass(frozen=True, slots=True)
class Script:
    backend: Responder
    classifier: Responder


class Scripts:
    """The replies each cell registered for its nonce; the wire thread reads them while the cells write them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_nonce: dict[str, Script] = {}  # mutable-ok: cells register scripts while the shared proxy serves

    def register(self, request_nonce: str, backend: Responder, classifier: Responder | None = None) -> None:
        with self._lock:
            self._by_nonce[request_nonce] = Script(backend, classifier or classifier_flagging_everything)

    def lookup(self, body: bytes) -> Script | None:
        match: Final = _NONCE.search(body)
        if match is None:
            return None
        with self._lock:
            return self._by_nonce.get(match.group().decode())


def tavily_reply(request: Request) -> Reply:
    query: Final = string_value(request_json(request).get("query", ""))
    results: Final = [{"title": "Build cleanup", "url": "https://example.test/clean", "content": f"Found for {query}"}]
    return Reply(body=json.dumps({"query": query, "results": results}).encode())


def dispatch(scripts: Scripts) -> Responder:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=_MODEL_DISCOVERY_BODY)
        if request.target.startswith(TAVILY_TARGET):
            return tavily_reply(request)
        script: Final = scripts.lookup(request.body)
        if script is None:
            return Reply(
                status=500, body=json.dumps({"error": {"message": f"no script for {request.target}"}}).encode()
            )
        if request.target.startswith("/classifier/"):
            return script.classifier(request)
        return script.backend(request)

    return respond


def deployment(name: str, model: str, api_base: str, **extra: JsonValue) -> Mapping[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": api_base, "api_key": f"synthetic-{name}-key", **extra},
    }


def deployments(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return (
        deployment(
            CHAT_MODEL, "fireworks_ai/accounts/fireworks/models/kimi-k2p5", f"{wire_url}/fireworks/inference/v1"
        ),
        deployment(RESPONSES_MODEL, "openai/gpt-5.4", f"{wire_url}/openai/v1"),
        deployment(NATIVE_MODEL, "deepseek/deepseek-v4-pro", f"{wire_url}/deepseek"),
        deployment(CLAUDE_MODEL, "anthropic/claude-sonnet-4-6", f"{wire_url}/anthropic"),
        deployment(CLASSIFIER_MODEL, "openai/gpt-5.4-mini", f"{wire_url}/classifier/v1"),
        deployment("safeguards-tencent", "tencent/deepseek-v4-pro", f"{wire_url}/tencent"),
        deployment("safeguards-minimax", "minimax/MiniMax-M2.5", f"{wire_url}/minimax"),
        deployment("safeguards-mantle", "bedrock/mantle/deepseek.v3-v1:0", f"{wire_url}/mantle"),
        {
            **deployment("safeguards-openai-like", "hosted_vllm/qwen3-coder", f"{wire_url}/openailike"),
            "model_info": {"supported_endpoints": ["/v1/messages"]},
        },
        deployment("safeguards-edenai", "edenai/openai/gpt-4o", f"{wire_url}/edenai/v3"),
    )


def write_config(
    directory: Path,
    wire_url: str,
    *,
    classifier_model: JsonValue | Omitted = CLASSIFIER_MODEL,
    general_settings: Mapping[str, JsonValue] | None = None,
    extra_models: ExtraModels = no_extra_models,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    top_level: TopLevel = no_top_level,
) -> Path:
    base: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    classifier_setting: Final[Mapping[str, JsonValue]] = (
        {} if isinstance(classifier_model, Omitted) else {"safeguards_classifier_model": classifier_model}
    )
    configuration: Final = {
        **base,
        **top_level(wire_url),
        "model_list": [*list_value(base["model_list"]), *deployments(wire_url), *extra_models(wire_url)],
        "general_settings": {
            **object_value(base["general_settings"]),
            **classifier_setting,
            **(general_settings or {}),
        },
        "litellm_settings": {**object_value(base["litellm_settings"]), **(litellm_settings or {})},
    }
    path: Final = directory / f"safeguards-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    wire: Wire
    scripts: Scripts
    owned: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.owned.gateway

    def register(self, request_nonce: str, backend: Responder, classifier: Responder | None = None) -> None:
        self.scripts.register(request_nonce, backend, classifier)

    def post(
        self,
        body: Mapping[str, JsonValue],
        *,
        key: str | None = None,
        beta: str | None = DANGEROUS_TOOL_USE_BETA,
        path: str = "/v1/messages",
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        return self.gateway.client.post(
            path, json=body, headers={**self.headers(key, beta), **(headers or {})}, timeout=120
        )

    def stream(
        self,
        body: Mapping[str, JsonValue],
        *,
        key: str | None = None,
        beta: str | None = DANGEROUS_TOOL_USE_BETA,
        path: str = "/v1/messages",
    ) -> tuple[int, bytes]:
        with self.gateway.client.stream(
            "POST", path, json={**body, "stream": True}, headers=self.headers(key, beta), timeout=120
        ) as response:
            return response.status_code, response.read()

    def headers(self, key: str | None, beta: str | None) -> Mapping[str, str]:
        authorization: Final = {"Authorization": f"Bearer {key or self.gateway.key}"}
        return authorization if beta is None else {**authorization, "anthropic-beta": beta}

    def received(self, request_nonce: str) -> tuple[Request, ...]:
        return tuple(request for request in self.wire.drain() if request_nonce.encode() in request.body)

    def classifier_calls(self, requests: Sequence[Request]) -> tuple[Request, ...]:
        return tuple(request for request in requests if request.target == CLASSIFIER_TARGET)

    def backend_calls(self, requests: Sequence[Request]) -> tuple[Request, ...]:
        return tuple(request for request in requests if request.target != CLASSIFIER_TARGET)


@contextmanager
def safeguards_rig(
    directory: Path,
    *,
    environment: Callable[[str], Mapping[str, str]] = no_environment,
    remove_environment: tuple[str, ...] = (),
    workers: int = 2,
    classifier_model: JsonValue | Omitted = CLASSIFIER_MODEL,
    general_settings: Mapping[str, JsonValue] | None = None,
    extra_models: ExtraModels = no_extra_models,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    top_level: TopLevel = no_top_level,
) -> Iterator[Rig]:
    scripts: Final = Scripts()
    with gateway_from_environment() as shared, wire_server(dispatch(scripts)) as wire:
        path: Final = write_config(
            directory,
            wire.url,
            classifier_model=classifier_model,
            general_settings=general_settings,
            extra_models=extra_models,
            litellm_settings=litellm_settings,
            top_level=top_level,
        )
        with owned_proxy_process(
            shared,
            directory,
            dict(environment(wire.url)),
            config=path,
            remove_environment=remove_environment,
            workers=workers,
            state=OwnedState.SHARED_WITH_WORKER,
        ) as owned:
            yield Rig(wire, scripts, owned)


def messages_body(
    model: str,
    request_nonce: str,
    *,
    safeguards: JsonValue | None = None,
    tools: Sequence[JsonValue] | None = None,
    prompt: str = "clean the build directory",
    max_tokens: int = 256,
    **extra: JsonValue,
) -> Mapping[str, JsonValue]:
    body: Final[Mapping[str, JsonValue]] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": f"{prompt} {request_nonce}"}],
        "tools": list(tools) if tools is not None else [dict(BASH_TOOL)],
        **extra,
    }
    return body if safeguards is None else {**body, "safeguards": safeguards}


def auto_mode_body(model: str, request_nonce: str, **extra: JsonValue) -> Mapping[str, JsonValue]:
    return messages_body(model, request_nonce, safeguards=list(SAFEGUARDS), **extra)


def json_body(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def request_json(request: Request) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def sse_events(raw: bytes) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line.removeprefix(b"data: "))
        for line in raw.split(b"\n")
        if line.startswith(b"data: ") and line != b"data: [DONE]"
    )


def stream_results(raw: bytes) -> JsonValue | None:
    deltas: Final = tuple(
        object_value(event["delta"]) for event in sse_events(raw) if event.get("type") == "message_delta"
    )
    assert len(deltas) == 1, deltas
    return deltas[0].get("safeguard_results")


def stream_tool_use_ids(raw: bytes) -> tuple[str, ...]:
    starts: Final = tuple(
        object_value(event["content_block"]) for event in sse_events(raw) if event.get("type") == "content_block_start"
    )
    return tuple(string_value(block["id"]) for block in starts if block.get("type") == "tool_use")


def tool_use_ids(response_body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    blocks: Final = tuple(object_value(block) for block in list_value(response_body["content"]))
    return tuple(string_value(block["id"]) for block in blocks if block.get("type") == "tool_use")


def flagged(ids: Sequence[str]) -> Sequence[JsonValue]:
    verdicts: Final[Mapping[str, JsonValue]] = {
        identity: {"type": "evaluated", "outcome": "flagged", "explanation": f"destructive {identity}"}
        for identity in ids
    }
    return available(verdicts)


def available(tool_uses: Mapping[str, JsonValue]) -> Sequence[JsonValue]:
    return [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": dict(tool_uses)}}]


def unavailable(ids: Sequence[str], reason: str) -> Sequence[JsonValue]:
    return available({identity: {"type": "unavailable", "reason": reason} for identity in ids})


def ids_under_review(classifier_request: Request) -> tuple[str, ...]:
    under_review: Final = list_value(
        json.loads(classifier_prompt(classifier_request).split("Tool calls to classify:\n", 1)[1])
    )
    return tuple(string_value(object_value(entry)["id"]) for entry in under_review)


def classifier_prompt(classifier_request: Request) -> str:
    user: Final = object_value(list_value(request_json(classifier_request)["messages"])[-1])
    return string_value(user["content"])


def requested_max_tokens(body: Mapping[str, JsonValue]) -> JsonValue:
    return body["max_completion_tokens"] if "max_completion_tokens" in body else body.get("max_tokens")


def one_classifier_call(rig: Rig, request_nonce: str) -> tuple[Request, tuple[Request, ...]]:
    received: Final = rig.received(request_nonce)
    classifier_calls: Final = rig.classifier_calls(received)
    assert len(classifier_calls) == 1, [request.target for request in received]
    return classifier_calls[0], rig.backend_calls(received)


def chat_completion_reply(
    text: str, *, model: str = "gpt-5.4-mini", usage: Mapping[str, JsonValue] = CLASSIFIER_USAGE
) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": dict(usage),
            }
        ).encode()
    )


def classifier_flagging_everything(request: Request) -> Reply:
    ids: Final = ids_under_review(request)
    verdicts: Final = {identity: {"flagged": True, "explanation": f"destructive {identity}"} for identity in ids}
    return chat_completion_reply(json.dumps({"verdicts": verdicts}))


def classifier_answering(text: str) -> Responder:
    def respond(request: Request) -> Reply:
        return chat_completion_reply(text)

    return respond


def classifier_failing(
    status: int, *, message: str = "", error_type: str = "server_error", code: str | None = None
) -> Responder:
    def respond(request: Request) -> Reply:
        error: Final[Mapping[str, JsonValue]] = {
            "message": message or f"scripted {status}",
            "type": error_type,
            "code": code,
        }
        return Reply(status=status, body=json.dumps({"error": error}).encode())

    return respond


def classifier_dropping_the_connection(request: Request) -> Reply:
    return Reply(drop_connection=True)


def classifier_slow(seconds: float) -> Responder:
    def respond(request: Request) -> Reply:
        reply: Final = classifier_flagging_everything(request)
        return Reply(chunks=(reply.body[:1], reply.body[1:]), pause_between_chunks=seconds)

    return respond


def _chat_tool_call(identity: str, name: str, arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _data_frame(event: Mapping[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


def _chat_tool_call_frames(
    chunk: Mapping[str, JsonValue], call: Mapping[str, JsonValue], name: str, command: str
) -> bytes:
    opening: Final = {"index": 0, "id": call["id"], "type": "function", "function": {"name": name, "arguments": ""}}
    arguments: Final = {"index": 0, "function": {"arguments": command}}
    return _data_frame({**chunk, "choices": [{"index": 0, "delta": {"tool_calls": [opening]}}]}) + _data_frame(
        {**chunk, "choices": [{"index": 0, "delta": {"tool_calls": [arguments]}}]}
    )


def chat_backend(
    identity: str,
    *,
    stream: bool,
    name: str = "Bash",
    arguments: Mapping[str, JsonValue] | None = None,
    text: str | None = None,
    finish_reason: str = "tool_calls",
) -> Responder:
    command: Final[Mapping[str, JsonValue]] = arguments if arguments is not None else {"command": DANGEROUS_COMMAND}
    tool_calls: Final = [] if finish_reason == "stop" else [_chat_tool_call(identity, name, command)]
    usage: Final[Mapping[str, JsonValue]] = {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}
    base: Final = {"id": f"chatcmpl-{identity}", "object": "chat.completion", "created": 1, "model": "kimi-k2p5"}

    def respond(request: Request) -> Reply:
        if not stream:
            message: Final = {"role": "assistant", "content": text, "tool_calls": tool_calls or None}
            choice: Final = {"index": 0, "message": message, "finish_reason": finish_reason}
            return Reply(body=json.dumps({**base, "choices": [choice], "usage": usage}).encode())
        chunk: Final = {**base, "object": "chat.completion.chunk"}
        frames: Final = (
            _data_frame({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text or ""}}]}),
            *(_chat_tool_call_frames(chunk, call, name, json.dumps(command)) for call in tool_calls),
            _data_frame(
                {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": usage}
            )
            + b"data: [DONE]\n\n",
        )
        return Reply(content_type="text/event-stream", chunks=frames)

    return respond


def _responses_event(sequence_number: int, event: Mapping[str, JsonValue]) -> bytes:
    payload: Final = {**event, "sequence_number": sequence_number}
    return f"event: {event['type']}\ndata: {json.dumps(payload)}\n\n".encode()


def responses_backend(
    identity: str,
    *,
    stream: bool,
    name: str = "Bash",
    arguments: Mapping[str, JsonValue] | None = None,
    text: str | None = None,
    incomplete: bool = False,
) -> Responder:
    command: Final[Mapping[str, JsonValue]] = arguments if arguments is not None else {"command": DANGEROUS_COMMAND}
    item_id: Final = f"fc_{identity}"
    call_item: Final[Mapping[str, JsonValue]] = {
        "type": "function_call",
        "id": item_id,
        "call_id": identity,
        "name": name,
        "arguments": json.dumps(command),
        "status": "completed",
    }
    message_item: Final[Mapping[str, JsonValue]] = {
        "type": "message",
        "id": f"msg_{identity}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text or "", "annotations": []}],
    }
    output: Final[list[JsonValue]] = [message_item] if text is not None else [call_item]
    usage: Final[Mapping[str, JsonValue]] = {"input_tokens": 40, "output_tokens": 9, "total_tokens": 49}
    final_status: Final = "incomplete" if incomplete else "completed"
    details: Final[Mapping[str, JsonValue]] = (
        {"incomplete_details": {"reason": "max_output_tokens"}} if incomplete else {}
    )

    def response(status: str, with_output: bool) -> Mapping[str, JsonValue]:
        return {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1,
            "status": status,
            "model": "gpt-5.4",
            "output": output if with_output else [],
            "usage": usage if with_output else None,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            **(details if with_output else {}),
        }

    def item_frames() -> tuple[bytes, ...]:
        if text is not None:
            delta: Final = {"item_id": f"msg_{identity}", "output_index": 0, "content_index": 0, "delta": text}
            return (
                _responses_event(2, {"type": "response.output_item.added", "output_index": 0, "item": message_item}),
                _responses_event(3, {"type": "response.output_text.delta", **delta}),
                _responses_event(4, {"type": "response.output_item.done", "output_index": 0, "item": message_item}),
            )
        added: Final = {**call_item, "arguments": "", "status": "in_progress"}
        arguments_delta: Final = {"item_id": item_id, "output_index": 0, "delta": json.dumps(command)}
        arguments_done: Final = {"item_id": item_id, "output_index": 0, "arguments": json.dumps(command)}
        return (
            _responses_event(2, {"type": "response.output_item.added", "output_index": 0, "item": added}),
            _responses_event(3, {"type": "response.function_call_arguments.delta", **arguments_delta}),
            _responses_event(4, {"type": "response.function_call_arguments.done", **arguments_done}),
            _responses_event(5, {"type": "response.output_item.done", "output_index": 0, "item": call_item}),
        )

    def respond(request: Request) -> Reply:
        if not stream:
            return Reply(body=json.dumps(response(final_status, True)).encode())
        frames: Final = (
            _responses_event(0, {"type": "response.created", "response": response("in_progress", False)}),
            _responses_event(1, {"type": "response.in_progress", "response": response("in_progress", False)}),
            *item_frames(),
            _responses_event(6, {"type": f"response.{final_status}", "response": response(final_status, True)}),
        )
        return Reply(content_type="text/event-stream", chunks=frames)

    return respond


def sse(event: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def native_message(
    identity: str,
    *,
    content: Sequence[Mapping[str, JsonValue]],
    stop_reason: str = "tool_use",
    model: str = "deepseek-v4-pro",
    **extra: JsonValue,
) -> Mapping[str, JsonValue]:
    return {
        "id": f"msg_{identity}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [dict(block) for block in content],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 40, "output_tokens": 9},
        **extra,
    }


def tool_use_block(
    identity: str, *, name: str = "Bash", arguments: Mapping[str, JsonValue] | None = None
) -> Mapping[str, JsonValue]:
    command: Final[Mapping[str, JsonValue]] = arguments if arguments is not None else {"command": DANGEROUS_COMMAND}
    return {"type": "tool_use", "id": identity, "name": name, "input": dict(command)}


def text_block(text: str) -> Mapping[str, JsonValue]:
    return {"type": "text", "text": text}


def native_stream_frames(
    message: Mapping[str, JsonValue], *, delta_extra: Mapping[str, JsonValue] | None = None
) -> tuple[bytes, ...]:
    blocks: Final = tuple(object_value(block) for block in list_value(message["content"]))
    usage: Final = object_value(message["usage"])
    opening: Final = sse(
        {
            "type": "message_start",
            "message": {**message, "content": [], "stop_reason": None, "usage": {**usage, "output_tokens": 1}},
        }
    )
    block_frames: Final = tuple(_native_block_frames(index, block) for index, block in enumerate(blocks))
    closing: Final = (
        sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None, **(delta_extra or {})},
                "usage": {"output_tokens": usage["output_tokens"]},
            }
        ),
        sse({"type": "message_stop"}),
    )
    return (opening, *(frame for frames in block_frames for frame in frames), *closing)


def _native_block_frames(index: int, block: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    if block["type"] == "tool_use":
        return (
            sse({"type": "content_block_start", "index": index, "content_block": {**block, "input": {}}}),
            sse(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
                }
            ),
            sse({"type": "content_block_stop", "index": index}),
        )
    return (
        sse({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}),
        sse({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}}),
        sse({"type": "content_block_stop", "index": index}),
    )


def native_backend(
    message: Mapping[str, JsonValue], *, stream: bool, chunks: Sequence[bytes] | None = None
) -> Responder:
    def respond(request: Request) -> Reply:
        if not stream:
            return Reply(body=json.dumps(message).encode())
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(chunks) if chunks is not None else native_stream_frames(message),
        )

    return respond


def spend_rows(caller_key: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        read_rows(
            'SELECT model, call_type, spend, api_key, request_tags, cache_hit FROM "LiteLLM_SpendLogs"'
            ' WHERE api_key = %s ORDER BY "startTime"',
            (key_hash(caller_key),),
        )
    )


def spend_rows_for(caller_key: str, count: int, seconds: float = 60) -> tuple[Mapping[str, object], ...]:
    return tuple(eventually(lambda: spend_rows(caller_key), lambda rows: len(rows) >= count, seconds=seconds))


def classifier_rows(rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(row for row in rows if "gpt-5.4-mini" in str(row["model"]))
