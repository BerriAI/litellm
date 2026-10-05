import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, object_value
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

ATTACK_MARKER: Final = "synthetic-attack-marker"
MODERATION_MARKER: Final = "synthetic-moderation-marker"
AZURE_ERROR_MARKERS: Final = ("AZURE_500", "AZURE_403", "AZURE_404")
PROVIDER_401_MARKER: Final = "PROVIDER_401"
OVERSIZED_MARKER: Final = "OVERSIZED_INPUT"
SHIELD_TARGET_PREFIX: Final = "/contentsafety/text:shieldPrompt?api-version="
ANALYZE_TARGET_PREFIX: Final = "/contentsafety/text:analyze?api-version="

HOOKS_SOURCE: Final = """from __future__ import annotations

from typing import Final, cast

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.azure.prompt_shield import (
    AzureContentSafetyPromptShieldGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.azure.text_moderation import (
    AzureContentSafetyTextModerationGuardrail,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypesLiteral


class TupleWriter(CustomGuardrail):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        messages: Final = data.get("messages")
        if isinstance(messages, list):
            data["messages"] = tuple(messages)  # mutable-ok: the test hook rewrites messages to a tuple
        return data


class AllTurnsPromptShield(AzureContentSafetyPromptShieldGuardrail):
    def get_user_prompt(self, messages: list[AllMessageValues]) -> str:
        return "\\n".join(
            message["content"]
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        )


class AllTurnsTextModeration(AzureContentSafetyTextModerationGuardrail):
    def get_user_prompt(self, messages: list[AllMessageValues]) -> str:
        return "\\n".join(
            message["content"]
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        )


class RequiringPromptShield(AzureContentSafetyPromptShieldGuardrail):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object] | None:
        messages: Final = data.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="no user text")
        user_prompt: Final = self.get_user_prompt(cast(list[AllMessageValues], messages))  # cast-ok: chat messages
        if not user_prompt:
            raise HTTPException(status_code=400, detail="no user text")
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type)


class RequiringTextModeration(AzureContentSafetyTextModerationGuardrail):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object] | None:
        messages: Final = data.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="no user text")
        user_prompt: Final = self.get_user_prompt(cast(list[AllMessageValues], messages))  # cast-ok: chat messages
        if not user_prompt:
            raise HTTPException(status_code=400, detail="no user text")
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type)
"""


@dataclass(frozen=True, slots=True)
class AzureBehavior:
    delay_seconds: float = 0
    down: threading.Event | None = None
    entered: threading.Event | None = None
    arrived: threading.Semaphore | None = None
    release: threading.Event | None = None
    barrier_marker: str | None = None


def azure_text(request: Request) -> str:
    body: Final = object_value(json.loads(request.body))
    if request.target.startswith(SHIELD_TARGET_PREFIX):
        prompt: Final = body["userPrompt"]
        assert isinstance(prompt, str), body
        return prompt
    assert request.target.startswith(ANALYZE_TARGET_PREFIX), request.target
    text: Final = body["text"]
    assert isinstance(text, str), body
    return text


def azure_texts(azure: Wire) -> tuple[str, ...]:
    return tuple(azure_text(request) for request in azure.drain())


def provider_text(request: Request) -> str:
    body: Final = object_value(json.loads(request.body)) if request.body else {}
    target: Final = request.target.split("?", 1)[0]
    match target:
        case "/v1/chat/completions" | "/v1/messages":
            messages: Final = body["messages"]
            assert isinstance(messages, list), body
            return "\n".join(
                message_text(message["content"])
                for message in messages
                if isinstance(message, dict)
                and message.get("role") == "user"
                and "content" in message
            )
        case "/v1/responses":
            value: Final = body["input"]
            if not isinstance(value, list):
                return message_text(value)
            return "\n".join(
                message_text(message["content"])
                for message in value
                if isinstance(message, dict)
                and message.get("role") == "user"
                and "content" in message
            )
        case "/v1/embeddings":
            return message_text(body["input"])
        case "/v1/completions":
            return message_text(body["prompt"])
        case _:
            raise AssertionError(f"Unexpected provider target {request.target}")


def message_text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(part["text"])
            for part in value
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def provider_texts(provider: Wire) -> tuple[str, ...]:
    requests: Final = tuple(
        request
        for request in provider.drain()
        if request.method != "GET" or request.target.split("?", 1)[0] != "/v1/models"
    )
    return tuple(provider_text(request) for request in requests)


def provider_messages(provider: Wire) -> tuple[JsonValue, ...]:
    requests: Final = tuple(
        request
        for request in provider.drain()
        if request.method != "GET" or request.target.split("?", 1)[0] != "/v1/models"
    )
    targets: Final = tuple(request.target.split("?", 1)[0] for request in requests)
    assert all(target == "/v1/chat/completions" for target in targets), targets
    return tuple(object_value(json.loads(request.body))["messages"] for request in requests)


def azure_handler(behavior: AzureBehavior = AzureBehavior()) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request.method
        text: Final = azure_text(request)
        if behavior.entered is not None and (
            behavior.barrier_marker is None or behavior.barrier_marker in text
        ):
            behavior.entered.set()
            if behavior.arrived is not None:
                behavior.arrived.release()
        if behavior.release is not None and (
            behavior.barrier_marker is None or behavior.barrier_marker in text
        ):
            assert behavior.release.wait(timeout=30), "Azure barrier was not released"
        if behavior.down is not None and behavior.down.is_set():
            return Reply(status=503, body=b'{"error":"synthetic Azure outage"}')
        if behavior.delay_seconds:
            time.sleep(behavior.delay_seconds)
        status: Final = next(
            (code for marker, code in (("AZURE_500", 500), ("AZURE_403", 403), ("AZURE_404", 404)) if marker in text),
            200,
        )
        if status != 200:
            return Reply(
                status=status,
                body=json.dumps({"error": {"message": f"synthetic Azure error {status}"}}).encode(),
            )
        if request.target.startswith(SHIELD_TARGET_PREFIX):
            return Reply(
                body=json.dumps(
                    {
                        "userPromptAnalysis": {"attackDetected": ATTACK_MARKER in text},
                        "documentsAnalysis": [],
                    }
                ).encode()
            )
        assert request.target.startswith(ANALYZE_TARGET_PREFIX), request.target
        severity: Final = 4 if MODERATION_MARKER in text else 0
        return Reply(
            body=json.dumps(
                {
                    "blocklistsMatch": [],
                    "categoriesAnalysis": [
                        {"category": "Hate", "severity": severity},
                        {"category": "Sexual", "severity": 0},
                        {"category": "SelfHarm", "severity": 0},
                        {"category": "Violence", "severity": 0},
                    ],
                }
            ).encode()
        )

    return respond


def provider_handler(request: Request) -> Reply:
    path: Final = request.target.split("?", 1)[0]
    if request.method == "GET" and path == "/v1/models":
        return Reply(body=b'{"data":[]}')
    assert request.method == "POST", request.method
    text: Final = provider_text(request)
    if PROVIDER_401_MARKER in text:
        return Reply(status=401, body=b'{"error":{"message":"synthetic provider unauthorized"}}')
    if OVERSIZED_MARKER in text:
        return Reply(
            status=400,
            body=b'{"error":{"message":"synthetic context length exceeded","code":"context_length_exceeded"}}',
        )
    identity: Final = uuid.uuid4().hex
    match path:
        case "/v1/chat/completions":
            if b'"stream":true' in request.body.replace(b" ", b""):
                return Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))
            return Reply(
                body=json.dumps(
                    {
                "id": "chatcmpl-" + identity,
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "permitted response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                    }
                ).encode()
            )
        case "/v1/messages":
            if b'"stream":true' in request.body.replace(b" ", b""):
                return Reply(content_type="text/event-stream", chunks=_messages_chunks(identity))
            return Reply(
                body=json.dumps(
                    {
                "id": "msg_" + identity,
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [{"type": "text", "text": "permitted response"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 2},
                    }
                ).encode()
            )
        case "/v1/responses":
            if b'"stream":true' in request.body.replace(b" ", b""):
                return Reply(content_type="text/event-stream", chunks=_responses_chunks(identity))
            return Reply(
                body=json.dumps(
                    {
                "id": "resp_" + identity,
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_" + identity,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "permitted response", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
                    }
                ).encode()
            )
        case "/v1/embeddings":
            return Reply(
                body=json.dumps(
                    {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                ).encode()
            )
        case "/v1/completions":
            return Reply(
                body=json.dumps(
                    {
                "id": "cmpl-" + identity,
                "object": "text_completion",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [{"text": "permitted response", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    }
                ).encode()
            )
        case _:
            return Reply(status=404, body=json.dumps({"error": "unexpected provider target " + path}).encode())


def _chat_chunks(identity: str) -> tuple[bytes, ...]:
    return (
        _sse({"id": "chatcmpl-" + identity, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "permitted "}, "finish_reason": None}]}),
        _sse({"id": "chatcmpl-" + identity, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "response"}, "finish_reason": None}]}),
        _sse({"id": "chatcmpl-" + identity, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    )


def _messages_chunks(identity: str) -> tuple[bytes, ...]:
    return (
        _event("message_start", {"type": "message_start", "message": {"id": "msg_" + identity, "type": "message", "role": "assistant", "model": "claude-sonnet-4-5-20250929", "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 2, "output_tokens": 0}}}),
        _event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        _event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "permitted response"}}),
        _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _event("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}}),
        _event("message_stop", {"type": "message_stop"}),
    )


def _responses_chunks(identity: str) -> tuple[bytes, ...]:
    response: Final = {
        "id": "resp_" + identity,
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_" + identity,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "permitted response", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
    }
    return (
        _event("response.created", {"type": "response.created", "response": response}),
        _event("response.output_text.delta", {"type": "response.output_text.delta", "delta": "permitted response"}),
        _event("response.completed", {"type": "response.completed", "response": response}),
    )


def _sse(value: dict[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def _event(name: str, value: dict[str, JsonValue]) -> bytes:
    return f"event: {name}\n".encode() + _sse(value)


def guardrail_configs(azure_url: str, *, default_on: bool = False) -> tuple[dict[str, JsonValue], ...]:
    return (
        {
            "guardrail_name": "tuple-writer",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.TupleWriter",
                "mode": "pre_call",
                "default_on": default_on,
            },
        },
        {
            "guardrail_name": "shield",
            "litellm_params": {
                "guardrail": "azure/prompt_shield",
                "mode": "pre_call",
                "default_on": default_on,
                "api_base": azure_url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "moderation",
            "litellm_params": {
                "guardrail": "azure/text_moderations",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure_url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "all-turns-shield",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.AllTurnsPromptShield",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure_url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "all-turns-moderation",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.AllTurnsTextModeration",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure_url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "requiring-shield",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.RequiringPromptShield",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure_url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "requiring-moderation",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.RequiringTextModeration",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure_url,
                "api_key": "synthetic-azure-key",
            },
        },
    )


def write_dispatch_config(directory: Path, azure_url: str, *, default_on: bool = False) -> Path:
    (directory / "azure_dispatch_hooks.py").write_text(HOOKS_SOURCE)
    base_config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    general_settings: Final = {
        **base_config["general_settings"],
        "store_prompts_in_spend_logs": True,
    }
    config: Final = {
        **base_config,
        "guardrails": guardrail_configs(azure_url, default_on=default_on),
        "general_settings": general_settings,
    }
    config_path: Final = directory / "azure-content-safety-dispatch.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@contextmanager
def dispatch_proxy(
    gateway: Gateway,
    directory: Path,
    redis: OwnedRedis,
    azure_url: str,
    *,
    workers: int = 2,
    default_on: bool = False,
) -> Iterator[OwnedProxy]:
    config: Final = write_dispatch_config(directory, azure_url, default_on=default_on)
    with owned_proxy_process(
        gateway,
        directory,
        {
            "REDIS_HOST": redis.host,
            "REDIS_PORT": str(redis.port),
            "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
        },
        config=config,
        workers=workers,
    ) as owned:
        yield owned


@contextmanager
def dispatch_rig(
    gateway: Gateway,
    directory: Path,
    *,
    workers: int = 2,
    default_on: bool = False,
    behavior: AzureBehavior = AzureBehavior(),
) -> Iterator[tuple[OwnedProxy, Wire, Wire, OwnedRedis]]:
    with ExitStack() as stack:
        redis: Final = stack.enter_context(owned_redis(directory))
        azure: Final = stack.enter_context(wire_server(azure_handler(behavior)))
        provider: Final = stack.enter_context(wire_server(provider_handler))
        owned: Final = stack.enter_context(
            dispatch_proxy(gateway, directory, redis, azure.url, workers=workers, default_on=default_on)
        )
        yield owned, azure, provider, redis
