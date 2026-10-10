from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
import yaml
from anthropic import Anthropic, AsyncAnthropic
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows, write_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request, Wire
from integration._support.wire import wire_server as _wire_server
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.callback_utils import CALLBACK_VAR_ENCRYPTED_PREFIX
from litellm.proxy.guardrails.guardrail_registry import decrypt_guardrail_litellm_params
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

Endpoint = Literal["chat", "messages", "responses"]

ClientKind = Literal["openai_sync", "openai_async", "anthropic_sync", "anthropic_async", "httpx"]

Direction = Literal["request", "response"]

BASE_DEFAULT_NORMAL_DIRECTIONS: Final[Mapping[tuple[Endpoint, bool], tuple[Direction, ...]]] = MappingProxyType(
    {
        ("chat", False): ("request", "response"),
        ("chat", True): ("request", "response"),
        ("messages", False): ("request", "response"),
        ("messages", True): ("request", "response"),
        ("responses", False): ("request", "response"),
        ("responses", True): ("request", "response"),
    }
)

BASE_DEFAULT_CACHE_HIT_DIRECTIONS: Final[Mapping[Endpoint, tuple[Direction, ...]]] = MappingProxyType(
    {
        "chat": ("request", "response"),
        "messages": ("request", "response"),
        "responses": ("request", "response"),
    }
)

BASE_DEFAULT_UPSTREAM_FAILURE_DIRECTIONS: Final[Mapping[Endpoint, tuple[Direction, ...]]] = MappingProxyType(
    {
        "chat": (),
        "messages": (),
        "responses": (),
    }
)

_AUDIT_RESPONSE_IDS: Final[ContextVar[tuple[str, ...]]] = ContextVar("audit_response_ids", default=())

_AUDIT_POLICY_REQUEST_COUNT: Final[ContextVar[int]] = ContextVar("audit_policy_request_count", default=0)

_AUDIT_UPSTREAM_REQUEST_COUNT: Final[ContextVar[int]] = ContextVar("audit_upstream_request_count", default=0)


@dataclass(frozen=True, slots=True)
class CallerResult:
    status: int
    body: dict[str, JsonValue]
    response_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ChaosCall:
    index: int
    endpoint: Endpoint
    client_kind: ClientKind
    model: str
    stream: bool
    prompt: str
    call_id: str


@dataclass(frozen=True, slots=True)
class ChaosDeployment:
    model_name: str
    model: str
    api_base: str


def _chaos_models(scenario: Scenario, marker: str) -> tuple[ChaosDeployment, ...]:
    endpoints: Final[tuple[Endpoint, Endpoint, Endpoint]] = ("chat", "messages", "responses")
    specs: Final = tuple((endpoint, False) for endpoint in endpoints) + tuple(
        (endpoint, True) for endpoint in endpoints
    )
    handles: Final = tuple(
        register_scenario(
            f"{marker}-{endpoint}-{'stream' if stream else 'complete'}",
            _provider_response(endpoint, marker, f"synthetic K response {marker}", stream),
        )
        for endpoint, stream in specs
    )
    for handle in handles:
        scenario.cleanups.callback(delete_scenario, handle)

    return tuple(
        ChaosDeployment(
            model_name=f"integration-{marker}-{endpoint}-{'stream' if stream else 'complete'}",
            model={
                "chat": "openai/gpt-4o-mini",
                "messages": "anthropic/claude-3-7-sonnet-20250219",
                "responses": "openai/gpt-4.1-mini",
            }[endpoint],
            api_base=handle.api_base() if endpoint == "messages" else f"{handle.api_base()}/v1",
        )
        for (endpoint, stream), handle in zip(specs, handles)
    )


def _chaos_model_list(deployments: tuple[ChaosDeployment, ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "model_name": deployment.model_name,
            "litellm_params": {
                "model": deployment.model,
                "api_base": deployment.api_base,
                "api_key": "synthetic-provider-key",
            },
        }
        for deployment in deployments
    )


def _chaos_control_configuration(
    tmp_path: Path,
    identity: str,
    model_list: tuple[dict[str, JsonValue], ...],
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["model_list"] = list(model_list)
    config["guardrails"] = []
    path: Final = tmp_path / f"{identity}-models.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _chaos_calls(deployments: tuple[ChaosDeployment, ...], marker: str) -> tuple[ChaosCall, ...]:
    endpoints: Final[tuple[Endpoint, Endpoint, Endpoint]] = ("chat", "messages", "responses")

    def one(index: int) -> ChaosCall:
        endpoint_index: Final = index % 3
        endpoint: Final = endpoints[endpoint_index]
        stream: Final = (index // 3) % 2 == 1
        client_kind: Final[ClientKind] = (
            "openai_async"
            if endpoint == "chat" and stream
            else "openai_sync"
            if endpoint in ("chat", "responses")
            else "anthropic_async"
            if stream
            else "anthropic_sync"
        )
        return ChaosCall(
            index=index,
            endpoint=endpoint,
            client_kind=client_kind,
            model=deployments[endpoint_index + 3 * int(stream)].model_name,
            stream=stream,
            prompt=f"synthetic K burst {marker}-{index}",
            call_id=f"{marker}-k-{index}",
        )

    return tuple(one(index) for index in range(30))


def _chaos_spend_minimums(
    models: tuple[str, ...], calls: tuple[ChaosCall, ...], baseline_count: int
) -> tuple[int, ...]:
    return tuple(baseline_count + sum(call.model == model for call in calls) for model in models)


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _is_base_audit_leg() -> bool:
    leg: Final = os.environ.get("LITELLM_LOGGING_ONLY_SCOPE_AUDIT_LEG", "head")
    assert leg in ("base", "head"), leg
    return leg == "base"


def _record_response_id(response_id: str) -> None:
    response_ids: Final = _AUDIT_RESPONSE_IDS.get()
    _record_response_ids((response_id,) if response_id not in response_ids else ())


def _record_response_ids(response_ids: tuple[str, ...]) -> None:
    current: Final = _AUDIT_RESPONSE_IDS.get()
    _AUDIT_RESPONSE_IDS.set(tuple(dict.fromkeys((*current, *response_ids))))


def _record_policy_request_count(count: int) -> None:
    _AUDIT_POLICY_REQUEST_COUNT.set(_AUDIT_POLICY_REQUEST_COUNT.get() + count)


def _record_upstream_request_count(count: int) -> None:
    _AUDIT_UPSTREAM_REQUEST_COUNT.set(_AUDIT_UPSTREAM_REQUEST_COUNT.get() + count)


@contextmanager
def wire_server(respond: Callable[[Request], Reply], port: int = 0, *, policy_edge: bool = True) -> Iterator[Wire]:
    received: Final[SimpleQueue[Request]] = SimpleQueue()

    def record(request: Request) -> Reply:
        received.put(request)
        return respond(request)

    try:
        with _wire_server(record, port=port) as server:
            yield server
    finally:
        if policy_edge:
            _record_policy_request_count(received.qsize())


def _directions_for_scope(base_default: tuple[Direction, ...], scope: str | None) -> tuple[Direction, ...]:
    if scope is None or scope == "both":
        return base_default
    selected_direction: Final = "request" if scope == "input" else "response"
    return tuple(direction for direction in base_default if direction == selected_direction)


def _directions_for_audit_leg(base_default: tuple[Direction, ...], scope: str | None) -> tuple[Direction, ...]:
    if _is_base_audit_leg():
        return base_default
    return _directions_for_scope(base_default, scope)


def _response_text(endpoint: Endpoint, body: Mapping[str, JsonValue]) -> str:
    if endpoint == "chat":
        choices: Final = body.get("choices")
        assert isinstance(choices, list) and choices, body
        message: Final = object_value(object_value(choices[0])["message"])
        return str(message["content"])
    if endpoint == "messages":
        content: Final = body.get("content")
        assert isinstance(content, list), body
        return "".join(
            str(object_value(block)["text"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    output: Final = body.get("output")
    assert isinstance(output, list), body
    return "".join(
        _response_text_from_blocks(object_value(item).get("content"))
        for item in output
        if isinstance(item, dict) and object_value(item).get("type") == "message"
    )


def _response_text_from_blocks(value: JsonValue | None) -> str:
    if not isinstance(value, list):
        return ""
    return "".join(
        str(object_value(block)["text"])
        for block in value
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def _chat_chunk_text(chunk: ChatCompletionChunk) -> str:
    return "".join(choice.delta.content for choice in chunk.choices if isinstance(choice.delta.content, str))


def _caller_result(endpoint: Endpoint, status: int, body: Mapping[str, JsonValue]) -> CallerResult:
    response_id: Final = body.get("id")
    assert isinstance(response_id, str), body
    _record_response_id(response_id)
    normalized: Final = JSON_OBJECT.validate_python(dict(body))
    return CallerResult(status, normalized, response_id, _response_text(endpoint, normalized))


def _stream_result(endpoint: Endpoint, response_id: str, text: str) -> CallerResult:
    _record_response_id(response_id)
    body: Final = JSON_OBJECT.validate_python({"id": response_id, "text": text})
    return CallerResult(200, body, response_id, text)


def _response_body_without_ids(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _response_body_without_ids(item) for key, item in value.items() if key != "id"}
    if isinstance(value, list):
        return [_response_body_without_ids(item) for item in value]
    return value


def _call_sync(
    client_kind: ClientKind,
    endpoint: Endpoint,
    proxy_url: str,
    key: str,
    model: str,
    prompt: str,
    stream: bool,
    call_id: str,
) -> CallerResult:
    if client_kind == "httpx":
        body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        with httpx.Client(base_url=proxy_url, timeout=30, trust_env=False) as client:
            response: Final = client.post(
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {key}", "x-litellm-call-id": call_id},
            )
        return _caller_result(endpoint, response.status_code, JSON_OBJECT.validate_json(response.content))
    if client_kind == "anthropic_sync":
        with Anthropic(
            base_url=proxy_url,
            api_key=key,
            max_retries=0,
            http_client=httpx.Client(timeout=30, trust_env=False),
        ) as client:
            if stream:
                with client.messages.stream(
                    model=model,
                    max_tokens=32,
                    messages=[{"role": "user", "content": prompt}],
                    extra_headers={"x-litellm-call-id": call_id},
                ) as stream_response:
                    message: Final = stream_response.get_final_message()
                return _stream_result(
                    endpoint,
                    message.id,
                    "".join(block.text for block in message.content if block.type == "text"),
                )
            message: Final = client.messages.create(
                model=model,
                max_tokens=32,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={"x-litellm-call-id": call_id},
            )
            return _caller_result(endpoint, 200, JSON_OBJECT.validate_python(message.model_dump(mode="json")))
    assert client_kind == "openai_sync", client_kind
    with OpenAI(
        base_url=f"{proxy_url}/v1",
        api_key=key,
        max_retries=0,
        http_client=httpx.Client(timeout=30, trust_env=False),
    ) as client:
        headers: Final = {"x-litellm-call-id": call_id}
        if endpoint == "chat":
            if stream:
                chunks: Final = tuple(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        extra_headers=headers,
                    )
                )
                response_id: Final = chunks[0].id
                text: Final = "".join(_chat_chunk_text(chunk) for chunk in chunks)
                return _stream_result(endpoint, response_id, text)
            response: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_headers=headers,
            )
            return _caller_result(endpoint, 200, JSON_OBJECT.validate_python(response.model_dump(mode="json")))
        assert endpoint == "responses", endpoint
        if stream:
            events: Final = tuple(
                client.responses.create(model=model, input=prompt, stream=True, extra_headers=headers)
            )
            completed: Final = next(event.response for event in events if event.type == "response.completed")
            body: Final = JSON_OBJECT.validate_python(completed.model_dump(mode="json"))
            return _stream_result(endpoint, str(body["id"]), _response_text(endpoint, body))
        completion: Final = client.responses.create(model=model, input=prompt, extra_headers=headers)
        return _caller_result(endpoint, 200, JSON_OBJECT.validate_python(completion.model_dump(mode="json")))


async def _call_async(
    client_kind: ClientKind,
    endpoint: Endpoint,
    proxy_url: str,
    key: str,
    model: str,
    prompt: str,
    stream: bool,
    call_id: str,
) -> CallerResult:
    if client_kind == "anthropic_async":
        async with AsyncAnthropic(
            base_url=proxy_url,
            api_key=key,
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=30, trust_env=False),
        ) as client:
            if stream:
                async with client.messages.stream(
                    model=model,
                    max_tokens=32,
                    messages=[{"role": "user", "content": prompt}],
                    extra_headers={"x-litellm-call-id": call_id},
                ) as stream_response:
                    message: Final = await stream_response.get_final_message()
                return _stream_result(
                    endpoint,
                    message.id,
                    "".join(block.text for block in message.content if block.type == "text"),
                )
            message: Final = await client.messages.create(
                model=model,
                max_tokens=32,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={"x-litellm-call-id": call_id},
            )
            return _caller_result(endpoint, 200, JSON_OBJECT.validate_python(message.model_dump(mode="json")))
    assert client_kind == "openai_async", client_kind
    async with AsyncOpenAI(
        base_url=f"{proxy_url}/v1",
        api_key=key,
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=30, trust_env=False),
    ) as client:
        headers: Final = {"x-litellm-call-id": call_id}
        if endpoint == "chat":
            assert stream, "The audit only uses the async OpenAI chat client for streaming rows"
            stream_response: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_headers=headers,
            )
            chunks: Final = tuple([chunk async for chunk in stream_response])
            response_id: Final = chunks[0].id
            text: Final = "".join(
                choice.delta.content
                for chunk in chunks
                for choice in chunk.choices
                if isinstance(choice.delta.content, str)
            )
            return _stream_result(endpoint, response_id, text)
        assert endpoint == "responses", endpoint
        if stream:
            responses_stream: Final = await client.responses.create(
                model=model, input=prompt, stream=True, extra_headers=headers
            )
            events: Final = tuple([event async for event in responses_stream])
            completed: Final = next(event.response for event in events if event.type == "response.completed")
            body: Final = JSON_OBJECT.validate_python(completed.model_dump(mode="json"))
            return _stream_result(endpoint, str(body["id"]), _response_text(endpoint, body))
        completion: Final = await client.responses.create(model=model, input=prompt, extra_headers=headers)
        return _caller_result(endpoint, 200, JSON_OBJECT.validate_python(completion.model_dump(mode="json")))


def _call_client(
    client_kind: ClientKind,
    endpoint: Endpoint,
    gateway: Gateway,
    model: str,
    prompt: str,
    stream: bool,
    call_id: str,
) -> CallerResult:
    if client_kind in ("openai_async", "anthropic_async"):
        return _record_caller_result(
            asyncio.run(
                _call_async(client_kind, endpoint, _proxy_url(gateway), gateway.key, model, prompt, stream, call_id)
            )
        )
    return _record_caller_result(
        _call_sync(client_kind, endpoint, _proxy_url(gateway), gateway.key, model, prompt, stream, call_id)
    )


def _record_caller_result(result: CallerResult) -> CallerResult:
    _record_response_id(result.response_id)
    return result


def _call_cache_client(endpoint: Endpoint, gateway: Gateway, model: str, prompt: str, call_id: str) -> CallerResult:
    path: Final = {
        "chat": "/v1/chat/completions",
        "messages": "/v1/messages",
        "responses": "/v1/responses",
    }[endpoint]
    body: Final = {
        "chat": {"model": model, "messages": [{"role": "user", "content": prompt}]},
        "messages": {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": prompt}]},
        "responses": {"model": model, "input": prompt},
    }[endpoint]
    with httpx.Client(timeout=30, trust_env=False) as client:
        response: Final = client.post(
            f"{_proxy_url(gateway)}{path}",
            json=body,
            headers={"Authorization": f"Bearer {gateway.key}", "x-litellm-call-id": call_id},
        )
    assert response.status_code == 200, response.text
    return _caller_result(
        endpoint,
        response.status_code,
        JSON_OBJECT.validate_json(response.content),
    )


def _provider_response(endpoint: Endpoint, _scenario_id: str, reply: str, stream: bool) -> JsonResponse | SseResponse:
    response_id: Final = {
        "chat": "chatcmpl-$UNIQUE_ID",
        "messages": "msg_$UNIQUE_ID",
        "responses": "resp_$UNIQUE_ID",
    }[endpoint]
    if endpoint == "chat":
        if stream:
            return SseResponse(
                content_type="text/event-stream",
                frames=(
                    f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': 1, 'model': 'gpt-4o-mini', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': reply}, 'finish_reason': None}]})}",
                    f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': 1, 'model': 'gpt-4o-mini', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}",
                    "data: [DONE]",
                ),
            )
        return JsonResponse(
            content_type="application/json",
            body={
                "id": response_id,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
            },
        )
    if endpoint == "messages":
        if stream:
            return SseResponse(
                content_type="text/event-stream",
                frames=(
                    f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': response_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': 'claude-3-7-sonnet-20250219', 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 9, 'output_tokens': 0}}})}",
                    f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}",
                    f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': reply}})}",
                    f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}",
                    f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 5}})}",
                    f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}",
                ),
            )
        return JsonResponse(
            content_type="application/json",
            body={
                "id": response_id,
                "type": "message",
                "role": "assistant",
                "model": "claude-3-7-sonnet-20250219",
                "content": [{"type": "text", "text": reply}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 9, "output_tokens": 5},
            },
        )
    if stream:
        completed: Final = {
            "id": response_id,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4.1-mini",
            "output": [
                {
                    "id": "msg_$UNIQUE_ID",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": reply, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 9, "output_tokens": 5, "total_tokens": 14},
        }
        return SseResponse(
            content_type="text/event-stream",
            frames=(
                f"data: {json.dumps({'type': 'response.created', 'response': {'id': response_id, 'object': 'response', 'created_at': 1, 'status': 'in_progress', 'model': 'gpt-4.1-mini', 'output': []}})}",
                f"data: {json.dumps({'type': 'response.output_text.delta', 'item_id': 'msg_$UNIQUE_ID', 'output_index': 0, 'content_index': 0, 'delta': reply})}",
                f"data: {json.dumps({'type': 'response.completed', 'response': completed})}",
            ),
        )
    return JsonResponse(
        content_type="application/json",
        body={
            "id": response_id,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4.1-mini",
            "output": [
                {
                    "id": "msg_$UNIQUE_ID",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": reply, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 9, "output_tokens": 5, "total_tokens": 14},
        },
    )


def _configuration(
    tmp_path: Path,
    identity: str,
    policy_url: str,
    scope: str | None,
    *,
    include_scope: bool = True,
    continue_on_input_failure: bool | None = None,
    default_on: bool = True,
    mode: str | list[str] = "logging_only",
    cache: bool = False,
    num_retries: int | None = None,
    model_list: tuple[dict[str, JsonValue], ...] = (),
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = cache
    if num_retries is not None:
        config["litellm_settings"]["num_retries"] = num_retries
    config["model_list"] = list(model_list)
    params: Final = {
        "guardrail": "generic_guardrail_api",
        "mode": mode,
        "default_on": default_on,
        "api_base": policy_url,
        "api_key": "synthetic-guardrail-key",
        "extra_headers": ["x-litellm-call-id"],
        **({"logging_only_scope": scope} if include_scope else {}),
        **(
            {"logging_only_continue_on_input_failure": continue_on_input_failure}
            if continue_on_input_failure is not None
            else {}
        ),
    }
    config["guardrails"] = [{"guardrail_name": identity, "litellm_params": params}]
    scope_name: Final = scope if scope is not None else "unset"
    path: Final = tmp_path / f"{identity}-{scope_name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _content_filter_configuration(tmp_path: Path, identity: str, scope: str, blocked_word: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["guardrails"] = [
        {
            "guardrail_name": identity,
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "logging_only",
                "logging_only_scope": scope,
                "default_on": True,
                "blocked_words": [{"keyword": blocked_word, "action": "BLOCK"}],
            },
        }
    ]
    path: Final = tmp_path / f"{identity}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _presidio_configuration(
    tmp_path: Path,
    identity: str,
    analyzer_api_base: str,
    anonymizer_api_base: str,
    scope: str | None,
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    params: Final = {
        "guardrail": "presidio",
        "mode": "logging_only",
        "default_on": True,
        "presidio_analyzer_api_base": analyzer_api_base,
        "presidio_anonymizer_api_base": anonymizer_api_base,
        "pii_entities_config": {"PERSON": "MASK"},
        **({"logging_only_scope": scope} if scope is not None else {}),
    }
    config["guardrails"] = [{"guardrail_name": identity, "litellm_params": params}]
    scope_name: Final = scope if scope is not None else "unset"
    path: Final = tmp_path / f"{identity}-{scope_name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _empty_proxy_configuration(tmp_path: Path, identity: str, reload_seconds: int = 30) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["guardrails"] = []
    config["general_settings"]["proxy_config_reload_interval_seconds"] = reload_seconds
    path: Final = tmp_path / f"{identity}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _model_armor_configuration(tmp_path: Path, identity: str, api_endpoint: str, token_uri: str, scope: str) -> Path:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key_pem: Final = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    credentials: Final = {
        "type": "service_account",
        "project_id": "synthetic-model-armor-project",
        "private_key_id": "synthetic-key-id",
        "private_key": private_key_pem,
        "client_email": "integration-model-armor@synthetic-project.iam.gserviceaccount.com",
        "client_id": "123456789012345678901",
        "token_uri": token_uri,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/integration",
    }
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["guardrails"] = [
        {
            "guardrail_name": identity,
            "litellm_params": {
                "guardrail": "model_armor",
                "mode": "logging_only",
                "logging_only_scope": scope,
                "default_on": True,
                "template_id": "synthetic-template",
                "project_id": "synthetic-model-armor-project",
                "location": "us-central1",
                "credentials": json.dumps(credentials),
                "api_endpoint": api_endpoint,
            },
        }
    ]
    path: Final = tmp_path / f"{identity}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _insert_database_guardrail(
    identity: str,
    policy_url: str,
    scope: str | None,
    *,
    mode: str = "pre_call",
    default_on: bool = True,
    continue_on_input_failure: bool | None = None,
) -> None:
    params: Final = {
        "guardrail": "generic_guardrail_api",
        "mode": mode,
        "default_on": default_on,
        "api_base": policy_url,
        "api_key": "synthetic-guardrail-key",
        "extra_headers": ["x-litellm-call-id"],
        **({"logging_only_scope": scope} if scope is not None else {}),
        **(
            {"logging_only_continue_on_input_failure": continue_on_input_failure}
            if continue_on_input_failure is not None
            else {}
        ),
    }
    write_rows(
        'INSERT INTO "LiteLLM_GuardrailsTable" '
        "(guardrail_id, guardrail_name, litellm_params, guardrail_info, updated_at) "
        "VALUES (%s, %s, %s::jsonb, %s::jsonb, NOW())",
        (str(uuid.uuid5(uuid.NAMESPACE_URL, identity)), identity, json.dumps(params), "{}"),
    )


@contextmanager
def _database_guardrail(
    identity: str,
    policy_url: str,
    scope: str | None,
    *,
    mode: str = "pre_call",
    default_on: bool = True,
) -> Iterator[None]:
    _insert_database_guardrail(identity, policy_url, scope, mode=mode, default_on=default_on)
    try:
        yield
    finally:
        _delete_database_guardrail(identity)


def _delete_database_guardrail(identity: str) -> None:
    write_rows('DELETE FROM "LiteLLM_GuardrailsTable" WHERE guardrail_name=%s', (identity,))


def _post_guardrail_body(
    identity: str,
    provider: str,
    mode: str | list[str],
    api_base: str,
    scope: JsonValue,
    include_scope: bool = True,
    continue_on_input_failure: bool | None = None,
) -> dict[str, JsonValue]:
    params: Final = {
        "guardrail": provider,
        "mode": mode,
        "default_on": True,
        **(
            {
                "presidio_analyzer_api_base": api_base,
                "presidio_anonymizer_api_base": api_base,
                "pii_entities_config": {"PERSON": "MASK"},
            }
            if provider == "presidio"
            else {"api_base": api_base, "api_key": "synthetic-guardrail-key"}
        ),
        **({"extra_headers": ["x-litellm-call-id"]} if provider == "generic_guardrail_api" else {}),
        **({"logging_only_scope": scope} if include_scope else {}),
        **(
            {"logging_only_continue_on_input_failure": continue_on_input_failure}
            if continue_on_input_failure is not None
            else {}
        ),
    }
    return {
        "guardrail": {
            "guardrail_name": identity,
            "litellm_params": params,
            "guardrail_info": {"description": "phase-12 logging scope audit"},
        }
    }


def _create_guardrail(candidate: Gateway, identity: str, params: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    guardrail_params: Final = {
        **params,
        **({"extra_headers": ["x-litellm-call-id"]} if params.get("guardrail") == "generic_guardrail_api" else {}),
    }
    response: Final = candidate.request(
        "POST",
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": identity,
                "litellm_params": guardrail_params,
                "guardrail_info": {"description": "phase-12 logging scope audit"},
            }
        },
    )
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)


def _management_guardrail_rows(identity: str) -> tuple[dict[str, JsonValue], ...]:
    rows: Final = tuple(
        object_value(row)
        for row in read_rows(
            "SELECT guardrail_id, guardrail_name, litellm_params, guardrail_info "
            'FROM "LiteLLM_GuardrailsTable" WHERE guardrail_name=%s',
            (identity,),
        )
    )
    return tuple({**row, "litellm_params": _decrypted_management_litellm_params(row["litellm_params"])} for row in rows)


def _decrypted_management_litellm_params(stored_value: JsonValue) -> dict[str, JsonValue]:
    stored: Final = object_value(stored_value)
    salt_key: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("LITELLM_SALT_KEY", salt_key)
        decrypted: Final = JSON_OBJECT.validate_python(decrypt_guardrail_litellm_params(stored))
    assert _decryption_only_changes_encrypted_values(stored, decrypted)
    return decrypted


def _decryption_only_changes_encrypted_values(stored: JsonValue, decrypted: JsonValue) -> bool:
    if isinstance(stored, dict) and isinstance(decrypted, dict):
        return stored.keys() == decrypted.keys() and all(
            _decryption_only_changes_encrypted_values(value, decrypted[key]) for key, value in stored.items()
        )
    if isinstance(stored, list) and isinstance(decrypted, list):
        return len(stored) == len(decrypted) and all(
            _decryption_only_changes_encrypted_values(stored_value, decrypted_value)
            for stored_value, decrypted_value in zip(stored, decrypted)
        )
    if isinstance(stored, str) and stored.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX):
        return isinstance(decrypted, str) and not decrypted.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX)
    return stored == decrypted


def _drain_upstream(upstream_url: str) -> tuple[dict[str, JsonValue], ...]:
    response: Final = httpx.get(f"{upstream_url.rstrip('/')}/__observations", trust_env=False, timeout=15)
    response.raise_for_status()
    requests: Final = object_value(JSON_OBJECT.validate_python(response.json())).get("requests")
    assert isinstance(requests, list), response.text
    observations: Final = tuple(object_value(request) for request in requests)
    forwarded_requests: Final = tuple(request for request in observations if request.get("method", "POST") != "GET")
    _record_upstream_request_count(len(forwarded_requests))
    return forwarded_requests


def _json_contains_exact_string(value: JsonValue, expected: str) -> bool:
    if isinstance(value, str):
        return value == expected
    if isinstance(value, list):
        return any(_json_contains_exact_string(item, expected) for item in value)
    if isinstance(value, dict):
        return any(_json_contains_exact_string(item, expected) for item in value.values())
    return False


@pytest.fixture(autouse=True)
def _record_audit_properties(
    request: pytest.FixtureRequest,
    record_property: Callable[[str, object], None],
    gateway: Gateway,
) -> Iterator[None]:
    response_ids_token: Final = _AUDIT_RESPONSE_IDS.set(())
    policy_count_token: Final = _AUDIT_POLICY_REQUEST_COUNT.set(0)
    upstream_count_token: Final = _AUDIT_UPSTREAM_REQUEST_COUNT.set(0)
    try:
        response: Final = httpx.get(f"{gateway.upstream_url.rstrip('/')}/__observations", trust_env=False, timeout=15)
        response.raise_for_status()
        yield
    finally:
        node_id: Final = request.node.nodeid
        inventory_ids: Final = re.findall(r"[A-Z]{1,2}\d+", node_id)
        record_property("node_id", node_id)
        record_property("inventory_id", inventory_ids[0] if inventory_ids else "support")
        record_property("response_ids", ",".join(_AUDIT_RESPONSE_IDS.get()))
        record_property("policy_edge_request_count", str(_AUDIT_POLICY_REQUEST_COUNT.get()))
        record_property("upstream_request_count", str(_AUDIT_UPSTREAM_REQUEST_COUNT.get()))
        _AUDIT_RESPONSE_IDS.reset(response_ids_token)
        _AUDIT_POLICY_REQUEST_COUNT.reset(policy_count_token)
        _AUDIT_UPSTREAM_REQUEST_COUNT.reset(upstream_count_token)


def _policy_call_id_matches(payload: Mapping[str, JsonValue], call_id: str) -> bool:
    actual: Final = payload.get("litellm_call_id")
    if actual == call_id:
        return True
    headers: Final = payload.get("request_headers")
    return isinstance(headers, dict) and any(
        key.lower() == "x-litellm-call-id" and value == call_id for key, value in headers.items()
    )


def _policy_call_id(payload: Mapping[str, JsonValue]) -> str | None:
    actual: Final = payload.get("litellm_call_id")
    if isinstance(actual, str):
        return actual
    headers: Final = payload.get("request_headers")
    if not isinstance(headers, dict):
        return None
    return next(
        (value for key, value in headers.items() if key.lower() == "x-litellm-call-id" and isinstance(value, str)),
        None,
    )


def _chat_request(gateway: Gateway, model: str, prompt: str, call_id: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}]},
        headers={"x-litellm-call-id": call_id},
    )


def _direction(payload: Mapping[str, JsonValue]) -> str:
    value: Final = payload.get("input_type")
    assert value in ("request", "response"), payload
    return str(value)


def _cache_hit(value: JsonValue) -> bool:
    return value is True or value == "True"


def _guardrail_mode_values(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(mode) for mode in value)
    return (str(value),)


def _guardrail_mode_status_pairs(
    entries: tuple[dict[str, JsonValue], ...],
) -> tuple[tuple[tuple[str, ...], str], ...]:
    return tuple((_guardrail_mode_values(entry["guardrail_mode"]), str(entry["guardrail_status"])) for entry in entries)


def _spend_row_for_response_id(response_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (response_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    row: Final = object_value(rows[0])
    _record_response_ids((str(row["request_id"]),))
    return row


def _spend_rows(model: str, minimum: int) -> tuple[dict[str, JsonValue], ...]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, metadata, cache_hit FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
            (model,),
        ),
        lambda rows: len(rows) >= minimum,
        seconds=70,
    )
    response_ids: Final = tuple(str(row["request_id"]) for row in rows)
    _record_response_ids(response_ids)
    return rows


def _spend_rows_for_calls(
    models: tuple[str, ...],
    expected: tuple[tuple[str, str, str], ...],
    *,
    tolerate_missing: bool = False,
) -> tuple[dict[str, JsonValue], ...]:
    model_placeholders: Final = ", ".join("%s" for _ in models)
    query: Final = (
        "SELECT model_group, request_id, metadata, cache_hit "
        f'FROM "LiteLLM_SpendLogs" WHERE model_group IN ({model_placeholders})'
    )
    rows: Final = eventually(
        lambda: tuple(read_rows(query, models)),
        lambda values: all(
            len(_spend_rows_matching_call(values, model, call_id)) == 1 for model, _, call_id in expected
        ),
        seconds=45 if tolerate_missing else 70,
        return_last_on_timeout=tolerate_missing,
    )
    _record_response_ids(tuple(response_id for _, response_id, _ in expected))
    return rows


def _spend_rows_matching_call(
    rows: tuple[dict[str, JsonValue], ...],
    model: str,
    call_id: str,
) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        row
        for row in rows
        if row["model_group"] == model and object_value(row["metadata"]).get("litellm_call_id") == call_id
    )


def _spend_row_for_call_id(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (call_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert str(rows[0]["request_id"]) == call_id, rows
    row: Final = object_value(rows[0])
    _record_response_ids((str(row["request_id"]),))
    return row


def _guardrail_entries(row: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    metadata: Final = object_value(row["metadata"])
    entries: Final = metadata.get("guardrail_information")
    if not isinstance(entries, list):
        return ()
    return tuple(object_value(entry) for entry in entries)


def _response_id_matches(endpoint: Endpoint, request_id: str, response_id: str, scenario_id: str) -> bool:
    if request_id == response_id:
        return True
    if endpoint != "responses" or not request_id.startswith("resp_"):
        return False
    encoded: Final = request_id.removeprefix("resp_")
    padding: Final = "=" * (-len(encoded) % 4)
    try:
        decoded: Final = base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    return response_id in decoded or scenario_id in decoded


def _assert_response_id(endpoint: Endpoint, request_id: str, response_id: str, scenario_id: str) -> None:
    assert _response_id_matches(endpoint, request_id, response_id, scenario_id), (
        endpoint,
        request_id,
        response_id,
        scenario_id,
    )
