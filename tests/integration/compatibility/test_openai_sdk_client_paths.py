from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, TypeVar

import anthropic
import httpx
import openai
import pytest
import yaml
from anthropic.types import RawContentBlockDeltaEvent, RawMessageStartEvent, RawMessageStreamEvent, TextDelta
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply, openai_error, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.responses_vendor import response_identities
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types import Completion, ModerationCreateResponse
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import ResponseStreamEvent
from pydantic import JsonValue

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

PROXY_WORKERS: Final = int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1"))
WORKER_SYNC_SECONDS: Final = 0.0 if PROXY_WORKERS == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS + 5.0
ROOT: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
TEXT: Final = "openai sdk client path"
PROVIDER_KEY: Final = "integration-provider-key"
PROVIDER_AUTHORIZATION: Final = f"Bearer {PROVIDER_KEY}"
UPSTREAM_MODEL: Final = "gpt-4o-mini"
DEPLOYMENT_MODEL: Final = f"openai/{UPSTREAM_MODEL}"
INSTRUCT_MODEL: Final = "gpt-3.5-turbo-instruct"
INSTRUCT_DEPLOYMENT_MODEL: Final = f"openai/{INSTRUCT_MODEL}"
NO_CACHE: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
USAGE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
)
FIVE_KB: Final = "x" * 5120
REJECTED_CHAT_TIMEOUTS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"list": [1, 2], "dict": {"read": 5}, "5 KB string": FIVE_KB}
)
ACCEPTED_CHAT_TIMEOUTS: Final[Mapping[str, JsonValue]] = MappingProxyType({"empty string": "", "int": 30})
NO_EXTRA: Final[Mapping[str, JsonValue]] = MappingProxyType({})
REJECTED_DEPLOYMENT_TIMEOUTS: Final[Mapping[str, JsonValue]] = MappingProxyType({"list": [1, 2], "dict": {"read": 5}})
ACCEPTED_DEPLOYMENT_TIMEOUTS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"empty string": "", "5 KB string": FIVE_KB}
)
MODERATION_CATEGORIES: Final = ("hate", "harassment", "self-harm", "sexual", "violence")
LISTED_JOB: Final = "ftjob-listed"
ASSISTANT: Final = "asst_integration"
CLOUDFLARE_GATEWAY: Final = "/gateway.ai.cloudflare.com/v1/account/gateway/azure-openai/resource"
AZURE_API_VERSION: Final = "2024-10-21"
STREAM_HEAD: Final = b"h" * 65536
STREAM_TAIL: Final = b'{"custom_id": "tail", "response": {"status_code": 200}}\n'
SPEECH_AUDIO: Final = b"OggS" + bytes(range(60))
SPEECH_UPSTREAM_MODEL: Final = "gpt-4o-mini-tts"
PAYMENT_REQUIRED_DEPLOYMENTS: Final[Mapping[str, str]] = MappingProxyType(
    {"openai": "openai/gpt-4o-mini", "anthropic": "anthropic/claude-sonnet-4-5"}
)
_R: Final = TypeVar("_R")


def _identity(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _path(request: Request) -> str:
    return request.target.split("?", 1)[0]


def _body(request: Request) -> Mapping[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def _streams(request: Request) -> bool:
    return _body(request).get("stream") is True


def _upstream_model(request: Request) -> str:
    return string_value(_body(request).get("model") or UPSTREAM_MODEL)


def _json(
    payload: Mapping[str, JsonValue], *, status: int = 200, headers: Mapping[str, str] = MappingProxyType({})
) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode(), headers=headers)


def _peer(route: Callable[[Request], Reply | None]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET" and _path(request).endswith("/models"):
            return _json({"object": "list", "data": []})
        reply: Final = route(request)
        if reply is None:
            return _json({"error": {"message": f"unrouted {request.method} {request.target}"}}, status=404)
        return reply

    return respond


def _is_model_discovery(request: Request) -> bool:
    return request.method == "GET" and _path(request).endswith("/models")


def _served(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if not _is_model_discovery(request))


def _routes(served: tuple[Request, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((request.method, _path(request)) for request in served)


def _is_chat(request: Request) -> bool:
    return request.method == "POST" and _path(request).endswith("/chat/completions")


def _chat_peer(identity: str) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        if not _is_chat(request):
            return None
        return chat_reply(identity, _upstream_model(request), TEXT, stream=_streams(request))

    return _peer(route)


def _fresh_chat_peer(served: SimpleQueue[str]) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        if not _is_chat(request):
            return None
        identity: Final = _identity("chatcmpl")
        served.put(identity)
        return chat_reply(identity, _upstream_model(request), TEXT, stream=False)

    return _peer(route)


def _held_chat_peer(gate: threading.Event, identity: str) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        if not _is_chat(request):
            return None
        assert gate.wait(timeout=10), "the timeout cell never released its peer"
        return chat_reply(identity, _upstream_model(request), TEXT, stream=False)

    return _peer(route)


def _status_peer(reply: Reply) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        return reply if request.method == "POST" else None

    return _peer(route)


def _completion_frame(identity: str, model: str, text: str, finish_reason: str | None) -> bytes:
    frame: Final = {
        "id": identity,
        "object": "text_completion",
        "created": 1,
        "model": model,
        "choices": [{"text": text, "index": 0, "finish_reason": finish_reason, "logprobs": None}],
    }
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def _completion_reply(identity: str, model: str, *, stream: bool) -> Reply:
    if stream:
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _completion_frame(identity, model, TEXT, None),
                _completion_frame(identity, model, "", "stop") + b"data: [DONE]\n\n",
            ),
        )
    return _json(
        {
            "id": identity,
            "object": "text_completion",
            "created": 1,
            "model": model,
            "choices": [{"text": TEXT, "index": 0, "finish_reason": "stop", "logprobs": None}],
            "usage": dict(USAGE),
        }
    )


def _completion_peer(identity: str) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        if request.method != "POST" or _path(request) != "/v1/completions":
            return None
        return _completion_reply(identity, _upstream_model(request), stream=_streams(request))

    return _peer(route)


def _responses_peer(identity: str) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        if request.method != "POST" or _path(request) != "/v1/responses":
            return None
        return responses_reply(identity, _upstream_model(request), TEXT, stream=_streams(request))

    return _peer(route)


def _moderation(identity: str, model: str) -> Mapping[str, JsonValue]:
    return {
        "id": identity,
        "model": model,
        "results": [
            {
                "flagged": False,
                "categories": {category: False for category in MODERATION_CATEGORIES},
                "category_scores": {category: 0.0 for category in MODERATION_CATEGORIES},
                "category_applied_input_types": {category: ["text"] for category in MODERATION_CATEGORIES},
            }
        ],
    }


def _moderation_peer(identity: str) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        if request.method != "POST" or _path(request) != "/v1/moderations":
            return None
        return _json(_moderation(identity, _upstream_model(request)))

    return _peer(route)


def _batch_line(custom_id: str) -> bytes:
    line: Final = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": UPSTREAM_MODEL, "messages": [_user_message()]},
    }
    return json.dumps(line).encode() + b"\n"


def _file_object(file_id: str, size: int) -> Mapping[str, JsonValue]:
    return {
        "id": file_id,
        "object": "file",
        "purpose": "batch",
        "bytes": size,
        "created_at": 1,
        "filename": "batch.jsonl",
        "status": "processed",
    }


def _files_peer(file_id: str, content: bytes, chunks: tuple[bytes, ...] | None = None) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        path: Final = _path(request)
        if request.method == "POST" and path == "/v1/files":
            return _json(_file_object(file_id, len(content)))
        if request.method == "GET" and path == f"/v1/files/{file_id}":
            return _json(_file_object(file_id, len(content)))
        if request.method == "GET" and path == f"/v1/files/{file_id}/content":
            return Reply(content_type="application/octet-stream", body=content, chunks=chunks)
        return None

    return _peer(route)


def _batch_object(batch_id: str, input_file_id: str, status: str) -> Mapping[str, JsonValue]:
    return {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": input_file_id,
        "completion_window": "24h",
        "status": status,
        "output_file_id": None,
        "error_file_id": None,
        "created_at": 1,
        "in_progress_at": None,
        "expires_at": None,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "request_counts": {"total": 1, "completed": 0, "failed": 0},
        "metadata": None,
    }


def _batches_peer(file_id: str, batch_id: str) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        path: Final = _path(request)
        if request.method == "POST" and path == "/v1/files":
            return _json(_file_object(file_id, 1))
        if request.method == "POST" and path == "/v1/batches":
            return _json(_batch_object(batch_id, string_value(_body(request)["input_file_id"]), "validating"))
        if request.method == "GET" and path == f"/v1/batches/{batch_id}":
            return _json(_batch_object(batch_id, file_id, "in_progress"))
        if request.method == "GET" and path == "/v1/batches":
            return _json(
                {
                    "object": "list",
                    "data": [_batch_object(batch_id, file_id, "in_progress")],
                    "first_id": batch_id,
                    "last_id": batch_id,
                    "has_more": False,
                }
            )
        if request.method == "POST" and path == f"/v1/batches/{batch_id}/cancel":
            return _json(_batch_object(batch_id, file_id, "cancelling"))
        return None

    return _peer(route)


def _fine_tuning_job(job_id: str, status: str) -> Mapping[str, JsonValue]:
    return {
        "id": job_id,
        "object": "fine_tuning.job",
        "model": UPSTREAM_MODEL,
        "created_at": 1,
        "fine_tuned_model": None,
        "finished_at": None,
        "hyperparameters": {"n_epochs": 1, "batch_size": 1, "learning_rate_multiplier": 1.0},
        "organization_id": "org-integration",
        "result_files": [],
        "status": status,
        "trained_tokens": None,
        "training_file": f"file-{job_id.removeprefix('ftjob-')}",
        "validation_file": None,
        "seed": 1,
        "error": None,
    }


def _assistant(identity: str) -> Mapping[str, JsonValue]:
    return {
        "id": identity,
        "object": "assistant",
        "created_at": 1,
        "name": "integration",
        "description": None,
        "model": UPSTREAM_MODEL,
        "instructions": None,
        "tools": [],
        "metadata": {},
        "temperature": 1.0,
        "top_p": 1.0,
        "response_format": "auto",
        "tool_resources": None,
    }


def _configured_peer() -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        path: Final = _path(request)
        if request.method == "POST" and path == "/v1/fine_tuning/jobs":
            training_file: Final = string_value(_body(request)["training_file"])
            return _json(_fine_tuning_job(f"ftjob-{training_file.removeprefix('file-')}", "queued"))
        if request.method == "GET" and path == "/v1/fine_tuning/jobs":
            return _json({"object": "list", "data": [_fine_tuning_job(LISTED_JOB, "running")], "has_more": False})
        if request.method == "POST" and path.startswith("/v1/fine_tuning/jobs/") and path.endswith("/cancel"):
            return _json(
                _fine_tuning_job(path.removeprefix("/v1/fine_tuning/jobs/").removesuffix("/cancel"), "cancelled")
            )
        if request.method == "GET" and path.startswith("/v1/fine_tuning/jobs/"):
            return _json(_fine_tuning_job(path.removeprefix("/v1/fine_tuning/jobs/"), "running"))
        if request.method == "GET" and path == "/v1/assistants":
            return _json(
                {
                    "object": "list",
                    "data": [_assistant(ASSISTANT)],
                    "first_id": ASSISTANT,
                    "last_id": ASSISTANT,
                    "has_more": False,
                }
            )
        return None

    return _peer(route)


def _config_with_provider_settings(directory: Path, api_base: str) -> Path:
    shared: Final = object_value(yaml.safe_load((ROOT / "tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "proxy_config_with_provider_settings.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **shared,
                "finetune_settings": [{"custom_llm_provider": "openai", "api_base": api_base, "api_key": PROVIDER_KEY}],
                "assistant_settings": {
                    "custom_llm_provider": "openai",
                    "litellm_params": {"api_base": api_base, "api_key": PROVIDER_KEY},
                },
            }
        )
    )
    return path


@dataclass(frozen=True, slots=True)
class ConfiguredProxy:
    gateway: Gateway
    wire: Wire


@pytest.fixture(scope="module")
def configured_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ConfiguredProxy]:
    with gateway_from_environment() as rig_gateway, wire_server(_configured_peer()) as wire:
        directory: Final = tmp_path_factory.mktemp("openai_sdk_client_paths")
        config: Final = _config_with_provider_settings(directory, f"{wire.url}/v1")
        with owned_proxy(rig_gateway, directory, {}, config=config, workers=PROXY_WORKERS) as owned:
            yield ConfiguredProxy(owned, wire)


def _v1(gateway: Gateway) -> str:
    return f"{str(gateway.client.base_url).rstrip('/')}/v1"


def _origin(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


@contextmanager
def _openai(gateway: Gateway) -> Iterator[openai.OpenAI]:
    with (
        httpx.Client(timeout=15, trust_env=False) as transport,
        openai.OpenAI(api_key=gateway.key, base_url=_v1(gateway), max_retries=0, http_client=transport) as client,
    ):
        yield client


def _async_openai(gateway: Gateway, call: Callable[[openai.AsyncOpenAI], Awaitable[_R]]) -> _R:
    async def run() -> _R:
        async with (
            httpx.AsyncClient(timeout=15, trust_env=False) as transport,
            openai.AsyncOpenAI(
                api_key=gateway.key, base_url=_v1(gateway), max_retries=0, http_client=transport
            ) as client,
        ):
            return await call(client)

    return asyncio.run(run())


@contextmanager
def _anthropic(gateway: Gateway) -> Iterator[anthropic.Anthropic]:
    with (
        httpx.Client(timeout=15, trust_env=False) as transport,
        anthropic.Anthropic(
            api_key=gateway.key, base_url=_origin(gateway), max_retries=0, http_client=transport
        ) as client,
    ):
        yield client


def _async_anthropic(gateway: Gateway, call: Callable[[anthropic.AsyncAnthropic], Awaitable[_R]]) -> _R:
    async def run() -> _R:
        async with (
            httpx.AsyncClient(timeout=15, trust_env=False) as transport,
            anthropic.AsyncAnthropic(
                api_key=gateway.key, base_url=_origin(gateway), max_retries=0, http_client=transport
            ) as client,
        ):
            return await call(client)

    return asyncio.run(run())


def _user_message() -> Mapping[str, JsonValue]:
    return {"role": "user", "content": TEXT}


def _chat_body(model: str, extra: Mapping[str, JsonValue] = NO_EXTRA) -> Mapping[str, JsonValue]:
    return {"model": model, "messages": [_user_message()], **NO_CACHE, **extra}


def _duplicated_timeout_body(model: str) -> bytes:
    fields: Final = json.dumps(_chat_body(model)).removesuffix("}")
    return f'{fields}, "timeout": 30, "timeout": 30}}'.encode()


def _chat(gateway: Gateway, model: str, extra: Mapping[str, JsonValue] = NO_EXTRA) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", _chat_body(model, extra))


def _response_id(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def _error_message(response: httpx.Response) -> str:
    body: Final = JSON_OBJECT.validate_json(response.content)
    if "error" in body:
        return string_value(object_value(body["error"])["message"])
    detail: Final = body["detail"]
    if isinstance(detail, str):
        return detail
    assert isinstance(detail, list) and detail, response.text
    return string_value(object_value(detail[0])["msg"])


def _assert_rejected(response: httpx.Response, shape: str) -> None:
    assert response.status_code >= 400, (shape, response.status_code, response.text)
    assert _error_message(response), (shape, response.text)


def _assert_served(response: httpx.Response, identity: str, shape: str) -> None:
    assert response.status_code == 200, (shape, response.status_code, response.text)
    assert _response_id(response) == identity, (shape, response.text)


def _moderation_through_the_sdk(gateway: Gateway, model: str) -> ModerationCreateResponse | openai.APIStatusError:
    with _openai(gateway) as client:
        try:
            return client.moderations.create(model=model, input=TEXT)
        except openai.APIStatusError as error:
            return error


def _moderation_after_worker_sync(gateway: Gateway, model: str) -> ModerationCreateResponse | openai.APIStatusError:
    return eventually(
        lambda: _moderation_through_the_sdk(gateway, model),
        lambda outcome: not isinstance(outcome, openai.InternalServerError),
        seconds=WORKER_SYNC_SECONDS + 10,
    )


def _speech_peer(content_type: str) -> Callable[[Request], Reply]:
    def route(request: Request) -> Reply | None:
        if request.method != "POST" or not _path(request).endswith("/audio/speech"):
            return None
        return Reply(body=SPEECH_AUDIO, content_type=content_type)

    return _peer(route)


def _speech_through_the_sdk(gateway: Gateway, model: str) -> httpx.Response | openai.APIStatusError:
    with _openai(gateway) as client:
        try:
            return client.audio.speech.create(model=model, voice="alloy", input=TEXT, response_format="mp3").response
        except openai.APIStatusError as error:
            return error


def _speech_after_worker_sync(gateway: Gateway, model: str) -> httpx.Response | openai.APIStatusError:
    return eventually(
        lambda: _speech_through_the_sdk(gateway, model),
        lambda outcome: not isinstance(outcome, openai.InternalServerError),
        seconds=WORKER_SYNC_SECONDS + 10,
    )


def _attempt(call: Callable[[], _R]) -> _R | openai.APIStatusError:
    try:
        return call()
    except openai.APIStatusError as error:
        return error


def _unknown_to_the_serving_worker(outcome: object) -> bool:
    return isinstance(outcome, openai.BadRequestError) and "Invalid model name" in outcome.response.text


def _outcome_after_worker_sync(call: Callable[[], _R]) -> _R | openai.APIStatusError:
    return eventually(
        lambda: _attempt(call),
        lambda outcome: not _unknown_to_the_serving_worker(outcome),
        seconds=WORKER_SYNC_SECONDS + 10,
    )


def _after_worker_sync(call: Callable[[], _R]) -> _R:
    outcome: Final = _outcome_after_worker_sync(call)
    assert not isinstance(outcome, openai.APIStatusError), outcome.response.text
    return outcome


def _async_after_worker_sync(gateway: Gateway, call: Callable[[openai.AsyncOpenAI], Awaitable[_R]]) -> _R:
    return _after_worker_sync(lambda: _async_openai(gateway, call))


def _streamed_file_content(gateway: Gateway, file_id: str, model: str) -> tuple[int, tuple[bytes, ...]]:
    with gateway.client.stream(
        "GET",
        f"/v1/files/{file_id}/content",
        params={"model": model},
        headers={"Authorization": f"Bearer {gateway.key}"},
    ) as response:
        return response.status_code, tuple(response.iter_bytes())


def _upstream_id(managed: str, prefix: str) -> str:
    encoded: Final = managed.removeprefix(prefix)
    try:
        decoded: Final = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except (binascii.Error, UnicodeDecodeError):
        return managed
    return decoded.removeprefix("litellm:").split(";", 1)[0] if decoded.startswith("litellm:") else managed


def _spend_row(identity: str) -> Mapping[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, call_type, model FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (identity,)
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    return rows[0]


def _spend_row_keyed_by_any(identities: frozenset[str]) -> Mapping[str, JsonValue]:
    keys: Final = tuple(sorted(identities))
    placeholders: Final = ", ".join("%s" for _ in keys)
    rows: Final = eventually(
        lambda: read_rows(
            f'SELECT request_id, call_type, model FROM "LiteLLM_SpendLogs" WHERE request_id IN ({placeholders})', keys
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    return rows[0]


def _spend_request_ids(identities: tuple[str, str, str]) -> frozenset[str]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id IN (%s, %s, %s)', identities),
        lambda rows: len(rows) == 3,
        seconds=70,
    )
    return frozenset(string_value(row["request_id"]) for row in rows)


def _anthropic_text(event: RawMessageStreamEvent) -> str:
    if isinstance(event, RawContentBlockDeltaEvent) and isinstance(event.delta, TextDelta):
        return event.delta.text
    return ""


def _chunk_text(chunk: ChatCompletionChunk) -> str:
    return chunk.choices[0].delta.content or "" if chunk.choices else ""


def _chunk_finished(chunk: ChatCompletionChunk) -> bool:
    return bool(chunk.choices) and chunk.choices[0].finish_reason == "stop"


def _new_model(gateway: Gateway, api_base: str, timeout: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/model/new",
        {
            "model_name": f"integration-{uuid.uuid4().hex}",
            "litellm_params": {
                "model": f"openai/{UPSTREAM_MODEL}",
                "api_key": PROVIDER_KEY,
                "api_base": api_base,
                "timeout": timeout,
            },
            "model_info": {},
        },
    )


def _register_accepted_model(scenario: Scenario, response: httpx.Response, shape: str) -> None:
    assert response.status_code == 200, (shape, response.status_code, response.text)
    identity: Final = string_value(object_value(JSON_OBJECT.validate_json(response.content)["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)


def test_r01_chat_completions_non_stream_through_the_sync_openai_sdk(gateway: Gateway) -> None:
    identity: Final = _identity("chatcmpl")
    with gateway.scenario() as scenario, wire_server(_chat_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        with _openai(gateway) as client:
            response: Final = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": TEXT}], extra_body=dict(NO_CACHE)
            )
        assert response.id == identity, response
        assert response.choices[0].message.content == TEXT, response
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/chat/completions"),), served
        assert served[0].headers["authorization"] == PROVIDER_AUTHORIZATION, served[0].headers
        upstream_body: Final = _body(served[0])
        assert upstream_body["model"] == UPSTREAM_MODEL, upstream_body
        assert upstream_body["messages"] == [_user_message()], upstream_body
        assert _spend_row(identity)["model"] == DEPLOYMENT_MODEL


def test_r02_chat_completions_stream_through_the_async_openai_sdk(gateway: Gateway) -> None:
    identity: Final = _identity("chatcmpl")
    with gateway.scenario() as scenario, wire_server(_chat_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")

        async def stream(client: openai.AsyncOpenAI) -> tuple[ChatCompletionChunk, ...]:
            chunks: Final = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": TEXT}], stream=True, extra_body=dict(NO_CACHE)
            )
            return tuple([chunk async for chunk in chunks])

        received: Final = _async_openai(gateway, stream)
        assert {chunk.id for chunk in received} == {identity}, received
        assert "".join(_chunk_text(chunk) for chunk in received) == TEXT, received
        assert any(_chunk_finished(chunk) for chunk in received), received
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/chat/completions"),), served
        assert _body(served[0])["stream"] is True, served[0].body
        assert _spend_row(identity)["model"] == DEPLOYMENT_MODEL


def test_r03_chat_completions_through_raw_httpx_with_a_body_timeout(gateway: Gateway) -> None:
    identity: Final = _identity("chatcmpl")
    with gateway.scenario() as scenario, wire_server(_chat_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        assert _response_id(_chat(gateway, model, {"timeout": 30})) == identity
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/chat/completions"),), served
        assert "timeout" not in _body(served[0]), served[0].body
        assert _spend_row(identity)["model"] == DEPLOYMENT_MODEL


def test_r04_messages_non_stream_through_the_sync_anthropic_sdk(gateway: Gateway) -> None:
    identity: Final = _identity("resp")
    with gateway.scenario() as scenario, wire_server(_responses_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        with _anthropic(gateway) as client:
            message: Final = client.messages.create(
                model=model, max_tokens=64, messages=[{"role": "user", "content": TEXT}]
            )
        assert identity in response_identities(message.id), message.id
        first_block: Final = message.content[0]
        assert first_block.type == "text" and first_block.text == TEXT, message
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/responses"),), served
        assert TEXT in served[0].body.decode(), served[0].body
        assert _spend_row(message.id)["call_type"] == "anthropic_messages"


def test_r05_messages_stream_through_the_async_anthropic_sdk(gateway: Gateway) -> None:
    identity: Final = _identity("resp")
    with gateway.scenario() as scenario, wire_server(_responses_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")

        async def stream(client: anthropic.AsyncAnthropic) -> tuple[RawMessageStreamEvent, ...]:
            events: Final = await client.messages.create(
                model=model, max_tokens=64, messages=[{"role": "user", "content": TEXT}], stream=True
            )
            return tuple([event async for event in events])

        received: Final = _async_anthropic(gateway, stream)
        first: Final = received[0]
        assert isinstance(first, RawMessageStartEvent), received
        assert received[-1].type == "message_stop", received
        assert "".join(_anthropic_text(event) for event in received) == TEXT, received
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/responses"),), served
        assert _body(served[0])["stream"] is True, served[0].body
        assert _spend_row(first.message.id)["call_type"] == "anthropic_messages"


def test_r06_responses_non_stream_through_the_sync_openai_sdk(gateway: Gateway) -> None:
    identity: Final = _identity("resp")
    with gateway.scenario() as scenario, wire_server(_responses_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        with _openai(gateway) as client:
            response: Final = client.responses.create(model=model, input=TEXT, extra_body=dict(NO_CACHE))
        assert identity in response_identities(response.id), response.id
        assert response.output_text == TEXT, response
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/responses"),), served
        upstream_body: Final = _body(served[0])
        assert upstream_body["model"] == UPSTREAM_MODEL, upstream_body
        assert upstream_body["input"] == TEXT, upstream_body
        assert _spend_row(response.id)["model"] == DEPLOYMENT_MODEL


def test_r07_responses_stream_through_the_async_openai_sdk(gateway: Gateway) -> None:
    identity: Final = _identity("resp")
    with gateway.scenario() as scenario, wire_server(_responses_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")

        async def stream(client: openai.AsyncOpenAI) -> tuple[ResponseStreamEvent, ...]:
            events: Final = await client.responses.create(
                model=model, input=TEXT, stream=True, extra_body=dict(NO_CACHE)
            )
            return tuple([event async for event in events])

        received: Final = _async_openai(gateway, stream)
        assert received[0].type == "response.created", received
        last: Final = received[-1]
        assert last.type == "response.completed", received
        assert identity in response_identities(last.response.id), last
        assert last.response.output_text == TEXT, last
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/responses"),), served
        assert _body(served[0])["stream"] is True, served[0].body
        assert _spend_row_keyed_by_any(response_identities(last.response.id))["model"] == DEPLOYMENT_MODEL


def test_r08_completions_non_stream_through_the_sync_openai_sdk(gateway: Gateway) -> None:
    identity: Final = _identity("cmpl")
    with gateway.scenario() as scenario, wire_server(_completion_peer(identity)) as wire:
        model: Final = scenario.model(model=INSTRUCT_DEPLOYMENT_MODEL, api_base=f"{wire.url}/v1")
        with _openai(gateway) as client:
            completion: Final = client.completions.create(model=model, prompt=TEXT, extra_body=dict(NO_CACHE))
        assert completion.id == identity, completion
        assert completion.choices[0].text == TEXT, completion
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/completions"),), served
        upstream_body: Final = _body(served[0])
        assert upstream_body["model"] == INSTRUCT_MODEL, upstream_body
        assert upstream_body["prompt"] == TEXT, upstream_body
        assert _spend_row(identity)["model"] == INSTRUCT_DEPLOYMENT_MODEL


def test_r09_completions_stream_through_the_async_openai_sdk(gateway: Gateway) -> None:
    identity: Final = _identity("cmpl")
    with gateway.scenario() as scenario, wire_server(_completion_peer(identity)) as wire:
        model: Final = scenario.model(model=INSTRUCT_DEPLOYMENT_MODEL, api_base=f"{wire.url}/v1")

        async def stream(client: openai.AsyncOpenAI) -> tuple[Completion, ...]:
            chunks: Final = await client.completions.create(
                model=model, prompt=TEXT, stream=True, extra_body=dict(NO_CACHE)
            )
            return tuple([chunk async for chunk in chunks])

        received: Final = _async_openai(gateway, stream)
        assert {chunk.id for chunk in received} == {identity}, received
        assert "".join(chunk.choices[0].text or "" for chunk in received if chunk.choices) == TEXT, received
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/completions"),), served
        assert _body(served[0])["stream"] is True, served[0].body
        assert _spend_row(identity)["model"] == INSTRUCT_DEPLOYMENT_MODEL


def test_r10_moderations_through_the_sync_openai_sdk(gateway: Gateway) -> None:
    identity: Final = _identity("modr")
    with gateway.scenario() as scenario, wire_server(_moderation_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        moderation: Final = _moderation_after_worker_sync(gateway, model)
        assert isinstance(moderation, ModerationCreateResponse), moderation
        assert moderation.id == identity, moderation
        assert [result.flagged for result in moderation.results] == [False], moderation
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/moderations"),), served
        assert served[0].headers["authorization"] == PROVIDER_AUTHORIZATION, served[0].headers
        assert _body(served[0])["input"] == TEXT, served[0].body
        assert _spend_row(moderation.id)["call_type"] == "amoderation"


def test_r11_files_create_retrieve_and_content_through_the_sync_openai_sdk(gateway: Gateway) -> None:
    file_id: Final = _identity("file")
    content: Final = _batch_line(file_id)
    with gateway.scenario() as scenario, wire_server(_files_peer(file_id, content)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        with _openai(gateway) as client:
            created: Final = _after_worker_sync(
                lambda: client.files.create(file=("batch.jsonl", content), purpose="batch", extra_body={"model": model})
            )
            retrieved: Final = _after_worker_sync(
                lambda: client.files.retrieve(created.id, extra_query={"model": model})
            )
            downloaded: Final = _after_worker_sync(
                lambda: client.files.content(created.id, extra_query={"model": model})
            ).content
        assert _upstream_id(created.id, "file-") == file_id, created
        assert _upstream_id(retrieved.id, "file-") == file_id, retrieved
        assert retrieved.bytes == len(content), retrieved
        assert downloaded == content, downloaded
        served: Final = _served(wire)
        assert _routes(served) == (
            ("POST", "/v1/files"),
            ("GET", f"/v1/files/{file_id}"),
            ("GET", f"/v1/files/{file_id}/content"),
        ), served
        assert content in served[0].body, served[0].body
        assert {request.headers["authorization"] for request in served} == {PROVIDER_AUTHORIZATION}, served


def test_r12_file_content_streams_through_raw_httpx(gateway: Gateway) -> None:
    file_id: Final = _identity("file")
    content: Final = STREAM_HEAD + STREAM_TAIL
    with (
        gateway.scenario() as scenario,
        wire_server(_files_peer(file_id, content, chunks=(STREAM_HEAD, STREAM_TAIL))) as wire,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        status, received = eventually(
            lambda: _streamed_file_content(gateway, file_id, model),
            lambda outcome: not (outcome[0] == 400 and b"Invalid model name" in b"".join(outcome[1])),
            seconds=WORKER_SYNC_SECONDS + 10,
        )
        assert status == 200, b"".join(received).decode()
        assert b"".join(received) == content, (len(received), sum(len(chunk) for chunk in received))
        served: Final = _served(wire)
        assert _routes(served) == (("GET", f"/v1/files/{file_id}/content"),), served
        assert served[0].headers["authorization"] == PROVIDER_AUTHORIZATION, served[0].headers


def test_r13_batches_create_retrieve_list_and_cancel_through_the_async_openai_sdk(gateway: Gateway) -> None:
    file_id: Final = _identity("file")
    batch_id: Final = _identity("batch")
    with gateway.scenario() as scenario, wire_server(_batches_peer(file_id, batch_id)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        uploaded: Final = _async_after_worker_sync(
            gateway,
            lambda client: client.files.create(
                file=("batch.jsonl", _batch_line("r1")), purpose="batch", extra_body={"model": model}
            ),
        )
        created: Final = _async_after_worker_sync(
            gateway,
            lambda client: client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                extra_body={"model": model},
            ),
        )
        retrieved: Final = _async_after_worker_sync(gateway, lambda client: client.batches.retrieve(created.id))
        listed: Final = _async_after_worker_sync(
            gateway, lambda client: client.batches.list(extra_query={"model": model})
        )
        cancelled: Final = _async_after_worker_sync(gateway, lambda client: client.batches.cancel(created.id))
        listed_ids: Final = tuple(batch.id for batch in listed.data)
        assert _upstream_id(uploaded.id, "file-") == file_id, uploaded
        assert _upstream_id(created.id, "batch_") == batch_id, created
        assert _upstream_id(retrieved.id, "batch_") == batch_id, retrieved
        assert batch_id in [_upstream_id(listed_id, "batch_") for listed_id in listed_ids], listed_ids
        assert _upstream_id(cancelled.id, "batch_") == batch_id, cancelled
        assert cancelled.status == "cancelling", cancelled
        served: Final = _served(wire)
        assert _routes(served) == (
            ("POST", "/v1/files"),
            ("POST", "/v1/batches"),
            ("GET", f"/v1/batches/{batch_id}"),
            ("POST", f"/v1/batches/{batch_id}/cancel"),
        ), served
        assert _body(served[1])["input_file_id"] == file_id, served[1].body
        assert {request.headers["authorization"] for request in served} == {PROVIDER_AUTHORIZATION}, served


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_r14_fine_tuning_jobs_create_list_retrieve_and_cancel_through_the_sync_openai_sdk(
    configured_proxy: ConfiguredProxy,
) -> None:
    suffix: Final = uuid.uuid4().hex
    job_id: Final = f"ftjob-{suffix}"
    with _openai(configured_proxy.gateway) as client:
        created: Final = client.fine_tuning.jobs.create(
            model=UPSTREAM_MODEL, training_file=f"file-{suffix}", extra_body={"custom_llm_provider": "openai"}
        )
        listed: Final = client.fine_tuning.jobs.list(extra_query={"custom_llm_provider": "openai"})
        retrieved: Final = client.fine_tuning.jobs.retrieve(created.id, extra_query={"custom_llm_provider": "openai"})
        cancelled: Final = client.fine_tuning.jobs.cancel(created.id, extra_body={"custom_llm_provider": "openai"})
    assert created.id == job_id, created
    assert created.status == "queued", created
    assert [job.id for job in listed.data] == [LISTED_JOB], listed
    assert retrieved.id == job_id, retrieved
    assert retrieved.training_file == f"file-{suffix}", retrieved
    assert cancelled.id == job_id and cancelled.status == "cancelled", cancelled
    served: Final = _served(configured_proxy.wire)
    assert _routes(served) == (
        ("POST", "/v1/fine_tuning/jobs"),
        ("GET", "/v1/fine_tuning/jobs"),
        ("GET", f"/v1/fine_tuning/jobs/{job_id}"),
        ("POST", f"/v1/fine_tuning/jobs/{job_id}/cancel"),
    ), served
    assert _body(served[0])["training_file"] == f"file-{suffix}", served[0].body
    assert {request.headers["authorization"] for request in served} == {PROVIDER_AUTHORIZATION}, served


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_r15_assistants_list_through_raw_httpx(configured_proxy: ConfiguredProxy) -> None:
    response: Final = configured_proxy.gateway.request("GET", "/v1/assistants")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert body["object"] == "list", response.text
    listed: Final = body["data"]
    assert isinstance(listed, list), response.text
    assert [string_value(object_value(entry)["id"]) for entry in listed] == [ASSISTANT], response.text
    served: Final = _served(configured_proxy.wire)
    assert _routes(served) == (("GET", "/v1/assistants"),), served
    assert served[0].headers["authorization"] == PROVIDER_AUTHORIZATION, served[0].headers


def test_r17_chat_completions_on_an_azure_cloudflare_gateway_deployment_through_the_sync_openai_sdk(
    gateway: Gateway,
) -> None:
    identity: Final = _identity("chatcmpl")
    deployment: Final = f"gpt-4o-mini-{uuid.uuid4().hex[:8]}"
    with gateway.scenario() as scenario, wire_server(_chat_peer(identity)) as wire:
        model: Final = scenario.model(
            model=f"azure/{deployment}", api_base=f"{wire.url}{CLOUDFLARE_GATEWAY}", api_version=AZURE_API_VERSION
        )
        with _openai(gateway) as client:
            response: Final = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": TEXT}], extra_body=dict(NO_CACHE)
            )
            streamed: Final = tuple(
                client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": TEXT}], stream=True, extra_body=dict(NO_CACHE)
                )
            )
        assert response.id == identity, response
        assert response.choices[0].message.content == TEXT, response
        assert {chunk.id for chunk in streamed} == {identity}, streamed
        assert "".join(_chunk_text(chunk) for chunk in streamed) == TEXT, streamed
        served: Final = _served(wire)
        assert _routes(served) == (("POST", f"{CLOUDFLARE_GATEWAY}/{deployment}/chat/completions"),) * 2, served
        assert {request.target.split("?", 1)[1] for request in served} == {f"api-version={AZURE_API_VERSION}"}, served
        assert {request.headers["api-key"] for request in served} == {PROVIDER_KEY}, served
        assert [_streams(request) for request in served] == [False, True], served
        assert _spend_row(identity)["model"] == f"azure/{deployment}"


def test_r18_audio_speech_answers_with_the_upstream_audio_type_through_the_sync_openai_sdk(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, wire_server(_speech_peer("audio/ogg")) as wire:
        model: Final = scenario.model(model=f"openai/{SPEECH_UPSTREAM_MODEL}", api_base=f"{wire.url}/v1")
        speech: Final = _speech_after_worker_sync(gateway, model)
        assert not isinstance(speech, openai.APIStatusError), speech
        assert speech.status_code == 200, speech.text
        assert speech.headers["content-type"] == "audio/ogg", dict(speech.headers)
        assert speech.content == SPEECH_AUDIO, speech.content[:16]
        served: Final = _served(wire)
        assert _routes(served) == (("POST", "/v1/audio/speech"),), served
        assert served[0].headers["authorization"] == PROVIDER_AUTHORIZATION, served[0].headers
        upstream_body: Final = _body(served[0])
        assert (upstream_body["model"], upstream_body["input"], upstream_body["response_format"]) == (
            SPEECH_UPSTREAM_MODEL,
            TEXT,
            "mp3",
        ), upstream_body
        assert _spend_row(speech.headers["x-litellm-call-id"])["call_type"] == "aspeech"


def test_s01_chat_completions_upstream_400_reaches_the_sync_openai_sdk_as_a_bad_request(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, wire_server(_status_peer(openai_error(400))) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", max_retries=0)
        with _openai(gateway) as client, pytest.raises(openai.BadRequestError) as raised:
            client.chat.completions.create(model=model, messages=[_user_message()], extra_body=dict(NO_CACHE))
        assert raised.value.status_code == 400, raised.value
        assert "scripted 400" in raised.value.response.text, raised.value.response.text
        assert _routes(_served(wire)) == (("POST", "/v1/chat/completions"),)


def test_s02_chat_completions_upstream_429_keeps_retry_after_through_raw_httpx(gateway: Gateway) -> None:
    limited: Final = Reply(status=429, body=openai_error(429).body, headers={"retry-after": "7"})
    with gateway.scenario() as scenario, wire_server(_status_peer(limited)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", max_retries=0)
        response: Final = _chat(gateway, model)
        assert response.status_code == 429, response.text
        assert response.headers.get("llm_provider-retry-after") == "7", dict(response.headers)
        assert "scripted 429" in _error_message(response), response.text
        assert _routes(_served(wire)) == (("POST", "/v1/chat/completions"),)


def test_s03_chat_completions_upstream_500_reaches_the_sync_openai_sdk_as_a_server_error(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, wire_server(_status_peer(openai_error(500))) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", max_retries=0)
        with _openai(gateway) as client, pytest.raises(openai.InternalServerError) as raised:
            client.chat.completions.create(model=model, messages=[_user_message()], extra_body=dict(NO_CACHE))
        assert raised.value.status_code == 500, raised.value
        assert "scripted 500" in raised.value.response.text, raised.value.response.text
        assert _routes(_served(wire)) == (("POST", "/v1/chat/completions"),)


def test_s04_chat_completions_body_timeout_shapes_answer_without_breaking_the_proxy(gateway: Gateway) -> None:
    identity: Final = _identity("chatcmpl")
    with gateway.scenario() as scenario, wire_server(_chat_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        for shape, timeout in REJECTED_CHAT_TIMEOUTS.items():
            _assert_rejected(_chat(gateway, model, {"timeout": timeout}), shape)
        for shape, timeout in ACCEPTED_CHAT_TIMEOUTS.items():
            _assert_served(_chat(gateway, model, {"timeout": timeout}), identity, shape)
        duplicated: Final = gateway.client.post(
            "/v1/chat/completions",
            content=_duplicated_timeout_body(model),
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        _assert_served(duplicated, identity, "duplicated key")
        _assert_served(_chat(gateway, model), identity, "follow-up")
        assert len(_served(wire)) == len(ACCEPTED_CHAT_TIMEOUTS) + 2


def test_s05_model_new_timeout_shapes_are_rejected_or_stored_without_breaking_the_proxy(gateway: Gateway) -> None:
    identity: Final = _identity("chatcmpl")
    with gateway.scenario() as scenario, wire_server(_chat_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        for shape, timeout in REJECTED_DEPLOYMENT_TIMEOUTS.items():
            _assert_rejected(_new_model(gateway, f"{wire.url}/v1", timeout), shape)
        for shape, timeout in ACCEPTED_DEPLOYMENT_TIMEOUTS.items():
            _register_accepted_model(scenario, _new_model(gateway, f"{wire.url}/v1", timeout), shape)
        _assert_served(_chat(gateway, model), identity, "follow-up")
        assert _routes(_served(wire)) == (("POST", "/v1/chat/completions"),)


def test_s06_upstream_errors_on_completions_moderations_and_files_reach_the_sync_openai_sdk(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        with wire_server(_status_peer(openai_error(401))) as unauthorized:
            model: Final = scenario.model(
                model=INSTRUCT_DEPLOYMENT_MODEL, api_base=f"{unauthorized.url}/v1", max_retries=0
            )
            with _openai(gateway) as client, pytest.raises(openai.AuthenticationError) as completion_error:
                client.completions.create(model=model, prompt=TEXT, extra_body=dict(NO_CACHE))
            assert completion_error.value.status_code == 401, completion_error.value
            assert "scripted 401" in completion_error.value.response.text, completion_error.value.response.text
            assert _routes(_served(unauthorized)) == (("POST", "/v1/completions"),)
        with wire_server(_status_peer(openai_error(400))) as rejecting:
            rejected_model: Final = scenario.model(api_base=f"{rejecting.url}/v1", max_retries=0)
            moderation_error: Final = _moderation_after_worker_sync(gateway, rejected_model)
            assert isinstance(moderation_error, openai.BadRequestError), moderation_error
            assert "scripted 400" in moderation_error.response.text, moderation_error.response.text
            with _openai(gateway) as client:
                file_error: Final = _outcome_after_worker_sync(
                    lambda: client.files.create(
                        file=("batch.jsonl", _batch_line("rejected")),
                        purpose="batch",
                        extra_body={"model": rejected_model},
                    )
                )
            assert isinstance(file_error, openai.BadRequestError), file_error
            assert "scripted 400" in file_error.response.text, file_error.response.text
            assert _routes(_served(rejecting)) == (("POST", "/v1/moderations"), ("POST", "/v1/files"))


@pytest.mark.parametrize("provider", tuple(PAYMENT_REQUIRED_DEPLOYMENTS))
def test_s07_chat_completions_upstream_402_reaches_the_sync_openai_sdk_as_payment_required(
    gateway: Gateway, provider: str
) -> None:
    with gateway.scenario() as scenario, wire_server(_status_peer(openai_error(402))) as wire:
        model: Final = scenario.model(model=PAYMENT_REQUIRED_DEPLOYMENTS[provider], api_base=wire.url, max_retries=0)
        with _openai(gateway) as client, pytest.raises(openai.APIStatusError) as raised:
            client.chat.completions.create(model=model, messages=[_user_message()], extra_body=dict(NO_CACHE))
        assert raised.value.status_code == 402, raised.value
        assert "scripted 402" in raised.value.response.text, raised.value.response.text
        assert len(_served(wire)) == 1


def test_e01_deployment_timeout_answers_408_and_a_body_timeout_on_the_same_deployment_recovers(
    gateway: Gateway,
) -> None:
    identity: Final = _identity("chatcmpl")
    gate: Final = threading.Event()
    with gateway.scenario() as scenario, wire_server(_held_chat_peer(gate, identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", timeout=0.5, max_retries=0)
        timed_out: Final = _chat(gateway, model)
        gate.set()
        assert timed_out.status_code == 408, timed_out.text
        assert "timeout" in _error_message(timed_out).lower(), timed_out.text
        recovered: Final = _chat(gateway, model, {"timeout": 5})
        assert _response_id(recovered) == identity, recovered.text
        assert _routes(_served(wire)) == (("POST", "/v1/chat/completions"),) * 2


def test_e02_three_identical_no_cache_requests_through_raw_httpx_land_three_spend_rows(gateway: Gateway) -> None:
    served_ids: Final[SimpleQueue[str]] = SimpleQueue()
    with gateway.scenario() as scenario, wire_server(_fresh_chat_peer(served_ids)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        received: Final = (
            _response_id(_chat(gateway, model)),
            _response_id(_chat(gateway, model)),
            _response_id(_chat(gateway, model)),
        )
        assert len(set(received)) == 3, received
        assert _routes(_served(wire)) == (("POST", "/v1/chat/completions"),) * 3
        assert {served_ids.get_nowait() for _ in range(3)} == set(received), received
        assert _spend_request_ids(received) == frozenset(received)


def test_e03_deployment_timeout_as_a_string_still_serves_through_raw_httpx(gateway: Gateway) -> None:
    identity: Final = _identity("chatcmpl")
    with gateway.scenario() as scenario, wire_server(_chat_peer(identity)) as wire:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", timeout="30")
        assert _response_id(_chat(gateway, model)) == identity
        assert _routes(_served(wire)) == (("POST", "/v1/chat/completions"),)
        assert _spend_row(identity)["model"] == DEPLOYMENT_MODEL
