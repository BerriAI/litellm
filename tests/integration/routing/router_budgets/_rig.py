"""Owned rig for router budget cells (provider, deployment and tag budgets in ``RouterBudgetLimiting``).

- ``upstream``: OpenAI wire double for chat (plain and SSE) and Responses (plain and SSE); every reply
  bills ``PROMPT_TOKENS`` + ``COMPLETION_TOKENS``. A body carrying ``PROVIDER_FAILURE`` gets HTTP 500.
- ``deployment(...)``: a model_list entry priced at ``PRICE`` per token, so one call costs ``CALL_COST``
  exactly in binary floating point and boundary comparisons are exact.
- ``budget_proxy(...)``: owned Redis plus an owned two-worker proxy booted from a config built here. Each
  proxy owns its Redis, so provider spend keys never leak between files.
- ``send(...)``: one raw httpx call per endpoint shape.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.wire import Reply, Request
from pydantic import JsonValue
from redis import Redis

PRICE: Final = 2**-10
PROMPT_TOKENS: Final = 20
COMPLETION_TOKENS: Final = 12
CALL_COST: Final = (PROMPT_TOKENS + COMPLETION_TOKENS) * PRICE
PROVIDER_FAILURE: Final = "router-budget-provider-failure"
BUDGET_ERROR: Final = "No deployments available - crossed budget"
ENDPOINTS: Final = ("chat", "chat_stream", "messages", "messages_stream", "responses", "responses_stream")


def _sse(events: Sequence[Mapping[str, JsonValue]], *, named: bool) -> bytes:
    frames: Final = (
        (f"event: {event['type']}\n" if named else "") + f"data: {json.dumps(event)}\n\n" for event in events
    )
    return ("".join(frames) + ("" if named else "data: [DONE]\n\n")).encode()


def _chat(body: Mapping[str, JsonValue]) -> Reply:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    usage: Final = {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
    }
    base: Final = {"id": identity, "created": 1, "model": str(body.get("model"))}
    if body.get("stream") is True:
        chunks: Final = (
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "budget"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        )
        return Reply(
            body=_sse([{**base, "object": "chat.completion.chunk", **chunk} for chunk in chunks], named=False),
            content_type="text/event-stream",
        )
    return Reply(
        body=json.dumps(
            {
                **base,
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "budget"}, "finish_reason": "stop"}
                ],
                "usage": usage,
            }
        ).encode()
    )


def _response_object(model: str, status: str) -> dict[str, JsonValue]:
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_budget",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "budget", "annotations": []}],
            }
        ]
        if status == "completed"
        else [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": PROMPT_TOKENS,
            "output_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        }
        if status == "completed"
        else None,
    }


def _responses(body: Mapping[str, JsonValue]) -> Reply:
    model: Final = str(body.get("model"))
    if body.get("stream") is not True:
        return Reply(body=json.dumps(_response_object(model, "completed")).encode())
    events: Final = (
        {"type": "response.created", "sequence_number": 0, "response": _response_object(model, "in_progress")},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_budget",
            "output_index": 0,
            "content_index": 0,
            "delta": "budget",
        },
        {"type": "response.completed", "sequence_number": 2, "response": _response_object(model, "completed")},
    )
    return Reply(body=_sse(events, named=True), content_type="text/event-stream")


def upstream(request: Request) -> Reply:
    if PROVIDER_FAILURE.encode() in request.body:
        return Reply(status=500, body=b'{"error":{"type":"server_error","message":"scripted provider failure"}}')
    body: Final = json.loads(request.body or b"{}")
    if request.target.split("?", 1)[0].endswith("/responses"):
        return _responses(body)
    return _chat(body)


def deployment(
    model_name: str,
    model: str,
    upstream_url: str,
    *,
    model_id: str,
    **litellm_params: JsonValue,
) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": model,
            "api_base": f"{upstream_url}/v1",
            "api_key": "router-budget-provider-key",
            "input_cost_per_token": PRICE,
            "output_cost_per_token": PRICE,
            "max_retries": 0,
            **litellm_params,
        },
        "model_info": {"id": model_id},
    }


def write_config(
    path: Path,
    model_list: Sequence[Mapping[str, JsonValue]],
    *,
    provider_budget_config: Mapping[str, JsonValue] | None = None,
    tag_budget_config: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": list(model_list),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                },
                "litellm_settings": {
                    **({"tag_budget_config": dict(tag_budget_config)} if tag_budget_config is not None else {}),
                    **(litellm_settings or {}),
                },
                "router_settings": {
                    "redis_host": "os.environ/REDIS_HOST",
                    "redis_port": "os.environ/REDIS_PORT",
                    "disable_cooldowns": True,
                    "num_retries": 0,
                    **(
                        {"provider_budget_config": dict(provider_budget_config)}
                        if provider_budget_config is not None
                        else {}
                    ),
                },
            }
        )
    )
    return path


@dataclass(frozen=True, slots=True)
class BudgetRig:
    gateway: Gateway
    redis: OwnedRedis

    def redis_float(self, key: str) -> float | None:
        with Redis(host=self.redis.host, port=self.redis.port, socket_timeout=2) as client:
            value: Final = client.get(key)
        return None if value is None else float(value)

    def settled(self, key: str, expected: float, seconds: float = 15) -> float | None:
        return eventually(
            lambda: self.redis_float(key), lambda spend: spend == expected, seconds=seconds, return_last_on_timeout=True
        )


@contextmanager
def redis_for(tmp_path: Path) -> Iterator[OwnedRedis]:
    directory: Final = tmp_path / f"redis-{uuid.uuid4().hex[:8]}"
    directory.mkdir()
    with owned_redis(directory) as cache:
        yield cache


@contextmanager
def proxy_on(
    gateway: Gateway,
    tmp_path: Path,
    config: Path,
    cache: OwnedRedis,
    *,
    workers: int = 2,
    extra: Mapping[str, str] | None = None,
) -> Iterator[Gateway]:
    with owned_proxy(
        gateway,
        tmp_path,
        {
            "REDIS_HOST": cache.host,
            "REDIS_PORT": str(cache.port),
            "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
            **(extra or {}),
        },
        config=config,
        workers=workers,
    ) as candidate:
        yield candidate


@contextmanager
def budget_proxy(gateway: Gateway, tmp_path: Path, config: Path, *, workers: int = 2) -> Iterator[BudgetRig]:
    with redis_for(tmp_path) as cache, proxy_on(gateway, tmp_path, config, cache, workers=workers) as candidate:
        yield BudgetRig(candidate, cache)


def body_for(endpoint: str, model: str, text: str) -> dict[str, JsonValue]:
    stream: Final = {"stream": True} if endpoint.endswith("_stream") else {}
    if endpoint.startswith("responses"):
        return {"model": model, "input": text, **stream}
    if endpoint.startswith("messages"):
        return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}], **stream}
    return {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        **({"stream": True, "stream_options": {"include_usage": True}} if stream else {}),
    }


def path_for(endpoint: str) -> str:
    if endpoint.startswith("responses"):
        return "/v1/responses"
    if endpoint.startswith("messages"):
        return "/v1/messages"
    return "/v1/chat/completions"


def send(
    gateway: Gateway,
    endpoint: str,
    model: str,
    text: str,
    extra: Mapping[str, JsonValue] | None = None,
) -> httpx.Response:
    return gateway.request("POST", path_for(endpoint), {**body_for(endpoint, model, text), **(extra or {})})


def chat(gateway: Gateway, model: str, text: str, extra: Mapping[str, JsonValue] | None = None) -> httpx.Response:
    return send(gateway, "chat", model, text, extra)


def is_budget_rejection(response: httpx.Response) -> bool:
    return response.status_code != 200 and BUDGET_ERROR in response.text


def until_rejected(
    gateway: Gateway, model: str, text: str, extra: Mapping[str, JsonValue] | None = None, seconds: float = 30
) -> httpx.Response:
    return eventually(lambda: chat(gateway, model, text, extra), is_budget_rejection, seconds=seconds)


def probe_text(label: str) -> str:
    return f"{label} {PROVIDER_FAILURE} {uuid.uuid4().hex}"


def json_body(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def proxy_logs_mentioning(directory: Path, needle: str) -> tuple[str, ...]:
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    logs: Final = (path.read_text(errors="replace") for path in output.glob("owned-proxy-*.log"))
    return tuple(log for log in logs if needle in log)
