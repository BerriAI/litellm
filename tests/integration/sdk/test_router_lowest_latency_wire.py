from __future__ import annotations

import json
import random
import threading
import uuid
from typing import Final

import pytest
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, wire_server
from litellm import Router
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MODEL: Final = "gpt-4o-mini"
_API_KEY: Final = "scripted-lowest-latency-key"


def _peer(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    if "/slow/" in request.target:
        return Reply(
            content_type="application/json",
            chunks=(b"{",),
            gate_after_first=threading.Event(),
        )
    return chat_reply(
        f"chatcmpl-{uuid.uuid4().hex}",
        _MODEL,
        request.target.split("/")[1],
        stream=body.get("stream") is True,
    )


def _deployment(model_group: str, api_base: str, deployment_id: str) -> dict[str, JsonValue]:
    return {
        "model_name": model_group,
        "litellm_params": {
            "model": f"openai/{_MODEL}",
            "api_base": api_base,
            "api_key": _API_KEY,
        },
        "model_info": {"id": deployment_id},
    }


@pytest.mark.asyncio
async def test_first_latency_pick_distribution_reaches_multiple_deployments() -> None:
    random.seed(2025)
    with wire_server(_peer) as wire:
        router: Final = Router(
            model_list=[
                _deployment("distribution", f"{wire.url}/one/v1", "one"),
                _deployment("distribution", f"{wire.url}/two/v1", "two"),
                _deployment("distribution", f"{wire.url}/three/v1", "three"),
            ],
            routing_strategy="latency-based-routing",
            routing_strategy_args={"ttl": 0},
            num_retries=0,
            disable_cooldowns=True,
        )
        responses: Final = tuple(
            [
                await router.acompletion(
                    model="distribution",
                    messages=[{"role": "user", "content": f"pick {index}"}],
                )
                for index in range(24)
            ]
        )
        requests: Final = wire.drain()

    picked: Final = frozenset(response._hidden_params["model_id"] for response in responses)
    assert picked == frozenset({"one", "two", "three"})
    assert len(requests) == 24
    assert all(request.target.endswith("/v1/chat/completions") for request in requests)


@pytest.mark.asyncio
async def test_latency_routing_avoids_timed_out_deployment() -> None:
    with wire_server(_peer) as wire:
        router: Final = Router(
            model_list=[
                {
                    **_deployment("latency", f"{wire.url}/slow/v1", "slow"),
                    "litellm_params": {
                        "model": f"openai/{_MODEL}",
                        "api_base": f"{wire.url}/slow/v1",
                        "api_key": _API_KEY,
                        "timeout": 0.5,
                    },
                },
                _deployment("latency", f"{wire.url}/fast/v1", "fast"),
            ],
            routing_strategy="latency-based-routing",
            num_retries=1,
            allowed_fails=0,
            cooldown_time=60,
        )
        router.lowestlatency_logger.log_success_event(
            kwargs={
                "litellm_params": {
                    "metadata": {"model_group": "latency"},
                    "model_info": {"id": "slow"},
                }
            },
            response_obj={"usage": {"total_tokens": 1}},
            start_time=100.0,
            end_time=100.1,
        )
        router.lowestlatency_logger.log_success_event(
            kwargs={
                "litellm_params": {
                    "metadata": {"model_group": "latency"},
                    "model_info": {"id": "fast"},
                }
            },
            response_obj={"usage": {"total_tokens": 1}},
            start_time=100.0,
            end_time=101.0,
        )
        responses: Final = tuple(
            [
                await router.acompletion(
                    model="latency",
                    messages=[{"role": "user", "content": f"timeout routing {index}"}],
                )
                for index in range(10)
            ]
        )
        requests: Final = wire.drain()

    assert tuple(response._hidden_params["model_id"] for response in responses) == (
        "fast",
    ) * 10
    assert sum("/slow/" in request.target for request in requests) == 1
    assert sum("/fast/" in request.target for request in requests) == 10
    assert all(
        json.loads(request.body)["model"] == _MODEL
        for request in requests
        if "/fast/" in request.target
    )
