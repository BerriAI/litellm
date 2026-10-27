from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Iterator
from typing import Final

import litellm
import pytest
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, wire_server
from litellm import Router
from litellm.types.utils import BudgetConfig
from pydantic import JsonValue, TypeAdapter
from redis import Redis

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MODEL: Final = "gpt-4o-mini"
_API_KEY: Final = "scripted-budget-test-key"


@pytest.fixture
def redis_client() -> Iterator[Redis]:
    with Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        decode_responses=True,
    ) as client:
        yield client


def _peer(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    provider: Final = request.target.split("/")[1]
    return chat_reply(
        f"chatcmpl-{uuid.uuid4().hex}",
        str(body.get("model") or _MODEL),
        provider,
        stream=body.get("stream") is True,
    )


def _deployment(
    model_name: str,
    deployment_id: str,
    api_base: str,
    model: str,
    max_budget: float | None = None,
) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": model,
            "api_base": api_base,
            "api_key": _API_KEY,
            **({"api_version": "2024-10-21"} if model.startswith("azure/") else {}),
            **(
                {"max_budget": max_budget, "budget_duration": "1d"}
                if max_budget is not None
                else {}
            ),
        },
        "model_info": {"id": deployment_id},
    }


def _router(
    model_list: list[dict[str, JsonValue]],
    redis_client: Redis,
    provider_budget_config: dict[str, BudgetConfig] | None = None,
) -> Router:
    return Router(
        model_list=model_list,
        redis_host=os.environ["REDIS_HOST"],
        redis_port=int(os.environ["REDIS_PORT"]),
        provider_budget_config=provider_budget_config,
        num_retries=0,
        disable_cooldowns=True,
    )


@pytest.mark.asyncio
async def test_provider_budget_routes_to_unlimited_provider(
    redis_client: Redis,
) -> None:
    redis_client.set("provider_spend:openai:1d", "1.0", ex=300)
    with wire_server(_peer) as wire:
        router: Final = _router(
            [
                _deployment(
                    "provider-budget",
                    "openai-deployment",
                    f"{wire.url}/openai/v1",
                    "openai/gpt-4o-mini",
                ),
                _deployment(
                    "provider-budget",
                    "azure-deployment",
                    f"{wire.url}/azure",
                    "azure/gpt-4o-mini",
                ),
            ],
            redis_client=redis_client,
            provider_budget_config={
                "openai": BudgetConfig(budget_duration="1d", max_budget=0.5)
            },
        )
        response: Final = await router.acompletion(
            model="provider-budget",
            messages=[{"role": "user", "content": "provider budget routing"}],
        )
        requests: Final = wire.drain()

    assert response._hidden_params["model_id"] == "azure-deployment"
    assert tuple(request.target for request in requests) == (
        "/azure/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-21",
    )


@pytest.mark.asyncio
async def test_provider_budget_rejects_when_every_provider_is_exhausted(
    redis_client: Redis,
) -> None:
    redis_client.set("provider_spend:openai:1d", "1.0", ex=300)
    with wire_server(_peer) as wire:
        router: Final = _router(
            [
                _deployment(
                    "provider-budget",
                    "openai-deployment",
                    f"{wire.url}/openai/v1",
                    "openai/gpt-4o-mini",
                )
            ],
            redis_client=redis_client,
            provider_budget_config={
                "openai": BudgetConfig(budget_duration="1d", max_budget=0.5)
            },
        )
        with pytest.raises(ValueError, match="budget"):
            await router.acompletion(
                model="provider-budget",
                messages=[{"role": "user", "content": "provider budget exhausted"}],
            )
        requests: Final = wire.drain()

    assert requests == ()


@pytest.mark.asyncio
async def test_deployment_budget_rejects_when_every_deployment_is_exhausted(
    redis_client: Redis,
) -> None:
    deployment_id: Final = f"spent-{uuid.uuid4().hex}"
    redis_client.set(f"deployment_spend:{deployment_id}:1d", "1.0", ex=300)
    with wire_server(_peer) as wire:
        router: Final = _router(
            [
                _deployment(
                    "deployment-budget",
                    deployment_id,
                    f"{wire.url}/spent/v1",
                    "openai/gpt-4o-mini",
                    max_budget=1.0,
                )
            ],
            redis_client=redis_client,
        )
        with pytest.raises(ValueError, match="budget"):
            await router.acompletion(
                model="deployment-budget",
                messages=[{"role": "user", "content": "deployment budget exhausted"}],
            )
        requests: Final = wire.drain()

    assert requests == ()


@pytest.mark.asyncio
async def test_tag_budget_blocks_exhausted_tag_but_allows_another(
    redis_client: Redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_tag: Final = f"chunk3-a-{uuid.uuid4().hex}"
    second_tag: Final = f"chunk3-b-{uuid.uuid4().hex}"
    monkeypatch.setattr(
        litellm,
        "tag_budget_config",
        {
            first_tag: BudgetConfig(budget_duration="1d", max_budget=1.0),
            second_tag: BudgetConfig(budget_duration="1d", max_budget=1.0),
        },
    )
    redis_client.set(f"tag_spend:{first_tag}:1d", "1.0", ex=300)
    redis_client.set(f"tag_spend:{second_tag}:1d", "0.0", ex=300)
    with wire_server(_peer) as wire:
        router: Final = _router(
            [
                _deployment(
                    "tag-budget",
                    f"tag-deployment-{uuid.uuid4().hex}",
                    f"{wire.url}/openai/v1",
                    "openai/gpt-4o-mini",
                )
            ],
            redis_client=redis_client,
        )
        with pytest.raises(ValueError, match="budget"):
            await router.acompletion(
                model="tag-budget",
                messages=[{"role": "user", "content": "blocked tag"}],
                metadata={"tags": [first_tag]},
            )
        response: Final = await router.acompletion(
            model="tag-budget",
            messages=[{"role": "user", "content": "available tag"}],
            metadata={"tags": [second_tag]},
        )
        requests: Final = wire.drain()

    assert response.choices[0].message.content == "openai"
    assert len(requests) == 1
    assert json.loads(requests[0].body)["model"] == _MODEL



_ISOLATED_SPEND_KEY: Final = "provider_spend:hosted_vllm:1d"
_INPUT_COST_PER_TOKEN: Final = 0.001
_OUTPUT_COST_PER_TOKEN: Final = 0.002


def _isolated_provider_deployment(api_base: str) -> dict[str, JsonValue]:
    """Only these two tests route to hosted_vllm, so success callbacks still draining from earlier tests cannot land here."""
    return {
        "model_name": "isolated-budget",
        "litellm_params": {
            "model": "hosted_vllm/llama-3",
            "api_base": api_base,
            "api_key": _API_KEY,
            "input_cost_per_token": _INPUT_COST_PER_TOKEN,
            "output_cost_per_token": _OUTPUT_COST_PER_TOKEN,
        },
        "model_info": {"id": "vllm-deployment"},
    }


def _isolated_budget_router(api_base: str, redis_client: Redis | None) -> Router:
    redis_params: Final = (
        {}
        if redis_client is None
        else {"redis_host": os.environ["REDIS_HOST"], "redis_port": int(os.environ["REDIS_PORT"])}
    )
    return Router(
        model_list=[_isolated_provider_deployment(api_base)],
        provider_budget_config={"hosted_vllm": BudgetConfig(budget_duration="1d", max_budget=10.0)},
        num_retries=0,
        disable_cooldowns=True,
        **redis_params,
    )


async def _complete_and_wait_for_spend(router: Router, previous: float) -> float:
    response: Final = await router.acompletion(
        model="isolated-budget",
        messages=[{"role": "user", "content": "budget tracking"}],
    )
    cost: Final = (
        response.usage.prompt_tokens * _INPUT_COST_PER_TOKEN + response.usage.completion_tokens * _OUTPUT_COST_PER_TOKEN
    )
    deadline: Final = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        spend: float | None = router.cache.in_memory_cache.get_cache(_ISOLATED_SPEND_KEY)
        if spend is not None and spend > previous:
            assert spend == pytest.approx(previous + cost)
            return spend
        await asyncio.sleep(0.05)
    raise AssertionError(f"{_ISOLATED_SPEND_KEY} never rose above {previous}")


@pytest.mark.asyncio
async def test_router_without_redis_tracks_spend_but_queues_no_redis_increments() -> None:
    """A Router with no Redis never flushes, so anything it queues for Redis is retained for the life of the process."""
    with wire_server(_peer) as wire:
        router: Final = _isolated_budget_router(f"{wire.url}/openai/v1", redis_client=None)
        first: Final = await _complete_and_wait_for_spend(router, previous=0.0)
        second: Final = await _complete_and_wait_for_spend(router, previous=first)
        third: Final = await _complete_and_wait_for_spend(router, previous=second)
        wire.drain()

    assert third > second > first > 0.0
    assert router.router_budget_logger is not None
    assert router.router_budget_logger.redis_increment_operation_queue == []


async def _wait_for_redis_spend(redis_client: Redis, at_least: float) -> float:
    deadline: Final = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        raw: str | None = redis_client.get(_ISOLATED_SPEND_KEY)
        if raw is not None and float(raw) >= at_least:
            return float(raw)
        await asyncio.sleep(0.1)
    raise AssertionError(f"{_ISOLATED_SPEND_KEY} in Redis never reached {at_least}")


@pytest.mark.asyncio
async def test_router_with_redis_flushes_queued_spend_increments(redis_client: Redis) -> None:
    redis_client.delete(_ISOLATED_SPEND_KEY)
    with wire_server(_peer) as wire:
        router: Final = _isolated_budget_router(f"{wire.url}/openai/v1", redis_client=redis_client)
        first: Final = await _complete_and_wait_for_spend(router, previous=0.0)
        second: Final = await _complete_and_wait_for_spend(router, previous=first)
        redis_spend: Final = await _wait_for_redis_spend(redis_client, at_least=second)
        wire.drain()

    assert redis_spend == pytest.approx(second)
