from __future__ import annotations

import json
import os
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
