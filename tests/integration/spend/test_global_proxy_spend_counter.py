import json
import os
import uuid
from itertools import product
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.redis_process import owned_redis
from pydantic import JsonValue, TypeAdapter
from redis import Redis

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import GLOBAL_PROXY_SPEND_CACHE_KEY, LITELLM_PROXY_BUDGET_NAME
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.proxy_server import update_cache

BUDGET_MODEL: Final = "proxy-budget-model"
CONFIG: Final = TypeAdapter(dict[str, JsonValue])
SECTION: Final = TypeAdapter(dict[str, JsonValue])
OWNED_PROXY_SECONDS: Final = 2 * graceful_stop_seconds() + 120


def _proxy_budget_config(gateway: Gateway, directory: Path, max_budget: float) -> Path:
    loaded: Final = CONFIG.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )
    model: Final[JsonValue] = {
        "model_name": BUDGET_MODEL,
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": "integration-provider-key",
            "api_base": f"{gateway.upstream_url}/v1",
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
        },
    }
    path: Final = directory / "proxy_budget_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **loaded,
                "model_list": [model],
                "general_settings": {
                    **SECTION.validate_python(loaded["general_settings"]),
                    "user_api_key_cache_ttl": 600,
                },
                "litellm_settings": {
                    **SECTION.validate_python(loaded["litellm_settings"]),
                    "max_budget": max_budget,
                    "budget_duration": "30d",
                },
            }
        )
    )
    return path


def _chat(proxy: Gateway, key: str, marker: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": BUDGET_MODEL, "max_tokens": 20, "messages": [{"role": "user", "content": marker}]},
        key=key,
    )


def _served_id(proxy: Gateway, key: str) -> str:
    response: Final = _chat(proxy, key, f"proxy budget {uuid.uuid4().hex}")
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    assert isinstance(identity, str), response.text
    return identity


def _logged_spend(proxy: Gateway, request_ids: tuple[str, ...]) -> tuple[float, ...]:
    rows: Final = eventually(
        lambda: proxy.rows(
            'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s::text[])',
            ("{" + ",".join(request_ids) + "}",),
        ),
        lambda found: len(found) == len(request_ids),
        seconds=70,
    )
    return tuple(float(str(row["spend"])) for row in rows)


def _proxy_budget_spend(proxy: Gateway) -> float | None:
    rows: Final = proxy.rows('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id = %s', (LITELLM_PROXY_BUDGET_NAME,))
    return float(str(rows[0]["spend"])) if rows else None


def _counter_key(redis: Redis) -> str:
    found: Final = tuple(str(key, "utf-8") for key in redis.scan_iter(match=f"*{GLOBAL_PROXY_SPEND_CACHE_KEY}"))
    assert len(found) == 1, found
    return found[0]


def _upstream_saw(proxy: Gateway, marker: str) -> bool:
    with httpx.Client(base_url=proxy.upstream_url, timeout=5, trust_env=False) as upstream:
        response: Final = upstream.get("/__observations")
        response.raise_for_status()
        return marker in json.dumps(object_value(response.json())["requests"])


async def _add_request_cost(cache: UserApiKeyCache, response_cost: float) -> None:
    await update_cache(
        token=None,
        user_id=None,
        end_user_id=None,
        team_id=None,
        response_cost=response_cost,
        parent_otel_span=None,
        cache=cache,
    )


@pytest.mark.asyncio
async def test_global_proxy_spend_adds_every_instance_cost_and_stays_cleared_after_a_budget_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "max_budget", 100.0)
    redis_cache: Final = RedisCache(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        namespace=f"global-proxy-spend-{uuid.uuid4().hex}",
    )
    gateway: Final = UserApiKeyCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)
    peer: Final = UserApiKeyCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)

    try:
        await gateway.async_set_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, value=10.0, ttl=60)
        await _add_request_cost(peer, 0.5)
        await _add_request_cost(gateway, 0.25)
        before_reset: Final = (
            gateway.get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, local_only=True),
            await redis_cache.async_get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY),
        )
        ttl_before_reset: Final = await redis_cache.async_get_ttl(key=GLOBAL_PROXY_SPEND_CACHE_KEY)

        await gateway.async_delete_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY)
        await _add_request_cost(peer, 0.5)
        await _add_request_cost(gateway, 0.25)

        assert before_reset == (10.75, 10.75)
        assert ttl_before_reset is not None and 0 < ttl_before_reset <= 60
        assert await redis_cache.async_get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY) is None
    finally:
        await redis_cache.async_delete_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY)


@pytest.mark.timeout(OWNED_PROXY_SECONDS)
def test_a_key_with_no_user_spends_toward_the_proxy_max_budget(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_proxy(gateway, tmp_path, {}, config=_proxy_budget_config(gateway, tmp_path, max_budget=0.01)) as owned,
        owned.scenario() as scenario,
    ):
        key: Final = scenario.key(models=[BUDGET_MODEL])
        (spent,) = _logged_spend(owned, (_served_id(owned, key),))
        assert spent > 0.01, spent
        eventually(
            lambda: _proxy_budget_spend(owned),
            lambda recorded: recorded is not None and recorded > 0.01,
            seconds=70,
        )

        marker: Final = f"over the proxy budget {uuid.uuid4().hex}"
        refused: Final = _chat(owned, key, marker)

        assert refused.status_code != 200, refused.text
        assert object_value(refused.json()["error"])["type"] == "budget_exceeded", refused.text
        assert not _upstream_saw(owned, marker), refused.text


@pytest.mark.timeout(2 * OWNED_PROXY_SECONDS)
def test_two_proxies_add_every_request_to_one_proxy_spend_counter(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _proxy_budget_config(gateway, tmp_path, max_budget=1000.0)
    with owned_redis(tmp_path / "redis") as cache, Redis(host=cache.host, port=cache.port) as redis:
        shared_redis: Final = {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)}
        with owned_proxy(gateway, tmp_path / "first", shared_redis, config=config) as first:
            assert first.database_url is not None
            with (
                owned_proxy(
                    gateway, tmp_path / "second", {**shared_redis, "DATABASE_URL": first.database_url}, config=config
                ) as second,
                first.scenario() as scenario,
            ):
                user_key: Final = scenario.key(user_id=scenario.user(), models=[BUDGET_MODEL])
                userless_key: Final = scenario.key(models=[BUDGET_MODEL])
                counter: Final = _counter_key(redis)
                ttl_before: Final = redis.ttl(counter)

                request_ids: Final = tuple(
                    _served_id(proxy, key) for key, proxy in product((user_key, userless_key), (first, second))
                )
                total: Final = sum(_logged_spend(first, request_ids))
                persisted: Final = eventually(
                    lambda: _proxy_budget_spend(first),
                    lambda spend: spend is not None and spend == pytest.approx(total),
                    seconds=70,
                )

                assert persisted == pytest.approx(total)
                eventually(
                    lambda: redis.get(counter),
                    lambda shared: shared is not None and float(str(shared, "utf-8")) == pytest.approx(total),
                    seconds=70,
                )
                assert 0 < redis.ttl(counter) <= ttl_before
