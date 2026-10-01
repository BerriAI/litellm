"""
The cooldown read for one request covers that request's model group, not every
deployment on the router (#44050).
"""

from unittest.mock import patch

import pytest

from litellm import Router
from litellm.router_utils.cooldown_cache import CooldownCache


def _router() -> Router:
    model_list = [
        {
            "model_name": group,
            "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "api_base": "http://localhost:1"},
            "model_info": {"id": f"{group}-{n}"},
        }
        for group, count in (("group-a", 2), ("group-b", 3))
        for n in range(count)
    ]
    return Router(model_list=model_list)


def _pass_through_router() -> Router:
    router = _router()
    for deployment in router.model_list:
        deployment["litellm_params"]["use_in_pass_through"] = True
    return router


def _cooldown(router: Router, deployment_id: str) -> None:
    router.cooldown_cache.add_deployment_to_cooldown(
        model_id=deployment_id,
        original_exception=Exception("boom"),
        exception_status=429,
        cooldown_time=60.0,
    )


def _read_keys(spy) -> set[str]:
    return {key for call in spy.call_args_list for key in call.kwargs["keys"]}


def test_sync_selection_reads_cooldown_only_for_its_group_and_skips_the_cooled_one():
    router = _router()
    _cooldown(router, "group-a-0")

    store = router.cooldown_cache.cooldown_store
    with patch.object(store, "batch_get_cache", wraps=store.batch_get_cache) as spy:
        picked = {router.get_available_deployment(model="group-a")["model_info"]["id"] for _ in range(10)}

    assert picked == {"group-a-1"}
    assert _read_keys(spy) == {CooldownCache.get_cooldown_cache_key(i) for i in ("group-a-0", "group-a-1")}


@pytest.mark.asyncio
async def test_async_selection_reads_cooldown_only_for_its_group_and_skips_the_cooled_one():
    router = _router()
    _cooldown(router, "group-b-1")

    store = router.cooldown_cache.cooldown_store
    with patch.object(store, "async_batch_get_cache", wraps=store.async_batch_get_cache) as spy:
        picked = {
            (await router.async_get_available_deployment(model="group-b", request_kwargs={}))["model_info"]["id"]
            for _ in range(20)
        }

    assert "group-b-1" not in picked
    assert _read_keys(spy) <= {CooldownCache.get_cooldown_cache_key(f"group-b-{n}") for n in range(3)}
    assert _read_keys(spy)


def test_lookup_without_ids_still_reads_every_deployment():
    from litellm.router_utils.cooldown_handlers import _get_cooldown_deployments

    router = _router()
    _cooldown(router, "group-b-2")

    store = router.cooldown_cache.cooldown_store
    with patch.object(store, "batch_get_cache", wraps=store.batch_get_cache) as spy:
        cooled = _get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None)

    assert cooled == ["group-b-2"]
    assert len(_read_keys(spy)) == 5


def test_healthy_deployment_lookup_reads_only_its_group():
    router = _router()
    _cooldown(router, "group-a-1")

    store = router.cooldown_cache.cooldown_store
    with patch.object(store, "batch_get_cache", wraps=store.batch_get_cache) as spy:
        healthy, every = router._get_healthy_deployments(model="group-a", parent_otel_span=None)

    assert [d["model_info"]["id"] for d in healthy] == ["group-a-0"]
    assert len(every) == 2
    assert _read_keys(spy) == {CooldownCache.get_cooldown_cache_key(i) for i in ("group-a-0", "group-a-1")}


@pytest.mark.asyncio
async def test_async_healthy_deployment_lookup_reads_only_its_group():
    router = _router()
    _cooldown(router, "group-b-0")

    store = router.cooldown_cache.cooldown_store
    with patch.object(store, "async_batch_get_cache", wraps=store.async_batch_get_cache) as spy:
        healthy, every = await router._async_get_healthy_deployments(model="group-b", parent_otel_span=None)

    assert sorted(d["model_info"]["id"] for d in healthy) == ["group-b-1", "group-b-2"]
    assert len(every) == 3
    assert _read_keys(spy) == {CooldownCache.get_cooldown_cache_key(f"group-b-{n}") for n in range(3)}


def test_pass_through_selection_reads_only_its_group_and_skips_the_cooled_one():
    router = _pass_through_router()
    _cooldown(router, "group-a-0")

    store = router.cooldown_cache.cooldown_store
    with patch.object(store, "batch_get_cache", wraps=store.batch_get_cache) as spy:
        picked = {
            router.get_available_deployment_for_pass_through(model="group-a")["model_info"]["id"] for _ in range(10)
        }

    assert picked == {"group-a-1"}
    assert _read_keys(spy) == {CooldownCache.get_cooldown_cache_key(i) for i in ("group-a-0", "group-a-1")}
