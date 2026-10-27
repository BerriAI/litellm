"""
Test that per-deployment custom pricing does not pollute the shared backend
model key in litellm.model_cost.

When two deployments share the same backend model (e.g. vertex_ai/gemini-2.5-flash)
and one has explicit zero-cost pricing in model_info, the other deployment
should still use the built-in pricing.
"""

import asyncio
import copy
import itertools
import json
import logging
import os
import pathlib
import queue
import re
import sys
import threading
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_MAX_LRU_CACHE_SIZE
from litellm.litellm_core_utils.ptu_pricing import ptu_config_error
from litellm.litellm_core_utils.llm_cost_calc.utils import SERVICE_TIER_COST_KEY_SUFFIXES
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_SECONDS
from litellm.types.router import (
    Deployment,
    DeploymentTypedDict,
    LiteLLM_Params,
    LiteLLMParamsTypedDict,
    ModelInfo,
)
from litellm.utils import (
    _invalidate_model_cost_lowercase_map,
    reapply_runtime_model_cost_registrations,
)


_COPILOT_TOKEN_EXPIRES_AT: Final = 4_102_444_800


def _simulate_price_data_reload(fetched_catalog):
    """Drive what a price data reload does to this process's litellm state.

    Mirrors `litellm.proxy.proxy_server._swap_in_model_cost_map`, which is the
    one place both reload paths adopt a freshly fetched catalog; that wiring is
    covered in the proxy's own tests, so these exercise the replay itself
    without dragging the proxy in. The provider model sets that helper also
    repopulates are left alone, since nothing here reads them and rebuilding
    them from a two-entry catalog would outlive the test.
    """
    litellm.model_cost = fetched_catalog
    _invalidate_model_cost_lowercase_map()
    reapply_runtime_model_cost_registrations()


def _nested_container_ids(value: object) -> frozenset[int]:
    """Identities of every dict/list reachable from `value`, so two structures can be
    checked for shared mutable state without writing into either one."""
    if isinstance(value, dict):
        return frozenset({id(value)} | {i for v in value.values() for i in _nested_container_ids(v)})
    if isinstance(value, list):
        return frozenset({id(value)} | {i for v in value for i in _nested_container_ids(v)})
    return frozenset()


def _restore_model_cost_entries(original_entries):
    for key, value in original_entries.items():
        if value is None:
            litellm.model_cost.pop(key, None)
        else:
            litellm.model_cost[key] = value
    _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize("initial_count", (1, DEFAULT_MAX_LRU_CACHE_SIZE + 1))
async def test_discovered_limits_survive_deployment_growth_and_removal(
    initial_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    deployments: Final = tuple(
        Deployment(
            model_name=f"local-{index}",
            litellm_params=LiteLLM_Params(
                model="hosted_vllm/local-model", api_base="https://capacity.test/v1", api_key="local-key"
            ),
            model_info=ModelInfo(id=f"capacity-{index}"),
        )
        for index in range(DEFAULT_MAX_LRU_CACHE_SIZE + 2)
    )
    router: Final = Router(model_list=[deployment.to_json() for deployment in deployments[:initial_count]])
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": 4096}]})
        )
    ) as client:
        handler.client = client
        await router.arefresh_model_info(client=handler)
        assert all(
            router.get_configured_token_limits(deployment.model_name) == (4096, 4096)
            for deployment in deployments[:initial_count]
        )
        for deployment in deployments[initial_count:]:
            router.add_deployment(deployment)
            await router._arefresh_deployment_model_info(router.model_list[-1], client=handler)
        assert all(
            router.get_configured_token_limits(deployment.model_name) == (4096, 4096) for deployment in deployments
        )
        for deployment in deployments[-2:]:
            router.delete_deployment(deployment.model_info.id or "")
        await router._arefresh_deployment_model_info(router.model_list[0], client=handler)
        assert all(
            router.get_configured_token_limits(deployment.model_name) == (4096, 4096) for deployment in deployments[:-2]
        )
    _invalidate_model_cost_lowercase_map()


async def test_discovery_discards_metadata_for_a_replaced_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    router: Final = Router(
        model_list=[
            {
                "model_name": "local",
                "litellm_params": {
                    "model": "hosted_vllm/local-model",
                    "api_base": "https://original.test/v1",
                    "api_key": "local-key",
                },
                "model_info": {"id": "replaced-deployment"},
            }
        ]
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "original.test":
            router.upsert_deployment(
                Deployment(
                    model_name="local",
                    litellm_params=LiteLLM_Params(
                        model="hosted_vllm/local-model",
                        api_base="https://replacement.test/v1",
                        api_key="local-key",
                    ),
                    model_info=ModelInfo(id="replaced-deployment"),
                )
            )
            return httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": 8192}]})
        assert request.url.host == "replacement.test"
        return httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": 2048}]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        await router._arefresh_deployment_model_info(router.model_list[0], client=handler)
        assert router.get_configured_token_limits("local") == (None, None)
        await router.arefresh_model_info(client=handler)
        assert router.get_configured_token_limits("local") == (2048, 2048)
    _invalidate_model_cost_lowercase_map()


async def test_discovery_sends_the_secret_behind_an_oidc_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    token: Final = tmp_path / "token"
    token.write_text("local-secret")
    router: Final = Router(model_list=[{
        "model_name": "local",
        "litellm_params": {
            "model": "hosted_vllm/local-model",
            "api_base": "https://discovery.test/v1",
            "api_key": f"oidc/file/{token}",
        },
        "model_info": {"id": "oidc-deployment"},
    }])
    authorizations: Final[list[str | None]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": 4096}]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        await router._arefresh_deployment_model_info(router.model_list[0], client=handler)
    assert authorizations == ["Bearer local-secret"]
    _invalidate_model_cost_lowercase_map()


async def test_discovery_is_isolated_across_routers_and_reused_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    first, second = tuple(
        Router(
            model_list=[
                {
                    "model_name": "local",
                    "litellm_params": {
                        "model": "hosted_vllm/local-model",
                        "api_base": f"https://{host}.test/v1",
                        "api_key": "local-key",
                    },
                    "model_info": {"id": "shared-discovery-id"},
                }
            ]
        )
        for host in ("first", "second")
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "unavailable.test":
            return httpx.Response(503)
        limit: Final = 8192 if request.url.host == "first.test" else 2048
        return httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": limit}]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        await first.arefresh_model_info(client=handler)
        assert second.get_configured_token_limits("local") == (None, None)
        await second.arefresh_model_info(client=handler)
        assert first.get_discovered_model_info("shared-discovery-id")["max_input_tokens"] == 8192
        assert first.get_configured_token_limits("local") == (8192, 8192)
        assert second.get_configured_token_limits("local") == (2048, 2048)
        assert litellm.model_cost["shared-discovery-id"].get("max_input_tokens") is None
        first.upsert_deployment(
            Deployment(
                model_name="local",
                litellm_params=LiteLLM_Params(
                    model="hosted_vllm/local-model",
                    api_base="https://unavailable.test/v1",
                    api_key="local-key",
                ),
                model_info=ModelInfo(id="shared-discovery-id"),
            )
        )
        assert first.get_configured_token_limits("local") == (None, None)
        await first.arefresh_model_info(client=handler)
        assert first.get_configured_token_limits("local") == (None, None)
        assert second.get_configured_token_limits("local") == (2048, 2048)
    _invalidate_model_cost_lowercase_map()


async def test_discovery_refreshes_other_endpoints_while_one_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    second_started: Final = asyncio.Event()
    router: Final = Router(
        model_list=[
            {
                "model_name": host,
                "litellm_params": {
                    "model": "hosted_vllm/local-model",
                    "api_base": f"https://{host}.test/v1",
                    "api_key": "local-key",
                },
            }
            for host in ("first", "second", "third")
        ]
    )

    async def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "first.test":
            await second_started.wait()
        if request.url.host == "second.test":
            second_started.set()
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": 2048}]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        await asyncio.wait_for(router.arefresh_model_info(client=handler), timeout=2)
        assert router.get_configured_token_limits("first") == (2048, 2048)
        assert router.get_configured_token_limits("second") == (None, None)
        assert router.get_configured_token_limits("third") == (2048, 2048)
    _invalidate_model_cost_lowercase_map()


async def test_discovered_limits_expire_after_the_last_successful_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    clock: Final = Mock(return_value=0.0)
    router: Final = Router(
        model_list=[
            {
                "model_name": "local",
                "litellm_params": {
                    "model": "hosted_vllm/local-model",
                    "api_base": "https://expiry.test/v1",
                    "api_key": "local-key",
                },
                "model_info": {"id": "expiring-discovery"},
            }
        ]
    )
    router._discovered_model_info_cache = InMemoryCache(clock=clock, default_ttl=2 * MODEL_INFO_REFRESH_SECONDS)
    responses: Final = iter(
        (
            httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": 4096}]}),
            httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": 8192}]}),
            httpx.Response(503),
        )
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(responses))) as client:
        handler.client = client
        await router.arefresh_model_info(client=handler)
        clock.return_value = MODEL_INFO_REFRESH_SECONDS
        router.cache.in_memory_cache.flush_cache()
        await router.arefresh_model_info(client=handler)
        clock.return_value = 2 * MODEL_INFO_REFRESH_SECONDS + 1
        router.cache.in_memory_cache.flush_cache()
        await router.arefresh_model_info(client=handler)
        assert router.get_configured_token_limits("local") == (8192, 8192)
        group: Final = router.get_model_group_info("local")
        assert group is not None
        assert group.max_input_tokens == 8192
        clock.return_value = 3 * MODEL_INFO_REFRESH_SECONDS + 1
        await router.arefresh_model_info(client=handler)
        assert router.get_configured_token_limits("local") == (None, None)
        expired_group: Final = router.get_model_group_info("local")
        assert expired_group is not None
        assert expired_group.max_input_tokens is None
    _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize("provider", ("hosted_vllm", "openai", "openai_like", "text-completion-openai"))
async def test_discovered_limits_are_isolated_overridable_and_refreshable(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    upstream_limit: Final = iter((8192, 4096, 16384, 2048))

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["authorization"] == "Bearer local-key"
        return httpx.Response(200, json={"data": [{"id": "org/local-model", "max_model_len": next(upstream_limit)}]})

    router: Final = Router(
        model_list=[
            {
                "model_name": "local",
                "litellm_params": {
                    "model": f"{provider}/org/local-model",
                    "api_base": f"https://{host}.test/v1",
                    "api_key": "local-key",
                },
                "model_info": {"id": host, **overrides},
            }
            for host, overrides in (("one", {}), ("two", {"max_output_tokens": 512}))
        ],
        enable_pre_call_checks=True,
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        await router.arefresh_model_info(client=handler)
        first: Final = router.get_router_model_info(id="one", deployment=None, received_model_name="local")
        second: Final = router.get_router_model_info(id="two", deployment=None, received_model_name="local")
        assert (first["max_input_tokens"], first["max_output_tokens"]) == (8192, 8192)
        assert (second["max_input_tokens"], second["max_output_tokens"]) == (4096, 512)
        group: Final = router.get_model_group_info("local")
        assert group is not None
        assert group.max_input_tokens == 8192
        listing: Final = router.get_model_listing_info("local")
        assert listing is not None
        assert listing.max_input_tokens == 8192
        assert router.get_configured_token_limits("local") == (8192, 8192)
        assert router._deployment_max_input_tokens("local", router.model_list[1]) == 4096
        allowed: Final = router._pre_call_checks(
            model="local", healthy_deployments=router.model_list, input="prompt", input_token_count=5000
        )
        assert [deployment["model_info"]["id"] for deployment in allowed] == ["one"]
        assert router.model_list[0]["model_info"].get("max_input_tokens") is None
        assert litellm.model_cost[f"{provider}/org/local-model"].get("max_input_tokens") is None
        router.cache.in_memory_cache.flush_cache()
        await router.arefresh_model_info(client=handler)
        refreshed: Final = router.get_model_group_info("local")
        assert refreshed is not None
        assert refreshed.max_input_tokens == 16384
        assert (
            router.get_router_model_info(id="two", deployment=None, received_model_name="local")["max_output_tokens"]
            == 512
        )
    _invalidate_model_cost_lowercase_map()


async def test_discovery_preserves_input_overrides_and_survives_outages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    responses: Final = iter(
        (
            httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": 4096}]}),
            httpx.Response(503),
        )
    )

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "backend.test"
        assert request.headers["authorization"] == "Bearer local-key"
        assert request.headers["x-tenant"] == "tenant"
        return next(responses)

    router: Final = Router(
        model_list=[
            {
                "model_name": "configured",
                "litellm_params": {
                    "model": "hosted_vllm/local-model",
                    "api_base": "https://backend.test/v1",
                    "api_key": "unused-key",
                    "extra_headers": {"authorization": "Bearer local-key", "X-Tenant": "tenant"},
                },
                "model_info": {"id": "configured", "max_input_tokens": 1024},
            },
            {
                "model_name": "byok",
                "litellm_params": {
                    "model": "openai/local-model",
                    "api_base": "https://caller.test/v1",
                    "use_clientside_credentials": True,
                },
            },
            {"model_name": "default-openai", "litellm_params": {"model": "openai/local-model", "api_key": "unused"}},
        ]
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    responder: Final = Mock(side_effect=respond)
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        handler.client = client
        await router.arefresh_model_info(client=handler)
        assert router.get_configured_token_limits("configured") == (1024, 4096)
        router.cache.in_memory_cache.flush_cache()
        await router.arefresh_model_info(client=handler)
        assert router.get_configured_token_limits("configured") == (1024, 4096)
        assert router.get_configured_token_limits("byok") == (None, None)
        assert next(responses, None) is None
        assert responder.call_count == 2
    _invalidate_model_cost_lowercase_map()


def test_should_not_pollute_shared_key_with_zero_cost_pricing():
    """
    When deployment A has input_cost_per_token=0 and deployment B has no
    custom pricing, deployment B should still report the built-in pricing
    (not zero).
    """
    backend_model = "vertex_ai/gemini-2.5-flash"

    # Grab built-in pricing before creating any router
    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_input_cost = builtin_info["input_cost_per_token"]
    builtin_output_cost = builtin_info["output_cost_per_token"]

    # Sanity: built-in pricing should be non-zero for this model
    assert builtin_input_cost > 0, "Test requires a model with non-zero built-in pricing"
    assert builtin_output_cost > 0, "Test requires a model with non-zero built-in pricing"

    router = Router(
        model_list=[
            # Deployment A: explicit zero-cost pricing
            {
                "model_name": "custom-zero-cost-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-1",
                },
                "model_info": {
                    "id": "deployment-a-zero-cost",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
            # Deployment B: no custom pricing, relies on built-in
            {
                "model_name": "standard-cost-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-2",
                },
                "model_info": {
                    "id": "deployment-b-builtin-cost",
                },
            },
        ],
    )

    # Deployment A: should report zero pricing via its unique model_id
    info_a = router.get_deployment_model_info(
        model_id="deployment-a-zero-cost",
        model_name=backend_model,
    )
    assert info_a is not None
    assert info_a["input_cost_per_token"] == 0.0
    assert info_a["output_cost_per_token"] == 0.0

    # Deployment B: should report built-in pricing, NOT zero
    info_b = router.get_deployment_model_info(
        model_id="deployment-b-builtin-cost",
        model_name=backend_model,
    )
    assert info_b is not None
    assert info_b["input_cost_per_token"] == builtin_input_cost, (
        f"Deployment B should use built-in input cost {builtin_input_cost}, got {info_b['input_cost_per_token']}"
    )
    assert info_b["output_cost_per_token"] == builtin_output_cost, (
        f"Deployment B should use built-in output cost {builtin_output_cost}, got {info_b['output_cost_per_token']}"
    )


def test_should_not_pollute_shared_key_with_custom_nonzero_pricing():
    """
    A deployment with custom (non-zero) pricing should not overwrite
    the shared backend key's built-in pricing.
    """
    backend_model = "vertex_ai/gemini-2.5-flash"

    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_input_cost = builtin_info["input_cost_per_token"]

    router = Router(
        model_list=[
            # Deployment with custom high pricing
            {
                "model_name": "expensive-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-3",
                },
                "model_info": {
                    "id": "deployment-expensive",
                    "input_cost_per_token": 0.99,
                    "output_cost_per_token": 0.99,
                },
            },
            # Deployment relying on built-in pricing
            {
                "model_name": "standard-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-4",
                },
                "model_info": {
                    "id": "deployment-standard",
                },
            },
        ],
    )

    # Custom pricing deployment should see its custom values
    info_expensive = router.get_deployment_model_info(
        model_id="deployment-expensive",
        model_name=backend_model,
    )
    assert info_expensive is not None
    assert info_expensive["input_cost_per_token"] == 0.99
    assert info_expensive["output_cost_per_token"] == 0.99

    # Standard deployment should still see built-in pricing
    info_standard = router.get_deployment_model_info(
        model_id="deployment-standard",
        model_name=backend_model,
    )
    assert info_standard is not None
    assert info_standard["input_cost_per_token"] == builtin_input_cost, (
        f"Standard deployment should use built-in pricing {builtin_input_cost}, "
        f"got {info_standard['input_cost_per_token']}"
    )


@pytest.mark.parametrize("flag", ["supports_regex_lookaround", "supports_assistant_prefill", "supports_sampling_params"])
def test_capability_flag_stays_on_the_deployment_that_set_it(flag: str) -> None:
    """A deployment's capability override must not land on the shared
    ``{provider}/{model}`` key, or every sibling deployment of that model would inherit it."""
    backend_model = "bedrock/us.xai.grok-4.6"
    deploy_id = "grok-deploy-keep-regex"

    builtin_flag = litellm.get_model_info(model=backend_model).get(flag)
    model_keys = {
        deploy_id: litellm.model_cost.get(deploy_id),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "grok-keep-regex",
                    "litellm_params": {"model": backend_model},
                    "model_info": {"id": deploy_id, flag: not builtin_flag},
                }
            ],
        )

        assert litellm.model_cost[deploy_id][flag] is (not builtin_flag)
        assert litellm.get_model_info(model=backend_model).get(flag) is builtin_flag
    finally:
        _restore_model_cost_entries(model_keys)


def test_token_count_settings_stay_on_the_deployment_that_set_them() -> None:
    """A deployment's ``token_count_model`` and ``token_count_framing`` must not land on
    the shared ``{provider}/{model}`` key, or a sibling deployment of that model would count with them."""
    from litellm.proxy.common_utils.token_count_settings import cost_map_entry

    backend_model: Final = "bedrock/us.anthropic.claude-sonnet-5-5"
    deploy_id: Final = "sonnet-deploy-keep-token-count"
    configured: Final = {
        "token_count_model": {"model": "anthropic.claude-opus-5-5", "region": "us-west-2"},
        "token_count_framing": 7,
    }
    builtin: Final = {field: cost_map_entry(backend_model).get(field) for field in configured}
    assert all(builtin[field] != value for field, value in configured.items()), builtin
    model_keys: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key))
        for key in (deploy_id, backend_model, backend_model.removeprefix("bedrock/"))
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "sonnet-keep-token-count",
                    "litellm_params": {"model": backend_model},
                    "model_info": {"id": deploy_id, **configured},
                }
            ],
        )

        assert {field: litellm.model_cost[deploy_id][field] for field in configured} == configured
        assert {field: cost_map_entry(backend_model).get(field) for field in configured} == builtin
    finally:
        _restore_model_cost_entries(model_keys)


def test_should_store_full_pricing_under_deployment_model_id():
    """
    Per-deployment pricing (including zero) should be stored and
    retrievable via the unique model_id key in litellm.model_cost.
    """
    backend_model = "vertex_ai/gemini-2.5-flash"

    router = Router(
        model_list=[
            {
                "model_name": "zero-cost-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-5",
                },
                "model_info": {
                    "id": "deployment-zero-check",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
        ],
    )

    # The model_id entry should exist and have the zero pricing
    entry = litellm.model_cost.get("deployment-zero-check")
    assert entry is not None, "Deployment should be registered by model_id"
    assert entry["input_cost_per_token"] == 0.0
    assert entry["output_cost_per_token"] == 0.0


def test_should_preserve_builtin_pricing_regardless_of_deployment_order():
    """
    The built-in pricing should be preserved no matter which deployment
    is processed first (zero-cost first, or standard first).
    """
    backend_model = "vertex_ai/gemini-2.5-flash"

    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_input_cost = builtin_info["input_cost_per_token"]
    builtin_output_cost = builtin_info["output_cost_per_token"]

    # Order 1: standard first, then zero-cost
    router1 = Router(
        model_list=[
            {
                "model_name": "standard-first",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-6",
                },
                "model_info": {"id": "order1-standard"},
            },
            {
                "model_name": "zero-cost-second",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-7",
                },
                "model_info": {
                    "id": "order1-zero",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
        ],
    )

    info_std_1 = router1.get_deployment_model_info(model_id="order1-standard", model_name=backend_model)
    assert info_std_1["input_cost_per_token"] == builtin_input_cost
    assert info_std_1["output_cost_per_token"] == builtin_output_cost

    # Order 2: zero-cost first, then standard
    router2 = Router(
        model_list=[
            {
                "model_name": "zero-cost-first",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-8",
                },
                "model_info": {
                    "id": "order2-zero",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
            {
                "model_name": "standard-second",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-9",
                },
                "model_info": {"id": "order2-standard"},
            },
        ],
    )

    info_std_2 = router2.get_deployment_model_info(model_id="order2-standard", model_name=backend_model)
    assert info_std_2["input_cost_per_token"] == builtin_input_cost, (
        f"Order should not matter. Expected {builtin_input_cost}, got {info_std_2['input_cost_per_token']}"
    )
    assert info_std_2["output_cost_per_token"] == builtin_output_cost, (
        f"Order should not matter. Expected {builtin_output_cost}, got {info_std_2['output_cost_per_token']}"
    )


def test_responses_prefix_stripped_alias_registered_for_model_list():
    """
    Register ``litellm.model_cost`` under the backend key with ``responses/`` and
    under the stripped key (``responses_api_bridge_check`` removes that segment).
    """
    uid = "responses-strip-alias-test-a1b2c3d4"
    Router(
        model_list=[
            {
                "model_name": "azure-responses-strip-test",
                "litellm_params": {
                    "model": "responses/gpt-strip-test-a1b2c3d4",
                    "custom_llm_provider": "azure",
                    "api_key": "fake-key-strip",
                },
                "model_info": {
                    "id": uid,
                    "supports_native_streaming": True,
                },
            }
        ],
    )
    assert "azure/responses/gpt-strip-test-a1b2c3d4" in litellm.model_cost
    assert "azure/gpt-strip-test-a1b2c3d4" in litellm.model_cost
    assert litellm.model_cost["azure/gpt-strip-test-a1b2c3d4"].get("supports_native_streaming") is True


def test_responses_prefix_stripped_alias_registered_for_add_deployment():
    """Dynamic ``add_deployment`` must mirror ``_create_deployment`` registration."""
    uid = "add-dep-responses-strip-e5f6a7b8"
    router = Router(model_list=[])
    deployment = Deployment(
        model_name="dyn-responses-strip",
        litellm_params=LiteLLM_Params(
            model="responses/gpt-add-strip-e5f6a7b8",
            custom_llm_provider="azure",
            api_key="fake-key-add",
        ),
        model_info=ModelInfo(id=uid, supports_native_streaming=True),
    )
    router.add_deployment(deployment=deployment)
    assert "azure/responses/gpt-add-strip-e5f6a7b8" in litellm.model_cost
    assert "azure/gpt-add-strip-e5f6a7b8" in litellm.model_cost
    assert litellm.model_cost["azure/gpt-add-strip-e5f6a7b8"].get("supports_native_streaming") is True


def test_should_not_downgrade_chatgpt_shared_key_mode_with_alias_override():
    """
    ChatGPT aliases that share the same backend model should not be able to
    downgrade the shared backend key from responses -> chat during router setup.
    """
    from litellm.main import responses_api_bridge_check

    backend_model = "chatgpt/gpt-5.4"
    model_keys = {
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
        "chatgpt-shared-mode-base": copy.deepcopy(litellm.model_cost.get("chatgpt-shared-mode-base")),
        "chatgpt-shared-mode-alias": copy.deepcopy(litellm.model_cost.get("chatgpt-shared-mode-alias")),
    }

    try:
        backend_entry = copy.deepcopy(model_keys[backend_model]) or {}
        backend_entry["litellm_provider"] = "chatgpt"
        backend_entry["mode"] = "responses"
        litellm.model_cost[backend_model] = backend_entry
        _invalidate_model_cost_lowercase_map()

        router = Router(model_list=[])
        with patch.object(Router, "_add_deployment", lambda self, deployment: deployment):
            router._create_deployment(
                deployment_info={},
                _model_name="chatgpt/gpt-5.4",
                _litellm_params={
                    "model": "gpt-5.4",
                    "custom_llm_provider": "chatgpt",
                },
                _model_info={
                    "id": "chatgpt-shared-mode-base",
                    "mode": "responses",
                },
            )
            router._create_deployment(
                deployment_info={},
                _model_name="chatgpt/gpt-5.4-medium",
                _litellm_params={
                    "model": "gpt-5.4",
                    "custom_llm_provider": "chatgpt",
                },
                _model_info={
                    "id": "chatgpt-shared-mode-alias",
                    "mode": "chat",
                },
            )

        assert litellm.model_cost[backend_model]["mode"] == "responses"
        assert "mode" in litellm.model_cost[backend_model]

        bridge_model_info, bridge_model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="chatgpt",
        )
        assert bridge_model == "gpt-5.4"
        assert bridge_model_info["mode"] == "responses"
    finally:
        _restore_model_cost_entries(model_keys)


@pytest.mark.parametrize("service_tier", ("default", "priority", "flex"))
@pytest.mark.parametrize(
    "field",
    ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost", "input_cost_per_token_priority"),
)
@pytest.mark.parametrize("rate", (0.0, 0.004))
@pytest.mark.parametrize("explicit_base", (False, True))
@pytest.mark.parametrize("provider", (None, "bedrock"))
@pytest.mark.parametrize("registration_path", ("direct", "startup", "reload"))
def test_bedrock_deployment_inherits_invoked_profile_around_custom_prices(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    rate: float,
    explicit_base: bool,
    provider: str | None,
    registration_path: str,
    service_tier: str,
) -> None:
    from litellm.cost_calculator import completion_cost
    from litellm.types.utils import ModelResponse, Usage, PromptTokensDetailsWrapper

    model: Final = "openai.gpt-5.6-sol"
    profile: Final = f"us.{model}"
    base_model: Final = f"global.{model}"
    profile_prices: Final = {
        "litellm_provider": "bedrock",
        "mode": "chat",
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.02,
        "cache_read_input_token_cost": 0.001,
        "input_cost_per_token_priority": 0.05,
        "output_cost_per_token_priority": 0.06,
        "input_cost_per_token_flex": 0.007,
        "output_cost_per_token_flex": 0.008,
    }
    base_prices: Final = {**profile_prices, "input_cost_per_token": 0.03, "output_cost_per_token": 0.04}
    monkeypatch.setattr(
        litellm, "model_cost", {**litellm.model_cost, profile: dict(profile_prices), base_model: dict(base_prices)}
    )
    deployment: Final = Deployment(
        model_name="profile-priced",
        litellm_params=LiteLLM_Params(
            model=f"converse/{model}" if provider else f"bedrock/converse/{model}",
            model_id=profile,
            custom_llm_provider=provider,
        ),
        model_info=ModelInfo(
            **{
                "id": "profile-priced-deployment",
                field: rate,
                **({"base_model": f"bedrock/{base_model}"} if explicit_base else {}),
            }
        ),
    )
    if registration_path == "direct":
        Router._register_deployment_pricing(deployment)
    else:
        router: Final = Router(model_list=[deployment.to_json(exclude_none=True)])
        if registration_path == "reload":
            router.replay_model_cost_registrations()
    response: Final = ModelResponse(
        model=model,
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=3,
            total_tokens=103,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=5),
        ),
    )
    catalog: Final = base_prices if explicit_base else profile_prices
    input_field: Final = "input_cost_per_token" + (f"_{service_tier}" if service_tier != "default" else "")
    output_field: Final = "output_cost_per_token" + (f"_{service_tier}" if service_tier != "default" else "")
    input_rate: Final = rate if field == input_field else catalog[input_field]
    output_rate: Final = rate if field == output_field else catalog[output_field]
    cache_rate: Final = rate if field == "cache_read_input_token_cost" else 0.001
    assert completion_cost(
        model=model,
        completion_response=response,
        custom_llm_provider="bedrock",
        custom_pricing=True,
        router_model_id="profile-priced-deployment",
        service_tier=service_tier,
    ) == pytest.approx(95 * input_rate + 3 * output_rate + 5 * cache_rate)
    assert litellm.model_cost[profile] == profile_prices


@pytest.mark.parametrize("catalog_entry", (None, {"litellm_provider": 1, "mode": "chat"}))
@pytest.mark.parametrize("known_model", (False, True))
@pytest.mark.parametrize("explicit_base", (False, True))
@pytest.mark.parametrize("service_tier", ("default", "priority", "flex"))
@pytest.mark.parametrize("override", (None, "cache_read_input_token_cost", "input_cost_per_token_priority"))
@pytest.mark.parametrize("reload", (False, True))
def test_bedrock_deployment_keeps_custom_prices_when_profile_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
    catalog_entry: dict[str, object] | None,
    known_model: bool,
    explicit_base: bool,
    service_tier: str,
    override: str | None,
    reload: bool,
) -> None:
    from litellm.cost_calculator import completion_cost
    from litellm.types.utils import ModelResponse, Usage, PromptTokensDetailsWrapper

    model: Final = "anthropic.unmapped-profile-pricing-test"
    profile: Final = (
        "us.anthropic.invalid-profile-pricing-test"
        if catalog_entry is not None
        else "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/opaque-test"
    )
    configured_prices: Final = {
        "litellm_provider": "bedrock",
        "mode": "chat",
        "input_cost_per_token": 0.03,
        "output_cost_per_token": 0.04,
        "cache_read_input_token_cost": 0.001,
        "input_cost_per_token_priority": 0.05,
        "output_cost_per_token_priority": 0.06,
        "input_cost_per_token_flex": 0.007,
        "output_cost_per_token_flex": 0.008,
    }
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.model_cost,
            **({model: dict(configured_prices)} if known_model else {}),
            **({profile: catalog_entry} if catalog_entry is not None else {}),
        },
    )
    deployment: Final = Deployment(
        model_name="custom-priced",
        litellm_params=LiteLLM_Params(model=f"bedrock/converse/{model}", model_id=profile),
        model_info=ModelInfo(
            id="unmapped-profile-deployment",
            input_cost_per_token=0.01,
            output_cost_per_token=0.02,
            **({override: 0.0} if override else {}),
            **({"base_model": f"bedrock/{profile}"} if explicit_base else {}),
        ),
    )
    router: Final = Router(model_list=[deployment.to_json(exclude_none=True)])
    if reload:
        router.replay_model_cost_registrations()
    inherit_configured: Final = known_model and not explicit_base
    tier_rates: Final = inherit_configured and service_tier != "default"
    input_rate: Final = configured_prices[f"input_cost_per_token_{service_tier}"] if tier_rates else 0.01
    output_rate: Final = configured_prices[f"output_cost_per_token_{service_tier}"] if tier_rates else 0.02
    effective_input: Final = 0.0 if override == "input_cost_per_token_priority" and service_tier == "priority" else input_rate
    cache_rate: Final = 0.0 if override == "cache_read_input_token_cost" else (0.001 if inherit_configured else effective_input)
    assert completion_cost(
        model=model,
        completion_response=ModelResponse(
            model=model,
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=3,
                total_tokens=103,
                prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=80),
            ),
        ),
        custom_llm_provider="bedrock",
        custom_pricing=True,
        router_model_id="unmapped-profile-deployment",
        service_tier=service_tier,
    ) == pytest.approx(20 * effective_input + 80 * cache_rate + 3 * output_rate)
    if known_model:
        assert {
            key: value for key, value in litellm.model_cost[model].items() if key in configured_prices
        } == configured_prices


def test_partial_custom_pricing_inherits_builtin_cache_pricing():
    """A deployment that overrides only input/output cost on a cache-supporting
    model must still bill cache_read and cache_creation tokens. Before the
    fix the deploy-id entry was registered with the user's two fields and
    nothing else, so the cost calculator silently billed cache tokens at 0.
    Regression for the prompt-caching cost dropout reported by the customer.
    """
    backend_model = "anthropic/claude-sonnet-4-5-20250929"
    deploy_id = "claude-deploy-partial-pricing"

    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_cache_create = builtin_info["cache_creation_input_token_cost"]
    builtin_cache_read = builtin_info["cache_read_input_token_cost"]
    assert builtin_cache_create is not None and builtin_cache_create > 0
    assert builtin_cache_read is not None and builtin_cache_read > 0

    model_keys = {
        deploy_id: litellm.model_cost.get(deploy_id),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "claude-custom",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key",
                    },
                    "model_info": {
                        "id": deploy_id,
                        "input_cost_per_token": 0.000003,
                        "output_cost_per_token": 0.000015,
                    },
                }
            ],
        )

        entry = litellm.model_cost[deploy_id]
        assert entry["input_cost_per_token"] == 0.000003
        assert entry["output_cost_per_token"] == 0.000015
        assert entry.get("cache_creation_input_token_cost") == builtin_cache_create
        assert entry.get("cache_read_input_token_cost") == builtin_cache_read
    finally:
        _restore_model_cost_entries(model_keys)


def test_partial_pricing_does_not_overwrite_explicit_cache_fields():
    """When the user explicitly sets cache_*_input_token_cost on a deployment,
    those values must not be replaced by the built-in fallback.
    """
    backend_model = "anthropic/claude-sonnet-4-5-20250929"
    deploy_id = "claude-deploy-explicit-cache"

    explicit_cache_create = 0.00001
    explicit_cache_read = 0.0000005
    builtin_info = litellm.get_model_info(model=backend_model)
    assert builtin_info["cache_creation_input_token_cost"] != explicit_cache_create
    assert builtin_info["cache_read_input_token_cost"] != explicit_cache_read

    model_keys = {
        deploy_id: litellm.model_cost.get(deploy_id),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "claude-custom-explicit",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key",
                    },
                    "model_info": {
                        "id": deploy_id,
                        "input_cost_per_token": 0.000003,
                        "output_cost_per_token": 0.000015,
                        "cache_creation_input_token_cost": explicit_cache_create,
                        "cache_read_input_token_cost": explicit_cache_read,
                    },
                }
            ],
        )

        entry = litellm.model_cost[deploy_id]
        assert entry.get("cache_creation_input_token_cost") == explicit_cache_create
        assert entry.get("cache_read_input_token_cost") == explicit_cache_read
    finally:
        _restore_model_cost_entries(model_keys)


@pytest.mark.parametrize(
    ("deployment_cache_prices", "billed_cache_read_rate"),
    [({}, 2.5e-07), ({"cache_read_input_token_cost": 1e-07}, 1e-07)],
    ids=["inherits-builtin-explicit-rate", "own-cache-read-price-wins"],
)
def test_partial_custom_pricing_bills_explicit_cache_hits(
    deployment_cache_prices: dict[str, float], billed_cache_read_rate: float
):
    backend_model = "dashscope/qwen-explicit-cache-inherit-backend"
    deploy_id = "qwen-deploy-partial-pricing"
    model_keys = {
        deploy_id: litellm.model_cost.get(deploy_id),
        backend_model: litellm.model_cost.get(backend_model),
    }
    litellm.model_cost[backend_model] = {
        "litellm_provider": "dashscope",
        "mode": "chat",
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 7.5e-06,
        "cache_creation_input_token_cost": 3.125e-06,
        "cache_read_input_token_cost": 5e-07,
        "explicit_cache_read_input_token_cost": 2.5e-07,
    }
    _invalidate_model_cost_lowercase_map()
    try:
        Router(
            model_list=[
                {
                    "model_name": "qwen-custom",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key",
                        "input_cost_per_token": 2e-06,
                        "output_cost_per_token": 6e-06,
                        **deployment_cache_prices,
                    },
                    "model_info": {"id": deploy_id},
                }
            ],
        )
        response = litellm.ModelResponse(
            model=backend_model,
            usage=litellm.Usage(
                prompt_tokens=1000,
                completion_tokens=0,
                prompt_tokens_details={"cached_tokens": 1000, "cache_type": "ephemeral"},
            ),
        )

        cost = litellm.completion_cost(
            completion_response=response,
            model=backend_model,
            custom_pricing=True,
            router_model_id=deploy_id,
        )

        assert cost == pytest.approx(1000 * billed_cache_read_rate)
    finally:
        _restore_model_cost_entries(model_keys)


def test_inherit_builtin_cache_pricing_fills_only_missing_fields():
    """Direct unit test of the helper: missing cache fields are filled from the
    backend model's built-in entry, while an explicitly set cache field and the
    user's input/output pricing are left untouched.
    """
    backend_model = "anthropic/claude-sonnet-4-5-20250929"
    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_cache_create = builtin_info["cache_creation_input_token_cost"]
    builtin_cache_read = builtin_info["cache_read_input_token_cost"]
    assert builtin_cache_create is not None and builtin_cache_create > 0
    assert builtin_cache_read is not None and builtin_cache_read > 0

    explicit_cache_read = builtin_cache_read + 1
    model_info = {
        "input_cost_per_token": 0.000003,
        "cache_read_input_token_cost": explicit_cache_read,
    }

    Router._inherit_builtin_cache_pricing(
        model_info=model_info,
        backend_model=backend_model,
        custom_llm_provider="anthropic",
    )

    assert model_info["input_cost_per_token"] == 0.000003
    assert model_info["cache_read_input_token_cost"] == explicit_cache_read
    assert model_info["cache_creation_input_token_cost"] == builtin_cache_create


def test_inherit_builtin_cache_pricing_noop_for_unknown_backend():
    """No canonical entry for the backend model means the helper leaves the
    passed-in dict unchanged rather than raising.
    """
    model_info = {"input_cost_per_token": 0.000003}

    Router._inherit_builtin_cache_pricing(
        model_info=model_info,
        backend_model="this-backend-model-does-not-exist-x9y8z7",
        custom_llm_provider=None,
    )

    assert model_info == {"input_cost_per_token": 0.000003}


_TIER_BACKEND_MODEL: Final = "tier-priced-backend"
_TIER_BACKEND_KEY: Final = f"openai/{_TIER_BACKEND_MODEL}"
_CUSTOM_STANDARD_INPUT_RATE: Final = 0.00011
_CUSTOM_STANDARD_OUTPUT_RATE: Final = 0.00022
_TIER_BACKEND_ENTRY: Final = {
    "key": _TIER_BACKEND_KEY,
    "litellm_provider": "openai",
    "mode": "chat",
    "max_tokens": 123456,
    "input_cost_per_token": 0.00021,
    "output_cost_per_token": 0.00032,
    "input_cost_per_token_ultrafast": 0.00031,
    "output_cost_per_token_ultrafast": 0.00042,
    "input_cost_per_token_priority": 0.00051,
    "output_cost_per_token_priority": 0.00062,
    "input_cost_per_token_flex": 0.00071,
    "output_cost_per_token_flex": 0.00082,
    "input_cost_per_token_balanced": 0.00091,
    "output_cost_per_token_balanced": 0.00102,
    "cache_read_input_token_cost_ultrafast": 0.00013,
    "input_cost_per_token_above_272k_tokens_ultrafast": 0.00014,
    "output_cost_per_token_above_272k_tokens_ultrafast": 0.00015,
    "input_cost_per_token_batches": 0.00016,
    "input_cost_per_token_above_272k_tokens": 0.00017,
}
_AZURE_TIER_BACKEND_KEY: Final = "azure/tier-priced-backend"
_AZURE_TIER_BACKEND_ENTRY: Final = {
    **_TIER_BACKEND_ENTRY,
    "key": _AZURE_TIER_BACKEND_KEY,
    "litellm_provider": "azure",
}


def _register_tier_backend() -> None:
    litellm.model_cost[_TIER_BACKEND_KEY] = copy.deepcopy(_TIER_BACKEND_ENTRY)
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()


def _register_azure_tier_backend() -> None:
    litellm.model_cost[_AZURE_TIER_BACKEND_KEY] = copy.deepcopy(_AZURE_TIER_BACKEND_ENTRY)
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()


def test_inherit_builtin_service_tier_pricing_fills_only_missing_fields() -> None:
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key)) for key in (_TIER_BACKEND_KEY, _TIER_BACKEND_MODEL)
    }
    try:
        _register_tier_backend()
        model_info: Final = {
            "id": "custom-priced-tier-deployment",
            "input_cost_per_token": _CUSTOM_STANDARD_INPUT_RATE,
            "output_cost_per_token": _CUSTOM_STANDARD_OUTPUT_RATE,
            "output_cost_per_token_ultrafast": 0.00999,
        }

        Router._inherit_builtin_service_tier_pricing(
            model_info=model_info,
            backend_model=_TIER_BACKEND_MODEL,
            custom_llm_provider="openai",
        )

        assert model_info == {
            "id": "custom-priced-tier-deployment",
            "input_cost_per_token": _CUSTOM_STANDARD_INPUT_RATE,
            "output_cost_per_token": _CUSTOM_STANDARD_OUTPUT_RATE,
            "input_cost_per_token_ultrafast": _TIER_BACKEND_ENTRY["input_cost_per_token_ultrafast"],
            "output_cost_per_token_ultrafast": 0.00999,
            "input_cost_per_token_priority": _TIER_BACKEND_ENTRY["input_cost_per_token_priority"],
            "output_cost_per_token_priority": _TIER_BACKEND_ENTRY["output_cost_per_token_priority"],
            "input_cost_per_token_flex": _TIER_BACKEND_ENTRY["input_cost_per_token_flex"],
            "output_cost_per_token_flex": _TIER_BACKEND_ENTRY["output_cost_per_token_flex"],
            "input_cost_per_token_balanced": _TIER_BACKEND_ENTRY["input_cost_per_token_balanced"],
            "output_cost_per_token_balanced": _TIER_BACKEND_ENTRY["output_cost_per_token_balanced"],
            "cache_read_input_token_cost_ultrafast": _TIER_BACKEND_ENTRY["cache_read_input_token_cost_ultrafast"],
            "input_cost_per_token_above_272k_tokens_ultrafast": _TIER_BACKEND_ENTRY[
                "input_cost_per_token_above_272k_tokens_ultrafast"
            ],
            "output_cost_per_token_above_272k_tokens_ultrafast": _TIER_BACKEND_ENTRY[
                "output_cost_per_token_above_272k_tokens_ultrafast"
            ],
        }
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


def test_inherit_builtin_service_tier_pricing_noop_without_base_rate_or_backend() -> None:
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key)) for key in (_TIER_BACKEND_KEY, _TIER_BACKEND_MODEL)
    }
    try:
        _register_tier_backend()
        model_info_without_base_rate: Final = {
            "id": "custom-priced-no-base-rate",
            "input_cost_per_token_ultrafast": 0.00031,
        }
        expected_without_base_rate: Final = copy.deepcopy(model_info_without_base_rate)
        Router._inherit_builtin_service_tier_pricing(
            model_info=model_info_without_base_rate,
            backend_model=_TIER_BACKEND_MODEL,
            custom_llm_provider="openai",
        )

        model_info_with_unknown_backend: Final = {
            "id": "custom-priced-unknown-backend",
            "input_cost_per_token": _CUSTOM_STANDARD_INPUT_RATE,
            "output_cost_per_token": _CUSTOM_STANDARD_OUTPUT_RATE,
        }
        expected_with_unknown_backend: Final = copy.deepcopy(model_info_with_unknown_backend)
        Router._inherit_builtin_service_tier_pricing(
            model_info=model_info_with_unknown_backend,
            backend_model="tier-priced-backend-unknown",
            custom_llm_provider="openai",
        )

        assert model_info_without_base_rate == expected_without_base_rate
        assert model_info_with_unknown_backend == expected_with_unknown_backend
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


def test_router_completion_uses_custom_standard_and_backend_ultrafast_pricing() -> None:
    model_id: Final = "tier-priced-deployment"
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key)) for key in (_TIER_BACKEND_KEY, _TIER_BACKEND_MODEL, model_id)
    }
    try:
        _register_tier_backend()
        router: Final = Router(
            model_list=[
                {
                    "model_name": "tier-priced-router",
                    "litellm_params": {
                        "model": _TIER_BACKEND_MODEL,
                        "custom_llm_provider": "openai",
                        "api_key": "sk-tier-pricing-not-used",
                        "input_cost_per_token": _CUSTOM_STANDARD_INPUT_RATE,
                        "output_cost_per_token": _CUSTOM_STANDARD_OUTPUT_RATE,
                    },
                    "model_info": {
                        "id": model_id,
                        "input_cost_per_token": _CUSTOM_STANDARD_INPUT_RATE,
                        "output_cost_per_token": _CUSTOM_STANDARD_OUTPUT_RATE,
                    },
                }
            ]
        )

        ultrafast_response: Final = router.completion(
            model="tier-priced-router",
            messages=[{"role": "user", "content": "tiered pricing"}],
            service_tier="ultrafast",
            mock_response=litellm.ModelResponse(
                model=_TIER_BACKEND_MODEL,
                service_tier="ultrafast",
                usage=litellm.Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
            ),
        )
        standard_response: Final = router.completion(
            model="tier-priced-router",
            messages=[{"role": "user", "content": "standard pricing"}],
            mock_response=litellm.ModelResponse(
                model=_TIER_BACKEND_MODEL,
                usage=litellm.Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
            ),
        )

        assert isinstance(ultrafast_response, litellm.ModelResponse)
        assert ultrafast_response._hidden_params["response_cost"] == pytest.approx(
            1000 * _TIER_BACKEND_ENTRY["input_cost_per_token_ultrafast"]
            + 100 * _TIER_BACKEND_ENTRY["output_cost_per_token_ultrafast"]
        )
        assert isinstance(standard_response, litellm.ModelResponse)
        assert standard_response._hidden_params["response_cost"] == pytest.approx(
            1000 * _CUSTOM_STANDARD_INPUT_RATE + 100 * _CUSTOM_STANDARD_OUTPUT_RATE
        )
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


def test_router_completion_uses_backend_ultrafast_long_context_rates() -> None:
    model_id: Final = "tier-priced-long-context-deployment"
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key)) for key in (_TIER_BACKEND_KEY, _TIER_BACKEND_MODEL, model_id)
    }
    try:
        _register_tier_backend()
        router: Final = Router(
            model_list=[
                {
                    "model_name": "tier-priced-long-context-router",
                    "litellm_params": {
                        "model": _TIER_BACKEND_MODEL,
                        "custom_llm_provider": "openai",
                        "api_key": "sk-tier-pricing-not-used",
                        "input_cost_per_token": _CUSTOM_STANDARD_INPUT_RATE,
                        "output_cost_per_token": _CUSTOM_STANDARD_OUTPUT_RATE,
                    },
                    "model_info": {
                        "id": model_id,
                        "input_cost_per_token": _CUSTOM_STANDARD_INPUT_RATE,
                        "output_cost_per_token": _CUSTOM_STANDARD_OUTPUT_RATE,
                    },
                }
            ]
        )

        response: Final = router.completion(
            model="tier-priced-long-context-router",
            messages=[{"role": "user", "content": "long context tiered pricing"}],
            service_tier="ultrafast",
            mock_response=litellm.ModelResponse(
                model=_TIER_BACKEND_MODEL,
                service_tier="ultrafast",
                usage=litellm.Usage(prompt_tokens=300_000, completion_tokens=100, total_tokens=300_100),
            ),
        )

        assert isinstance(response, litellm.ModelResponse)
        assert response._hidden_params["response_cost"] == pytest.approx(
            300_000 * _TIER_BACKEND_ENTRY["input_cost_per_token_above_272k_tokens_ultrafast"]
            + 100 * _TIER_BACKEND_ENTRY["output_cost_per_token_above_272k_tokens_ultrafast"]
        )
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("ptu_enabled", (True, False))
def test_ptu_service_tier_pricing_is_disabled_only_when_attribution_is_enabled(
    monkeypatch: pytest.MonkeyPatch, ptu_enabled: bool
) -> None:
    model_id: Final = f"ptu-tier-deployment-{ptu_enabled}"
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key)) for key in (_TIER_BACKEND_KEY, model_id)
    }
    try:
        _register_tier_backend()
        monkeypatch.setenv("LITELLM_ENABLE_PTU_COST_ATTRIBUTION", "True" if ptu_enabled else "")
        router: Final = Router(
            model_list=[
                {
                    "model_name": f"ptu-tier-model-{ptu_enabled}",
                    "litellm_params": {
                        "model": _TIER_BACKEND_MODEL,
                        "custom_llm_provider": "openai",
                        "api_key": "sk-tier-pricing-not-used",
                        "input_cost_per_token": _CUSTOM_STANDARD_INPUT_RATE,
                        "output_cost_per_token": _CUSTOM_STANDARD_OUTPUT_RATE,
                    },
                    "model_info": {**_PTU_MODEL_INFO, "id": model_id},
                }
            ]
        )
        registered: Final = litellm.model_cost[model_id]
        tier_fields: Final = tuple(
            field for field in _TIER_BACKEND_ENTRY if field.endswith(SERVICE_TIER_COST_KEY_SUFFIXES)
        )
        if ptu_enabled:
            assert all(field not in registered for field in tier_fields)
        else:
            assert all(field in registered for field in tier_fields)

        response: Final = router.completion(
            model=f"ptu-tier-model-{ptu_enabled}",
            messages=[{"role": "user", "content": "ptu service tier pricing"}],
            service_tier="priority",
            mock_response=litellm.ModelResponse(
                model=_TIER_BACKEND_MODEL,
                service_tier="priority",
                usage=litellm.Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
            ),
        )

        assert isinstance(response, litellm.ModelResponse)
        expected_cost: Final = (
            0.0
            if ptu_enabled
            else 1000 * _TIER_BACKEND_ENTRY["input_cost_per_token_priority"]
            + 100 * _TIER_BACKEND_ENTRY["output_cost_per_token_priority"]
        )
        assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


def test_azure_base_model_inherits_service_tier_pricing_for_registration_and_payload() -> None:
    model_id: Final = "azure-tier-priced-alias"
    payload_id: Final = "azure-tier-priced-payload"
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key)) for key in (_AZURE_TIER_BACKEND_KEY, model_id, payload_id)
    }
    try:
        _register_azure_tier_backend()
        router: Final = Router(
            model_list=[
                {
                    "model_name": "azure/tier-priced-alias",
                    "litellm_params": {
                        "model": "azure/tier-priced-alias",
                        "custom_llm_provider": "azure",
                        "api_key": "sk-tier-pricing-not-used",
                        "api_base": "https://tier-priced.azure.invalid",
                    },
                    "model_info": {
                        "id": model_id,
                        "base_model": _AZURE_TIER_BACKEND_KEY,
                        "input_cost_per_token": _CUSTOM_STANDARD_INPUT_RATE,
                        "output_cost_per_token": _CUSTOM_STANDARD_OUTPUT_RATE,
                    },
                }
            ]
        )

        response: Final = router.completion(
            model="azure/tier-priced-alias",
            messages=[{"role": "user", "content": "azure base model pricing"}],
            service_tier="priority",
            allowed_openai_params=["service_tier"],
            mock_response=litellm.ModelResponse(
                model=_AZURE_TIER_BACKEND_KEY,
                service_tier="priority",
                usage=litellm.Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
            ),
        )

        assert isinstance(response, litellm.ModelResponse)
        assert response._hidden_params["response_cost"] == pytest.approx(
            1000 * _AZURE_TIER_BACKEND_ENTRY["input_cost_per_token_priority"]
            + 100 * _AZURE_TIER_BACKEND_ENTRY["output_cost_per_token_priority"]
        )

        payload: Final = Router._deployment_model_cost_payload(
            deployment=Deployment(
                model_name="azure/tier-priced-alias-from-params",
                litellm_params=LiteLLM_Params(
                    model="azure/tier-priced-alias",
                    custom_llm_provider="azure",
                    base_model=_AZURE_TIER_BACKEND_KEY,
                    input_cost_per_token=_CUSTOM_STANDARD_INPUT_RATE,
                    output_cost_per_token=_CUSTOM_STANDARD_OUTPUT_RATE,
                ),
                model_info=ModelInfo(id=payload_id),
            )
        )

        assert payload["input_cost_per_token_priority"] == _AZURE_TIER_BACKEND_ENTRY["input_cost_per_token_priority"]
        assert payload["output_cost_per_token_priority"] == _AZURE_TIER_BACKEND_ENTRY["output_cost_per_token_priority"]
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


@pytest.mark.parametrize(
    ("model_info_base_model", "params_base_model", "model", "expected"),
    (
        pytest.param(
            "azure/tier-priced-model-info-base",
            "azure/tier-priced-params-base",
            "azure/tier-priced-deployment-alias",
            "azure/tier-priced-model-info-base",
            id="model-info-base-model-wins",
        ),
        pytest.param(
            None,
            "azure/tier-priced-params-base",
            "azure/tier-priced-deployment-alias",
            "azure/tier-priced-params-base",
            id="params-base-model-fallback",
        ),
        pytest.param(
            None,
            None,
            "azure/tier-priced-deployment-alias",
            "azure/tier-priced-deployment-alias",
            id="model-fallback",
        ),
        pytest.param(
            "",
            "azure/tier-priced-params-base",
            "azure/tier-priced-deployment-alias",
            "azure/tier-priced-params-base",
            id="empty-model-info-base-model-falls-through",
        ),
    ),
)
def test_cost_map_backend_model_uses_canonical_model_precedence(
    model_info_base_model: str | None,
    params_base_model: str | None,
    model: str,
    expected: str,
) -> None:
    deployment: Final = Deployment(
        model_name="azure/tier-priced-cost-map-backend",
        litellm_params=LiteLLM_Params(model=model, base_model=params_base_model),
        model_info=ModelInfo(id="tier-priced-cost-map-backend", base_model=model_info_base_model),
    )

    assert Router._cost_map_backend_model(deployment) == expected


def test_inherit_builtin_base_rates_for_off_peak_fills_missing_rates():
    """Direct unit test of the helper: an entry carrying only an
    off_peak_pricing block inherits the backend model's built-in base token
    rates, so cost lookup via the deployment id can bill standard rates
    outside the windows.
    """
    backend_model = "gpt-4o-mini"
    builtin_info = litellm.get_model_info(model=backend_model, custom_llm_provider="openai")
    off_peak_block = {
        "hours_utc": "00:00-00:00",
        "input_cost_per_token": 5e-07,
        "output_cost_per_token": 1e-06,
    }
    model_info = {"off_peak_pricing": off_peak_block}

    Router._inherit_builtin_base_rates_for_off_peak(
        model_info=model_info,
        backend_model=backend_model,
        custom_llm_provider="openai",
    )

    assert model_info["input_cost_per_token"] == builtin_info["input_cost_per_token"]
    assert model_info["output_cost_per_token"] == builtin_info["output_cost_per_token"]
    assert model_info["off_peak_pricing"] == off_peak_block


def test_inherit_builtin_base_rates_for_off_peak_carries_threshold_rates():
    """A backend with above-threshold pricing hands the whole rate structure to
    the deployment entry, so peak-hour billing of large prompts through that
    entry matches the shared backend entry instead of flattening to the base
    rate.
    """
    backend_model = "gemini/gemini-2.5-pro"
    builtin_info = litellm.get_model_info(model=backend_model)
    assert builtin_info["input_cost_per_token_above_200k_tokens"] is not None

    model_info = {
        "off_peak_pricing": {"hours_utc": "00:00-00:00", "input_cost_per_token": 5e-07},
    }

    Router._inherit_builtin_base_rates_for_off_peak(
        model_info=model_info,
        backend_model=backend_model,
        custom_llm_provider="gemini",
    )

    assert model_info["input_cost_per_token"] == builtin_info["input_cost_per_token"]
    assert (
        model_info["input_cost_per_token_above_200k_tokens"] == builtin_info["input_cost_per_token_above_200k_tokens"]
    )
    assert (
        model_info["output_cost_per_token_above_200k_tokens"] == builtin_info["output_cost_per_token_above_200k_tokens"]
    )


def test_inherit_builtin_base_rates_for_off_peak_carries_companion_billing_fields():
    """Billing rules that are not literal cost rates, like the web search
    billing unit, must ride along, or grounding and regional uplifts would
    bill differently through the deployment entry than through the shared
    backend entry.
    """
    backend_model = "gemini-3-pro-image"
    raw_entry = litellm.model_cost[backend_model]
    assert raw_entry.get("web_search_billing_unit") is not None

    model_info = {
        "off_peak_pricing": {"hours_utc": "00:00-00:00", "input_cost_per_token": 5e-07},
    }

    Router._inherit_builtin_base_rates_for_off_peak(
        model_info=model_info,
        backend_model=backend_model,
        custom_llm_provider=None,
    )

    assert model_info["web_search_billing_unit"] == raw_entry["web_search_billing_unit"]
    assert model_info["input_cost_per_token"] == raw_entry["input_cost_per_token"]


def test_inherit_builtin_base_rates_for_off_peak_tiered_only_backend_stores_no_zero():
    """A tiered-only backend has no flat token rates; get_model_info synthesizes
    zeros for them, and storing those would mark the deployment explicitly
    priced free. The tier table itself must carry over as an isolated copy so
    mutating the deployment entry never touches the shared cost map.
    """
    backend_model = "dashscope/qwen-flash"
    raw_tiers = litellm.model_cost[backend_model]["tiered_pricing"]

    model_info = {
        "off_peak_pricing": {"hours_utc": "00:00-00:00", "input_cost_per_token": 5e-07},
    }

    Router._inherit_builtin_base_rates_for_off_peak(
        model_info=model_info,
        backend_model=backend_model,
        custom_llm_provider="dashscope",
    )

    assert model_info.get("input_cost_per_token") != 0
    assert model_info.get("output_cost_per_token") != 0
    assert model_info["tiered_pricing"] == raw_tiers
    assert model_info["tiered_pricing"] is not raw_tiers
    assert model_info["tiered_pricing"][0] is not raw_tiers[0]

    original_first_tier = copy.deepcopy(raw_tiers[0])
    model_info["tiered_pricing"][0]["input_cost_per_token"] = 123.0
    assert raw_tiers[0] == original_first_tier


def test_inherit_builtin_base_rates_for_off_peak_leaves_explicit_rates_alone():
    """An entry that sets its own base rate beside the block already counts as
    a full custom pricing entry; the helper must not mix builtin rates into it.
    """
    model_info = {
        "off_peak_pricing": {"hours_utc": "00:00-00:00", "input_cost_per_token": 5e-07},
        "input_cost_per_token": 3e-06,
    }

    Router._inherit_builtin_base_rates_for_off_peak(
        model_info=model_info,
        backend_model="gpt-4o-mini",
        custom_llm_provider="openai",
    )

    assert model_info["input_cost_per_token"] == 3e-06
    assert "output_cost_per_token" not in model_info


def test_inherit_builtin_base_rates_for_off_peak_noop_without_block_or_backend():
    """Nothing happens without an off_peak_pricing block, and an unmapped
    backend model leaves the entry unchanged rather than raising.
    """
    plain_info = {"id": "dep-1"}
    Router._inherit_builtin_base_rates_for_off_peak(
        model_info=plain_info,
        backend_model="gpt-4o-mini",
        custom_llm_provider="openai",
    )
    assert plain_info == {"id": "dep-1"}

    off_peak_info = {"off_peak_pricing": {"hours_utc": "00:00-00:00", "input_cost_per_token": 5e-07}}
    Router._inherit_builtin_base_rates_for_off_peak(
        model_info=off_peak_info,
        backend_model="this-backend-model-does-not-exist-x9y8z7",
        custom_llm_provider=None,
    )
    assert "input_cost_per_token" not in off_peak_info


def test_custom_pricing_field_denylist_covers_all_builtin_pricing_fields():
    """The shared-backend-key stripping in Router relies on
    CustomPricingLiteLLMParams enumerating every per-deployment pricing field.
    If a new pricing field is added to ModelInfoBase but not mirrored here, a
    deployment override on that field leaks into the shared backend key and
    every sibling deployment reads the wrong rate (LIT-3897). This guard fails
    fast when the two drift apart.
    """
    import typing

    from litellm.types.utils import CustomPricingLiteLLMParams, ModelInfoBase

    pricing_markers = ("cost", "price", "uplift", "vector_size", "tiered_pricing")
    builtin_pricing_fields = {
        name for name in typing.get_type_hints(ModelInfoBase) if any(marker in name for marker in pricing_markers)
    }
    denylisted_fields = set(CustomPricingLiteLLMParams.model_fields.keys())

    uncovered = sorted(builtin_pricing_fields - denylisted_fields)
    assert not uncovered, (
        "ModelInfoBase pricing fields missing from CustomPricingLiteLLMParams; "
        f"these would leak into shared backend keys: {uncovered}"
    )


def test_tiered_pricing_override_isolated_from_sibling_via_model_info_lookup():
    """LIT-3897: a deployment that overrides a tiered pricing field
    (input_cost_per_token_above_272k_tokens) must not pollute the shared
    backend key, so a sibling sharing the same backend resolves its pricing
    via litellm.get_model_info (the path /model/info uses) without seeing the
    override.
    """
    backend_model = "gemini/gemini-2.5-flash"
    override = 0.000999

    builtin_info = litellm.get_model_info(model=backend_model)
    assert builtin_info.get("input_cost_per_token_above_272k_tokens") != override

    model_keys = {
        "lit3897-tiered-custom": litellm.model_cost.get("lit3897-tiered-custom"),
        "lit3897-tiered-sibling": litellm.model_cost.get("lit3897-tiered-sibling"),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "custom-priced-flash",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-tiered-1",
                    },
                    "model_info": {
                        "id": "lit3897-tiered-custom",
                        "input_cost_per_token_above_272k_tokens": override,
                        "cache_read_input_token_cost_above_272k_tokens": override,
                    },
                },
                {
                    "model_name": "gemini-2.5-flash",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-tiered-2",
                    },
                    "model_info": {"id": "lit3897-tiered-sibling"},
                },
            ],
        )

        shared = litellm.get_model_info(model=backend_model)
        assert shared.get("input_cost_per_token_above_272k_tokens") != override, (
            "Tiered override leaked into the shared backend key; siblings read the wrong rate via /model/info"
        )
        assert shared.get("cache_read_input_token_cost_above_272k_tokens") != override

        custom_entry = litellm.model_cost["lit3897-tiered-custom"]
        assert custom_entry["input_cost_per_token_above_272k_tokens"] == override
        assert custom_entry["cache_read_input_token_cost_above_272k_tokens"] == override
    finally:
        _restore_model_cost_entries(model_keys)


def test_custom_pricing_isolated_from_sibling_via_proxy_model_info_path():
    """LIT-3897 end to end through the proxy resolution helper: the override
    deployment reports its custom input rate while the sibling keeps the
    canonical gemini rate when /model/info resolves each deployment. Mirrors the
    ticket config where the override is set on litellm_params.
    """
    from litellm.proxy.proxy_server import _get_proxy_model_info

    backend_model = "gemini/gemini-2.5-flash"
    override_input = 5e-05
    override_output = 1e-04

    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_input = builtin_info["input_cost_per_token"]
    assert builtin_input != override_input

    model_keys = {
        "lit3897-proxy-custom": litellm.model_cost.get("lit3897-proxy-custom"),
        "lit3897-proxy-sibling": litellm.model_cost.get("lit3897-proxy-sibling"),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "custom-priced-flash",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-proxy-1",
                        "input_cost_per_token": override_input,
                        "output_cost_per_token": override_output,
                    },
                    "model_info": {"id": "lit3897-proxy-custom"},
                },
                {
                    "model_name": "gemini-2.5-flash",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-proxy-2",
                    },
                    "model_info": {"id": "lit3897-proxy-sibling"},
                },
            ],
        )

        resolved = {
            m["model_name"]: _get_proxy_model_info(model=copy.deepcopy(m))["model_info"]["input_cost_per_token"]
            for m in router.model_list
        }

        assert resolved["custom-priced-flash"] == override_input
        assert resolved["gemini-2.5-flash"] == builtin_input
        assert resolved["gemini-2.5-flash"] != resolved["custom-priced-flash"]
    finally:
        _restore_model_cost_entries(model_keys)


def test_custom_model_info_metadata_not_leaked_to_shared_backend_key():
    """LIT-4544: two deployments share the same backend model but carry
    different custom model_info (arbitrary keys, access_via_team_ids, ids).
    None of that per-deployment metadata may land on the shared backend key in
    litellm.model_cost (served raw by /public/litellm_model_cost_map);
    before the fix it was merged last-write-wins so values flipped randomly.
    """
    backend_model = "openai/gpt-4o-mini"
    shared_keys = ("gpt-4o-mini", backend_model)
    leak_fields = ("id", "additionalProp1", "access_via_team_ids", "db_model")

    model_keys = {
        key: copy.deepcopy(litellm.model_cost.get(key))
        for key in (*shared_keys, "lit4544-deploy-a", "lit4544-deploy-b")
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "alias-unrestricted",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-a",
                    },
                    "model_info": {
                        "id": "lit4544-deploy-a",
                        "additionalProp1": {"restricted": False, "model_location": "EU"},
                    },
                },
                {
                    "model_name": "alias-restricted",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-b",
                    },
                    "model_info": {
                        "id": "lit4544-deploy-b",
                        "additionalProp1": {"restricted": True, "model_location": "US"},
                        "access_via_team_ids": ["team-b-only"],
                    },
                },
            ],
        )

        for shared_key in shared_keys:
            shared_entry = litellm.model_cost.get(shared_key) or {}
            leaked = [field for field in leak_fields if field in shared_entry]
            assert not leaked, f"per-deployment metadata {leaked} leaked onto shared key {shared_key}: {shared_entry}"

        entry_a = litellm.model_cost["lit4544-deploy-a"]
        assert entry_a["additionalProp1"] == {"restricted": False, "model_location": "EU"}
        entry_b = litellm.model_cost["lit4544-deploy-b"]
        assert entry_b["additionalProp1"] == {"restricted": True, "model_location": "US"}
        assert entry_b["access_via_team_ids"] == ["team-b-only"]
    finally:
        _restore_model_cost_entries(model_keys)


def test_add_deployment_does_not_leak_custom_metadata_to_shared_backend_key():
    """LIT-4544 dynamic path: deployments added at runtime (e.g. loaded from
    the DB every scheduler cycle) must not re-pollute the shared backend key
    with per-deployment metadata either.
    """
    backend_model = "openai/gpt-4o-mini"
    shared_keys = ("gpt-4o-mini", backend_model)
    deploy_id = "lit4544-add-deployment"

    model_keys = {key: copy.deepcopy(litellm.model_cost.get(key)) for key in (*shared_keys, deploy_id)}
    try:
        router = Router(model_list=[])
        router.add_deployment(
            deployment=Deployment(
                model_name="alias-dynamic",
                litellm_params=LiteLLM_Params(
                    model=backend_model,
                    api_key="fake-key-dynamic",
                ),
                model_info=ModelInfo(
                    id=deploy_id,
                    additionalProp1={"restricted": True},
                    access_via_team_ids=["team-dynamic"],
                ),
            )
        )

        for shared_key in shared_keys:
            shared_entry = litellm.model_cost.get(shared_key) or {}
            leaked = [
                field for field in ("id", "additionalProp1", "access_via_team_ids", "db_model") if field in shared_entry
            ]
            assert not leaked, f"per-deployment metadata {leaked} leaked onto shared key {shared_key}: {shared_entry}"

        assert litellm.model_cost[deploy_id]["access_via_team_ids"] == ["team-dynamic"]
    finally:
        _restore_model_cost_entries(model_keys)


def test_shared_backend_model_info_keeps_schema_fields_and_drops_the_rest():
    """Unit test of the whitelist helper: cost-map schema fields survive,
    custom pricing overrides and per-deployment metadata do not.
    """
    from litellm.types.utils import shared_backend_model_info

    filtered = shared_backend_model_info(
        {
            "mode": "chat",
            "litellm_provider": "openai",
            "max_tokens": 128000,
            "supports_vision": True,
            "supported_endpoints": ["/v1/responses"],
            "use_openai_responses_path": True,
            "input_cost_per_token": 0.99,
            "output_cost_per_token": 0.99,
            "id": "deploy-a",
            "db_model": False,
            "access_via_team_ids": ["team-a"],
            "additionalProp1": {"restricted": True},
            "base_model": "gpt-4o-mini",
        }
    )

    assert filtered == {
        "mode": "chat",
        "litellm_provider": "openai",
        "max_tokens": 128000,
        "supports_vision": True,
        "supported_endpoints": ["/v1/responses"],
        "use_openai_responses_path": True,
    }


def test_capability_flags_propagate_from_deployment_model_info_to_shared_key():
    """Backend-model capability facts (supported_endpoints,
    use_openai_responses_path) declared in a deployment's model_info must reach
    the shared backend key: the Bedrock Mantle routing gates read them raw off
    litellm.model_cost and document proxy model_info as an override path for
    models missing from the built-in cost map.
    """
    from litellm.llms.bedrock_mantle.common_utils import (
        mantle_base_segment,
        mantle_supports_responses,
    )

    bare_model = "somelab.lit4544-unmapped-model"
    backend_model = f"bedrock_mantle/{bare_model}"
    deploy_id = "lit4544-mantle-deploy"

    model_keys = {key: copy.deepcopy(litellm.model_cost.get(key)) for key in (bare_model, backend_model, deploy_id)}
    try:
        Router(
            model_list=[
                {
                    "model_name": "mantle-alias",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key",
                    },
                    "model_info": {
                        "id": deploy_id,
                        "supported_endpoints": ["/v1/responses"],
                        "use_openai_responses_path": True,
                    },
                },
            ],
        )

        shared_entry = litellm.model_cost.get(backend_model) or {}
        assert shared_entry.get("supported_endpoints") == ["/v1/responses"]
        assert shared_entry.get("use_openai_responses_path") is True
        assert "id" not in shared_entry
        assert mantle_supports_responses(bare_model, litellm.model_cost) is True
        assert mantle_base_segment(bare_model, litellm.model_cost) == "openai/v1"
    finally:
        _restore_model_cost_entries(model_keys)


def test_wildcard_zero_cost_request_does_not_poison_named_deployment_pricing():
    """LIT-3991 end to end: a proxy has a named text-embedding-3-small
    deployment relying on built-in pricing plus an ``openai/*`` wildcard with
    explicit zero pricing. One embedding call routed through the wildcard must
    not clobber the shared ``openai/text-embedding-3-small`` pricing; requests
    to the named deployment afterwards must still cost non-zero.
    """
    shared_key = "openai/text-embedding-3-small"
    model_keys = {
        shared_key: copy.deepcopy(litellm.model_cost.get(shared_key)),
        "text-embedding-3-small": copy.deepcopy(litellm.model_cost.get("text-embedding-3-small")),
        "openai/*": copy.deepcopy(litellm.model_cost.get("openai/*")),
        "lit3991-named": litellm.model_cost.get("lit3991-named"),
        "lit3991-wildcard": litellm.model_cost.get("lit3991-wildcard"),
    }
    builtin_input_cost = litellm.get_model_info(model=shared_key)["input_cost_per_token"]
    assert builtin_input_cost > 0

    try:
        router = Router(
            model_list=[
                {
                    "model_name": "text-embedding-3-small",
                    "litellm_params": {
                        "model": "openai/text-embedding-3-small",
                        "api_key": "fake-key-named",
                    },
                    "model_info": {"id": "lit3991-named"},
                },
                {
                    "model_name": "openai/*",
                    "litellm_params": {
                        "model": "openai/*",
                        "api_key": "fake-key-wildcard",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "lit3991-wildcard"},
                },
            ],
        )

        router.embedding(
            model="openai/text-embedding-3-small",
            input=["hello"],
            mock_response=[0.1, 0.2],
        )

        assert litellm.get_model_info(model=shared_key)["input_cost_per_token"] == builtin_input_cost, (
            f"one call through the zero-cost wildcard poisoned the shared {shared_key} pricing for the named deployment"
        )

        named_response = router.embedding(
            model="text-embedding-3-small",
            input=["hello"],
            mock_response=[0.1, 0.2],
        )
        named_cost = litellm.completion_cost(completion_response=named_response, call_type="embedding")
        assert named_cost == pytest.approx(10 * builtin_input_cost)
    finally:
        _restore_model_cost_entries(model_keys)


def test_price_data_reload_preserves_router_registered_model_info(monkeypatch):
    """
    A price-data reload replaces litellm.model_cost wholesale. Deployment
    model_info registered by the Router is not in the fetched catalog, so
    without a replay of runtime registrations the reload silently strips
    max_input_tokens / max_output_tokens from every custom model group and
    /model_group/info starts reporting nulls.
    """
    from litellm import utils as litellm_utils

    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "custom-alias",
                "litellm_params": {"model": "hosted_vllm/not-in-the-catalog"},
                "model_info": {
                    "id": "custom-alias-id",
                    "max_input_tokens": 128000,
                    "max_output_tokens": 16384,
                },
            }
        ],
    )

    before = router.get_model_group_info(model_group="custom-alias")
    assert before is not None
    assert before.max_input_tokens == 128000
    assert before.max_output_tokens == 16384

    saved_model_cost = litellm.model_cost
    try:
        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )

        after = router.get_model_group_info(model_group="custom-alias")
        assert after is not None
        assert after.max_input_tokens == 128000
        assert after.max_output_tokens == 16384
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_price_data_reload_preserves_custom_override_of_a_catalog_model(monkeypatch):
    """
    A deployment whose backend model IS in the catalog is the quieter half of
    the same bug: the reload does not blank the metadata, it reverts the
    operator's model_info override to the upstream catalog values.
    """
    from litellm import utils as litellm_utils

    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "capped-gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {
                    "id": "capped-gpt-4o-id",
                    "max_input_tokens": 12345,
                    "max_output_tokens": 678,
                },
            }
        ],
    )

    saved_model_cost = litellm.model_cost
    try:
        _simulate_price_data_reload(
            {
                "openai/gpt-4o": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "max_input_tokens": 999999,
                    "max_output_tokens": 888888,
                }
            },
        )

        after = router.get_model_group_info(model_group="capped-gpt-4o")
        assert after is not None
        assert after.max_input_tokens == 12345
        assert after.max_output_tokens == 678
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_deleted_deployments_are_not_replayed_onto_later_reloads(monkeypatch):
    """
    Runtime registrations are replayed onto every price data reload, so a
    deleted deployment has to be withdrawn or it is re-asserted for the life of
    the process and the registry grows with every create/delete cycle. A backend
    key that another live deployment still points at must survive the same
    deletion.
    """
    from litellm import utils as litellm_utils

    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "doomed",
                "litellm_params": {"model": "hosted_vllm/shared-backend"},
                "model_info": {"id": "doomed-id", "max_input_tokens": 111},
            },
            {
                "model_name": "kept",
                "litellm_params": {"model": "hosted_vllm/shared-backend"},
                "model_info": {"id": "kept-id", "max_input_tokens": 222},
            },
            {
                "model_name": "solo",
                "litellm_params": {"model": "hosted_vllm/solo-backend"},
                "model_info": {"id": "solo-id", "max_input_tokens": 333},
            },
        ],
    )

    saved_model_cost = litellm.model_cost
    try:
        assert router.delete_deployment(id="doomed-id") is not None
        assert router.delete_deployment(id="solo-id") is not None

        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )

        assert "doomed-id" not in litellm.model_cost
        assert "solo-id" not in litellm.model_cost
        assert "hosted_vllm/solo-backend" not in litellm.model_cost

        surviving = litellm.model_cost["kept-id"]
        assert surviving["max_input_tokens"] == 222
        assert "hosted_vllm/shared-backend" in litellm.model_cost
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_deleting_a_deployment_leaves_catalog_pricing_for_its_backend_model(monkeypatch):
    """
    A backend key is shared with the fetched catalog, so withdrawing the entries
    a deleted deployment owns must not take real upstream pricing down with it.
    """
    from litellm import utils as litellm_utils

    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    backend_model = "gemini/gemini-2.5-pro"
    catalog_entry = litellm.get_model_info(model=backend_model)
    catalog_input_cost = catalog_entry["input_cost_per_token"]
    assert catalog_input_cost > 0, "Test requires a catalog model with non-zero pricing"

    saved_catalog = litellm.model_cost
    fetched_catalog = copy.deepcopy(litellm.model_cost)
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "doomed-gemini",
                    "litellm_params": {"model": backend_model, "api_key": "sk-fake"},
                    "model_info": {"id": "doomed-gemini-id"},
                }
            ],
        )

        assert router.delete_deployment(id="doomed-gemini-id") is not None

        _simulate_price_data_reload(
            copy.deepcopy(fetched_catalog),
        )

        assert "doomed-gemini-id" not in litellm.model_cost
        assert litellm.model_cost[backend_model]["input_cost_per_token"] == catalog_input_cost
    finally:
        litellm.model_cost = saved_catalog
        _invalidate_model_cost_lowercase_map()


def test_repointing_a_deployment_drops_its_previous_backend_key(monkeypatch):
    """
    An update that moves a deployment onto a different backend model leaves the
    old backend key behind, and a replayed registry would re-assert it onto every
    later catalog for the life of the process.
    """
    from litellm import utils as litellm_utils

    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "moving-target",
                "litellm_params": {"model": "hosted_vllm/old-backend"},
                "model_info": {"id": "moving-target-id"},
            }
        ],
    )

    saved_model_cost = litellm.model_cost
    try:
        router.upsert_deployment(
            deployment=Deployment(
                model_name="moving-target",
                litellm_params=LiteLLM_Params(model="hosted_vllm/new-backend"),
                model_info=ModelInfo(id="moving-target-id"),
            )
        )

        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )

        assert "hosted_vllm/old-backend" not in litellm.model_cost
        assert "hosted_vllm/new-backend" in litellm.model_cost
        assert "moving-target-id" in litellm.model_cost
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize(
    "model, custom_llm_provider, expected",
    [
        ("gpt-4o", None, ("gpt-4o",)),
        ("gpt-4o", "openai", ("openai/gpt-4o",)),
        ("openai/gpt-4o", None, ("openai/gpt-4o",)),
        ("responses/gpt-4o", "openai", ("openai/responses/gpt-4o", "openai/gpt-4o")),
        ("responses/gpt-4o", None, ("responses/gpt-4o", "gpt-4o")),
    ],
)
def test_backend_cost_map_keys_matches_what_registration_writes(model, custom_llm_provider, expected):
    """
    The withdrawal path drops exactly the keys the registration wrote, so the two
    have to agree on the provider prefix and on the responses/ alias. The first
    key is also the one the registration uses as the shared backend key, so its
    position is load-bearing rather than incidental.
    """
    keys = Router._backend_cost_map_keys(model=model, custom_llm_provider=custom_llm_provider)
    assert keys == expected
    assert keys[0] == (model if custom_llm_provider is None else f"{custom_llm_provider}/{model}")


def test_mantle_chat_deployment_does_not_rewrite_the_shared_responses_row():
    """
    A Bedrock Mantle deployment that pins ``mode: chat`` routes its own chat
    requests to Chat Completions, while the shared cost-map row keeps
    ``mode: responses`` for every other deployment and plain SDK calls.
    """
    from litellm.main import responses_api_bridge_check

    backend_model = "bedrock_mantle/openai.gpt-5.6-sol"
    cost_map: Final = cast(dict[str, dict[str, object]], litellm.model_cost)
    model_keys = {
        key: copy.deepcopy(cost_map.get(key))
        for key in (backend_model, f"bedrock_mantle/{backend_model}", "mantle-chat-pinned")
    }

    try:
        litellm.model_cost[backend_model] = {
            **(model_keys[backend_model] or {}),
            "litellm_provider": "bedrock_mantle",
            "mode": "responses",
        }
        _invalidate_model_cost_lowercase_map()

        Router(
            model_list=[
                {
                    "model_name": "mantle-chat",
                    "litellm_params": {
                        "model": backend_model,
                        "custom_llm_provider": "bedrock_mantle",
                        "api_key": "test-bearer",
                    },
                    "model_info": {"id": "mantle-chat-pinned", "mode": "chat"},
                }
            ]
        )

        assert litellm.model_cost[backend_model]["mode"] == "responses"

        shared_info, _ = responses_api_bridge_check(model=backend_model, custom_llm_provider="bedrock_mantle")
        assert shared_info["mode"] == "responses"
        pinned_info, _ = responses_api_bridge_check(
            model=backend_model,
            custom_llm_provider="bedrock_mantle",
            deployment_model_info={"id": "mantle-chat-pinned", "mode": "chat"},
        )
        assert pinned_info["mode"] == "chat"
    finally:
        _restore_model_cost_entries(model_keys)


def _register_routing_test_row(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    litellm.model_cost["routing-test-model"] = {
        "litellm_provider": "openai",
        "mode": mode,
        "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
        "max_input_tokens": 1000,
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
    }
    _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize("registered_mode, pinned_mode", [("responses", "chat"), ("chat", "responses")])
@pytest.mark.parametrize(
    "model, custom_llm_provider",
    [
        ("openai/routing-test-model", "openai"),
        ("openai/routing-test-model", None),
        ("routing-test-model", "openai"),
        ("openai/responses/routing-test-model", None),
    ],
)
def test_a_deployment_mode_never_rewrites_the_shared_row(
    monkeypatch: pytest.MonkeyPatch,
    registered_mode: str,
    pinned_mode: str,
    model: str,
    custom_llm_provider: str | None,
) -> None:
    """A ``mode`` pin routes only its own deployment. Every key registration merges into the shared
    ``<name>`` row, the ``responses/``-stripped twin included, has to keep the mode the SDK and the
    sibling deployments route by, whichever way the pin points."""
    from litellm.main import responses_api_bridge_check

    _register_routing_test_row(monkeypatch, registered_mode)
    litellm_params: Final = {"model": model, "api_key": "sk-test"} | (
        {"custom_llm_provider": custom_llm_provider} if custom_llm_provider else {}
    )

    Router(
        model_list=[
            {
                "model_name": "pinned",
                "litellm_params": litellm_params,
                "model_info": {"id": "pinned-id", "mode": pinned_mode},
            },
            {"model_name": "sibling", "litellm_params": {"model": "openai/routing-test-model", "api_key": "sk-test"}},
        ]
    )

    assert litellm.model_cost["routing-test-model"]["mode"] == registered_mode
    shared_info, _ = responses_api_bridge_check(model="routing-test-model", custom_llm_provider="openai")
    assert shared_info["mode"] == registered_mode


def test_a_custom_priced_chat_pin_request_leaves_its_unpinned_sibling_on_responses(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    _register_routing_test_row(monkeypatch, "responses")
    chat_route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "routing-test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    responses_route: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "routing-test-model",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "priced-chat-pin",
                "litellm_params": {
                    "model": "openai/routing-test-model",
                    "api_key": "sk-test",
                    "input_cost_per_token": 3e-06,
                    "output_cost_per_token": 4e-06,
                },
                "model_info": {"id": "priced-chat-pin-id", "mode": "chat"},
            },
            {"model_name": "sibling", "litellm_params": {"model": "openai/routing-test-model", "api_key": "sk-test"}},
        ]
    )

    router.completion(model="priced-chat-pin", messages=[{"role": "user", "content": "Say ok"}])
    router.completion(model="sibling", messages=[{"role": "user", "content": "Say ok"}])

    assert (chat_route.call_count, responses_route.call_count) == (1, 1)


def test_a_prefixed_deployment_override_leaves_the_shared_row_limits_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment that names its provider twice, in ``custom_llm_provider`` and in the model, registers
    under its own key, so its ``max_input_tokens`` override stays off the row its siblings read."""
    _register_routing_test_row(monkeypatch, "responses")
    shared_before: Final = copy.deepcopy(litellm.model_cost["routing-test-model"])

    Router(
        model_list=[
            {
                "model_name": "small-window",
                "litellm_params": {
                    "model": "openai/routing-test-model",
                    "custom_llm_provider": "openai",
                    "api_key": "sk-test",
                },
                "model_info": {"id": "small-window-id", "max_input_tokens": 10},
            }
        ]
    )

    assert litellm.model_cost["routing-test-model"] == shared_before


@pytest.mark.parametrize("pins", [("chat", "responses"), ("responses", "chat")], ids=["chat first", "responses first"])
def test_an_unmapped_model_routes_the_same_whichever_pin_registers_first(
    monkeypatch: pytest.MonkeyPatch, pins: tuple[str, str]
) -> None:
    """A model missing from the cost map gets its shared row from the router, so conflicting pins must not
    leave a row that depends on deployment order, and a ``responses`` pin keeps opting the model in."""
    from litellm.main import responses_api_bridge_check

    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    Router(
        model_list=[
            {
                "model_name": f"pinned-{mode}",
                "litellm_params": {"model": "openai/unmapped-routing-test-model", "api_key": "sk-test"},
                "model_info": {"id": f"pinned-{mode}", "mode": mode},
            }
            for mode in pins
        ]
    )

    shared_info, _ = responses_api_bridge_check(model="unmapped-routing-test-model", custom_llm_provider="openai")
    assert shared_info.get("mode") == "responses"


_FINE_TUNE: Final = "openai/ft:gpt-4o-mini-2024-07-18:routing-test::abc123"


def _fine_tune_routing(
    monkeypatch: pytest.MonkeyPatch, pristine_cost_map: dict[str, object], order: tuple[str | None, ...]
) -> tuple[object, ...]:
    from litellm.main import responses_api_bridge_check

    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(pristine_cost_map))
    Router(
        model_list=[
            {
                "model_name": f"ft-{mode}",
                "litellm_params": {"model": _FINE_TUNE, "api_key": "sk-test"},
                "model_info": {"id": f"ft-{mode}"} if mode is None else {"id": f"ft-{mode}", "mode": mode},
            }
            for mode in order
        ]
    )
    return tuple(
        responses_api_bridge_check(
            model=_FINE_TUNE.removeprefix("openai/"),
            custom_llm_provider="openai",
            deployment_model_info=None if mode is None else {"mode": mode},
        )[0].get("mode")
        for mode in (None, "chat", "responses")
    )


@pytest.mark.parametrize("order", list(itertools.permutations(("chat", "responses", None))), ids=str)
def test_a_fine_tune_routes_the_same_whichever_deployment_registers_first(
    monkeypatch: pytest.MonkeyPatch, order: tuple[str | None, ...]
) -> None:
    """A fine-tune missing from the cost map inherits its base model's row on its first registration, so a
    ``responses`` pin must not opt the shared row in only when it happens to register before its siblings."""
    pristine_cost_map: Final = litellm.model_cost

    assert _fine_tune_routing(monkeypatch, pristine_cost_map, order) == _fine_tune_routing(
        monkeypatch, pristine_cost_map, ("responses", "chat", None)
    )


def test_a_non_routing_mode_pin_still_reaches_the_shared_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ``chat`` and ``responses`` pins are per-deployment routing. Other modes, such as a realtime
    model pinned to ``audio_transcription``, are still read from the shared row by name."""
    _register_routing_test_row(monkeypatch, "chat")

    Router(
        model_list=[
            {
                "model_name": "transcriber",
                "litellm_params": {"model": "openai/routing-test-model", "api_key": "sk-test"},
                "model_info": {"id": "transcriber-id", "mode": "audio_transcription"},
            }
        ]
    )

    assert litellm.get_model_info(model="routing-test-model", custom_llm_provider="openai")["mode"] == (
        "audio_transcription"
    )


def test_a_non_routing_mode_pin_leaves_a_responses_row_on_the_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.main import responses_api_bridge_check

    _register_routing_test_row(monkeypatch, "responses")

    Router(
        model_list=[
            {
                "model_name": "transcriber",
                "litellm_params": {"model": "openai/routing-test-model", "api_key": "sk-test"},
                "model_info": {"id": "transcriber-id", "mode": "audio_transcription"},
            }
        ]
    )

    shared_info, _ = responses_api_bridge_check(model="routing-test-model", custom_llm_provider="openai")
    assert shared_info["mode"] == "responses"


@pytest.mark.parametrize("registered_mode", ["responses", "chat"])
@pytest.mark.parametrize("malformed_mode", [5, ["chat"], {"mode": "chat"}], ids=["int", "list", "object"])
def test_a_malformed_mode_never_reaches_the_shared_row(
    monkeypatch: pytest.MonkeyPatch, registered_mode: str, malformed_mode: object
) -> None:
    from litellm.main import responses_api_bridge_check

    _register_routing_test_row(monkeypatch, registered_mode)

    Router(
        model_list=[
            {
                "model_name": "malformed",
                "litellm_params": {"model": "openai/routing-test-model", "api_key": "sk-test"},
                "model_info": {"id": "malformed-id", "mode": malformed_mode},
            }
        ]
    )

    assert litellm.model_cost["routing-test-model"]["mode"] == registered_mode
    shared_info, _ = responses_api_bridge_check(model="routing-test-model", custom_llm_provider="openai")
    assert shared_info["mode"] == registered_mode


def test_a_github_copilot_chat_pin_opts_a_dual_endpoint_model_out_of_the_responses_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """``github_copilot`` sends a model listing both endpoints to ``/v1/responses`` unless its row says ``chat``,
    so a ``chat`` pin has to reach the row to keep that model on chat completions."""
    from litellm.llms.github_copilot.responses.transformation import github_copilot_supports_responses_api

    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    (tmp_path / "api-key.json").write_text(json.dumps({"token": "tid=test", "expires_at": _COPILOT_TOKEN_EXPIRES_AT}))
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    litellm.model_cost["github_copilot/routing-test-model"] = {
        "litellm_provider": "github_copilot",
        "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
        "max_input_tokens": 1000,
    }
    _invalidate_model_cost_lowercase_map()
    assert github_copilot_supports_responses_api("github_copilot/routing-test-model") is True

    Router(
        model_list=[
            {
                "model_name": "copilot",
                "litellm_params": {"model": "github_copilot/routing-test-model"},
                "model_info": {"id": "copilot-chat-pinned", "mode": "chat"},
            }
        ]
    )

    assert github_copilot_supports_responses_api("github_copilot/routing-test-model") is False


@pytest.mark.parametrize("registered_mode, pinned_mode", [("chat", "responses"), ("responses", "chat")])
@pytest.mark.parametrize(
    "model, custom_llm_provider", [("github_copilot/routing-test-model", None), ("routing-test-model", "github_copilot")]
)
def test_a_github_copilot_responses_pin_still_opts_the_model_into_the_responses_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    registered_mode: str,
    pinned_mode: str,
    model: str,
    custom_llm_provider: str | None,
) -> None:
    """``github_copilot`` dispatches ``/v1/responses`` natively by the shared row's mode, and a ``responses``
    pin is its documented opt-in, while a ``chat`` pin must not move a Responses-only row off it."""
    from litellm.llms.github_copilot.responses.transformation import github_copilot_supports_responses_api

    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    (tmp_path / "api-key.json").write_text(json.dumps({"token": "tid=test", "expires_at": _COPILOT_TOKEN_EXPIRES_AT}))
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    litellm.model_cost["github_copilot/routing-test-model"] = {
        "litellm_provider": "github_copilot",
        "mode": registered_mode,
        "supported_endpoints": ["/v1/responses"] if registered_mode == "responses" else ["/v1/chat/completions"],
        "max_input_tokens": 1000,
    }
    _invalidate_model_cost_lowercase_map()

    Router(
        model_list=[
            {
                "model_name": "copilot",
                "litellm_params": {"model": model}
                | ({"custom_llm_provider": custom_llm_provider} if custom_llm_provider else {}),
                "model_info": {"id": "copilot-pinned", "mode": pinned_mode},
            }
        ]
    )

    assert github_copilot_supports_responses_api("github_copilot/routing-test-model") is True


def test_a_discarded_router_stops_contributing_to_later_reloads(monkeypatch):
    """
    `_route_user_config_request` builds a Router per request from caller-supplied
    config and discards it. Nothing can withdraw entries on its behalf afterwards,
    so a rebuild driven off live routers is what keeps a caller from growing the
    cost map one request at a time.
    """
    saved_model_cost = litellm.model_cost
    try:
        kept = Router(
            model_list=[
                {
                    "model_name": "kept",
                    "litellm_params": {"model": "hosted_vllm/kept-backend"},
                    "model_info": {"id": "kept-router-id", "max_input_tokens": 4242},
                }
            ],
        )
        throwaway = Router(
            model_list=[
                {
                    "model_name": "throwaway",
                    "litellm_params": {"model": "hosted_vllm/throwaway-backend"},
                    "model_info": {"id": "throwaway-router-id", "max_input_tokens": 111},
                }
            ],
        )
        throwaway.discard()

        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )

        assert "throwaway-router-id" not in litellm.model_cost
        assert "hosted_vllm/throwaway-backend" not in litellm.model_cost
        assert litellm.model_cost["kept-router-id"]["max_input_tokens"] == 4242
        assert "hosted_vllm/kept-backend" in litellm.model_cost
        assert kept.model_list  # keep the live router referenced for the duration
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_a_reload_rebuilds_exactly_what_a_fresh_boot_registered() -> None:
    """
    The rebuild is only correct if it reproduces the entries the original
    registration wrote, including the pieces that are derived rather than stored:
    custom pricing carried on litellm_params, and the cache pricing inherited from
    the built-in cost map.
    """
    saved_catalog = litellm.model_cost
    fetched_catalog = copy.deepcopy(litellm.model_cost)
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "priced",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000123,
                        "output_cost_per_token": 0.000456,
                    },
                    "model_info": {"id": "priced-id", "max_input_tokens": 4242},
                }
            ],
        )
        at_boot = copy.deepcopy(litellm.model_cost["priced-id"])
        assert at_boot["input_cost_per_token"] == 0.000123
        assert at_boot["cache_read_input_token_cost"] is not None
        assert "member_auto_router" not in litellm.model_cost["gpt-4o"]

        _simulate_price_data_reload(
            copy.deepcopy(fetched_catalog),
        )

        rebuilt = litellm.model_cost["priced-id"]
        assert at_boot == rebuilt
        assert "member_auto_router" not in litellm.model_cost["gpt-4o"]
        assert router.model_list
    finally:
        litellm.model_cost = saved_catalog
        _invalidate_model_cost_lowercase_map()


def test_replay_model_cost_registrations_survives_a_malformed_deployment():
    """
    The rebuild reads whatever dicts are sitting in model_list, so one entry that
    cannot be rebuilt into a Deployment must not stop the rest being restored.
    """
    saved_model_cost = litellm.model_cost
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "healthy",
                    "litellm_params": {"model": "hosted_vllm/healthy-backend"},
                    "model_info": {"id": "healthy-id", "max_input_tokens": 777},
                }
            ],
        )
        router.model_list.insert(0, {"litellm_params": {}})

        litellm.model_cost = {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}}
        _invalidate_model_cost_lowercase_map()
        router.replay_model_cost_registrations()

        assert litellm.model_cost["healthy-id"]["max_input_tokens"] == 777
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_deployment_model_cost_payload_folds_in_litellm_params_pricing():
    """
    Custom pricing is configured on litellm_params but has to land in the
    cost-map entry, and setting it pulls in the built-in cache pricing for the
    backend model. Both are what make the entry reproducible from a deployment.
    """
    payload = Router._deployment_model_cost_payload(
        deployment=Deployment(
            model_name="priced",
            litellm_params=LiteLLM_Params(
                model="gemini/gemini-2.5-pro",
                input_cost_per_token=0.000123,
            ),
            model_info=ModelInfo(id="payload-id", max_input_tokens=4242),
        )
    )

    assert payload["id"] == "payload-id"
    assert payload["max_input_tokens"] == 4242
    assert payload["input_cost_per_token"] == 0.000123
    assert payload["cache_read_input_token_cost"] > 0


def test_deployment_model_cost_payload_includes_builtin_service_tier_pricing() -> None:
    model_id: Final = "tier-priced-payload"
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key)) for key in (_TIER_BACKEND_KEY, _TIER_BACKEND_MODEL, model_id)
    }
    try:
        _register_tier_backend()
        payload: Final = Router._deployment_model_cost_payload(
            deployment=Deployment(
                model_name="tier-priced-payload",
                litellm_params=LiteLLM_Params(
                    model=_TIER_BACKEND_MODEL,
                    custom_llm_provider="openai",
                    input_cost_per_token=_CUSTOM_STANDARD_INPUT_RATE,
                    output_cost_per_token=_CUSTOM_STANDARD_OUTPUT_RATE,
                ),
                model_info=ModelInfo(id=model_id),
            )
        )

        assert payload["input_cost_per_token_ultrafast"] == _TIER_BACKEND_ENTRY["input_cost_per_token_ultrafast"]
        assert payload["output_cost_per_token_ultrafast"] == _TIER_BACKEND_ENTRY["output_cost_per_token_ultrafast"]
        assert payload["input_cost_per_token_balanced"] == _TIER_BACKEND_ENTRY["input_cost_per_token_balanced"]
        assert payload["input_cost_per_token"] == _CUSTOM_STANDARD_INPUT_RATE
        assert payload["output_cost_per_token"] == _CUSTOM_STANDARD_OUTPUT_RATE
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


def test_register_deployment_in_model_cost_writes_both_key_families():
    """
    A deployment contributes its full model_info under its unique id and the
    cost-map subset under the shared backend key, and the shared key must not
    pick up the deployment's private metadata.
    """
    model_keys = {
        "both-families-id": copy.deepcopy(litellm.model_cost.get("both-families-id")),
        "hosted_vllm/both-families-backend": copy.deepcopy(litellm.model_cost.get("hosted_vllm/both-families-backend")),
    }
    try:
        Router._register_deployment_in_model_cost(
            model_id="both-families-id",
            model_info={"id": "both-families-id", "max_input_tokens": 999, "litellm_provider": "hosted_vllm"},
            model="hosted_vllm/both-families-backend",
            custom_llm_provider=None,
        )

        assert litellm.model_cost["both-families-id"]["max_input_tokens"] == 999
        shared = litellm.model_cost["hosted_vllm/both-families-backend"]
        assert shared["max_input_tokens"] == 999
        assert "id" not in shared
    finally:
        _restore_model_cost_entries(model_keys)


def test_router_registration_keeps_ultrafast_long_context_deployment_pricing() -> None:
    model_id: Final = "ultrafast-long-context-pricing-id"
    backend_key: Final = "openai/gpt-6-astra"
    rates: Final = {
        "input_cost_per_token_above_272k_tokens_ultrafast": 0.00012,
        "output_cost_per_token_above_272k_tokens_ultrafast": 0.00045,
        "cache_read_input_token_cost_above_272k_tokens_ultrafast": 1.2e-05,
        "cache_creation_input_token_cost_above_272k_tokens_ultrafast": 0.00015,
    }
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key)) for key in (model_id, backend_key, "gpt-6-astra")
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "ultrafast-long-context-pricing",
                    "litellm_params": {"model": backend_key, **rates},
                    "model_info": {"id": model_id},
                }
            ]
        )

        assert {key: litellm.model_cost[model_id][key] for key in rates} == rates
    finally:
        _restore_model_cost_entries(model_cost_entries)


def test_reload_keeps_custom_pricing_configured_on_litellm_params_for_a_db_model():
    """
    A deployment added at runtime, which is what /model/new does, configures its
    custom pricing on litellm_params rather than on model_info. A price data
    reload must not revert that to the catalog's pricing.
    """
    saved_catalog = litellm.model_cost
    fetched_catalog = copy.deepcopy(litellm.model_cost)
    try:
        router = Router(model_list=[])
        router.add_deployment(
            deployment=Deployment(
                model_name="db-priced",
                litellm_params=LiteLLM_Params(
                    model="openai/gpt-4o",
                    api_key="sk-fake",
                    input_cost_per_token=0.000123,
                    output_cost_per_token=0.000456,
                ),
                model_info=ModelInfo(id="db-priced-id"),
            )
        )

        assert litellm.model_cost["db-priced-id"]["input_cost_per_token"] == 0.000123

        _simulate_price_data_reload(
            copy.deepcopy(fetched_catalog),
        )

        assert litellm.model_cost["db-priced-id"]["input_cost_per_token"] == 0.000123
        assert litellm.model_cost["db-priced-id"]["output_cost_per_token"] == 0.000456
    finally:
        litellm.model_cost = saved_catalog
        _invalidate_model_cost_lowercase_map()


def test_reload_keeps_custom_pricing_of_a_db_model_on_a_router_built_without_a_model_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A proxy whose config.yaml has no model_list builds its router without one and
    takes every deployment from the DB afterwards. A price data reload must restore
    that deployment's custom pricing exactly as it does for a config router.
    """
    fetched_catalog: Final = copy.deepcopy(litellm.model_cost)
    router: Final = Router()
    router.add_deployment(
        deployment=Deployment(
            model_name="db-only-priced",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4o",
                api_key="sk-fake",
                input_cost_per_token=0.000123,
                output_cost_per_token=0.000456,
            ),
            model_info=ModelInfo(id="db-only-priced-id"),
        )
    )
    assert litellm.model_cost["db-only-priced-id"]["input_cost_per_token"] == 0.000123

    _simulate_price_data_reload_with_provider_sets(monkeypatch, fetched_catalog)

    assert litellm.model_cost["db-only-priced-id"]["input_cost_per_token"] == 0.000123
    assert litellm.model_cost["db-only-priced-id"]["output_cost_per_token"] == 0.000456


def test_replay_live_router_model_cost_rebuilds_every_live_router():
    """
    A process can hold more than one Router, so the rebuild has to fan out across
    all of them rather than restoring whichever one happens to be reachable.
    """
    from litellm.router import _replay_live_router_model_cost

    saved_model_cost = litellm.model_cost
    try:
        first = Router(
            model_list=[
                {
                    "model_name": "first",
                    "litellm_params": {"model": "hosted_vllm/first-backend"},
                    "model_info": {"id": "first-id", "max_input_tokens": 111},
                }
            ],
        )
        second = Router(
            model_list=[
                {
                    "model_name": "second",
                    "litellm_params": {"model": "hosted_vllm/second-backend"},
                    "model_info": {"id": "second-id", "max_input_tokens": 222},
                }
            ],
        )

        litellm.model_cost = {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}}
        _invalidate_model_cost_lowercase_map()
        _replay_live_router_model_cost()

        assert litellm.model_cost["first-id"]["max_input_tokens"] == 111
        assert litellm.model_cost["second-id"]["max_input_tokens"] == 222
        assert first.model_list and second.model_list
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_strategy_router_alias_pricing_never_enters_model_cost(monkeypatch):
    """
    A strategy-router alias is never the deployment actually called or billed,
    so custom pricing configured on it must not be registered under its
    model_id - an explicit zero there makes the budget check treat the alias
    as a genuinely free model while requests bill as a real deployment. The
    strip must also survive a price-data reload, which rebuilds entries by
    walking the live routers.
    """
    from litellm import utils as litellm_utils

    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router/smart-router",
                    "complexity_router_default_model": "paid-model",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "complexity_router_config": {"tiers": {"simple": "paid-model"}},
                },
                "model_info": {"id": "strategy-alias-id", "max_input_tokens": 128000},
            },
            {
                "model_name": "paid-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
                "model_info": {"id": "strategy-alias-paid-id"},
            },
        ],
    )

    def _assert_alias_unpriced():
        entry = litellm.model_cost.get("strategy-alias-id")
        assert entry is not None, "Alias metadata should still be registered"
        assert entry["max_input_tokens"] == 128000
        assert "input_cost_per_token" not in entry
        assert "output_cost_per_token" not in entry

    _assert_alias_unpriced()

    saved_model_cost = litellm.model_cost
    try:
        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )
        _assert_alias_unpriced()
        assert router.model_list
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_inherit_builtin_tiered_output_rate_fills_the_backend_flat_rate():
    """
    A deployment entry whose custom tiers publish only input rates would bill
    completions at 0, so the backend model's flat output rate is copied in at
    registration.
    """
    model_info = {"tiered_pricing": [{"range": [0, 3000], "input_cost_per_token": 3.25e-07}]}

    Router._inherit_builtin_tiered_output_rate(
        model_info=model_info,
        backend_model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
    )

    backend_rate = litellm.get_model_info(model="claude-haiku-4-5", custom_llm_provider="anthropic")[
        "output_cost_per_token"
    ]
    assert backend_rate > 0
    assert model_info["output_cost_per_token"] == backend_rate


def test_inherit_builtin_tiered_output_rate_never_stores_a_synthesized_zero():
    """
    Regression: get_model_info reports output_cost_per_token 0 for a backend that
    only publishes tiered rates (e.g. dashscope/qwen-flash), and storing that zero
    would mark the deployment as explicitly priced free.
    """
    backend_info = litellm.get_model_info(model="qwen-flash", custom_llm_provider="dashscope")
    assert backend_info["output_cost_per_token"] == 0

    model_info = {"tiered_pricing": [{"range": [0, 3000], "input_cost_per_token": 3.25e-07}]}
    Router._inherit_builtin_tiered_output_rate(
        model_info=model_info,
        backend_model="qwen-flash",
        custom_llm_provider="dashscope",
    )

    assert "output_cost_per_token" not in model_info


def test_inherit_builtin_tiered_output_rate_leaves_a_user_rate_alone():
    model_info = {
        "tiered_pricing": [{"range": [0, 3000], "input_cost_per_token": 3.25e-07}],
        "output_cost_per_token": 9e-07,
    }

    Router._inherit_builtin_tiered_output_rate(
        model_info=model_info,
        backend_model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
    )

    assert model_info["output_cost_per_token"] == 9e-07


# --- a config.yaml PTU deployment must not also bill per token ------------------

_PTU_MODEL_INFO = {
    "id": "ptu-alpha-eastus",
    "team_id": "team-alpha",
    "ptu_count": 100,
    "cost_per_ptu_per_hour": 0.02,
    "ptu_effective_from": "2026-01-01T00:00:00Z",
}


def _ptu_router(model_info=None, litellm_params=None, ptu_enabled=True):
    """A router built the way loading config.yaml builds one."""
    with patch.dict(os.environ, {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True" if ptu_enabled else ""}, clear=False):
        return Router(
            model_list=[
                {
                    "model_name": "gpt-4o-ptu",
                    "litellm_params": {
                        "model": "anthropic/claude-sonnet-4-5-20250929",
                        "api_key": "sk-not-used",
                        **(litellm_params or {}),
                    },
                    "model_info": dict(_PTU_MODEL_INFO if model_info is None else model_info),
                }
            ]
        )


def test_a_config_ptu_deployment_bills_nothing_per_token():
    """Reserved capacity is already billed by the hour, so charging its traffic bills the
    same tokens twice. Left unset the rate falls back to the public cost map, which makes
    the double charge the default rather than an opt-in."""
    router = _ptu_router(litellm_params={"input_cost_per_token": 5e-06, "output_cost_per_token": 1.5e-05})
    entry = router.model_list[0]

    assert entry["litellm_params"]["input_cost_per_token"] == 0.0
    assert entry["litellm_params"]["output_cost_per_token"] == 0.0
    assert entry["model_info"]["input_cost_per_token"] == 0.0
    assert litellm.model_cost[entry["model_info"]["id"]]["input_cost_per_token"] == 0.0


@pytest.mark.parametrize(
    "backend",
    ["anthropic/claude-sonnet-4-5-20250929", "azure/gpt-4o", "gemini/gemini-2.5-flash"],
)
def test_a_config_ptu_deployment_imports_no_cache_rate_from_its_backend(backend):
    """The cache back-fill runs whenever input_cost_per_token is set, and 0.0 is set, so a
    partially zeroed deployment would silently inherit the backend model's real cache rates.
    Every backend here publishes non-zero ones, which is what makes the assertion mean
    something."""
    cache_fields = (
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
        "cache_creation_input_token_cost_above_200k_tokens",
        "cache_read_input_token_cost",
        "cache_read_input_token_cost_above_200k_tokens",
    )
    builtin = litellm.get_model_info(model=backend)
    assert any(builtin.get(field) for field in cache_fields), "backend publishes no cache pricing to leak"

    router = _ptu_router(litellm_params={"model": backend})
    priced = litellm.model_cost[router.model_list[0]["model_info"]["id"]]

    assert [field for field in cache_fields if priced.get(field)] == []


def test_zeroing_a_ptu_deployment_leaves_its_backend_model_priced():
    """A sibling deployment on the same backend must keep billing normally."""
    backend = "anthropic/claude-sonnet-4-5-20250929"
    builtin = litellm.get_model_info(model=backend)["input_cost_per_token"]
    assert builtin > 0

    _ptu_router(litellm_params={"model": backend})

    assert litellm.get_model_info(model=backend)["input_cost_per_token"] == builtin


def test_the_registered_id_is_the_one_the_operator_declared():
    """Registration must key the deployment by the declared id, not by a hash of params that
    zeroing has just rewritten. The id keys cooldowns, budgets and every spend row already
    written, so minting one here would move all of them.

    A derived id is no longer reachable for a reservation: zeroing requires PTU terms and
    PTU terms now require a declared id, so the two never combine."""
    params = {"input_cost_per_token": 5e-06}
    priced = _ptu_router(litellm_params=params, ptu_enabled=False).model_list[0]["model_info"]["id"]
    zeroed = _ptu_router(litellm_params=params).model_list[0]["model_info"]["id"]

    assert priced == zeroed == "ptu-alpha-eastus"


def test_a_database_backed_deployment_is_left_alone():
    """The write endpoints already zero those, and they answer 400 rather than silently
    rewriting a rate the caller sent."""
    entry = _ptu_router(model_info={**_PTU_MODEL_INFO, "db_model": True}).model_list[0]

    assert entry["litellm_params"].get("input_cost_per_token") is None


def test_nothing_is_zeroed_while_the_feature_is_off():
    """No flat cost accrues with the flag off, so zeroing would serve the traffic free."""
    entry = _ptu_router(litellm_params={"input_cost_per_token": 5e-06}, ptu_enabled=False).model_list[0]

    assert entry["litellm_params"]["input_cost_per_token"] == 5e-06


@pytest.mark.parametrize("dropped", ["team_id", "ptu_effective_from"], ids=["no team_id", "no ptu_effective_from"])
def test_an_incomplete_reservation_is_refused_rather_than_served(dropped):
    """POST /model/new answers 400 for exactly this config, so config.yaml must not quietly
    accept it. Serving it would bill per token while accruing no flat cost, which is the
    state the operator was trying to leave."""
    incomplete = {k: v for k, v in _PTU_MODEL_INFO.items() if k != dropped}

    with pytest.raises(ValueError, match="PTU configuration on model 'gpt") as raised:
        _ptu_router(model_info=incomplete, litellm_params={"input_cost_per_token": 5e-06})

    assert "gpt-4o-ptu" in str(raised.value)


@pytest.mark.parametrize(
    "dropped, expected",
    [
        ("team_id", "team_id or ptu_shares is required when PTU fields are set"),
        ("cost_per_ptu_per_hour", "ptu_count and cost_per_ptu_per_hour must be set together"),
    ],
    ids=["no team_id", "count without rate"],
)
def test_the_refusal_reason_is_the_one_the_model_endpoint_answers_with(dropped, expected):
    """One rule, stated once. If these drift, an operator gets contradictory guidance
    depending on which path they used."""
    incomplete = {k: v for k, v in _PTU_MODEL_INFO.items() if k != dropped}

    assert ptu_config_error(incomplete) == expected
    with pytest.raises(ValueError, match="PTU configuration on model 'gpt") as raised:
        _ptu_router(model_info=incomplete)

    assert expected in str(raised.value)


@pytest.mark.parametrize("dropped", ["team_id", "ptu_effective_from"], ids=["no team_id", "no ptu_effective_from"])
def test_an_incomplete_reservation_is_left_alone_while_the_feature_is_off(dropped):
    """Nothing accrues with the flag off, so refusing a deployment there would take a
    serving model away from an operator who never opted in."""
    incomplete = {k: v for k, v in _PTU_MODEL_INFO.items() if k != dropped}
    entry = _ptu_router(
        model_info=incomplete, litellm_params={"input_cost_per_token": 5e-06}, ptu_enabled=False
    ).model_list[0]

    assert entry["litellm_params"]["input_cost_per_token"] == 5e-06


@pytest.mark.parametrize("dropped", ["team_id", "ptu_effective_from"], ids=["no team_id", "no ptu_effective_from"])
def test_the_proxy_drops_the_deployment_rather_than_failing_to_boot(dropped):
    """The proxy builds its router with ignore_invalid_deployments, so one bad entry must
    cost that entry and not the whole config."""
    incomplete = {k: v for k, v in _PTU_MODEL_INFO.items() if k != dropped}
    with patch.dict(os.environ, {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True"}, clear=False):
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-ptu",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-5-20250929", "api_key": "sk-not-used"},
                    "model_info": dict(incomplete),
                },
                {
                    "model_name": "plain-sibling",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-5-20250929", "api_key": "sk-not-used"},
                },
            ],
            ignore_invalid_deployments=True,
        )

    assert [entry["model_name"] for entry in router.model_list] == ["plain-sibling"]


def test_a_complete_reservation_still_registers():
    """The refusal must be scoped to a broken reservation, not to PTU configuration."""
    entry = _ptu_router().model_list[0]

    assert entry["model_name"] == "gpt-4o-ptu"
    assert entry["litellm_params"]["input_cost_per_token"] == 0.0


def test_nested_custom_model_info_does_not_pollute_shared_backend():
    backend_model = "gpt-4o-search-preview"
    custom_id = "lit5471-search-custom"
    sibling_id = "lit5471-search-sibling"
    builtin_info = copy.deepcopy(litellm.get_model_info(model=backend_model))
    expected_nested = copy.deepcopy(builtin_info["search_context_cost_per_query"])
    model_keys = {
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
        custom_id: copy.deepcopy(litellm.model_cost.get(custom_id)),
        sibling_id: copy.deepcopy(litellm.model_cost.get(sibling_id)),
    }
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "search-custom",
                    "litellm_params": {"model": backend_model, "api_key": "fake-key"},
                    "model_info": {
                        "id": custom_id,
                        "search_context_cost_per_query": {
                            "search_context_size_low": 0.123,
                        },
                    },
                },
                {
                    "model_name": "search-sibling",
                    "litellm_params": {"model": backend_model, "api_key": "fake-key"},
                    "model_info": {"id": sibling_id},
                },
            ],
        )

        custom_info = router.get_deployment_model_info(model_id=custom_id, model_name=backend_model)
        sibling_info = router.get_deployment_model_info(model_id=sibling_id, model_name=backend_model)

        assert custom_info is not None
        assert custom_info["search_context_cost_per_query"]["search_context_size_low"] == 0.123
        assert litellm.model_cost[backend_model]["search_context_cost_per_query"] == expected_nested
        assert sibling_info is not None
        assert sibling_info["search_context_cost_per_query"] == expected_nested
    finally:
        _restore_model_cost_entries(model_keys)
        litellm.get_model_info.cache_clear()


def test_base_model_custom_info_does_not_pollute_cached_base_model():
    base_model = "azure/gpt-4o"
    deployment_id = "lit5471-base-model"
    base_model_info = copy.deepcopy(litellm.get_model_info(model=base_model))
    model_keys = {
        "azure/gpt-4o": copy.deepcopy(litellm.model_cost.get("azure/gpt-4o")),
        deployment_id: copy.deepcopy(litellm.model_cost.get(deployment_id)),
    }
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "azure-custom",
                    "litellm_params": {
                        "model": "gpt-4o",
                        "custom_llm_provider": "azure",
                        "api_key": "fake-key",
                    },
                    "model_info": {
                        "id": deployment_id,
                        "base_model": base_model,
                        "input_cost_per_token": 0.777,
                    },
                }
            ],
        )

        info = router.get_deployment_model_info(model_id=deployment_id, model_name=base_model)

        assert info is not None
        assert info["input_cost_per_token"] == 0.777
        assert litellm.get_model_info(model=base_model) == base_model_info
    finally:
        _restore_model_cost_entries(model_keys)
        litellm.get_model_info.cache_clear()


def test_builtin_only_deployment_info_is_not_the_cached_object():
    backend_model = "gpt-4o-search-preview"
    deployment_id = "lit5471-builtin-only"
    litellm.get_model_info.cache_clear()
    model_keys = {deployment_id: copy.deepcopy(litellm.model_cost.get(deployment_id))}
    try:
        cached_info = litellm.get_model_info(model=backend_model)
        assert cached_info["search_context_cost_per_query"]

        info = Router(model_list=[]).get_deployment_model_info(model_id=deployment_id, model_name=backend_model)

        assert info is not None
        assert info["search_context_cost_per_query"] == cached_info["search_context_cost_per_query"]
        assert _nested_container_ids(info).isdisjoint(_nested_container_ids(cached_info))
    finally:
        _restore_model_cost_entries(model_keys)
        litellm.get_model_info.cache_clear()


def test_custom_only_deployment_info_is_not_the_registry_entry():
    unknown_backend = "openai/lit5471-unknown-backend"
    deployment_id = "lit5471-custom-only"
    nested_pricing = {"search_context_size_low": 0.123}
    model_keys = {
        unknown_backend: copy.deepcopy(litellm.model_cost.get(unknown_backend)),
        deployment_id: copy.deepcopy(litellm.model_cost.get(deployment_id)),
    }
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "custom-only",
                    "litellm_params": {"model": unknown_backend, "api_key": "fake-key"},
                    "model_info": {"id": deployment_id, "search_context_cost_per_query": dict(nested_pricing)},
                }
            ],
        )
        registry_entry = litellm.model_cost[deployment_id]

        info = router.get_deployment_model_info(model_id=deployment_id, model_name=unknown_backend)

        assert info is not None
        assert info["search_context_cost_per_query"] == nested_pricing
        assert _nested_container_ids(info).isdisjoint(_nested_container_ids(registry_entry))
    finally:
        _restore_model_cost_entries(model_keys)
        litellm.get_model_info.cache_clear()


def test_router_model_info_deep_copies_nested_cached_metadata():
    model = "openai/gpt-4o-search-preview"
    litellm.get_model_info.cache_clear()
    try:
        cached_info = litellm.get_model_info(model=model)
        assert cached_info is not None
        expected_nested = copy.deepcopy(cached_info["search_context_cost_per_query"])
        assert expected_nested

        router = Router(model_list=[])
        merged_info = router.get_router_model_info(
            deployment={
                "model_name": "search",
                "litellm_params": {"model": "gpt-4o-search-preview"},
                "model_info": {"id": "lit5471-router-model-info"},
            },
            received_model_name="search",
        )

        assert merged_info["search_context_cost_per_query"] == expected_nested
        assert _nested_container_ids(merged_info).isdisjoint(_nested_container_ids(cached_info))
        assert litellm.get_model_info(model=model)["search_context_cost_per_query"] == expected_nested
    finally:
        litellm.get_model_info.cache_clear()


# --- a config.yaml reservation must carry an id its operator owns --------------------


def test_a_reservation_without_a_declared_id_is_refused():
    """Left underived the id is a hash of the resolved litellm_params, so rotating the
    credential mints a second identity and the catch-up bills the window again under it.
    The flat cost is keyed by that id and a written charge is never retracted, so the
    duplicate is permanent."""
    anonymous = {k: v for k, v in _PTU_MODEL_INFO.items() if k != "id"}

    with pytest.raises(ValueError, match=re.escape("model_info.id is required")):
        _ptu_router(model_info=anonymous)


def test_the_id_rule_does_not_reach_a_deployment_without_ptu_config():
    """An ordinary deployment keeps deriving its id, which is most of every config.yaml."""
    entry = _ptu_router(model_info={"team_id": "team-alpha"}).model_list[0]

    assert entry["model_info"]["id"]


def test_a_reservation_is_left_alone_while_the_feature_is_off():
    anonymous = {k: v for k, v in _PTU_MODEL_INFO.items() if k != "id"}
    entry = _ptu_router(model_info=anonymous, ptu_enabled=False).model_list[0]

    assert entry["model_info"]["id"]


def test_two_reservations_cannot_share_one_id():
    """Both would key the same sentinel row, so the second upsert overwrites the first and
    one reservation is billed at the other's rate."""
    with patch.dict(os.environ, {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True"}, clear=False):
        with pytest.raises(ValueError, match="declared on more than one deployment"):
            Router(
                model_list=[
                    {
                        "model_name": "azure-ptu",
                        "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://e.azure.com"},
                        "model_info": dict(_PTU_MODEL_INFO),
                    },
                    {
                        "model_name": "azure-ptu-west",
                        "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://w.azure.com"},
                        "model_info": dict(_PTU_MODEL_INFO),
                    },
                ]
            )


def test_two_reservations_with_distinct_ids_both_register():
    """The refusal must be scoped to a collision, not to a team running two regions."""
    with patch.dict(os.environ, {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True"}, clear=False):
        router = Router(
            model_list=[
                {
                    "model_name": "azure-ptu",
                    "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://e.azure.com"},
                    "model_info": dict(_PTU_MODEL_INFO),
                },
                {
                    "model_name": "azure-ptu-west",
                    "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://w.azure.com"},
                    "model_info": {**_PTU_MODEL_INFO, "id": "ptu-alpha-westus"},
                },
            ]
        )

    assert sorted(m["model_info"]["id"] for m in router.model_list) == ["ptu-alpha-eastus", "ptu-alpha-westus"]


@pytest.mark.parametrize("declared", ["dup-id", 12345], ids=["string id", "numeric id"])
def test_a_duplicate_id_is_caught_whatever_yaml_parsed_it_as(declared):
    """An unquoted id in config.yaml arrives as an int, and ModelInfo stores it as a string,
    so both deployments would still key one flat-cost row."""

    def entry(name, region):
        return {
            "model_name": name,
            "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": f"https://{region}.azure.com"},
            "model_info": {**_PTU_MODEL_INFO, "id": declared},
        }

    with patch.dict(os.environ, {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True"}, clear=False):
        with pytest.raises(ValueError, match="declared on more than one deployment"):
            Router(model_list=[entry("a", "eastus"), entry("b", "westus")])


def test_a_bare_yaml_date_bound_does_not_escape_the_id_rule():
    """`ptu_effective_to: 2027-01-01` unquoted loads as a date. While that failed to parse,
    the reservation was invisible to PTU entirely: no id rule, no zeroing, no flat cost."""
    import datetime as _dt

    windowed = {k: v for k, v in _PTU_MODEL_INFO.items() if k != "id"}

    with pytest.raises(ValueError, match=re.escape("model_info.id is required")):
        _ptu_router(model_info={**windowed, "ptu_effective_to": _dt.date(2027, 1, 1)})


def test_a_reservation_declaring_id_zero_registers():
    """0 is stable and unique, so reading it as absent refused a correct config."""
    entry = _ptu_router(model_info={**_PTU_MODEL_INFO, "id": 0}).model_list[0]

    assert entry["model_info"]["id"] == "0"


def test_a_falsy_id_is_still_scanned_for_collisions():
    """The duplicate scan skipped falsy ids, so a reservation on '0' could share its key with
    an ordinary deployment and the id index would keep only the last one registered."""
    with patch.dict(os.environ, {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True"}, clear=False):
        with pytest.raises(ValueError, match="declared on more than one deployment"):
            Router(
                model_list=[
                    {
                        "model_name": "azure-ptu",
                        "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://e.azure.com"},
                        "model_info": {**_PTU_MODEL_INFO, "id": "0"},
                    },
                    {
                        "model_name": "plain-sibling",
                        "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://w.azure.com"},
                        "model_info": {"id": 0},
                    },
                ]
            )


# --- a reservation declared while the feature is off says so ------------------------


def _ptu_warnings(caplog):
    return tuple(
        record.getMessage()
        for record in caplog.records
        if record.name == "LiteLLM Router" and record.levelno == logging.WARNING and "PTU" in record.getMessage()
    )


def test_a_reservation_declared_while_the_feature_is_off_is_warned_about(caplog):
    """The deployment serves and bills per token, so without this the operator believes they
    reserved capacity and sees no signal anywhere that nothing accrues."""
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        _ptu_router(ptu_enabled=False)

    warnings = _ptu_warnings(caplog)

    assert len(warnings) == 1
    assert "gpt-4o-ptu" in warnings[0]
    assert "LITELLM_ENABLE_PTU_COST_ATTRIBUTION" in warnings[0]


def test_a_reservation_is_not_warned_about_while_the_feature_is_on(caplog):
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        _ptu_router()

    assert _ptu_warnings(caplog) == ()


def test_a_deployment_carrying_no_ptu_field_is_not_warned_about(caplog):
    """Most of every config.yaml, so warning here would fire on proxies that never asked."""
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        _ptu_router(model_info={"team_id": "team-alpha"}, ptu_enabled=False)

    assert _ptu_warnings(caplog) == ()


def test_a_half_written_reservation_is_warned_about(caplog):
    """A count with no rate is not a chargeable reservation, but the operator still meant to
    declare one, so what they wrote is what decides whether they hear about it."""
    half_written = {k: v for k, v in _PTU_MODEL_INFO.items() if k != "cost_per_ptu_per_hour"}

    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        _ptu_router(model_info=half_written, ptu_enabled=False)

    assert len(_ptu_warnings(caplog)) == 1


@pytest.mark.parametrize(
    "typo",
    [
        {"ptu_count": 0},
        {"ptu_count": 0, "cost_per_ptu_per_hour": 0, "ptu_effective_from": None},
    ],
    ids=["count out of range", "every value still a zero placeholder"],
)
def test_a_reservation_dropped_by_a_typo_is_warned_about(caplog, typo):
    """An out-of-range value fails ModelInfo before the flag is ever consulted, so the
    deployment stops serving on a proxy that never enabled PTU. The warning is what tells the
    operator which feature the entry that vanished belonged to.

    Built the way proxy_server builds it, since dropping rather than raising is what
    ``ignore_invalid_deployments`` does and config.yaml is loaded with it on.
    """
    with patch.dict(os.environ, {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": ""}, clear=False):
        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            router = Router(
                ignore_invalid_deployments=True,
                model_list=[
                    {
                        "model_name": "gpt-4o-ptu",
                        "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://e.azure.com"},
                        "model_info": {**_PTU_MODEL_INFO, **typo},
                    }
                ],
            )

    assert router.model_list == []
    assert len(_ptu_warnings(caplog)) == 1


def test_a_db_backed_reservation_is_not_warned_about(caplog):
    """/model/new already answered the caller with a 400, so repeating it on every reload
    would report the operator's own rejected write back to them as a standing problem."""
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        _ptu_router(model_info={**_PTU_MODEL_INFO, "db_model": True}, ptu_enabled=False)

    assert _ptu_warnings(caplog) == ()


def test_every_declaring_deployment_is_named(caplog):
    """One line naming all of them, so a reload does not bury the config in repeats."""
    with patch.dict(os.environ, {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": ""}, clear=False):
        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            Router(
                model_list=[
                    {
                        "model_name": "azure-ptu-east",
                        "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://e.azure.com"},
                        "model_info": dict(_PTU_MODEL_INFO),
                    },
                    {
                        "model_name": "azure-ptu-west",
                        "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://w.azure.com"},
                        "model_info": {**_PTU_MODEL_INFO, "id": "ptu-alpha-westus"},
                    },
                    {
                        "model_name": "plain-gpt-4o",
                        "litellm_params": {"model": "azure/gpt-4o", "api_key": "k", "api_base": "https://p.azure.com"},
                        "model_info": {"id": "plain"},
                    },
                ]
            )

    warnings = _ptu_warnings(caplog)

    assert len(warnings) == 1
    assert "azure-ptu-east" in warnings[0]
    assert "azure-ptu-west" in warnings[0]
    assert "plain-gpt-4o" not in warnings[0]


def _simulate_price_data_reload_with_provider_sets(monkeypatch, fetched_catalog):
    """Like `_simulate_price_data_reload`, plus the provider model-set refresh the proxy's
    `_swap_in_model_cost_map` does before replaying, so bare names in the new catalog resolve."""
    monkeypatch.setattr(litellm, "model_cost", fetched_catalog)
    _invalidate_model_cost_lowercase_map()
    litellm.add_known_models(model_cost_map=fetched_catalog)
    reapply_runtime_model_cost_registrations()


def test_a_config_deployment_dropped_by_a_stale_cost_map_comes_back_on_reload(monkeypatch):
    """
    Booting on the bundled backup, a bare model that only the remote catalog knows
    cannot be provider-resolved, so the proxy router (ignore_invalid_deployments) drops
    it. Once a reload brings in a catalog that knows the model, the deployment must be
    served again with its access groups, and exactly once however many reloads follow.
    """
    backend = "lit-5766-only-in-remote-catalog"
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "new-model",
                    "litellm_params": {"model": backend, "api_key": "k"},
                    "model_info": {"id": "new-id", "access_groups": ["team-models"]},
                },
                {
                    "model_name": "control-model",
                    "litellm_params": {"model": "hosted_vllm/control-backend", "api_key": "k"},
                    "model_info": {"id": "control-id", "access_groups": ["team-models"]},
                },
            ],
            ignore_invalid_deployments=True,
        )
        assert router.get_model_names() == ["control-model"]
        assert router.get_model_access_groups(model_name="new-model") == {}

        fresh_catalog = {**litellm.model_cost, backend: {"litellm_provider": "openai", "mode": "chat"}}
        _simulate_price_data_reload_with_provider_sets(monkeypatch, fresh_catalog)
        _simulate_price_data_reload_with_provider_sets(monkeypatch, fresh_catalog)

        assert sorted(router.get_model_names()) == ["control-model", "new-model"]
        assert router.get_model_access_groups(model_name="new-model") == {"team-models": ["new-model"]}
        assert [d["model_info"]["id"] for d in router.model_list] == ["control-id", "new-id"]
        assert "new-id" in litellm.model_cost
    finally:
        litellm.open_ai_chat_completion_models.discard(backend)
        litellm.models_by_provider["openai"].discard(backend)


def test_a_config_deployment_dropped_for_a_permanent_reason_is_not_retried_on_reload(monkeypatch):
    """
    Only provider-resolution drops can be healed by a fresh catalog. A deployment that
    fails after its provider resolved (here a pass-through vertex entry with no project)
    has already touched router state, so replaying it on every reload would leak into
    `deployment_names` each time.
    """
    router = Router(
        model_list=[
            {
                "model_name": "vertex-passthrough",
                "litellm_params": {"model": "vertex_ai/gemini-2.5-flash", "use_in_pass_through": True},
                "model_info": {"id": "vertex-id"},
            },
            {
                "model_name": "control-model",
                "litellm_params": {"model": "hosted_vllm/control-backend", "api_key": "k"},
                "model_info": {"id": "control-id"},
            },
        ],
        ignore_invalid_deployments=True,
    )
    assert router.get_model_names() == ["control-model"]
    names_after_boot = list(router.deployment_names)

    _simulate_price_data_reload_with_provider_sets(monkeypatch, dict(litellm.model_cost))

    assert router.get_model_names() == ["control-model"]
    assert router.deployment_names == names_after_boot


def test_price_data_reload_refreshes_the_cached_model_group_and_deployment_info(monkeypatch):
    """
    Budget reservation reads pricing through the router's lru-cached group and
    deployment lookups. A reload swaps the catalog without touching model_list, so
    unless the replay clears those caches the next reservation prices against the
    old catalog until some unrelated model-list change happens to evict it.
    """
    router = Router(
        model_list=[
            {
                "model_name": "grp",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k"},
                "model_info": {"id": "dep-a"},
            }
        ]
    )
    old_price = router.cached_model_group_info("grp").input_cost_per_token
    assert router.cached_deployment_model_info("dep-a", "openai/gpt-4o")["input_cost_per_token"] == old_price

    new_price = old_price * 10
    fresh_catalog = copy.deepcopy(litellm.model_cost)
    fresh_catalog["gpt-4o"]["input_cost_per_token"] = new_price
    _simulate_price_data_reload_with_provider_sets(monkeypatch, fresh_catalog)

    assert router.cached_model_group_info("grp").input_cost_per_token == new_price
    assert router.cached_deployment_model_info("dep-a", "openai/gpt-4o")["input_cost_per_token"] == new_price


def test_price_data_reload_keeps_geo_deployments_priced_while_requests_register_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    catalog: Final = litellm.get_model_cost_map(url="")
    monkeypatch.setattr(litellm, "model_cost", dict(catalog))
    _invalidate_model_cost_lowercase_map()
    backends: Final = tuple(
        f"bedrock/au.anthropic.{model}"
        for model in ("claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001-v1:0", "claude-sonnet-4-6")
    )
    canonical_prices: Final = MappingProxyType(
        {backend: catalog[backend.removeprefix("bedrock/")]["input_cost_per_token"] for backend in backends}
    )
    assert all(price > 0 for price in canonical_prices.values())
    router: Final = Router(
        model_list=[
            {
                "model_name": backend.rsplit(".", 1)[-1],
                "litellm_params": {"model": backend, "aws_region_name": "ap-southeast-2"},
                "model_info": {"id": f"lit-5853-dep-{index}"},
            }
            for index, backend in enumerate(backends)
        ]
    )
    stop: Final = threading.Event()
    writer_outcome: Final[queue.SimpleQueue[int | Exception]] = queue.SimpleQueue()

    def register_until_stopped() -> None:
        try:
            for i in itertools.count():
                if stop.is_set():
                    writer_outcome.put(i)
                    return
                litellm.register_model(
                    {
                        f"lit-5853-writer-{i}": {
                            "litellm_provider": "lit-5853-writer",
                            "mode": "chat",
                            "input_cost_per_token": 1e-07,
                            "output_cost_per_token": 4e-07,
                            "cache_read_input_token_cost": 1e-08,
                        }
                    },
                    persist_across_reloads=False,
                )
        except Exception as failure:
            writer_outcome.put(failure)

    writer: Final = threading.Thread(target=register_until_stopped, daemon=True)
    saved_switch_interval: Final = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    writer.start()
    try:
        for _ in range(10):
            _simulate_price_data_reload_with_provider_sets(monkeypatch, dict(catalog))
    finally:
        stop.set()
        writer.join()
        sys.setswitchinterval(saved_switch_interval)

    registrations_landed: Final = writer_outcome.get(timeout=5)
    try:
        assert isinstance(registrations_landed, int) and registrations_landed > 0, registrations_landed
        assert tuple(backend for backend in backends if backend in litellm.model_cost) == ()
        for backend in backends:
            assert litellm.get_model_info(backend)["input_cost_per_token"] == canonical_prices[backend]
        assert tuple(router.get_model_names()) == tuple(backend.rsplit(".", 1)[-1] for backend in backends)
    finally:
        _invalidate_model_cost_lowercase_map()


_COLLIDING_BUILTIN_KEY: Final = "baseten/zai-org/GLM-5.2"
_COLLIDING_SHARED_OPENAI_KEY: Final = "openai/zai-org/GLM-5.2"
_COLLIDING_MODEL_INFO: Final = {
    "input_cost_per_token": 0.00000096,
    "output_cost_per_token": 0.00000302,
    "cache_read_input_token_cost": 0.00000010,
    "mode": "chat",
}


def _colliding_id_deployment(model_id: str, custom_llm_provider: str) -> DeploymentTypedDict:
    litellm_params: Final = LiteLLMParamsTypedDict(
        model="zai-org/GLM-5.2",
        api_base="https://inference.baseten.co/v1",
        custom_llm_provider=custom_llm_provider,
    )
    model_info: Final[dict] = {"id": model_id, **_COLLIDING_MODEL_INFO}
    return DeploymentTypedDict(
        model_name="nvidia/zai-org/glm-5.2",
        litellm_params=litellm_params,
        model_info=model_info,
    )


@pytest.mark.parametrize("model_id", ("baseten/zai-org/glm-5.2",))
def test_deployment_id_colliding_with_another_providers_catalog_key_keeps_its_own_pricing(model_id: str) -> None:
    """A deployment id equal to another provider's catalog key must not merge
    into that row: the merge leaves litellm_provider=baseten on the entry, which
    _check_provider_match then rejects for the openai request, billing $0."""
    builtin_row_before: Final = copy.deepcopy(litellm.model_cost[_COLLIDING_BUILTIN_KEY])
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key))
        for key in (model_id, _COLLIDING_BUILTIN_KEY, _COLLIDING_SHARED_OPENAI_KEY)
    }
    try:
        router: Final = Router(model_list=[_colliding_id_deployment(model_id, "openai")])

        response: Final = router.completion(
            model="nvidia/zai-org/glm-5.2",
            messages=[{"role": "user", "content": "colliding id pricing"}],
            mock_response=litellm.ModelResponse(
                model="zai-org/GLM-5.2",
                usage=litellm.Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
            ),
        )

        assert isinstance(response, litellm.ModelResponse)
        assert response._hidden_params["response_cost"] == pytest.approx(1000 * 0.00000096 + 500 * 0.00000302)
        assert litellm.model_cost[_COLLIDING_BUILTIN_KEY] == builtin_row_before
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


def test_deployment_id_matching_its_own_providers_catalog_key_still_merges() -> None:
    """Same-provider collision keeps today's merge behavior: the deployment's
    custom prices land on the baseten row the openai-compatible baseten request
    matches against."""
    model_id: Final = "baseten/zai-org/glm-5.2"
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key)) for key in (model_id, _COLLIDING_BUILTIN_KEY)
    }
    try:
        router: Final = Router(model_list=[_colliding_id_deployment(model_id, "baseten")])

        response: Final = router.completion(
            model="nvidia/zai-org/glm-5.2",
            messages=[{"role": "user", "content": "same provider pricing"}],
            mock_response=litellm.ModelResponse(
                model="zai-org/GLM-5.2",
                usage=litellm.Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
            ),
        )

        assert isinstance(response, litellm.ModelResponse)
        assert response._hidden_params["response_cost"] == pytest.approx(1000 * 0.00000096 + 500 * 0.00000302)
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


_CAPACITY_TEST_DEPLOYMENTS: Final = (
    (
        "bedrock",
        "bedrock/converse/anthropic.claude-router-capacity-test",
        "bedrock/us-east-1/anthropic.claude-router-capacity-test",
    ),
    (
        "bedrock_mantle",
        "bedrock_mantle/openai.router-capacity-test",
        "bedrock_mantle/us-east-1/openai.router-capacity-test",
    ),
    ("openai", "openai/router-capacity-test", "openai/us/router-capacity-test"),
)


def _capacity_test_metadata(provider: str, max_input_tokens: int) -> dict[str, object]:
    return {
        "litellm_provider": provider,
        "mode": "chat",
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": 512,
        "max_tokens": 512,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    }


def _capacity_test_router(route: str, base: str, base_location: str | None, explicit_limit: int | None) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "capacity-test",
                "litellm_params": {
                    "model": route,
                    **({"base_model": base} if base_location == "litellm_params" else {}),
                },
                "model_info": {
                    "id": "capacity-test",
                    **({"base_model": base} if base_location == "model_info" else {}),
                    **({"max_input_tokens": explicit_limit} if explicit_limit is not None else {}),
                },
            }
        ],
        enable_pre_call_checks=True,
    )


def _deployments_allowed_for_2048_input_tokens(router: Router) -> int:
    try:
        return len(
            router._pre_call_checks(
                model="capacity-test",
                healthy_deployments=router.model_list,
                input="synthetic input",
                input_token_count=2048,
            )
        )
    except litellm.ContextWindowExceededError:
        return 0


def _effective_max_input_tokens(router: Router) -> object:
    return router.get_router_model_info(id="capacity-test", deployment=None, received_model_name="capacity-test")[
        "max_input_tokens"
    ]


@pytest.mark.parametrize(("provider", "route", "base"), _CAPACITY_TEST_DEPLOYMENTS)
@pytest.mark.parametrize("base_location", ("model_info", "litellm_params", None))
@pytest.mark.parametrize("explicit_limit", (None, 4096))
def test_router_capacity_widens_to_a_larger_base_model_window(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    route: str,
    base: str,
    base_location: str | None,
    explicit_limit: int | None,
) -> None:
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.model_cost,
            route: _capacity_test_metadata(provider, 1024),
            base: _capacity_test_metadata(provider, 8192),
        },
    )
    _invalidate_model_cost_lowercase_map()
    router: Final = _capacity_test_router(route, base, base_location, explicit_limit)
    expected: Final = explicit_limit if explicit_limit is not None else (1024 if base_location is None else 8192)
    assert _effective_max_input_tokens(router) == expected
    assert _deployments_allowed_for_2048_input_tokens(router) == (1 if expected >= 2048 else 0)
    _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize(("provider", "route", "base"), _CAPACITY_TEST_DEPLOYMENTS)
@pytest.mark.parametrize("base_location", ("model_info", "litellm_params"))
def test_router_capacity_never_narrows_to_a_smaller_base_model_window(
    monkeypatch: pytest.MonkeyPatch, provider: str, route: str, base: str, base_location: str
) -> None:
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.model_cost,
            route: _capacity_test_metadata(provider, 8192),
            base: _capacity_test_metadata(provider, 1024),
        },
    )
    _invalidate_model_cost_lowercase_map()
    router: Final = _capacity_test_router(route, base, base_location, None)
    assert _effective_max_input_tokens(router) == 8192
    assert _deployments_allowed_for_2048_input_tokens(router) == 1
    _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize(("provider", "route", "base"), _CAPACITY_TEST_DEPLOYMENTS)
def test_router_capacity_never_takes_a_window_from_base_model_when_the_route_is_unmapped(
    monkeypatch: pytest.MonkeyPatch, provider: str, route: str, base: str
) -> None:
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost, base: _capacity_test_metadata(provider, 1024)})
    _invalidate_model_cost_lowercase_map()
    router: Final = _capacity_test_router(f"{route}-not-in-cost-map", base, "model_info", None)
    assert _deployments_allowed_for_2048_input_tokens(router) == 1
    _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize(("provider", "route", "base"), _CAPACITY_TEST_DEPLOYMENTS)
def test_router_capacity_never_takes_a_window_from_base_model_when_the_route_declares_none(
    monkeypatch: pytest.MonkeyPatch, provider: str, route: str, base: str
) -> None:
    route_without_window: Final = {
        key: value
        for key, value in _capacity_test_metadata(provider, 0).items()
        if key not in ("max_input_tokens", "max_tokens")
    }
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {**litellm.model_cost, route: route_without_window, base: _capacity_test_metadata(provider, 1024)},
    )
    _invalidate_model_cost_lowercase_map()
    router: Final = _capacity_test_router(route, base, "model_info", None)
    assert _effective_max_input_tokens(router) is None
    assert _deployments_allowed_for_2048_input_tokens(router) == 1
    _invalidate_model_cost_lowercase_map()


_SAME_MODEL_ROUTE: Final = "bedrock/converse/anthropic.claude-routercapacitytest-9-9"
_SAME_MODEL_GLOBAL_ROUTE: Final = "bedrock/converse/global.anthropic.claude-routercapacitytest-9-9"
_SAME_MODEL_QUALIFIED_ROUTE: Final = "bedrock/converse/anthropic.claude-routercapacitytest-9-9@first"


@pytest.mark.parametrize(
    ("route", "base_model", "expected"),
    (
        (_SAME_MODEL_ROUTE, "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9", 8192),
        (_SAME_MODEL_ROUTE, "us-east-1/anthropic.claude-routercapacitytest-9-9", 1024),
        (_SAME_MODEL_ROUTE, "anthropic.claude-routercapacitytest-9-9", 1024),
        (_SAME_MODEL_GLOBAL_ROUTE, "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9", 8192),
        (_SAME_MODEL_ROUTE, "bedrock/anthropic.claude-routercapacitytest-9-9@global:us-east-1", 8192),
        (_SAME_MODEL_GLOBAL_ROUTE, "bedrock/anthropic.claude-routercapacitytest-9-9@global:us-east-1", 8192),
        (_SAME_MODEL_QUALIFIED_ROUTE, "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9@first", 8192),
        (_SAME_MODEL_QUALIFIED_ROUTE, "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9@second", 1024),
        (_SAME_MODEL_ROUTE, "bedrock/us-east-1/anthropic.claude-routercapacitytest-*", 1024),
        (_SAME_MODEL_ROUTE, "bedrock/us-east-1/anthropic.claude-routercapacitystandin-9-9", 1024),
        (_SAME_MODEL_ROUTE, "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9-extended", 1024),
        (_SAME_MODEL_ROUTE, "bedrock/eu-west-1/anthropic.claude-routercapacitytest-9-9", 1024),
    ),
)
def test_router_capacity_widens_only_for_an_exact_base_model_entry_of_the_same_model(
    monkeypatch: pytest.MonkeyPatch, route: str, base_model: str, expected: int
) -> None:
    larger_window: Final = _capacity_test_metadata("bedrock", 8192)
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.model_cost,
            _SAME_MODEL_ROUTE: _capacity_test_metadata("bedrock", 1024),
            _SAME_MODEL_GLOBAL_ROUTE: _capacity_test_metadata("bedrock", 1024),
            _SAME_MODEL_QUALIFIED_ROUTE: _capacity_test_metadata("bedrock", 1024),
            "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9@first": larger_window,
            "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9@second": larger_window,
            "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9": larger_window,
            "anthropic.claude-routercapacitytest-9-9": _capacity_test_metadata("bedrock", 1024),
            "bedrock/anthropic.claude-routercapacitytest-9-9": larger_window,
            "bedrock/anthropic.claude-routercapacitytest-9-9@global:us-east-1": larger_window,
            "bedrock/us-east-1/anthropic.claude-routercapacitytest-*": larger_window,
            "bedrock/us-east-1/anthropic.claude-routercapacitystandin-9-9": larger_window,
            "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9-extended": larger_window,
        },
    )
    _invalidate_model_cost_lowercase_map()
    router: Final = _capacity_test_router(route, base_model, "model_info", None)
    assert _effective_max_input_tokens(router) == expected
    assert _deployments_allowed_for_2048_input_tokens(router) == (1 if expected >= 2048 else 0)
    _invalidate_model_cost_lowercase_map()


def test_router_capacity_reads_a_model_info_base_model_before_a_litellm_params_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.model_cost,
            _SAME_MODEL_ROUTE: _capacity_test_metadata("bedrock", 1024),
            "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9": _capacity_test_metadata("bedrock", 8192),
            "bedrock/us-east-1/anthropic.claude-routercapacitystandin-9-9": _capacity_test_metadata("bedrock", 8192),
        },
    )
    _invalidate_model_cost_lowercase_map()
    router: Final = Router(
        model_list=[
            {
                "model_name": "capacity-test",
                "litellm_params": {
                    "model": _SAME_MODEL_ROUTE,
                    "base_model": "bedrock/us-east-1/anthropic.claude-routercapacitytest-9-9",
                },
                "model_info": {
                    "id": "capacity-test",
                    "base_model": "bedrock/us-east-1/anthropic.claude-routercapacitystandin-9-9",
                },
            }
        ],
        enable_pre_call_checks=True,
    )
    assert _effective_max_input_tokens(router) == 1024
    assert _deployments_allowed_for_2048_input_tokens(router) == 0
    _invalidate_model_cost_lowercase_map()


def test_deployment_id_equal_to_another_providers_catalog_key_bills_its_own_prices_under_pricing_provider() -> None:
    """Resolving the id by name under the routing provider lands on the other provider's row and the
    provider check rejects it, billing $0. ``pricing_provider`` names the row's provider, so the
    deployment's own prices (merged into that row) bill."""
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key))
        for key in (_COLLIDING_BUILTIN_KEY, _COLLIDING_SHARED_OPENAI_KEY)
    }
    try:
        router: Final = Router(
            model_list=[
                _pricing_provider_deployment(
                    _COLLIDING_BUILTIN_KEY,
                    {"input_cost_per_token": 0.00000096, "output_cost_per_token": 0.00000302},
                )
            ]
        )

        response: Final = _glm_completion(router)

        assert response._hidden_params["response_cost"] == pytest.approx(1000 * 0.00000096 + 500 * 0.00000302)
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


def _pricing_provider_deployment(model_id: str, model_info: dict) -> DeploymentTypedDict:
    litellm_params: Final = LiteLLMParamsTypedDict(
        model="zai-org/GLM-5.2",
        api_base="https://inference.baseten.co/v1",
        custom_llm_provider="openai",
    )
    return DeploymentTypedDict(
        model_name="nvidia/zai-org/glm-5.2",
        litellm_params=litellm_params,
        model_info={"id": model_id, "pricing_provider": "baseten", **model_info},
    )


def _glm_completion(router: Router) -> litellm.ModelResponse:
    response: Final = router.completion(
        model="nvidia/zai-org/glm-5.2",
        messages=[{"role": "user", "content": "priced under another provider"}],
        mock_response=litellm.ModelResponse(
            model="zai-org/GLM-5.2",
            usage=litellm.Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        ),
    )
    assert isinstance(response, litellm.ModelResponse)
    return response


def test_pricing_provider_without_custom_prices_bills_that_providers_catalog_rate() -> None:
    """An openai-compatible deployment pointed at another provider's host can name that provider for
    pricing: the request is priced from its catalog row, where the routing provider has none."""
    model_id: Final = "glm52-priced-as-baseten"
    catalog_row: Final = litellm.model_cost[_COLLIDING_BUILTIN_KEY]
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key))
        for key in (model_id, _COLLIDING_BUILTIN_KEY, _COLLIDING_SHARED_OPENAI_KEY)
    }
    try:
        response: Final = _glm_completion(Router(model_list=[_pricing_provider_deployment(model_id, {})]))

        assert response._hidden_params["response_cost"] == pytest.approx(
            1000 * catalog_row["input_cost_per_token"] + 500 * catalog_row["output_cost_per_token"]
        )
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()


def test_pricing_provider_fills_unset_rates_from_that_providers_catalog_row() -> None:
    """Rates the deployment sets win, rates it leaves out come from the pricing provider's row for the
    backend model rather than defaulting to 0 because the routing provider has no row."""
    model_id: Final = "glm52-priced-as-baseten-custom"
    catalog_row: Final = copy.deepcopy(litellm.model_cost[_COLLIDING_BUILTIN_KEY])
    model_cost_entries: Final = {
        key: copy.deepcopy(litellm.model_cost.get(key))
        for key in (model_id, _COLLIDING_BUILTIN_KEY, _COLLIDING_SHARED_OPENAI_KEY)
    }
    rates: Final = {"input_cost_per_token": 0.00000096, "output_cost_per_token": 0.00000302}
    try:
        router: Final = Router(model_list=[_pricing_provider_deployment(model_id, rates)])

        registered: Final = litellm.model_cost[model_id]
        assert {key: registered[key] for key in rates} == rates
        assert registered["cache_read_input_token_cost"] == catalog_row["cache_read_input_token_cost"]
        assert _glm_completion(router)._hidden_params["response_cost"] == pytest.approx(
            1000 * 0.00000096 + 500 * 0.00000302
        )
        assert litellm.model_cost[_COLLIDING_BUILTIN_KEY] == catalog_row
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.get_model_info.cache_clear()
