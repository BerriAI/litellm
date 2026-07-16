"""
Tests for Router(plugins=[...]) -- a pipeline of routing plugins that run
before the routing decision is made, narrowing the candidate deployment pool.

Discussion: https://github.com/BerriAI/litellm/discussions/32168
"""

import pytest

from litellm import Router
from litellm.types.router import RoutingContext


class LanguageDetector:
    async def run(self, context: RoutingContext) -> RoutingContext:
        context.signals["language-detector"] = {"lang": "en"}
        return context


class DomainClassifier:
    async def run(self, context: RoutingContext) -> RoutingContext:
        context.signals["domain-classifier"] = {"domain": "coding", "confidence": 0.93}
        return context


class TenantPolicy:
    ALLOWED_PROVIDERS = {"acme-corp": {"openai", "anthropic"}}

    async def run(self, context: RoutingContext) -> RoutingContext:
        tenant = context.metadata.get("tenant", "default")
        allowed = self.ALLOWED_PROVIDERS.get(tenant, {"openai", "anthropic", "self-hosted"})
        context.candidate_models = [m for m in context.candidate_models if m.split("/")[0] in allowed]
        context.signals["tenant-policy"] = {"tenant": tenant, "allowed_providers": sorted(allowed)}
        return context


class BudgetPolicy:
    COST_CAP_PER_TOKEN = 0.000005
    COST_BY_MODEL = {
        "openai/gpt-4o-mini": 0.00000015,
        "anthropic/claude-haiku-4-5": 0.000001,
        "openai/gpt-5.1": 0.00003,
    }

    async def run(self, context: RoutingContext) -> RoutingContext:
        context.candidate_models = [
            m for m in context.candidate_models if self.COST_BY_MODEL.get(m, 0) <= self.COST_CAP_PER_TOKEN
        ]
        context.signals["budget-policy"] = {"daily_limit": 100}
        return context


class BlockEverything:
    async def run(self, context: RoutingContext) -> RoutingContext:
        context.candidate_models = []
        return context


def _smart_router_model_list():
    return [
        {
            "model_name": "smart-router",
            "litellm_params": {"model": "openai/gpt-4o-mini", "mock_response": "cheap openai"},
            "model_info": {"tags": ["openai"]},
        },
        {
            "model_name": "smart-router",
            "litellm_params": {"model": "anthropic/claude-haiku-4-5", "mock_response": "anthropic"},
            "model_info": {"tags": ["anthropic"]},
        },
        {
            "model_name": "smart-router",
            "litellm_params": {"model": "openai/gpt-5.1", "mock_response": "expensive openai"},
            "model_info": {"tags": ["openai"]},
        },
        {
            "model_name": "smart-router",
            "litellm_params": {"model": "ollama/llama-3-70b", "mock_response": "self hosted"},
            "model_info": {"tags": ["self-hosted"]},
        },
    ]


@pytest.mark.asyncio
async def test_routing_plugin_pipeline_matches_jeann2013_e2e_scenario():
    """
    https://github.com/BerriAI/litellm/discussions/32168#discussioncomment-17608820

    language plugin -> domain classifier -> tenant policy (openai+anthropic only)
    -> budget policy (drops over-cap models) -> Router picks the best remaining
    candidate. Must never land on the self-hosted or over-budget deployment.
    """
    router = Router(
        model_list=_smart_router_model_list(),
        plugins=[LanguageDetector(), DomainClassifier(), TenantPolicy(), BudgetPolicy()],
    )

    response = await router.acompletion(
        model="smart-router",
        messages=[{"role": "user", "content": "Write a function to reverse a linked list."}],
        metadata={"tenant": "acme-corp"},
    )

    # response.model is the bare model name (litellm strips the provider/ prefix
    # on the response), so compare against bare names rather than litellm_params.model
    routed_model = response.model

    assert routed_model in {"gpt-4o-mini", "claude-haiku-4-5"}
    assert routed_model not in {"llama-3-70b", "gpt-5.1"}


@pytest.mark.asyncio
async def test_routing_plugin_narrowing_to_zero_candidates_raises():
    """A plugin narrowing to nothing is a policy decision -- must raise, not silently
    fall back to the unfiltered pool (that would defeat the policy it enforces)."""
    router = Router(
        model_list=_smart_router_model_list(),
        plugins=[BlockEverything()],
    )

    with pytest.raises(ValueError, match="No deployments left after routing-plugin filtering"):
        await router.acompletion(
            model="smart-router",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_sync_get_available_deployment_rejects_configured_plugins():
    """
    Router.completion() (and any other sync entry point) resolves deployments via
    the synchronous get_available_deployment(), which never runs the routing-plugin
    pipeline. Silently allowing that would let a deny-all policy plugin be bypassed
    just by calling the sync API -- must fail closed instead.
    """
    router = Router(model_list=_smart_router_model_list(), plugins=[TenantPolicy()])

    with pytest.raises(ValueError, match="routing-plugin pipeline"):
        router.get_available_deployment(model="smart-router", messages=[{"role": "user", "content": "hi"}])


def test_sync_router_completion_rejects_configured_plugins():
    """End-to-end: Router.completion() (the sync API) must not silently skip plugins either."""
    router = Router(model_list=_smart_router_model_list(), plugins=[TenantPolicy()])

    with pytest.raises(ValueError, match="routing-plugin pipeline"):
        router.completion(model="smart-router", messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_async_completion_with_unsupported_strategy_rejects_configured_plugins():
    """
    async_get_available_deployment() itself delegates to the synchronous selector
    for routing strategies outside {simple-shuffle, usage-based-routing-v2,
    cost-based-routing, latency-based-routing, least-busy} -- e.g. "usage-based-routing"
    (v1, not v2) -- which would silently bypass the plugin pipeline on the async path too.
    """
    router = Router(
        model_list=_smart_router_model_list(),
        plugins=[TenantPolicy()],
        routing_strategy="usage-based-routing",
    )

    with pytest.raises(ValueError, match="routing-plugin pipeline"):
        await router.acompletion(model="smart-router", messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_router_without_plugins_is_unaffected():
    """Regression guard: a Router with no `plugins` configured behaves exactly as before."""
    router = Router(
        model_list=[
            {
                "model_name": "smart-router",
                "litellm_params": {"model": "openai/gpt-4o-mini", "mock_response": "hi"},
            },
        ],
    )
    response = await router.acompletion(
        model="smart-router",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response.choices[0].message.content == "hi"


@pytest.mark.asyncio
async def test_run_routing_plugins_narrows_candidates_and_records_signals():
    """Unit-level check of _run_routing_plugins in isolation, independent of acompletion."""
    router = Router(
        model_list=_smart_router_model_list(),
        plugins=[LanguageDetector(), DomainClassifier(), TenantPolicy(), BudgetPolicy()],
    )
    request_kwargs = {"metadata": {"tenant": "acme-corp"}}

    context = await router._run_routing_plugins(
        model="smart-router",
        request_kwargs=request_kwargs,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert context.candidate_models == ["openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"]
    assert context.signals["domain-classifier"]["domain"] == "coding"
    assert request_kwargs["metadata"]["_routing_plugin_candidate_models"] == context.candidate_models


def test_filter_by_routing_plugin_candidates_narrows_and_raises_when_empty():
    """Unit-level check of _filter_by_routing_plugin_candidates in isolation."""
    router = Router(model_list=_smart_router_model_list(), plugins=[TenantPolicy()])
    healthy_deployments = router.model_list

    narrowed = router._filter_by_routing_plugin_candidates(
        healthy_deployments=healthy_deployments,
        request_kwargs={"metadata": {"_routing_plugin_candidate_models": ["openai/gpt-4o-mini"]}},
    )
    assert [d["litellm_params"]["model"] for d in narrowed] == ["openai/gpt-4o-mini"]

    with pytest.raises(ValueError, match="No deployments left after routing-plugin filtering"):
        router._filter_by_routing_plugin_candidates(
            healthy_deployments=healthy_deployments,
            request_kwargs={"metadata": {"_routing_plugin_candidate_models": ["nonexistent/model"]}},
        )


def test_json_default_stable_id_is_stable_across_instances():
    """_generate_model_id's json.dumps `default=` fallback must not embed an object's
    memory address (e.g. plain str() on an object with no custom __repr__ falls back
    to object.__repr__'s `<module.Class object at 0x...>`) -- that would make the
    deployment id churn on every process restart for any deployment whose
    litellm_params contain a live plugin instance."""
    router = Router(model_list=_smart_router_model_list())

    assert router._json_default_stable_id(LanguageDetector()) == router._json_default_stable_id(LanguageDetector())
    assert router._json_default_stable_id(LanguageDetector()) != router._json_default_stable_id(TenantPolicy())


def test_generate_model_id_is_stable_when_litellm_params_contain_a_plugin_instance():
    """End-to-end: a deployment id built from litellm_params containing a routing
    plugin instance (e.g. complexity_router_config.plugins) must be identical across
    separate calls, not just non-crashing."""
    router = Router(model_list=_smart_router_model_list())
    litellm_params = {
        "model": "auto_router/complexity_router",
        "complexity_router_config": {"plugins": [LanguageDetector()]},
    }

    id1 = router._generate_model_id("smart-router", litellm_params)
    id2 = router._generate_model_id(
        "smart-router",
        {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"plugins": [LanguageDetector()]},
        },
    )
    assert id1 == id2
