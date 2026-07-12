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
