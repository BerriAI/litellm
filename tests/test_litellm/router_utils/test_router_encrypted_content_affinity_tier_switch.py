"""
Router-level encrypted-content affinity for complexity-router tier switches (#40237).

`EncryptedContentAffinityCheck` (tested in
`pre_call_checks/test_encrypted_content_affinity_check.py`) filters deployments *within*
one model group. A complexity router picks the model group first, per turn, so a
follow-up carrying encrypted reasoning can be classified into a different tier and never
reach the originating group at all. These tests cover the Router pre-routing hook that
keeps such a follow-up on the deployment that produced the encrypted content.
"""

import pytest

import litellm
from litellm.responses.utils import ResponsesAPIRequestUtils

# ---------------------------------------------------------------------------
# Complexity-router tier switches (#40237)
#
# A complexity router classifies every turn on its own, so a follow-up carrying
# encrypted reasoning can be reclassified into a different tier and sent to a
# model group on the other side of the encryption boundary. TIER_SWITCHING_PROMPT
# is a prompt the classifier puts in a non-default tier; each test asserts that
# premise so a classifier change cannot quietly make these vacuous.
# ---------------------------------------------------------------------------

TIER_SWITCHING_PROMPT = "Think through this step by step and then explain your reasoning in detail."


def _complexity_router_model_list(simple_tier_model="simple-model"):
    return [
        {
            "model_name": "auto-router",
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_default_model": simple_tier_model,
                "complexity_router_config": {
                    "tiers": {
                        "SIMPLE": simple_tier_model,
                        "COMPLEX": "complex-model",
                        "REASONING": "complex-model",
                    }
                },
            },
        },
        {
            "model_name": "simple-model",
            "litellm_params": {
                "model": "openai/gpt-simple",
                "api_base": "https://base-a.example.com",
                "api_key": "key-a",
            },
            "model_info": {"id": "dep-simple-1"},
        },
        {
            "model_name": "complex-model",
            "litellm_params": {
                "model": "openai/gpt-complex",
                "api_base": "https://base-b.example.com",
                "api_key": "key-b",
            },
            "model_info": {"id": "dep-complex-1"},
        },
    ]


def _reasoning_input_for(model_id, prompt=TIER_SWITCHING_PROMPT):
    """A follow-up turn whose reasoning item carries a marker for ``model_id``."""
    return [
        {
            "type": "reasoning",
            "encrypted_content": ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
                "dummy_ciphertext", model_id
            ),
        },
        {"role": "user", "content": prompt},
    ]


async def _assert_prompt_switches_tier(router):
    """Guard the premise: without a marker this prompt leaves the default tier."""
    plain_input = [{"role": "user", "content": TIER_SWITCHING_PROMPT}]
    unpinned = await router.async_get_available_deployment(
        model="auto-router",
        input=plain_input,
        request_kwargs={"input": plain_input},
    )
    assert unpinned.get("model_name") == "complex-model", (
        "premise broken: the classifier no longer moves TIER_SWITCHING_PROMPT off the "
        "default tier, so this test would pass without any affinity handling"
    )


@pytest.mark.asyncio
async def test_complexity_router_encrypted_content_affinity_bypasses_tier_switch():
    """
    Issue #40237: a follow-up carrying encrypted reasoning stays on the deployment
    that produced it, instead of being reclassified into another tier whose
    deployment sits on a different encryption boundary.

    Without the pin this raises ServiceUnavailableError from
    EncryptedContentAffinityCheck, because the COMPLEX tier deployment does not
    share an (api_base, api_key) boundary with the originating one.
    """
    router = litellm.Router(
        model_list=_complexity_router_model_list(),
        routing_strategy="simple-shuffle",
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    try:
        await _assert_prompt_switches_tier(router)

        request_input = _reasoning_input_for("dep-simple-1")
        deployment = await router.async_get_available_deployment(
            model="auto-router",
            input=request_input,
            request_kwargs={"input": request_input},
        )

        assert deployment.get("model_name") == "simple-model"
        assert deployment.get("model_info", {}).get("id") == "dep-simple-1"
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_resolve_encrypted_content_affinity_hook_pins_to_originating_model_group():
    """The hook returns the originating deployment's model group as the pre-routing decision."""
    router = litellm.Router(
        model_list=_complexity_router_model_list(),
        routing_strategy="simple-shuffle",
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    try:
        selected_strategy = router._select_pre_routing_strategy(model="auto-router", request_kwargs={})
        assert selected_strategy is not None

        request_input = _reasoning_input_for("dep-simple-1")
        response = router._resolve_encrypted_content_affinity_hook(
            selected_strategy=selected_strategy,
            request_kwargs={"input": request_input},
            messages=None,
            input=request_input,
        )

        assert response is not None
        assert response.model == "simple-model"
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_resolve_encrypted_content_affinity_hook_reads_input_from_request_kwargs():
    """The marker is found in request_kwargs when the caller passes no explicit input."""
    router = litellm.Router(
        model_list=_complexity_router_model_list(),
        routing_strategy="simple-shuffle",
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    try:
        selected_strategy = router._select_pre_routing_strategy(model="auto-router", request_kwargs={})
        response = router._resolve_encrypted_content_affinity_hook(
            selected_strategy=selected_strategy,
            request_kwargs={"input": _reasoning_input_for("dep-simple-1")},
            messages=None,
            input=None,
        )

        assert response is not None
        assert response.model == "simple-model"
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_resolve_encrypted_content_affinity_hook_ignores_deployment_outside_router_tiers():
    """A marker naming a deployment the strategy cannot route to is not honored.

    ``encrypted_content`` is client supplied, so honoring an arbitrary deployment id
    would let a caller pin any deployment on the router and escape the tier config.
    """
    model_list = _complexity_router_model_list()
    model_list.append(
        {
            "model_name": "unrelated-model",
            "litellm_params": {
                "model": "openai/gpt-unrelated",
                "api_key": "key-c",
            },
            "model_info": {"id": "dep-unrelated-1"},
        }
    )
    router = litellm.Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    try:
        selected_strategy = router._select_pre_routing_strategy(model="auto-router", request_kwargs={})
        request_input = _reasoning_input_for("dep-unrelated-1")

        assert (
            router._resolve_encrypted_content_affinity_hook(
                selected_strategy=selected_strategy,
                request_kwargs={"input": request_input},
                messages=None,
                input=request_input,
            )
            is None
        )
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_resolve_encrypted_content_affinity_hook_returns_none_for_unknown_deployment():
    """A marker for a deployment the router no longer configures is not honored."""
    router = litellm.Router(
        model_list=_complexity_router_model_list(),
        routing_strategy="simple-shuffle",
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    try:
        selected_strategy = router._select_pre_routing_strategy(model="auto-router", request_kwargs={})
        request_input = _reasoning_input_for("dep-that-was-removed")

        assert (
            router._resolve_encrypted_content_affinity_hook(
                selected_strategy=selected_strategy,
                request_kwargs={"input": request_input},
                messages=None,
                input=request_input,
            )
            is None
        )
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_resolve_encrypted_content_affinity_hook_returns_none_without_a_marker():
    """A first turn carries no marker, so tier classification is left alone."""
    router = litellm.Router(
        model_list=_complexity_router_model_list(),
        routing_strategy="simple-shuffle",
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    try:
        selected_strategy = router._select_pre_routing_strategy(model="auto-router", request_kwargs={})
        plain_input = [{"role": "user", "content": TIER_SWITCHING_PROMPT}]

        assert (
            router._resolve_encrypted_content_affinity_hook(
                selected_strategy=selected_strategy,
                request_kwargs={"input": plain_input},
                messages=None,
                input=plain_input,
            )
            is None
        )
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_resolve_encrypted_content_affinity_hook_returns_none_when_affinity_not_enabled():
    """With no affinity callback registered the hook does nothing, marker or not."""
    router = litellm.Router(
        model_list=_complexity_router_model_list(),
        routing_strategy="simple-shuffle",
        num_retries=0,
    )

    try:
        from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
            EncryptedContentAffinityCheck,
        )

        assert not any(isinstance(cb, EncryptedContentAffinityCheck) for cb in (router.optional_callbacks or []))

        selected_strategy = router._select_pre_routing_strategy(model="auto-router", request_kwargs={})
        request_input = _reasoning_input_for("dep-simple-1")

        assert (
            router._resolve_encrypted_content_affinity_hook(
                selected_strategy=selected_strategy,
                request_kwargs={"input": request_input},
                messages=None,
                input=request_input,
            )
            is None
        )
    finally:
        router.discard()


def test_encrypted_affinity_allowed_deployment_ids_covers_tiers_and_default_model():
    """Every tier target counts, including a tier that names a pool of models."""
    model_list = _complexity_router_model_list()
    model_list[0]["litellm_params"]["complexity_router_config"]["tiers"] = {
        "SIMPLE": "simple-model",
        "COMPLEX": ["complex-model", "simple-model"],
    }
    model_list.append(
        {
            "model_name": "unrelated-model",
            "litellm_params": {"model": "openai/gpt-unrelated", "api_key": "key-c"},
            "model_info": {"id": "dep-unrelated-1"},
        }
    )
    router = litellm.Router(model_list=model_list, num_retries=0)

    try:
        selected_strategy = router._select_pre_routing_strategy(model="auto-router", request_kwargs={})
        allowed = router._encrypted_affinity_allowed_deployment_ids(selected_strategy.strategy)

        assert allowed == frozenset({"dep-simple-1", "dep-complex-1"})
    finally:
        router.discard()


def test_encrypted_affinity_allowed_deployment_ids_is_empty_for_other_strategies():
    """Only complexity routers reclassify across model groups, so others yield no ids."""
    router = litellm.Router(model_list=_complexity_router_model_list(), num_retries=0)

    try:
        assert router._encrypted_affinity_allowed_deployment_ids(object()) == frozenset()
    finally:
        router.discard()
