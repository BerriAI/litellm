"""
Tests for `routing_groups` — assigns named subsets of `model_name`s to a
per-group routing strategy. Models not claimed by an explicit group fall into
the implicit `"default"` group driven by the router's top-level
`routing_strategy` / `routing_strategy_args`.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import Router
from litellm.types.router import RoutingGroup, RoutingStrategy


def _model_list():
    return [
        {
            "model_name": "filtered-model",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "sk-test-1",
                "api_base": "https://example.invalid",
            },
            "model_info": {"id": "deploy-1"},
        },
        {
            "model_name": "filtered-model",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-test-2",
                "api_base": "https://example.invalid",
            },
            "model_info": {"id": "deploy-2"},
        },
        {
            "model_name": "other-model",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "sk-test-3",
                "api_base": "https://example.invalid",
            },
            "model_info": {"id": "deploy-3"},
        },
    ]


def _build_router(routing_strategy="simple-shuffle", routing_groups=None):
    return Router(
        model_list=_model_list(),
        routing_strategy=routing_strategy,
        routing_groups=routing_groups,
    )


def test_no_groups_uses_top_level_strategy_for_all_models():
    router = _build_router(routing_strategy="latency-based-routing")
    assert router._get_routing_context("filtered-model")[0] == "latency-based-routing"
    assert router._get_routing_context("other-model")[0] == "latency-based-routing"


def test_explicit_group_overrides_top_level():
    router = _build_router(
        routing_strategy="simple-shuffle",
        routing_groups=[
            {
                "group_name": "fast",
                "models": ["filtered-model"],
                "routing_strategy": "latency-based-routing",
            }
        ],
    )
    strategy, selector = router._get_routing_context("filtered-model")
    assert strategy == "latency-based-routing"
    assert selector is not None
    # other-model isn't in any explicit group, so it lands in the default fallback
    strategy_other, _ = router._get_routing_context("other-model")
    assert strategy_other == "simple-shuffle"


def test_default_group_is_simple_shuffle_when_top_level_strategy_unset():
    router = _build_router()
    strategy, selector = router._get_routing_context("other-model")
    assert strategy == "simple-shuffle"
    # simple-shuffle does not use a selector
    assert selector is None


def test_two_groups_same_strategy_have_independent_selectors():
    router = _build_router(
        routing_strategy="simple-shuffle",
        routing_groups=[
            {
                "group_name": "group_a",
                "models": ["filtered-model"],
                "routing_strategy": "latency-based-routing",
                "routing_strategy_args": {"ttl": 60},
            },
            {
                "group_name": "group_b",
                "models": ["other-model"],
                "routing_strategy": "latency-based-routing",
                "routing_strategy_args": {"ttl": 600},
            },
        ],
    )
    a_selector = router._group_selectors["group_a"]["latency-based-routing"]
    b_selector = router._group_selectors["group_b"]["latency-based-routing"]
    assert a_selector is not None and b_selector is not None
    assert id(a_selector) != id(b_selector)


def test_overlapping_models_across_groups_raises():
    with pytest.raises(ValueError, match="appears in"):
        _build_router(
            routing_groups=[
                {
                    "group_name": "g1",
                    "models": ["filtered-model"],
                    "routing_strategy": "latency-based-routing",
                },
                {
                    "group_name": "g2",
                    "models": ["filtered-model"],
                    "routing_strategy": "least-busy",
                },
            ],
        )


def test_reserved_default_group_name_raises():
    with pytest.raises(ValueError, match="reserved"):
        _build_router(
            routing_groups=[
                {
                    "group_name": "default",
                    "models": ["filtered-model"],
                    "routing_strategy": "latency-based-routing",
                }
            ],
        )


def test_invalid_strategy_in_group_raises():
    with pytest.raises(ValueError, match="Invalid routing_strategy"):
        _build_router(
            routing_groups=[
                {
                    "group_name": "g1",
                    "models": ["filtered-model"],
                    "routing_strategy": "not-a-real-strategy",
                }
            ],
        )


def test_unknown_model_in_group_warns_but_does_not_raise(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        router = _build_router(
            routing_groups=[
                {
                    "group_name": "g1",
                    "models": ["model-not-in-list"],
                    "routing_strategy": "latency-based-routing",
                }
            ],
        )
    assert "model-not-in-list" in caplog.text
    assert router._model_to_group.get("model-not-in-list") == "g1"


def test_duplicate_group_name_raises():
    with pytest.raises(ValueError, match="duplicate"):
        _build_router(
            routing_groups=[
                {
                    "group_name": "g1",
                    "models": ["filtered-model"],
                    "routing_strategy": "latency-based-routing",
                },
                {
                    "group_name": "g1",
                    "models": ["other-model"],
                    "routing_strategy": "least-busy",
                },
            ],
        )


def test_routing_group_object_input_accepted():
    router = _build_router(
        routing_groups=[
            RoutingGroup(
                group_name="g1",
                models=["filtered-model"],
                routing_strategy="latency-based-routing",
            )
        ],
    )
    assert router._model_to_group["filtered-model"] == "g1"


@pytest.mark.asyncio
async def test_async_dispatch_uses_group_strategy_for_grouped_model():
    router = _build_router(
        routing_strategy="simple-shuffle",
        routing_groups=[
            {
                "group_name": "fast",
                "models": ["filtered-model"],
                "routing_strategy": "latency-based-routing",
            }
        ],
    )

    group_selector = router._group_selectors["fast"]["latency-based-routing"]

    with (
        patch.object(
            group_selector,
            "async_get_available_deployments",
            wraps=group_selector.async_get_available_deployments,
        ) as latency_spy,
        patch(
            "litellm.router.simple_shuffle", wraps=litellm.router.simple_shuffle
        ) as shuffle_spy,
    ):
        await router.async_get_available_deployment(
            model="filtered-model", request_kwargs={}
        )

        assert (
            latency_spy.called
        ), "group's latency selector should run for grouped model"
        assert not shuffle_spy.called


@pytest.mark.asyncio
async def test_async_dispatch_falls_back_to_default_for_ungrouped_models():
    router = _build_router(
        routing_strategy="simple-shuffle",
        routing_groups=[
            {
                "group_name": "fast",
                "models": ["filtered-model"],
                "routing_strategy": "latency-based-routing",
            }
        ],
    )

    group_selector = router._group_selectors["fast"]["latency-based-routing"]

    with (
        patch.object(
            group_selector,
            "async_get_available_deployments",
            wraps=group_selector.async_get_available_deployments,
        ) as latency_spy,
        patch(
            "litellm.router.simple_shuffle", wraps=litellm.router.simple_shuffle
        ) as shuffle_spy,
    ):
        await router.async_get_available_deployment(
            model="other-model", request_kwargs={}
        )

        assert shuffle_spy.called, "default group's simple-shuffle should run"
        assert not latency_spy.called


def test_update_settings_round_trip_routing_groups():
    router = _build_router()
    assert router._model_to_group == {}

    router.update_settings(
        routing_groups=[
            {
                "group_name": "fast",
                "models": ["filtered-model"],
                "routing_strategy": "latency-based-routing",
            }
        ]
    )
    assert router._model_to_group == {"filtered-model": "fast"}
    assert router._get_routing_context("filtered-model")[0] == "latency-based-routing"

    settings = router.get_settings()
    assert any(g["group_name"] == "fast" for g in settings["routing_groups"])


def test_default_group_accepts_routing_strategy_enum_for_top_level_strategy():
    """
    The Router constructor accepts both string and RoutingStrategy enum forms.
    The default group must resolve a real selector for either input shape, or
    every ungrouped model raises NoDeploymentAvailable.
    """
    router = _build_router(routing_strategy=RoutingStrategy.LATENCY_BASED)
    strategy, selector = router._get_routing_context("other-model")
    assert strategy == "latency-based-routing"
    assert selector is not None
    assert router.lowestlatency_logger is selector


@pytest.mark.asyncio
async def test_async_dispatch_uses_default_selector_when_constructed_with_enum():
    router = _build_router(routing_strategy=RoutingStrategy.LATENCY_BASED)
    default_selector = router.lowestlatency_logger
    assert default_selector is not None

    # filtered-model has 2 deployments, so the single-deployment short-circuit
    # in async_get_available_deployment doesn't kick in and we actually hit
    # the latency selector.
    with (
        patch.object(
            default_selector,
            "async_get_available_deployments",
            wraps=default_selector.async_get_available_deployments,
        ) as latency_spy,
        patch(
            "litellm.router.simple_shuffle", wraps=litellm.router.simple_shuffle
        ) as shuffle_spy,
    ):
        await router.async_get_available_deployment(
            model="filtered-model", request_kwargs={}
        )

        assert (
            latency_spy.called
        ), "default group's latency selector must run when constructed with the enum"
        assert not shuffle_spy.called


def test_update_settings_does_not_leak_strategy_callbacks(monkeypatch):
    """
    Repeated `update_settings` calls must not accumulate stale selectors in
    `litellm.callbacks` / `litellm.input_callback`. Each rebuild owns the
    previous generation and is responsible for unregistering it by identity
    so the global lists don't grow without bound.
    """
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])

    router = _build_router(routing_strategy="latency-based-routing")
    initial_default_selector = router.lowestlatency_logger
    assert initial_default_selector is not None
    assert any(
        c is initial_default_selector for c in litellm.callbacks
    ), "fresh router should register its default selector"

    # Flip strategies back and forth — each transition triggers
    # routing_strategy_init and must replace, not accumulate.
    for _ in range(3):
        router.update_settings(routing_strategy="usage-based-routing")
        router.update_settings(routing_strategy="latency-based-routing")

    assert all(
        c is not initial_default_selector for c in litellm.callbacks
    ), "old default selector instance leaked into litellm.callbacks"

    # The set of router-owned selectors should be bounded — at most one per
    # strategy class. Without the cleanup fix this grows by one per toggle.
    router_owned_classes = (
        "LeastBusyLoggingHandler",
        "LowestTPMLoggingHandler",
        "LowestTPMLoggingHandler_v2",
        "LowestLatencyLoggingHandler",
        "LowestCostLoggingHandler",
    )
    router_owned = [
        c for c in litellm.callbacks if type(c).__name__ in router_owned_classes
    ]
    assert len(router_owned) <= len(
        router_owned_classes
    ), f"router selectors leaked: {[type(c).__name__ for c in router_owned]}"

    # Group selectors: capture the v1 instance, swap, confirm v1 is gone by id.
    router.update_settings(
        routing_groups=[
            {
                "group_name": "fast",
                "models": ["filtered-model"],
                "routing_strategy": "latency-based-routing",
            }
        ]
    )
    group_selector_v1 = router._group_selectors["fast"]["latency-based-routing"]
    assert group_selector_v1 is not None

    router.update_settings(
        routing_groups=[
            {
                "group_name": "fast",
                "models": ["filtered-model"],
                "routing_strategy": "latency-based-routing",
                "routing_strategy_args": {"ttl": 600},
            }
        ]
    )
    assert all(
        c is not group_selector_v1 for c in litellm.callbacks
    ), "old group selector instance leaked into litellm.callbacks"
    group_selector_v2 = router._group_selectors["fast"]["latency-based-routing"]
    assert group_selector_v2 is not group_selector_v1


def test_update_settings_unregisters_group_selectors_when_groups_removed(monkeypatch):
    """
    Setting routing_groups to an empty list (or omitting all groups) must
    unregister every previously-owned group selector.
    """
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])

    router = _build_router(
        routing_groups=[
            {
                "group_name": "fast",
                "models": ["filtered-model"],
                "routing_strategy": "least-busy",
            }
        ]
    )
    group_selector = router._group_selectors["fast"]["least-busy"]
    assert group_selector in litellm.callbacks
    assert group_selector in litellm.input_callback

    router.update_settings(routing_groups=[])

    assert all(c is not group_selector for c in litellm.callbacks)
    assert all(c is not group_selector for c in litellm.input_callback)
    assert router._group_selectors == {}


# ---------------------------------------------------------------------------
# Direct helper coverage
# ---------------------------------------------------------------------------


def test_normalize_strategy_handles_string_enum_and_none():
    assert Router._normalize_strategy(None) is None
    assert Router._normalize_strategy("simple-shuffle") == "simple-shuffle"
    assert (
        Router._normalize_strategy(RoutingStrategy.LATENCY_BASED)
        == RoutingStrategy.LATENCY_BASED.value
    )


def test_validate_routing_strategy_accepts_valid_and_rejects_invalid():
    router = _build_router()
    # Valid: string, enum, None
    router._validate_routing_strategy("simple-shuffle")
    router._validate_routing_strategy(RoutingStrategy.LATENCY_BASED)
    router._validate_routing_strategy(None)
    with pytest.raises(ValueError, match="Invalid routing_strategy"):
        router._validate_routing_strategy("not-a-real-strategy")


def test_build_strategy_selector_returns_none_for_simple_shuffle(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    router = _build_router()
    assert (
        router._build_strategy_selector(
            strategy="simple-shuffle",
            routing_strategy_args={},
            register_callbacks=False,
        )
        is None
    )


def test_build_strategy_selector_constructs_for_known_strategies(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    router = _build_router()
    selector = router._build_strategy_selector(
        strategy="latency-based-routing",
        routing_strategy_args={"ttl": 30},
        register_callbacks=False,
    )
    assert selector is not None


def test_unregister_router_selectors_removes_by_identity(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    router = _build_router()
    selector = router._build_strategy_selector(
        strategy="least-busy",
        routing_strategy_args={},
        register_callbacks=True,
    )
    assert selector in litellm.callbacks
    assert selector in litellm.input_callback
    router._unregister_router_selectors([selector])
    assert all(c is not selector for c in litellm.callbacks)
    assert all(c is not selector for c in litellm.input_callback)


def test_init_routing_groups_with_none_clears_state():
    router = _build_router(
        routing_groups=[
            {
                "group_name": "fast",
                "models": ["filtered-model"],
                "routing_strategy": "least-busy",
            }
        ]
    )
    assert router._routing_groups
    router._init_routing_groups(None)
    assert router._routing_groups == {}
    assert router._model_to_group == {}
    assert router._group_selectors == {}


@pytest.mark.asyncio
async def test_select_deployment_async_returns_none_without_selector():
    router = _build_router()
    result = await router._select_deployment_async(
        strategy="simple-shuffle",
        selector=None,
        model="filtered-model",
        healthy_deployments=[],
        messages=None,
        input=None,
        request_kwargs=None,
    )
    assert result is None


def test_select_deployment_sync_returns_none_without_selector():
    router = _build_router()
    result = router._select_deployment_sync(
        strategy="simple-shuffle",
        selector=None,
        model="filtered-model",
        healthy_deployments=[],
        messages=None,
        input=None,
        request_kwargs=None,
    )
    assert result is None


def test_select_deployment_sync_does_not_dispatch_cost_based_routing():
    """
    `LowestCostLoggingHandler` only implements `async_get_available_deployments`,
    so the sync dispatch must fall through to None for `cost-based-routing`
    instead of raising AttributeError.
    """
    router = _build_router(
        routing_strategy="simple-shuffle",
        routing_groups=[
            {
                "group_name": "cheap",
                "models": ["filtered-model"],
                "routing_strategy": "cost-based-routing",
            }
        ],
    )
    cost_selector = router._group_selectors["cheap"]["cost-based-routing"]
    assert not hasattr(cost_selector, "get_available_deployments"), (
        "test premise broken: LowestCostLoggingHandler unexpectedly grew a "
        "sync method — revisit the sync dispatch arm."
    )

    result = router._select_deployment_sync(
        strategy="cost-based-routing",
        selector=cost_selector,
        model="filtered-model",
        healthy_deployments=[],
        messages=None,
        input=None,
        request_kwargs=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_async_dispatch_falls_back_to_sync_for_usage_based_routing_v1():
    """
    `LowestTPMLoggingHandler` (v1) only implements the sync
    `get_available_deployments`. The async dispatch must call the sync method
    on the v1 selector instead of awaiting a non-existent async one.
    """
    router = _build_router(
        routing_strategy="simple-shuffle",
        routing_groups=[
            {
                "group_name": "v1",
                "models": ["filtered-model"],
                "routing_strategy": "usage-based-routing",
            }
        ],
    )
    v1_selector = router._group_selectors["v1"]["usage-based-routing"]
    assert not hasattr(v1_selector, "async_get_available_deployments"), (
        "test premise broken: LowestTPMLoggingHandler v1 unexpectedly grew an "
        "async method — revisit the async dispatch arm."
    )

    with patch.object(
        v1_selector,
        "get_available_deployments",
        wraps=v1_selector.get_available_deployments,
    ) as v1_spy:
        await router._select_deployment_async(
            strategy="usage-based-routing",
            selector=v1_selector,
            model="filtered-model",
            healthy_deployments=[
                {
                    "model_name": "filtered-model",
                    "litellm_params": {"model": "openai/gpt-4o"},
                    "model_info": {"id": "deploy-1"},
                }
            ],
            messages=[{"role": "user", "content": "hi"}],
            input=None,
            request_kwargs={},
        )

    assert v1_spy.called, "async dispatch must route v1 strategy through sync method"
