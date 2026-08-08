"""
Tests for `model_info.routing_strategy` — a per-model-group routing strategy
declared on the model definition itself. Applies to every request for that
model_name, sits between the per-request override and legacy `routing_groups`
in precedence, and never raises on bad stored values.
"""

import logging
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import Router
from litellm.router import _warn_model_group_strategy_once
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.types.router import Deployment


@pytest.fixture(autouse=True)
def _clear_warn_once_cache():
    _warn_model_group_strategy_once.cache_clear()
    yield
    _warn_model_group_strategy_once.cache_clear()


def _deployment(model_name, model, deployment_id, model_info=None):
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": model,
            "api_key": "sk-test",
            "api_base": "https://example.invalid",
        },
        "model_info": {"id": deployment_id, **(model_info or {})},
    }


def _build_router(model_list, routing_strategy="simple-shuffle", routing_groups=None):
    return Router(
        model_list=model_list,
        routing_strategy=routing_strategy,
        routing_groups=routing_groups,
    )


def test_model_info_strategy_overrides_top_level():
    router = _build_router(
        [
            _deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "cost-based-routing"}),
            _deployment("quality", "openai/gpt-4o-mini", "d2", {"routing_strategy": "cost-based-routing"}),
            _deployment("plain", "openai/gpt-4o-mini", "d3"),
        ]
    )
    strategy, selector = router._get_routing_context("quality")
    assert strategy == "cost-based-routing"
    assert isinstance(selector, LowestCostLoggingHandler)
    assert router._get_routing_context("plain") == ("simple-shuffle", None)


def test_strategy_on_one_deployment_covers_the_whole_model_group():
    router = _build_router(
        [
            _deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "cost-based-routing"}),
            _deployment("quality", "openai/gpt-4o-mini", "d2"),
        ]
    )
    strategy, _ = router._get_routing_context("quality")
    assert strategy == "cost-based-routing"


def test_request_override_beats_model_info():
    router = _build_router(
        [_deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "cost-based-routing"})]
    )
    strategy, selector = router._get_routing_context("quality", {"routing_strategy": "latency-based-routing"})
    assert strategy == "latency-based-routing"
    assert isinstance(selector, LowestLatencyLoggingHandler)


def test_model_info_beats_legacy_routing_group():
    router = _build_router(
        [
            _deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "cost-based-routing"}),
            _deployment("grouped-only", "openai/gpt-4o-mini", "d2"),
        ],
        routing_groups=[
            {
                "group_name": "legacy",
                "models": ["quality", "grouped-only"],
                "routing_strategy": "latency-based-routing",
            }
        ],
    )
    assert router._get_routing_context("quality")[0] == "cost-based-routing"
    assert router._get_routing_context("grouped-only")[0] == "latency-based-routing"


def test_simple_shuffle_in_model_info_overrides_nondefault_top_level():
    router = _build_router(
        [_deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "simple-shuffle"})],
        routing_strategy="latency-based-routing",
    )
    assert router._get_routing_context("quality") == ("simple-shuffle", None)


def test_conflicting_values_first_deployment_wins_and_warns_once(caplog):
    router = _build_router(
        [
            _deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "cost-based-routing"}),
            _deployment("quality", "openai/gpt-4o-mini", "d2", {"routing_strategy": "latency-based-routing"}),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        assert router._get_routing_context("quality")[0] == "cost-based-routing"
        assert router._get_routing_context("quality")[0] == "cost-based-routing"
    conflict_warnings = [r for r in caplog.records if "conflicting values" in r.getMessage()]
    assert len(conflict_warnings) == 1
    assert "quality" in conflict_warnings[0].getMessage()


def test_empty_string_counts_as_unset_without_warning(caplog):
    router = _build_router(
        [_deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": ""})],
        routing_strategy="latency-based-routing",
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        strategy, _ = router._get_routing_context("quality")
    assert strategy == "latency-based-routing"
    assert not any("model_info.routing_strategy" in r.getMessage() for r in caplog.records)


def test_invalid_value_ignored_with_warning(caplog):
    router = _build_router(
        [_deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "not-a-strategy"})],
        routing_strategy="latency-based-routing",
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        strategy, selector = router._get_routing_context("quality")
    assert strategy == "latency-based-routing"
    assert isinstance(selector, LowestLatencyLoggingHandler)
    assert any("unsupported value" in r.getMessage() for r in caplog.records)


def test_invalid_args_fall_back_without_failing_traffic(caplog):
    router = _build_router(
        [
            _deployment(
                "quality",
                "openai/gpt-4o",
                "d1",
                {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": "bogus"}},
            )
        ],
        routing_strategy="cost-based-routing",
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        strategy, selector = router._get_routing_context("quality")
        router._get_routing_context("quality")
    assert strategy == "cost-based-routing"
    assert selector is router.lowestcost_logger
    warnings = [r for r in caplog.records if "cannot initialize strategy" in r.getMessage()]
    assert len(warnings) == 1


def test_conflict_winner_is_stable_across_model_list_order():
    ordered = _build_router(
        [
            _deployment("quality", "openai/gpt-4o", "a-first", {"routing_strategy": "cost-based-routing"}),
            _deployment("quality", "openai/gpt-4o-mini", "b-second", {"routing_strategy": "latency-based-routing"}),
        ]
    )
    reversed_router = _build_router(
        [
            _deployment("quality", "openai/gpt-4o-mini", "b-second", {"routing_strategy": "latency-based-routing"}),
            _deployment("quality", "openai/gpt-4o", "a-first", {"routing_strategy": "cost-based-routing"}),
        ]
    )
    assert ordered._get_routing_context("quality")[0] == "cost-based-routing"
    assert reversed_router._get_routing_context("quality")[0] == "cost-based-routing"


def _upsert(router, model_name, model, deployment_id, model_info):
    router.upsert_deployment(
        Deployment(
            model_name=model_name,
            litellm_params={"model": model, "api_key": "sk-test", "api_base": "https://example.invalid"},
            model_info={"id": deployment_id, **model_info},
        )
    )


def test_invalid_args_evict_previously_cached_selector(caplog):
    router = _build_router(
        [
            _deployment(
                "quality",
                "openai/gpt-4o",
                "d1",
                {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": 120}},
            )
        ]
    )
    _, old_selector = router._get_routing_context("quality")
    assert any("|" in k for k in router._override_selectors)

    _upsert(
        router,
        "quality",
        "openai/gpt-4o",
        "d1",
        {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": "bogus"}},
    )

    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        strategy, _ = router._get_routing_context("quality")
    assert strategy == "simple-shuffle"
    assert not any("|" in k for k in router._override_selectors)
    assert all(c is not old_selector for c in litellm.callbacks)
    assert any("ttl" in r.getMessage() for r in caplog.records if "cannot initialize strategy" in r.getMessage())


def test_deleting_deployment_evicts_its_selector():
    router = _build_router(
        [
            _deployment(
                "quality",
                "openai/gpt-4o",
                "d1",
                {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": 120}},
            )
        ]
    )
    _, selector = router._get_routing_context("quality")
    assert any("|" in k for k in router._override_selectors)

    router.delete_deployment(id="d1")
    assert not any("|" in k for k in router._override_selectors)
    assert all(c is not selector for c in litellm.callbacks)


def test_strategy_config_is_cached_until_model_list_changes():
    router = _build_router(
        [_deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "cost-based-routing"})]
    )
    assert router._get_routing_context("quality")[0] == "cost-based-routing"

    for idx in router.model_name_to_deployment_indices["quality"]:
        router.model_list[idx]["model_info"]["routing_strategy"] = "least-busy"
    assert router._get_routing_context("quality")[0] == "cost-based-routing"

    _upsert(router, "quality", "openai/gpt-4o", "d1", {"routing_strategy": "latency-based-routing"})
    assert router._get_routing_context("quality")[0] == "latency-based-routing"


def test_config_and_args_warnings_fire_independently(caplog):
    router = _build_router(
        [
            _deployment(
                "quality",
                "openai/gpt-4o",
                "d1",
                {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": "bogus"}},
            ),
            _deployment("quality", "openai/gpt-4o-mini", "d2", {"routing_strategy": "cost-based-routing"}),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        strategy, _ = router._get_routing_context("quality")
        router._get_routing_context("quality")
    assert strategy == "simple-shuffle"
    conflict_warnings = [r for r in caplog.records if "conflicting values" in r.getMessage()]
    args_warnings = [r for r in caplog.records if "cannot initialize strategy" in r.getMessage()]
    assert len(conflict_warnings) == 1
    assert len(args_warnings) == 1


def test_changed_bad_args_warn_again(caplog):
    router = _build_router(
        [
            _deployment(
                "quality",
                "openai/gpt-4o",
                "d1",
                {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": "bogus"}},
            )
        ]
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        router._get_routing_context("quality")
        router._get_routing_context("quality")

        _upsert(
            router,
            "quality",
            "openai/gpt-4o",
            "d1",
            {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": "still-bogus"}},
        )
        router._get_routing_context("quality")
    args_warnings = [r for r in caplog.records if "cannot initialize strategy" in r.getMessage()]
    assert len(args_warnings) == 2


def test_simple_shuffle_with_args_keeps_shuffle_semantics(caplog):
    router = _build_router(
        [
            _deployment(
                "quality",
                "openai/gpt-4o",
                "d1",
                {"routing_strategy": "simple-shuffle", "routing_strategy_args": {"ignored": 1}},
            )
        ],
        routing_strategy="latency-based-routing",
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        assert router._get_routing_context("quality") == ("simple-shuffle", None)
    assert not any("cannot initialize strategy" in r.getMessage() for r in caplog.records)


def test_stale_selector_evicted_when_args_change():
    router = _build_router(
        [
            _deployment(
                "quality",
                "openai/gpt-4o",
                "d1",
                {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": 120}},
            ),
            _deployment(
                "other",
                "openai/gpt-4o-mini",
                "d2",
                {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": 600}},
            ),
        ]
    )
    _, old_selector = router._get_routing_context("quality")
    _, kept_selector = router._get_routing_context("other")
    old_keys = {k for k in router._override_selectors if "|" in k}
    assert len(old_keys) == 2

    _upsert(
        router,
        "quality",
        "openai/gpt-4o",
        "d1",
        {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": 240}},
    )

    _, new_selector = router._get_routing_context("quality")
    assert new_selector is not old_selector
    assert new_selector.routing_args.ttl == 240
    remaining_keys = {k for k in router._override_selectors if "|" in k}
    assert len(remaining_keys) == 2
    assert all(c is not old_selector for c in litellm.callbacks)
    assert router._get_routing_context("other")[1] is kept_selector


def test_selector_shared_across_model_groups_with_identical_config():
    router = _build_router(
        [
            _deployment("a", "openai/gpt-4o", "d1", {"routing_strategy": "latency-based-routing"}),
            _deployment("b", "openai/gpt-4o-mini", "d2", {"routing_strategy": "latency-based-routing"}),
            _deployment(
                "c",
                "openai/gpt-4o-mini",
                "d3",
                {"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": 120}},
            ),
        ]
    )
    _, selector_a = router._get_routing_context("a")
    _, selector_b = router._get_routing_context("b")
    _, selector_c = router._get_routing_context("c")
    assert selector_a is selector_b
    assert selector_c is not selector_a
    assert selector_c.routing_args.ttl == 120


def test_reuses_default_selector_when_config_matches_top_level():
    router = _build_router(
        [_deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "cost-based-routing"})],
        routing_strategy="cost-based-routing",
    )
    _, selector = router._get_routing_context("quality")
    assert selector is router.lowestcost_logger


@pytest.mark.asyncio
async def test_async_dispatch_uses_model_info_strategy():
    router = _build_router(
        [
            _deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "latency-based-routing"}),
            _deployment("quality", "openai/gpt-4o-mini", "d2"),
            _deployment("plain", "openai/gpt-4o-mini", "d3"),
        ]
    )
    _, selector = router._get_routing_context("quality")

    with (
        patch.object(
            selector,
            "async_get_available_deployments",
            wraps=selector.async_get_available_deployments,
        ) as latency_spy,
        patch("litellm.router.simple_shuffle", wraps=litellm.router.simple_shuffle) as shuffle_spy,
    ):
        await router.async_get_available_deployment(model="quality", request_kwargs={})
        assert latency_spy.called
        assert not shuffle_spy.called

        await router.async_get_available_deployment(model="plain", request_kwargs={})
        assert shuffle_spy.called


def test_update_settings_resets_cached_model_group_selectors():
    router = _build_router(
        [_deployment("quality", "openai/gpt-4o", "d1", {"routing_strategy": "latency-based-routing"})]
    )
    _, selector = router._get_routing_context("quality")
    assert selector in router._override_selectors.values()

    router.update_settings(routing_strategy="cost-based-routing")
    assert router._override_selectors == {}
    assert all(c is not selector for c in litellm.callbacks)

    _, rebuilt = router._get_routing_context("quality")
    assert isinstance(rebuilt, LowestLatencyLoggingHandler)
    assert rebuilt is not selector
