from __future__ import annotations

from collections.abc import Mapping

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.prompt_cache_prediction import (
    PromptCachePlan,
    make_plan,
    observe,
    predict,
)

_MODEL = "claude-opus-4-8"
_PREFIX = "stable prefix " * 900


@pytest.fixture(autouse=True)
def _local_model_cost_map_autouse(local_model_cost_map):
    yield


def _body(
    *,
    messages: list[dict[str, object]] | None = None,
    tools: list[dict[str, object]] | None = None,
    system: object = None,
    ttl: str = "5m",
    automatic: bool = False,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": _MODEL,
        "max_tokens": 128,
        "messages": messages
        or [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _PREFIX,
                        **({} if automatic else {"cache_control": {"type": "ephemeral", "ttl": ttl}}),
                    }
                ],
            }
        ],
    }
    if automatic:
        body["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    if tools is not None:
        body["tools"] = tools
    if system is not None:
        body["system"] = system
    return body


def _plan(body: Mapping[str, object]) -> PromptCachePlan:
    plan = make_plan(body)
    assert plan is not None
    return plan


def _write_usage(plan: PromptCachePlan) -> dict[str, object]:
    tokens = plan.final_boundary.estimated_tokens
    return {
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": tokens,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": tokens,
            "cache_creation_tokens": tokens,
            "cache_creation_token_details": {
                "ephemeral_5m_input_tokens": tokens if plan.final_ttl_seconds == 300 else 0,
                "ephemeral_1h_input_tokens": tokens if plan.final_ttl_seconds == 3600 else 0,
            },
        },
    }


@pytest.mark.asyncio
async def test_observed_exact_prefix_is_warm_and_append_is_partial():
    cache = DualCache()
    first = _plan(_body())
    await observe(cache, "scope-a", "deployment-a", first, _write_usage(first), 100.0)

    exact = await predict(cache, "scope-a", "deployment-a", first, 101.0)
    assert exact.state == "warm"
    assert exact.cache_read_input_tokens == first.final_boundary.estimated_tokens
    assert exact.cache_creation_input_tokens_5m == 0

    extended = _plan(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": _PREFIX}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "follow up", "cache_control": {"type": "ephemeral"}}],
                },
            ]
        )
    )
    partial = await predict(cache, "scope-a", "deployment-a", extended, 102.0)
    assert partial.state == "partial"
    assert partial.cache_read_input_tokens == first.final_boundary.estimated_tokens
    assert partial.cache_creation_input_tokens_5m > 0
    assert (
        partial.cache_read_input_tokens + partial.cache_creation_input_tokens_5m + partial.uncached_input_tokens
        == extended.total_estimated_tokens
    )


@pytest.mark.asyncio
async def test_deployment_tools_and_system_separate_observations():
    cache = DualCache()
    base = _plan(_body(system="stable system"))
    await observe(cache, "scope-a", "deployment-a", base, _write_usage(base), 100.0)

    assert (await predict(cache, "scope-a", "deployment-a", base, 101.0)).state == "warm"
    assert (await predict(cache, "scope-b", "deployment-a", base, 101.0)).state == "unknown"
    assert (await predict(cache, "scope-a", "deployment-b", base, 101.0)).state == "unknown"
    changed_system = _plan(_body(system="different system"))
    assert (await predict(cache, "scope-a", "deployment-a", changed_system, 101.0)).state == "unknown"

    tools = [
        {
            "name": "weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    changed_tools = _plan(_body(system="stable system", tools=tools))
    assert (await predict(cache, "scope-a", "deployment-a", changed_tools, 101.0)).state == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ttl", "ttl_seconds", "write_field"),
    (("5m", 300, "cache_creation_input_tokens_5m"), ("1h", 3600, "cache_creation_input_tokens_1h")),
)
async def test_ttl_controls_expiry_and_cold_budget(ttl: str, ttl_seconds: int, write_field: str):
    cache = DualCache()
    plan = _plan(_body(ttl=ttl))
    cold = await predict(cache, "scope-a", "deployment", plan, 99.0)
    assert cold.state == "unknown"
    assert getattr(cold, write_field) == plan.final_boundary.estimated_tokens
    assert cold.reason == "no_observation_cold_assumption"

    await observe(cache, "scope-a", "deployment", plan, _write_usage(plan), 100.0)
    live = await predict(cache, "scope-a", "deployment", plan, 100.0 + ttl_seconds - 0.1)
    stale = await predict(cache, "scope-a", "deployment", plan, 100.0 + ttl_seconds)
    assert live.state == "warm"
    assert stale.state == "stale"
    assert stale.expires_at == 100.0 + ttl_seconds
    assert getattr(stale, write_field) == plan.final_boundary.estimated_tokens


@pytest.mark.asyncio
async def test_pure_read_refreshes_only_matching_exact_observation():
    cache = DualCache()
    plan = _plan(_body())
    await observe(cache, "scope-a", "deployment", plan, _write_usage(plan), 100.0)
    read_tokens = plan.final_boundary.estimated_tokens

    await observe(
        cache,
        "scope-a",
        "deployment",
        plan,
        {"cache_read_input_tokens": read_tokens, "cache_creation_input_tokens": 0},
        200.0,
    )
    refreshed = await predict(cache, "scope-a", "deployment", plan, 499.0)
    assert refreshed.state == "warm"
    assert refreshed.as_of == 200.0
    assert refreshed.expires_at == 500.0

    await observe(
        cache,
        "scope-a",
        "deployment",
        plan,
        {"cache_read_input_tokens": read_tokens - 1, "cache_creation_input_tokens": 0},
        300.0,
    )
    unchanged = await predict(cache, "scope-a", "deployment", plan, 499.0)
    assert unchanged.as_of == 200.0


@pytest.mark.asyncio
async def test_zero_failed_and_inconsistent_usage_do_not_record():
    cache = DualCache()
    plan = _plan(_body())
    await observe(cache, "scope-a", "deployment", plan, {}, 100.0)
    await observe(
        cache,
        "scope-a",
        "deployment",
        plan,
        {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        101.0,
    )
    await observe(
        cache,
        "scope-a",
        "deployment",
        plan,
        {
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 10,
            "prompt_tokens_details": {
                "cache_write_tokens": "invalid",
                "cache_creation_token_details": {
                    "ephemeral_5m_input_tokens": 10,
                    "ephemeral_1h_input_tokens": 0,
                },
            },
        },
        102.0,
    )
    prediction = await predict(cache, "scope-a", "deployment", plan, 103.0)
    assert prediction.state == "unknown"
    assert prediction.reason == "no_observation_cold_assumption"


@pytest.mark.parametrize(
    "update",
    (
        {"thinking": {"type": "adaptive"}},
        {"context_management": {"edits": []}},
        {"tools": [{"type": "web_search_20260209", "name": "web_search"}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/image.png"},
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        },
    ),
)
def test_unsupported_transformations_media_and_server_tools_are_unknown(update: dict[str, object]):
    assert make_plan({**_body(), **update}) is None


def test_mixed_ttl_and_more_than_four_markers_are_unknown():
    mixed = _body(system=[{"type": "text", "text": _PREFIX, "cache_control": {"type": "ephemeral", "ttl": "1h"}}])
    assert make_plan(mixed) is None

    messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": [{"type": "text", "text": f"{index} {_PREFIX}", "cache_control": {"type": "ephemeral"}}],
        }
        for index in range(5)
    ]
    assert make_plan(_body(messages=messages)) is None


def test_root_automatic_cache_control_and_budget_bounds():
    plan = _plan(_body(automatic=True))
    assert plan.final_checkpoint_index == plan.boundaries[-1].index
    assert plan.final_ttl_seconds == 300
    cold = plan.cold_budget()
    warm = plan.fully_warm_budget()
    assert cold.cache_creation_input_tokens_5m == plan.final_boundary.estimated_tokens
    assert warm.cache_read_input_tokens == plan.final_boundary.estimated_tokens
    assert warm.uncached_input_tokens == cold.uncached_input_tokens


@pytest.mark.asyncio
async def test_ancestor_more_than_twenty_positions_back_is_not_considered():
    cache = DualCache()
    first = _plan(_body())
    await observe(cache, "scope-a", "deployment", first, _write_usage(first), 100.0)
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": _PREFIX}],
        }
    ]
    messages.extend(
        {
            "role": "assistant" if index % 2 == 0 else "user",
            "content": [{"type": "text", "text": f"block {index}"}],
        }
        for index in range(20)
    )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": "new final", "cache_control": {"type": "ephemeral"}}],
        }
    )
    far = _plan(_body(messages=messages))
    prediction = await predict(cache, "scope-a", "deployment", far, 101.0)
    assert prediction.state == "unknown"
    assert prediction.reason == "no_observation_cold_assumption"
