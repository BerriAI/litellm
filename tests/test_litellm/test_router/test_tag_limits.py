"""
Tests for tag-based rate limiting (`model_info.token_limits` / `request_limits` / `dollar_limits`).
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.router_strategy.tag_limits import (
    TagLimitCheck,
    get_tag_limits,
    has_tag_limits,
)


def _deployment(model_info: Optional[Dict[str, Any]] = None, model_id: str = "id-1") -> Dict[str, Any]:
    return {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"id": model_id, **(model_info or {})},
    }


def _limits(
    unit: str = "request_limits",
    name: str = "daily",
    tag_id: str = "end_user_id",
    limit: float = 2,
    period_days: float = 1,
) -> Dict[str, Any]:
    return {unit: {"limits": [{"name": name, "tag_id": tag_id, "limit": limit, "period_days": period_days}]}}


def _request_kwargs(tags: List[str]) -> Dict[str, Any]:
    return {"metadata": {"tags": tags}}


def _success_kwargs(
    model_info: Dict[str, Any],
    tags: List[str],
    total_tokens: int = 0,
    response_cost: float = 0.0,
) -> Dict[str, Any]:
    return {
        "litellm_params": {"model_info": {"id": "id-1", **model_info}, "metadata": {"tags": tags}},
        "standard_logging_object": {"total_tokens": total_tokens, "response_cost": response_cost},
    }


async def _filter(check: TagLimitCheck, deployments: List[Dict[str, Any]], tags: List[str]) -> List[dict]:
    return await check.async_filter_deployments(
        model="gpt-4o",
        healthy_deployments=deployments,
        messages=None,
        request_kwargs=_request_kwargs(tags),
    )


def test_get_tag_limits_parses_per_entry_tag_ids():
    model_info = {
        "token_limits": {
            "limits": [
                {"name": "daily", "tag_id": "end_user_id", "limit": 500000, "period_days": 1},
                {"name": "weekly", "tag_id": "team_id", "limit": 2000000, "period_days": 7},
            ]
        },
        "dollar_limits": {"limits": [{"name": "monthly", "tag_id": "team_id", "limit": 50.0, "period_days": 30}]},
    }
    limits = get_tag_limits(model_info)
    assert limits is not None
    assert [(unit, limit.name, limit.tag_id, limit.limit) for unit, limit in limits.limits_by_unit()] == [
        ("tokens", "daily", "end_user_id", 500000),
        ("tokens", "weekly", "team_id", 2000000),
        ("dollars", "monthly", "team_id", 50.0),
    ]


def test_get_tag_limits_ignores_deployments_without_limits_and_invalid_config():
    assert get_tag_limits({"id": "id-1"}) is None
    assert get_tag_limits(None) is None
    assert get_tag_limits({"request_limits": {"limits": [{"name": "daily", "limit": 5}]}}) is None
    assert has_tag_limits([_deployment()]) is False
    assert has_tag_limits([_deployment(_limits())]) is True


@pytest.mark.asyncio
async def test_request_without_matching_tag_is_unaffected():
    check = TagLimitCheck(dual_cache=DualCache())
    deployments = [_deployment(_limits(limit=1))]

    assert await _filter(check, deployments, []) == deployments
    assert await _filter(check, deployments, ["team_id:team-a"]) == deployments
    assert await _filter(check, deployments, ["end_user_id"]) == deployments

    # untagged usage never lands in a counter, so the deployment stays available
    await check.async_log_success_event(_success_kwargs(_limits(limit=1), ["team_id:team-a"]), None, None, None)
    assert await _filter(check, deployments, ["team_id:team-a"]) == deployments


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unit, field, usage_kwargs",
    [
        ("requests", "request_limits", {}),
        ("tokens", "token_limits", {"total_tokens": 1}),
        ("dollars", "dollar_limits", {"response_cost": 1.0}),
    ],
)
async def test_limit_blocks_with_429_once_exceeded(unit: str, field: str, usage_kwargs: Dict[str, Any]):
    model_info = _limits(unit=field, limit=2)
    check = TagLimitCheck(dual_cache=DualCache())
    deployments = [_deployment(model_info)]
    tags = ["end_user_id:user-123"]

    assert await _filter(check, deployments, tags) == deployments

    for _ in range(2):
        await check.async_log_success_event(_success_kwargs(model_info, tags, **usage_kwargs), None, None, None)

    with pytest.raises(litellm.RateLimitError) as exc_info:
        await _filter(check, deployments, tags)

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail == {
        "error": {
            "message": "tag_rate_limit_exceeded",
            "type": unit,
            "tag_id": "end_user_id",
            "tag_value": "user-123",
            "limit_name": "daily",
            "limit": 2.0,
            "period_days": 1.0,
        }
    }
    assert json.loads(exc_info.value.message.split("litellm.RateLimitError: ")[1]) == exc_info.value.detail

    # a different tag value has its own counter
    assert await _filter(check, deployments, ["end_user_id:user-456"]) == deployments


@pytest.mark.asyncio
async def test_usage_is_counted_per_limit_entry_tag_id():
    model_info = {
        "token_limits": {
            "limits": [
                {"name": "daily", "tag_id": "end_user_id", "limit": 10, "period_days": 1},
                {"name": "weekly", "tag_id": "team_id", "limit": 100, "period_days": 7},
            ]
        }
    }
    check = TagLimitCheck(dual_cache=DualCache())
    deployments = [_deployment(model_info)]
    tags = ["end_user_id:user-123", "team_id:team-a"]

    for _ in range(2):
        await check.async_log_success_event(_success_kwargs(model_info, tags, total_tokens=6), None, None, None)

    # end_user_id daily limit (10) is exceeded by 12 tokens; team_id weekly (100) is not
    with pytest.raises(litellm.RateLimitError) as exc_info:
        await _filter(check, deployments, tags)
    assert exc_info.value.detail["error"]["tag_id"] == "end_user_id"

    # another end user on the same team is still under both limits
    assert await _filter(check, deployments, ["end_user_id:user-999", "team_id:team-a"]) == deployments


@pytest.mark.asyncio
async def test_usage_from_a_previous_window_is_not_counted():
    model_info = _limits(limit=1, period_days=1)
    clock = {"now": 1_700_000_000.0}
    check = TagLimitCheck(dual_cache=DualCache(), now=lambda: clock["now"])
    deployments = [_deployment(model_info)]
    tags = ["end_user_id:user-123"]

    await check.async_log_success_event(_success_kwargs(model_info, tags), None, None, None)
    with pytest.raises(litellm.RateLimitError):
        await _filter(check, deployments, tags)

    clock["now"] += 86400
    assert await _filter(check, deployments, tags) == deployments


@pytest.mark.asyncio
async def test_only_deployments_over_their_own_limits_are_filtered_out():
    limited = _deployment(_limits(limit=1), model_id="limited")
    unlimited = _deployment(model_id="unlimited")
    check = TagLimitCheck(dual_cache=DualCache())
    tags = ["end_user_id:user-123"]

    await check.async_log_success_event(_success_kwargs(_limits(limit=1), tags), None, None, None)

    assert await _filter(check, [limited, unlimited], tags) == [unlimited]


@pytest.mark.asyncio
async def test_router_registers_and_enforces_tag_limits():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake", "mock_response": "hi"},
                "model_info": _limits(limit=1),
            }
        ],
        num_retries=0,
    )
    checks = [cb for cb in (router.optional_callbacks or []) if isinstance(cb, TagLimitCheck)]
    assert len(checks) == 1

    request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"tags": ["end_user_id:user-123"]},
    }
    assert (await router.acompletion(**request)).choices[0].message.content == "hi"

    for _ in range(50):
        if await checks[0].dual_cache.async_get_cache(
            key=next(iter(await _counter_keys(checks[0], request["metadata"]["tags"])))
        ):
            break
        await asyncio.sleep(0.05)

    with pytest.raises(litellm.RateLimitError) as exc_info:
        await router.acompletion(**request)
    assert exc_info.value.detail["error"]["message"] == "tag_rate_limit_exceeded"

    # a request without the configured tag id is unaffected
    assert (
        await router.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"tags": ["team_id:team-a"]},
        )
    ).choices[0].message.content == "hi"


async def _counter_keys(check: TagLimitCheck, tags: List[str]) -> List[str]:
    from litellm.router_strategy.tag_limits import _tag_values_from_request, _usage_key

    tag_values = _tag_values_from_request(_request_kwargs(tags))
    limits = get_tag_limits(_limits(limit=1))
    assert limits is not None
    return [
        _usage_key(unit, limit, tag_values[limit.tag_id], check.now())
        for unit, limit in limits.limits_by_unit()
        if limit.tag_id in tag_values
    ]
