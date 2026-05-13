import asyncio
import os
import sys
from typing import Any, cast

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.caching.dual_cache import DualCache
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
    PromptCachingDeploymentCheck,
    _openai_prompt_cache_affinity_cache_key,
)
from litellm.types.utils import (
    CallTypes,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)


def _long_user_messages() -> list:
    # Enough tokens for is_prompt_caching_valid_prompt (default >= 1024).
    return [{"role": "user", "content": "test long message here" * 1024}]


def _make_openai_success_payload() -> StandardLoggingPayload:
    # TypedDict 构造函数对「缺字段」很敏感；测试里用 dict + cast，避免维护完整 StandardLoggingMetadata 继承链。
    raw: dict[str, Any] = {
        "id": "test_id",
        "trace_id": "trace-1",
        "litellm_call_id": None,
        "call_type": CallTypes.acompletion.value,
        "stream": False,
        "response_cost": 0.1,
        "cost_breakdown": None,
        "response_cost_failure_debug_info": None,
        "status": "success",
        "status_fields": {},
        "custom_llm_provider": "openai",
        "total_tokens": 30,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "startTime": 1234567890.0,
        "endTime": 1234567891.0,
        "completionStartTime": 1234567890.5,
        "response_time": 1.0,
        "model_map_information": StandardLoggingModelInformation(
            model_map_key="gpt-4o-mini", model_map_value=None
        ),
        "model": "openai/gpt-4o-mini",
        "model_id": "dep-openai-1",
        "model_group": "gpt-4o-mini",
        "api_base": "https://api.openai.com",
        "metadata": {
            "user_api_key_hash": "hash-abc",
            "user_api_key_org_id": None,
            "user_api_key_alias": None,
            "user_api_key_team_id": None,
            "user_api_key_user_id": None,
            "user_api_key_team_alias": None,
            "spend_logs_metadata": None,
            "requester_ip_address": None,
            "requester_metadata": None,
        },
        "cache_hit": False,
        "cache_key": None,
        "saved_cache_cost": 0.0,
        "request_tags": [],
        "end_user": None,
        "requester_ip_address": None,
        "user_agent": None,
        "messages": _long_user_messages(),
        "response": {"choices": [{"message": {"content": "ok"}}]},
        "error_str": None,
        "error_information": None,
        "model_parameters": {},
        "hidden_params": {
            "model_id": "dep-openai-1",
            "cache_key": None,
            "api_base": "https://api.openai.com",
            "response_cost": "0.1",
            "additional_headers": None,
            "litellm_overhead_time_ms": None,
            "batch_models": None,
            "litellm_model_name": None,
            "usage_object": None,
        },
        "guardrail_information": None,
        "standard_built_in_tools_params": None,
    }
    return cast(StandardLoggingPayload, raw)


@pytest.mark.asyncio
async def test_openai_prompt_cache_key_affinity_writes_and_filters_deployment():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)

    payload = _make_openai_success_payload()
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "prompt_cache_key": "tenant-doc-v1",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    await asyncio.sleep(0.05)

    model_group = "gpt-4o-mini"
    redis_key = _openai_prompt_cache_affinity_cache_key(
        router_model=model_group,
        tenant_token="hash-abc",
        prompt_cache_key="tenant-doc-v1",
    )
    cached = await cache.async_get_cache(key=redis_key)
    assert cached is not None
    assert cached["model_id"] == "dep-openai-1"

    healthy = [
        {
            "model_name": model_group,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "model_info": {"id": "dep-openai-1"},
        },
        {
            "model_name": model_group,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "model_info": {"id": "dep-openai-2"},
        },
    ]

    filtered = await check.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy,
        messages=_long_user_messages(),
        request_kwargs={
            "prompt_cache_key": "tenant-doc-v1",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
    )
    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "dep-openai-1"


@pytest.mark.asyncio
async def test_openai_prompt_cache_key_affinity_tenant_isolation():
    """Same prompt_cache_key must not pin another tenant's deployment."""
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)

    payload = _make_openai_success_payload()
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "prompt_cache_key": "shared-doc-key",
            "metadata": {"user_api_key_hash": "hash-tenant-a"},
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    await asyncio.sleep(0.05)

    model_group = "gpt-4o-mini"
    healthy = [
        {
            "model_name": model_group,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "model_info": {"id": "dep-a"},
        },
        {
            "model_name": model_group,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "model_info": {"id": "dep-b"},
        },
    ]

    filtered = await check.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy,
        messages=_long_user_messages(),
        request_kwargs={
            "prompt_cache_key": "shared-doc-key",
            "metadata": {"user_api_key_hash": "hash-tenant-b"},
        },
    )
    assert len(filtered) == 2


@pytest.mark.asyncio
async def test_openai_without_prompt_cache_key_does_not_write_affinity_key():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)

    payload = _make_openai_success_payload()
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    await asyncio.sleep(0.05)

    would_be_key = _openai_prompt_cache_affinity_cache_key(
        router_model="gpt-4o-mini",
        tenant_token="hash-abc",
        prompt_cache_key="if-this-were-sent",
    )
    assert await cache.async_get_cache(key=would_be_key) is None


@pytest.mark.asyncio
async def test_anthropic_success_does_not_write_openai_affinity_key():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)

    payload = _make_openai_success_payload()
    payload["model"] = "anthropic/claude-3-5-sonnet-20240620"
    payload["custom_llm_provider"] = "anthropic"

    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "prompt_cache_key": "should-not-create-openai-key",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    await asyncio.sleep(0.05)

    openai_style_key = _openai_prompt_cache_affinity_cache_key(
        router_model="gpt-4o-mini",
        tenant_token="hash-abc",
        prompt_cache_key="should-not-create-openai-key",
    )
    assert await cache.async_get_cache(key=openai_style_key) is None
