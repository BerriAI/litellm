import asyncio
import os
import sys
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.caching.dual_cache import DualCache
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
    PromptCachingDeploymentCheck,
    _extract_prompt_cache_key,
    _openai_prompt_cache_affinity_cache_key,
    _parse_model_id_from_affinity_cache_value,
    _tenant_token_for_openai_pc_affinity,
)
from litellm.router_utils.prompt_caching_cache import PromptCachingCache
from litellm.types.utils import (
    CallTypes,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)


def _long_user_messages() -> list:
    # Enough tokens for is_prompt_caching_valid_prompt (default >= 1024).
    return [{"role": "user", "content": "test long message here" * 1024}]


def _anthropic_cache_control_messages() -> list:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement" * 400,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


def test_helper_extract_and_tenant_edge_cases():
    assert _extract_prompt_cache_key(None) is None
    assert _extract_prompt_cache_key({"prompt_cache_key": "   "}) is None
    assert _tenant_token_for_openai_pc_affinity(None) == ""
    assert (
        _tenant_token_for_openai_pc_affinity({"metadata": {"other": "x"}}) == ""
    )
    assert (
        _tenant_token_for_openai_pc_affinity(
            {"litellm_metadata": {"user_api_key_hash": "from-litellm-md"}}
        )
        == "from-litellm-md"
    )
    assert _parse_model_id_from_affinity_cache_value(99) is None


def _make_openai_success_payload() -> StandardLoggingPayload:
    # TypedDict is strict; build a dict and cast to avoid maintaining the full metadata hierarchy.
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


@pytest.mark.asyncio
async def test_async_log_success_skips_when_standard_logging_object_missing():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    await check.async_log_success_event(
        kwargs={"prompt_cache_key": "x"},
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    assert await cache.async_get_cache(
        key=_openai_prompt_cache_affinity_cache_key(
            router_model="gpt-4o-mini",
            tenant_token="",
            prompt_cache_key="x",
        )
    ) is None


@pytest.mark.asyncio
async def test_async_log_success_skips_embedding_call_type():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    payload = _make_openai_success_payload()
    payload["call_type"] = CallTypes.embedding.value
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "prompt_cache_key": "k",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    await asyncio.sleep(0.05)
    key = _openai_prompt_cache_affinity_cache_key(
        router_model="gpt-4o-mini",
        tenant_token="hash-abc",
        prompt_cache_key="k",
    )
    assert await cache.async_get_cache(key=key) is None


@pytest.mark.asyncio
async def test_async_log_success_openai_custom_llm_provider_from_kwargs():
    """When payload omits custom_llm_provider, fall back to kwargs."""
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    payload = _make_openai_success_payload()
    raw = cast(dict[str, Any], payload)
    raw.pop("custom_llm_provider", None)
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": cast(StandardLoggingPayload, raw),
            "custom_llm_provider": "openai",
            "prompt_cache_key": "doc-key",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    await asyncio.sleep(0.05)
    key = _openai_prompt_cache_affinity_cache_key(
        router_model="gpt-4o-mini",
        tenant_token="hash-abc",
        prompt_cache_key="doc-key",
    )
    cached = await cache.async_get_cache(key=key)
    assert cached is not None
    assert cached["model_id"] == "dep-openai-1"


@pytest.mark.asyncio
async def test_async_log_success_prompt_cache_key_from_optional_params():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    payload = _make_openai_success_payload()
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "optional_params": {"prompt_cache_key": "from-optional"},
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    await asyncio.sleep(0.05)
    key = _openai_prompt_cache_affinity_cache_key(
        router_model="gpt-4o-mini",
        tenant_token="hash-abc",
        prompt_cache_key="from-optional",
    )
    cached = await cache.async_get_cache(key=key)
    assert cached is not None
    assert cached["model_id"] == "dep-openai-1"


@pytest.mark.asyncio
async def test_async_log_success_openai_affinity_set_cache_failure_swallowed():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    payload = _make_openai_success_payload()
    with patch.object(
        cache,
        "async_set_cache",
        new_callable=AsyncMock,
        side_effect=RuntimeError("redis down"),
    ):
        await check.async_log_success_event(
            kwargs={
                "standard_logging_object": payload,
                "prompt_cache_key": "k",
                "metadata": {"user_api_key_hash": "hash-abc"},
            },
            response_obj={},
            start_time=0.0,
            end_time=1.0,
        )


@pytest.mark.asyncio
async def test_filter_deployments_optional_params_prompt_cache_key():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    payload = _make_openai_success_payload()
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "optional_params": {"prompt_cache_key": "opt-key"},
            "metadata": {"user_api_key_hash": "hash-abc"},
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
            "optional_params": {"prompt_cache_key": "opt-key"},
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
    )
    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "dep-openai-1"


@pytest.mark.asyncio
async def test_filter_openai_affinity_inferred_from_deployments_only():
    """Model string without openai/ prefix still applies affinity when deployments are OpenAI."""
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    payload = _make_openai_success_payload()
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "prompt_cache_key": "route-inf",
            "metadata": {"user_api_key_hash": "hash-abc"},
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
            "prompt_cache_key": "route-inf",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
    )
    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "dep-openai-1"


@pytest.mark.asyncio
async def test_filter_deployments_pinned_id_not_in_healthy_returns_all():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    model_group = "gpt-4o-mini"
    key = _openai_prompt_cache_affinity_cache_key(
        router_model=model_group,
        tenant_token="hash-abc",
        prompt_cache_key="orphan-pin",
    )
    await cache.async_set_cache(
        key,
        {"model_id": "dep-not-listed"},
        ttl=300,
    )
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
            "prompt_cache_key": "orphan-pin",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
    )
    assert len(filtered) == 2


@pytest.mark.asyncio
async def test_filter_deployments_cache_dict_without_model_id_returns_all():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    model_group = "gpt-4o-mini"
    key = _openai_prompt_cache_affinity_cache_key(
        router_model=model_group,
        tenant_token="hash-abc",
        prompt_cache_key="bad-dict",
    )
    await cache.async_set_cache(key, {"not_model_id": "x"}, ttl=300)
    healthy = [
        {
            "model_name": model_group,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "model_info": {"id": "dep-openai-1"},
        },
    ]
    filtered = await check.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy,
        messages=_long_user_messages(),
        request_kwargs={
            "prompt_cache_key": "bad-dict",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
    )
    assert len(filtered) == 1


@pytest.mark.asyncio
async def test_filter_deployments_cache_string_model_id():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    model_group = "gpt-4o-mini"
    key = _openai_prompt_cache_affinity_cache_key(
        router_model=model_group,
        tenant_token="hash-abc",
        prompt_cache_key="str-val",
    )
    await cache.async_set_cache(key, "dep-openai-1", ttl=300)
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
            "prompt_cache_key": "str-val",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
    )
    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "dep-openai-1"


@pytest.mark.asyncio
async def test_filter_deployments_short_prompt_returns_all_deployments():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    healthy = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"custom_llm_provider": "openai"},
            "model_info": {"id": "a"},
        },
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"custom_llm_provider": "openai"},
            "model_info": {"id": "b"},
        },
    ]
    filtered = await check.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy,
        messages=[{"role": "user", "content": "short"}],
        request_kwargs={
            "prompt_cache_key": "k",
            "metadata": {"user_api_key_hash": "h"},
        },
    )
    assert len(filtered) == 2


@pytest.mark.asyncio
async def test_async_log_success_completion_call_type_writes_affinity():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    payload = _make_openai_success_payload()
    payload["call_type"] = CallTypes.completion.value
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "prompt_cache_key": "sync-completion",
            "metadata": {"user_api_key_hash": "hash-abc"},
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    await asyncio.sleep(0.05)
    key = _openai_prompt_cache_affinity_cache_key(
        router_model="gpt-4o-mini",
        tenant_token="hash-abc",
        prompt_cache_key="sync-completion",
    )
    assert await cache.async_get_cache(key=key) is not None


@pytest.mark.asyncio
async def test_filter_deployments_anthropic_prefix_cache_pins_deployment():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    messages = _anthropic_cache_control_messages()
    prompt_cache = PromptCachingCache(cache=cache)
    await prompt_cache.async_add_model_id(
        model_id="dep-anthropic-1",
        messages=messages,
        tools=None,
    )
    healthy = [
        {
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "custom_llm_provider": "anthropic",
            },
            "model_info": {"id": "dep-anthropic-1"},
        },
        {
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "custom_llm_provider": "anthropic",
            },
            "model_info": {"id": "dep-anthropic-2"},
        },
    ]
    filtered = await check.async_filter_deployments(
        model="claude-model",
        healthy_deployments=healthy,
        messages=messages,
    )
    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "dep-anthropic-1"


@pytest.mark.asyncio
async def test_filter_deployments_no_prompt_cache_key_returns_all():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    healthy = [
        {
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "model_info": {"id": "dep-1"},
        },
        {
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "model_info": {"id": "dep-2"},
        },
    ]
    filtered = await check.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy,
        messages=_long_user_messages(),
        request_kwargs={"metadata": {"user_api_key_hash": "h"}},
    )
    assert len(filtered) == 2


@pytest.mark.asyncio
async def test_filter_skips_openai_affinity_when_model_and_deployments_unsupported():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    model_group = "router-alias"
    key = _openai_prompt_cache_affinity_cache_key(
        router_model=model_group,
        tenant_token="h",
        prompt_cache_key="k",
    )
    await cache.async_set_cache(key, {"model_id": "dep-1"}, ttl=300)
    healthy = [
        {"model_info": {"id": "dep-1"}},
        {
            "litellm_params": "not-a-dict",
            "model_info": {"id": "dep-2"},
        },
        {
            "litellm_params": {"custom_llm_provider": "ollama"},
            "model_info": {"id": "dep-3"},
        },
    ]
    with patch(
        "litellm.litellm_core_utils.get_supported_openai_params.get_supported_openai_params",
        return_value=["messages"],
    ):
        filtered = await check.async_filter_deployments(
            model=model_group,
            healthy_deployments=healthy,
            messages=_long_user_messages(),
            request_kwargs={
                "prompt_cache_key": "k",
                "metadata": {"user_api_key_hash": "h"},
                "custom_llm_provider": "ollama",
            },
        )
    assert len(filtered) == 3


@pytest.mark.asyncio
async def test_async_log_success_skips_when_messages_not_list():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    payload = _make_openai_success_payload()
    payload["messages"] = "not-a-list"
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "prompt_cache_key": "k",
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    key = _openai_prompt_cache_affinity_cache_key(
        router_model="gpt-4o-mini",
        tenant_token="hash-abc",
        prompt_cache_key="k",
    )
    assert await cache.async_get_cache(key=key) is None


@pytest.mark.asyncio
async def test_async_log_success_skips_when_model_id_none():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    payload = _make_openai_success_payload()
    payload["model_id"] = None
    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": payload,
            "prompt_cache_key": "k",
        },
        response_obj={},
        start_time=0.0,
        end_time=1.0,
    )
    key = _openai_prompt_cache_affinity_cache_key(
        router_model="gpt-4o-mini",
        tenant_token="hash-abc",
        prompt_cache_key="k",
    )
    assert await cache.async_get_cache(key=key) is None


@pytest.mark.asyncio
async def test_model_supports_prompt_cache_key_handles_get_params_exception():
    with patch(
        "litellm.litellm_core_utils.get_supported_openai_params.get_supported_openai_params",
        side_effect=RuntimeError("lookup failed"),
    ):
        filtered = await PromptCachingDeploymentCheck(
            cache=DualCache()
        ).async_filter_deployments(
            model="gpt-4o-mini",
            healthy_deployments=[
                {
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "custom_llm_provider": "openai",
                    },
                    "model_info": {"id": "dep-1"},
                },
            ],
            messages=_long_user_messages(),
            request_kwargs={"prompt_cache_key": "k"},
        )
    assert len(filtered) == 1


@pytest.mark.asyncio
async def test_filter_openai_affinity_when_only_deployments_support_prompt_cache_key():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    model_group = "router-alias-only"
    affinity_key = _openai_prompt_cache_affinity_cache_key(
        router_model=model_group,
        tenant_token="h",
        prompt_cache_key="dep-only",
    )
    await cache.async_set_cache(
        affinity_key,
        {"model_id": "dep-openai-1"},
        ttl=300,
    )

    def _supported_params(model: str, custom_llm_provider=None):
        if model == model_group:
            return ["messages"]
        return ["messages", "prompt_cache_key"]

    healthy = [
        {
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "model_info": {"id": "dep-openai-1"},
        },
        {
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "model_info": {"id": "dep-openai-2"},
        },
    ]
    with patch(
        "litellm.litellm_core_utils.get_supported_openai_params.get_supported_openai_params",
        side_effect=_supported_params,
    ):
        filtered = await check.async_filter_deployments(
            model=model_group,
            healthy_deployments=healthy,
            messages=_long_user_messages(),
            request_kwargs={
                "prompt_cache_key": "dep-only",
                "metadata": {"user_api_key_hash": "h"},
            },
        )
    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "dep-openai-1"
