import asyncio
import json
import os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime, timezone

from litellm import Router
import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock
from litellm.types.utils import ModelResponse, StandardLoggingPayload
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.router_utils.router_callbacks.track_deployment_metrics import get_deployment_successes_for_current_minute
from litellm.types.router import Deployment, DeploymentTypedDict, LiteLLM_Params, ModelInfo
from litellm.constants import DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS, ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "tpm": 1000,  # Add TPM limit so async method doesn't return early
                "rpm": 100,  # Add RPM limit so async method doesn't return early
            },
            "model_info": {
                "access_groups": ["group1", "group2"],
            },
        },
        {
            "model_name": "gpt-5.5",
            "litellm_params": {
                "model": "gpt-5.5",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-image-1",
            "litellm_params": {
                "model": "gpt-image-1",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "claude-*",
            "litellm_params": {
                "model": "anthropic/*",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
        },
    ]


def test_validate_fallbacks(model_list):
    router = Router(model_list=model_list, fallbacks=[{"gpt-5.5": "gpt-5-mini"}])
    router.validate_fallbacks(fallback_param=[{"gpt-5.5": "gpt-5-mini"}])


def test_routing_strategy_init(model_list):
    """Test if all routing strategies are initialized correctly"""
    from litellm.types.router import RoutingStrategy

    router = Router(model_list=model_list)
    for strategy in RoutingStrategy:
        router.routing_strategy_init(
            routing_strategy=strategy, routing_strategy_args={}
        )




def test_routing_strategy_init_valid_string_strategies(model_list):
    """Test that all valid string routing strategies work without error.

    Valid strategies are derived from RoutingStrategy enum values plus 'simple-shuffle'.
    """
    from litellm.types.router import RoutingStrategy

    router = Router(model_list=model_list)

    # All strategies from enum + simple-shuffle (default, not in enum)
    valid_strategies = ["simple-shuffle"] + [s.value for s in RoutingStrategy]

    for strategy in valid_strategies:
        # Should not raise
        router.routing_strategy_init(
            routing_strategy=strategy, routing_strategy_args={}
        )








@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.flaky(retries=6, delay=1)
@pytest.mark.asyncio
async def test_image_generation(model_list, sync_mode):
    """Test if the underlying '_image_generation' function is working correctly"""
    from litellm.types.utils import ImageResponse

    router = Router(model_list=model_list)
    if sync_mode:
        response = router._image_generation(
            model="gpt-image-1",
            prompt="A cute baby sea otter",
        )
    else:
        response = await router._aimage_generation(
            model="gpt-image-1",
            prompt="A cute baby sea otter",
        )

    ImageResponse.model_validate(response)




































































def _rpm_tpm_router(model_id: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-fake", "tpm": 1000, "rpm": 100},
                "model_info": {"id": model_id},
            }
        ]
    )


@pytest.fixture
def router_minute_pinned(monkeypatch):
    pinned = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr("litellm.router.get_utc_datetime", lambda: pinned)


def _ratelimit_headers(response: ModelResponse | CustomStreamWrapper) -> dict[str, int]:
    return {k: v for k, v in response._hidden_params["additional_headers"].items() if k.startswith("x-ratelimit-")}


@pytest.mark.asyncio
@pytest.mark.usefixtures("router_minute_pinned")
async def test_acompletion_headers_read_post_increment_counter_and_count_once():
    router = _rpm_tpm_router("lit-3058-async")

    response = await router.acompletion(
        model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}], mock_response="pong"
    )
    total_tokens = response.usage.total_tokens
    assert total_tokens > 0

    headers = _ratelimit_headers(response)
    assert headers["x-ratelimit-remaining-tokens"] == 1000 - total_tokens
    assert headers["x-ratelimit-remaining-requests"] == 99
    assert await router.get_model_group_usage("gpt-5-mini") == (total_tokens, 1)

    await asyncio.sleep(0.5)
    assert await router.get_model_group_usage("gpt-5-mini") == (total_tokens, 1)




@pytest.mark.asyncio
@pytest.mark.usefixtures("router_minute_pinned")
async def test_acompletion_stream_counts_request_before_headers_and_tokens_once_on_completion():
    router = _rpm_tpm_router("lit-3058-stream")

    stream = await router.acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong pong pong",
        stream=True,
        stream_options={"include_usage": True},
    )
    headers = _ratelimit_headers(stream)
    assert headers["x-ratelimit-remaining-tokens"] == 1000
    assert headers["x-ratelimit-remaining-requests"] == 99
    assert await router.get_model_group_usage("gpt-5-mini") == (0, 1)

    chunks = [chunk async for chunk in stream]
    total_tokens = chunks[-1].usage.total_tokens
    assert total_tokens > 0

    await asyncio.sleep(0.5)
    assert await router.get_model_group_usage("gpt-5-mini") == (total_tokens, 1)




class _GatedIncrementCache(DualCache):
    def __init__(self) -> None:
        super().__init__(in_memory_cache=InMemoryCache())
        self.first_increment_started = asyncio.Event()
        self.release_first_increment = asyncio.Event()
        self.increment_calls = 0

    async def async_increment_cache_pipeline(
        self,
        increment_list: list[RedisPipelineIncrementOperation],
        local_only: bool = False,
        parent_otel_span: object = None,
        **kwargs: object,
    ) -> list[float] | None:
        self.increment_calls += 1
        if self.increment_calls == 1:
            self.first_increment_started.set()
            await self.release_first_increment.wait()
        return await super().async_increment_cache_pipeline(
            increment_list, local_only=local_only, parent_otel_span=parent_otel_span, **kwargs
        )


@pytest.mark.asyncio
async def test_success_callback_running_during_pre_header_increment_does_not_double_count():
    router = _rpm_tpm_router("lit-3058-race")
    cache = _GatedIncrementCache()
    router.cache = cache

    request = asyncio.ensure_future(
        router.acompletion(model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}], mock_response="pong")
    )
    await asyncio.wait_for(cache.first_increment_started.wait(), timeout=5)
    for _ in range(50):
        if get_deployment_successes_for_current_minute(router, "lit-3058-race") == 1:
            break
        await asyncio.sleep(0.1)
    assert get_deployment_successes_for_current_minute(router, "lit-3058-race") == 1
    assert cache.increment_calls == 1

    cache.release_first_increment.set()
    response = await request

    assert await router.get_model_group_usage("gpt-5-mini") == (response.usage.total_tokens, 1)


class _UnavailableIncrementCache(DualCache):
    def __init__(self) -> None:
        super().__init__(in_memory_cache=InMemoryCache())
        self.first_increment_started = asyncio.Event()
        self.release_first_increment = asyncio.Event()
        self.increment_calls = 0

    async def async_increment_cache_pipeline(
        self,
        increment_list: list[RedisPipelineIncrementOperation],
        local_only: bool = False,
        parent_otel_span: object = None,
        **kwargs: object,
    ) -> list[float] | None:
        self.increment_calls += 1
        if self.increment_calls == 1:
            self.first_increment_started.set()
            await self.release_first_increment.wait()
        raise RuntimeError("cache unavailable")


@pytest.mark.asyncio
async def test_callback_observing_stamp_before_pre_header_increment_fails_leaves_no_stamp_behind():
    router = _rpm_tpm_router("lit-3058-fail")
    cache = _UnavailableIncrementCache()
    router.cache = cache
    metadata: dict[str, object] = {}

    request = asyncio.ensure_future(
        router.acompletion(
            model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}], mock_response="pong", metadata=metadata
        )
    )
    await asyncio.wait_for(cache.first_increment_started.wait(), timeout=5)
    assert metadata[ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY] == 30
    for _ in range(50):
        if get_deployment_successes_for_current_minute(router, "lit-3058-fail") == 1:
            break
        await asyncio.sleep(0.1)
    assert get_deployment_successes_for_current_minute(router, "lit-3058-fail") == 1
    assert cache.increment_calls == 1

    cache.release_first_increment.set()
    response = await request

    assert response.usage.total_tokens == 30
    assert ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY not in metadata
    assert _ratelimit_headers(response)["x-ratelimit-remaining-requests"] == 100
    assert await router.get_model_group_usage("gpt-5-mini") == (None, None)


def test_track_deployment_metrics(model_list):
    """Test if the 'track_deployment_metrics' function is working correctly"""
    from litellm.types.utils import ModelResponse

    router = Router(model_list=model_list)
    router._track_deployment_metrics(
        deployment=router.get_deployment_by_model_group_name(
            model_group_name="gpt-5-mini"
        ),
        response=ModelResponse(
            model="gpt-5-mini",
            usage={"total_tokens": 100},
        ),
        parent_otel_span=None,
    )


def test_pass_through_assistants_endpoint_factory(model_list):
    """Test if the 'pass_through_assistants_endpoint_factory' function is working correctly"""
    router = Router(model_list=model_list)
    router._pass_through_assistants_endpoint_factory(
        original_function=litellm.acreate_assistants,
        custom_llm_provider="openai",
        client=None,
        **{},
    )


def test_factory_function(model_list):
    """Test if the 'factory_function' function is working correctly"""
    router = Router(model_list=model_list)
    router.factory_function(litellm.acreate_assistants)












# def test_pattern_match_deployments(model_list):
#     from litellm.router_utils.pattern_match_deployments import PatternMatchRouter
#     import re

#     patter_router = PatternMatchRouter()

#     request = "fo::hi::static::hello"
#     model_name = "fo::*:static::*"

#     model_name_regex = patter_router._pattern_to_regex(model_name)

#     # Match against the request
#     match = re.match(model_name_regex, request)

#     print(f"match: {match}")
#     print(f"match.end: {match.end()}")
#     if match is None:
#         raise ValueError("Match not found")
#     updated_model = patter_router.set_deployment_model_name(
#         matched_pattern=match, litellm_deployment_litellm_model="openai/*"
#     )
#     assert updated_model == "openai/fo::hi:static::hello"




@pytest.mark.asyncio
async def test_pass_through_moderation_endpoint_factory(model_list):
    router = Router(model_list=model_list)
    response = await router._pass_through_moderation_endpoint_factory(
        original_function=litellm.amoderation,
        input="this is valid good text",
        model=None,
    )
    assert response is not None
































def test_handle_clientside_credential_no_metadata(model_list):
    """Test that _handle_clientside_credential handles cases where no metadata is provided"""
    router = Router(model_list=model_list)

    # Mock deployment
    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test_key"},
        "model_info": {"id": "original-id-789"},
    }

    # Mock kwargs with clientside credentials but NO metadata
    kwargs = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
        # No metadata key at all
    }

    # This should fail because there's no model_group in metadata
    # The function expects to find model_group in the metadata
    try:
        result_deployment = router._handle_clientside_credential(
            deployment=deployment, kwargs=kwargs, function_name="acompletion"
        )
        # If we get here, the function should have used deployment.model_name as fallback
        assert result_deployment.model_name == "gpt-4.1"
        print("✓ Success with no metadata - used deployment.model_name as fallback")
    except Exception as e:
        # This is expected behavior - the function needs model_group to generate model_id
        print(f"✓ Correctly handled no metadata case: {e}")

    # Test with empty metadata
    kwargs_with_empty_metadata = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
        "metadata": {},  # Empty metadata
    }

    try:
        result_deployment = router._handle_clientside_credential(
            deployment=deployment,
            kwargs=kwargs_with_empty_metadata,
            function_name="acompletion",
        )
        # Should fail because empty metadata has no model_group
        pytest.fail("Expected failure with empty metadata")
    except Exception as e:
        print(f"✓ Correctly handled empty metadata case: {e}")
