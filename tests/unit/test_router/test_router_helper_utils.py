import json
import os
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS, ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.router import Deployment, DeploymentTypedDict, LiteLLM_Params, ModelInfo
from litellm.types.utils import (
    ModelResponse,
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)


def create_standard_logging_payload() -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        call_type="completion",
        response_cost=0.1,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=30,
        prompt_tokens=20,
        completion_tokens=10,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        model_map_information=StandardLoggingModelInformation(model_map_key="gpt-5-mini", model_map_value=None),
        model="gpt-5-mini",
        model_id="model-123",
        model_group="openai-gpt",
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_org_id=None,
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_team_alias="test_team_alias",
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
        ),
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        model_parameters={"stream": True},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.1",
            additional_headers=None,
        ),
    )


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "tpm": 1000,
                "rpm": 100,
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


def test_routing_strategy_init_invalid_strategy(model_list):
    """Test that invalid routing_strategy raises ValueError with helpful message.

    See: https://github.com/BerriAI/litellm/issues/11330
    Invalid strategies like 'simple' (without '-shuffle') should fail fast
    with a clear error, not silently cause 'No deployments available' errors.
    """
    router = Router(model_list=model_list)

    with pytest.raises(ValueError, match="usage-based-routing', 'provider-budget-routing'\\]\\. Check") as exc_info:
        router.routing_strategy_init(routing_strategy="simple", routing_strategy_args={})

    error_msg = str(exc_info.value)
    assert "Invalid routing_strategy" in error_msg
    assert "simple" in error_msg
    assert "simple-shuffle" in error_msg

    assert "config.yaml" in error_msg
    assert "router_settings.routing_strategy" in error_msg
    assert "Router SDK" in error_msg

    with pytest.raises(ValueError, match="usage-based-routing', 'provider-budget-routing'\\]\\. Check") as exc_info:
        router.routing_strategy_init(routing_strategy="not-a-real-strategy", routing_strategy_args={})
    assert "Invalid routing_strategy" in str(exc_info.value)


@pytest.mark.usefixtures("fake_provider_credentials")
def test_print_deployment(model_list):
    """Test if the api key is masked correctly"""

    router = Router(model_list=model_list)
    deployment = {
        "model_name": "gpt-5-mini",
        "litellm_params": {
            "model": "gpt-5-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    }
    printed_deployment = router.print_deployment(deployment)
    assert 10 * "*" in printed_deployment["litellm_params"]["api_key"]


def test_print_deployment_with_redact_enabled(model_list):
    """Test if sensitive credentials are masked when redact_user_api_key_info is enabled"""
    import litellm

    router = Router(model_list=model_list)
    deployment = {
        "model_name": "bedrock-claude",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-v2",
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "aws_region_name": "us-west-2",
        },
    }

    original_setting = litellm.redact_user_api_key_info
    try:
        litellm.redact_user_api_key_info = True
        printed_deployment = router.print_deployment(deployment)

        assert "*" in printed_deployment["litellm_params"]["aws_access_key_id"]
        assert "*" in printed_deployment["litellm_params"]["aws_secret_access_key"]
        assert "us-west-2" == printed_deployment["litellm_params"]["aws_region_name"]
    finally:
        litellm.redact_user_api_key_info = original_setting


def test_completion(model_list):
    """Test if the completion function is working correctly"""
    router = Router(model_list=model_list)
    response = router._completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    assert response["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_acompletion_util(model_list):
    """Test if the underlying '_acompletion' function is working correctly"""
    router = Router(model_list=model_list)
    response = await router._acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    assert response["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_abatch_completion_one_model_multiple_requests_util(model_list):
    """Test if the 'abatch_completion_one_model_multiple_requests' function is working correctly"""
    router = Router(model_list=model_list)
    response = await router.abatch_completion_one_model_multiple_requests(
        model="gpt-5-mini",
        messages=[
            [{"role": "user", "content": "Hello, how are you?"}],
            [{"role": "user", "content": "Hello, how are you?"}],
        ],
        mock_response="I'm fine, thank you!",
    )
    print(response)
    assert response[0]["choices"][0]["message"]["content"] == "I'm fine, thank you!"
    assert response[1]["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_schedule_acompletion(model_list):
    """Test if the 'schedule_acompletion' function is working correctly"""
    router = Router(model_list=model_list)
    response = await router.schedule_acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
        priority=1,
    )
    assert response["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_schedule_atext_completion(model_list):
    """Test if the 'schedule_atext_completion' function is working correctly"""
    from litellm.types.utils import TextCompletionResponse

    router = Router(model_list=model_list)
    with patch.object(router, "_atext_completion", AsyncMock()) as mock_atext_completion:
        mock_atext_completion.return_value = TextCompletionResponse()
        response = await router.atext_completion(
            model="gpt-5-mini",
            prompt="Hello, how are you?",
            priority=1,
        )
        mock_atext_completion.assert_awaited_once()
        assert "priority" not in mock_atext_completion.call_args.kwargs


@pytest.mark.asyncio
async def test_router_schedule_factory(model_list):
    """Test if the 'schedule_atext_completion' function is working correctly"""
    from litellm.types.utils import TextCompletionResponse

    router = Router(model_list=model_list)
    with patch.object(router, "_atext_completion", AsyncMock()) as mock_atext_completion:
        mock_atext_completion.return_value = TextCompletionResponse()
        response = await router._schedule_factory(
            model="gpt-5-mini",
            args=(
                "gpt-5-mini",
                "Hello, how are you?",
            ),
            priority=1,
            kwargs={},
            original_function=router.atext_completion,
        )
        mock_atext_completion.assert_awaited_once()
        assert "priority" not in mock_atext_completion.call_args.kwargs


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_router_function_with_fallbacks(model_list, sync_mode):
    """Test if the router 'async_function_with_fallbacks' + 'function_with_fallbacks' are working correctly"""
    router = Router(model_list=model_list)
    data = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "mock_response": "I'm fine, thank you!",
        "num_retries": 0,
    }
    if sync_mode:
        response = router.function_with_fallbacks(
            original_function=router._completion,
            **data,
        )
    else:
        response = await router.async_function_with_fallbacks(
            original_function=router._acompletion,
            **data,
        )
    assert response.choices[0].message.content == "I'm fine, thank you!"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_router_function_with_retries(model_list, sync_mode):
    """Test if the router 'async_function_with_retries' + 'function_with_retries' are working correctly"""
    router = Router(model_list=model_list)
    data = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "mock_response": "I'm fine, thank you!",
        "num_retries": 0,
    }
    response = await router.async_function_with_retries(
        original_function=router._acompletion,
        **data,
    )

    assert response.choices[0].message.content == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_make_call(model_list):
    """Test if the router 'make_call' function is working correctly"""

    router = Router(model_list=model_list)
    response = await router.make_call(
        original_function=router._acompletion,
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    assert response.choices[0].message.content == "I'm fine, thank you!"

    response = await router.make_call(
        original_function=router._atext_completion,
        model="gpt-5-mini",
        prompt="Hello, how are you?",
        mock_response="I'm fine, thank you!",
    )
    assert response.choices[0].text == "I'm fine, thank you!"

    response = await router.make_call(
        original_function=router._aembedding,
        model="gpt-5-mini",
        input="Hello, how are you?",
        mock_response=[0.1, 0.2, 0.3],
    )
    assert response.data[0].embedding == [0.1, 0.2, 0.3]

    response = await router.make_call(
        original_function=router._aimage_generation,
        model="gpt-image-1",
        prompt="A cute baby sea otter",
        mock_response="https://example.com/image.png",
    )
    assert response.data[0].url == "https://example.com/image.png"


def test_update_kwargs_with_deployment(model_list):
    """Test if the '_update_kwargs_with_deployment' function is working correctly"""
    router = Router(model_list=model_list)
    kwargs: dict = {"metadata": {}}
    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")
    router._update_kwargs_with_deployment(
        deployment=deployment,
        kwargs=kwargs,
    )
    set_fields = ["deployment", "api_base", "model_info"]
    assert all(field in kwargs["metadata"] for field in set_fields)


def test_update_kwargs_with_default_litellm_params(model_list):
    """Test if the '_update_kwargs_with_default_litellm_params' function is working correctly"""
    router = Router(
        model_list=model_list,
        default_litellm_params={"api_key": "test", "metadata": {"key": "value"}},
    )
    kwargs: dict = {"metadata": {"key2": "value2"}}
    router._update_kwargs_with_default_litellm_params(kwargs=kwargs)
    assert kwargs["api_key"] == "test"
    assert kwargs["metadata"]["key"] == "value"
    assert kwargs["metadata"]["key2"] == "value2"


def test_get_timeout(model_list):
    """Test if the '_get_timeout' function is working correctly"""
    router = Router(model_list=model_list)
    timeout = router._get_timeout(kwargs={}, data={"timeout": 100})
    assert timeout == 100


@pytest.mark.parametrize(
    "fallback_kwarg, expected_error",
    [
        ("mock_testing_fallbacks", litellm.InternalServerError),
        ("mock_testing_context_fallbacks", litellm.ContextWindowExceededError),
        ("mock_testing_content_policy_fallbacks", litellm.ContentPolicyViolationError),
    ],
)
def test_handle_mock_testing_fallbacks(model_list, fallback_kwarg, expected_error):
    """Test if the '_handle_mock_testing_fallbacks' function is working correctly"""
    router = Router(model_list=model_list)
    data = {
        fallback_kwarg: True,
    }

    with pytest.raises(expected_error):
        router._handle_mock_testing_fallbacks(
            kwargs=data,
        )


def test_handle_mock_testing_rate_limit_error(model_list):
    """Test if the '_handle_mock_testing_rate_limit_error' function is working correctly"""
    router = Router(model_list=model_list)
    data = {
        "mock_testing_rate_limit_error": True,
    }

    with pytest.raises(litellm.RateLimitError):
        router._handle_mock_testing_rate_limit_error(
            kwargs=data,
        )


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_deployment_callback_on_success(sync_mode):
    """Test if the '_deployment_callback_on_success' function is working correctly"""
    import time

    model_list = [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "rpm": 100,
            },
            "model_info": {"id": "100"},
        }
    ]
    router = Router(model_list=model_list)
    # Get the actual deployment ID that was generated
    gpt_deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-5-mini"
    )
    deployment_id = gpt_deployment["model_info"]["id"]

    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["total_tokens"] = 100
    standard_logging_payload["model_id"] = "100"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-5-mini",
            },
            "model_info": {"id": deployment_id},
        },
        "standard_logging_object": standard_logging_payload,
    }
    response = litellm.ModelResponse(
        model="gpt-5-mini",
        usage={"total_tokens": 100},
    )
    if sync_mode:
        tpm_key = router.sync_deployment_callback_on_success(
            kwargs=kwargs,
            completion_response=response,
            start_time=time.time(),
            end_time=time.time(),
        )
    else:
        tpm_key = await router.deployment_callback_on_success(
            kwargs=kwargs,
            completion_response=response,
            start_time=time.time(),
            end_time=time.time(),
        )
    assert tpm_key is not None


@pytest.mark.asyncio
async def test_deployment_callback_on_success_tracks_tpm_for_io_deployment():
    """
    An IO-limited deployment (itpm/otpm, no tpm/rpm) must still record TPM usage
    in the router's routing counter so TPM-aware routing strategies see its real
    load in mixed model groups; its itpm/otpm enforcement runs separately.
    """
    import time

    model_list = [
        {
            "model_name": "opus",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-fake",
                "itpm": 1000,
            },
            "model_info": {"id": "io-100"},
        }
    ]
    router = Router(model_list=model_list)

    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["total_tokens"] = 100
    standard_logging_payload["model_id"] = "io-100"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "deployment": "openai/gpt-4o-mini",
                "model_group": "opus",
            },
            "model_info": {"id": "io-100"},
        },
        "standard_logging_object": standard_logging_payload,
    }
    response = litellm.ModelResponse(model="openai/gpt-4o-mini", usage={"total_tokens": 100})

    tpm_key = await router.deployment_callback_on_success(
        kwargs=kwargs,
        completion_response=response,
        start_time=time.time(),
        end_time=time.time(),
    )

    # The IO deployment is no longer skipped: its TPM routing counter is tracked.
    assert tpm_key is not None
    assert await router.cache.async_get_cache(key=tpm_key) == 100


@pytest.mark.asyncio
async def test_deployment_callback_on_failure(model_list):
    """Test if the '_deployment_callback_on_failure' function is working correctly"""
    import time

    router = Router(model_list=model_list)
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-5-mini",
            },
            "model_info": {"id": 100},
        },
    }
    result = router.deployment_callback_on_failure(
        kwargs=kwargs,
        completion_response=None,
        start_time=time.time(),
        end_time=time.time(),
    )
    assert isinstance(result, bool)
    assert result is False

    model_response = router.completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    result = await router.async_deployment_callback_on_failure(
        kwargs=kwargs,
        completion_response=model_response,
        start_time=time.time(),
        end_time=time.time(),
    )


def test_deployment_callback_respects_cooldown_time(model_list):
    """Ensure per-model cooldown_time is honored even when exception headers are present."""
    import httpx
    import time
    from unittest.mock import patch

    router = Router(model_list=model_list)

    class FakeException(Exception):
        def __init__(self):
            self.status_code = 429
            self.headers = httpx.Headers({"x-test": "1"})

    kwargs = {
        "exception": FakeException(),
        "litellm_params": {
            "metadata": {"model_group": "gpt-5-mini"},
            "model_info": {"id": 100},
            "cooldown_time": 0,
        },
    }

    with patch("litellm.router.set_cooldown_deployments") as mock_set:
        router.deployment_callback_on_failure(
            kwargs=kwargs,
            completion_response=None,
            start_time=time.time(),
            end_time=time.time(),
        )

        mock_set.assert_called_once()
        assert mock_set.call_args.kwargs["time_to_cooldown"] == 0


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_log_retry(model_list: list[DeploymentTypedDict], metadata_key: str) -> None:
    """log_retry appends one flat record per failed attempt, copies neither the request kwargs nor the
    request metadata into it, counts every failed attempt of the request independently of the
    per-hop attempted_retries, and never trusts a negative count planted before the first failure"""
    router = Router(model_list=model_list)
    rate_limit_error = litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-3.5-turbo")
    new_kwargs = router.log_retry(
        kwargs={
            "model": "gpt-3.5-turbo",
            "api_key": "sk-must-not-be-recorded",
            "messages": [{"role": "user", "content": "hi"}],
            metadata_key: {"model_info": {"id": "deployment-1"}, "attempted_retries": 2, "user_api_key": "sk-proxy"},
        },
        e=rate_limit_error,
    )
    assert json.loads(json.dumps(new_kwargs[metadata_key]["previous_models"])) == [
        {
            "model_group": "gpt-3.5-turbo",
            "deployment_id": "deployment-1",
            "exception_type": "RateLimitError",
            "exception_string": "litellm.RateLimitError: slow down",
            "attempted_retries": 2,
        }
    ]
    assert new_kwargs[metadata_key]["request_retry_count"] == 1
    assert router.log_retry(kwargs=new_kwargs, e=rate_limit_error)[metadata_key]["request_retry_count"] == 2
    planted_kwargs = {"model": "gpt-3.5-turbo", metadata_key: {"request_retry_count": -100}}
    assert router.log_retry(kwargs=planted_kwargs, e=rate_limit_error)[metadata_key]["request_retry_count"] == 1


@pytest.mark.usefixtures("router_minute_pinned")
def test_update_usage(model_list):
    """Test if the '_update_usage' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")
    deployment_id = deployment["model_info"]["id"]
    request_count = router._update_usage(deployment_id=deployment_id, parent_otel_span=None)
    assert request_count == 1

    request_count = router._update_usage(deployment_id=deployment_id, parent_otel_span=None)

    assert request_count == 2


@pytest.mark.parametrize("finish_reason, expected_fallback", [("content_filter", True), ("stop", False)])
@pytest.mark.parametrize("fallback_type", ["model-specific", "default"])
def test_should_raise_content_policy_error(model_list, finish_reason, expected_fallback, fallback_type):
    """Test if the '_should_raise_content_policy_error' function is working correctly"""
    router = Router(
        model_list=model_list,
        default_fallbacks=["gpt-5.5"] if fallback_type == "default" else None,
    )

    assert (
        router._should_raise_content_policy_error(
            model="gpt-5-mini",
            response=litellm.ModelResponse(
                model="gpt-5-mini",
                choices=[
                    {
                        "finish_reason": finish_reason,
                        "message": {"content": "I'm fine, thank you!"},
                    }
                ],
                usage={"total_tokens": 100},
            ),
            kwargs={
                "content_policy_fallbacks": ([{"gpt-5-mini": "gpt-5.5"}] if fallback_type == "model-specific" else None)
            },
        )
        is expected_fallback
    )


def test_get_healthy_deployments(model_list):
    """Test if the '_get_healthy_deployments' function is working correctly"""
    router = Router(model_list=model_list)
    deployments = router._get_healthy_deployments(model="gpt-5-mini", parent_otel_span=None)
    assert len(deployments) > 0


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_routing_strategy_pre_call_checks(model_list, sync_mode):
    """Test if the '_routing_strategy_pre_call_checks' function is working correctly"""
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.litellm_logging import Logging

    callback = CustomLogger()
    litellm.callbacks = [callback]

    router = Router(model_list=model_list)

    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-5-mini"
    )

    litellm_logging_obj = Logging(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )
    if sync_mode:
        router.routing_strategy_pre_call_checks(deployment)
    else:
        ## NO EXCEPTION
        await router.async_routing_strategy_pre_call_checks(
            deployment, litellm_logging_obj
        )

        ## WITH EXCEPTION - rate limit error
        with patch.object(
            callback,
            "async_pre_call_check",
            AsyncMock(
                side_effect=litellm.RateLimitError(
                    message="Rate limit error",
                    llm_provider="openai",
                    model="gpt-5-mini",
                )
            ),
        ):
            with pytest.raises(litellm.RateLimitError):
                await router.async_routing_strategy_pre_call_checks(
                    deployment, litellm_logging_obj
                )

        ## WITH EXCEPTION - generic error
        with patch.object(
            callback, "async_pre_call_check", AsyncMock(side_effect=Exception("Error"))
        ):
            with pytest.raises(Exception, match="Error"):
                await router.async_routing_strategy_pre_call_checks(
                    deployment, litellm_logging_obj
                )


@pytest.mark.parametrize(
    "set_supported_environments, supported_environments, is_supported",
    [(True, ["staging"], True), (False, None, True), (True, ["development"], False)],
)
def test_create_deployment(
    model_list, set_supported_environments, supported_environments, is_supported
):
    """Test if the '_create_deployment' function is working correctly"""
    router = Router(model_list=model_list)

    if set_supported_environments:
        os.environ["LITELLM_ENVIRONMENT"] = "staging"
    deployment = router._create_deployment(
        deployment_info={},
        _model_name="gpt-5-mini",
        _litellm_params={
            "model": "gpt-5-mini",
            "api_key": "test",
            "custom_llm_provider": "openai",
        },
        _model_info={
            "id": 100,
            "supported_environments": supported_environments,
        },
    )
    if is_supported:
        assert deployment is not None
    else:
        assert deployment is None


@pytest.mark.parametrize(
    "set_supported_environments, supported_environments, is_supported",
    [(True, ["staging"], True), (False, None, True), (True, ["development"], False)],
)
def test_deployment_is_active_for_environment(
    model_list, set_supported_environments, supported_environments, is_supported
):
    """Test if the '_deployment_is_active_for_environment' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-5-mini"
    )
    if set_supported_environments:
        os.environ["LITELLM_ENVIRONMENT"] = "staging"
    deployment["model_info"]["supported_environments"] = supported_environments
    if is_supported:
        assert (
            router.deployment_is_active_for_environment(deployment=deployment) is True
        )
    else:
        assert (
            router.deployment_is_active_for_environment(deployment=deployment) is False
        )


def test_set_model_list(model_list):
    """Test if the '_set_model_list' function is working correctly"""
    router = Router(model_list=model_list)
    router.set_model_list(model_list=model_list)
    assert len(router.model_list) == len(model_list)


def test_add_deployment(model_list):
    """Test if the '_add_deployment' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")
    deployment["model_info"]["id"] = "100"

    router.add_deployment(deployment=deployment)

    router._add_deployment(deployment=deployment)
    assert len(router.model_list) == len(model_list) + 1


def test_upsert_deployment(model_list):
    """Test if the 'upsert_deployment' function is working correctly"""
    router = Router(model_list=model_list)
    print("model list", len(router.model_list))
    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")
    deployment.litellm_params.model = "gpt-5.5"
    router.upsert_deployment(deployment=deployment)
    assert len(router.model_list) == len(model_list)


def test_delete_deployment(model_list):
    """Test if the 'delete_deployment' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")
    router.delete_deployment(id=deployment["model_info"]["id"])
    assert len(router.model_list) == len(model_list) - 1


def test_get_model_info(model_list):
    """Test if the 'get_model_info' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")
    model_info = router.get_model_info(id=deployment["model_info"]["id"])
    assert model_info is not None


def test_get_model_group(model_list):
    """Test if the 'get_model_group' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")
    model_group = router.get_model_group(id=deployment["model_info"]["id"])
    assert model_group is not None
    assert model_group[0]["model_name"] == "gpt-5-mini"


@pytest.mark.parametrize("user_facing_model_group_name", ["gpt-5-mini", "gpt-5.5"])
def test_set_model_group_info(model_list, user_facing_model_group_name):
    """Test if the 'set_model_group_info' function is working correctly"""
    router = Router(model_list=model_list)
    resp = router._set_model_group_info(
        model_group="gpt-5-mini",
        user_facing_model_group_name=user_facing_model_group_name,
    )
    assert resp is not None
    assert resp.model_group == user_facing_model_group_name


@pytest.mark.asyncio
async def test_set_response_headers(model_list):
    """Test if the 'set_response_headers' function is working correctly"""
    router = Router(model_list=model_list)
    resp = await router.set_response_headers(response=None, model_group=None)
    assert resp is None


@pytest.mark.asyncio
async def test_set_response_headers_passes_through_post_increment_counters(model_list):
    from pydantic import BaseModel

    class _Usage(BaseModel):
        total_tokens: int = 42

    class _Resp(BaseModel):
        usage: _Usage = _Usage()
        _hidden_params: dict = {}

    router = Router(model_list=model_list)
    router.get_remaining_model_group_usage = AsyncMock(
        return_value={
            "x-ratelimit-remaining-tokens": 958,
            "x-ratelimit-limit-tokens": 1000,
            "x-ratelimit-remaining-requests": 99,
            "x-ratelimit-limit-requests": 100,
            "x-ratelimit-remaining-input-tokens": 1000,
            "x-ratelimit-remaining-output-tokens": 500,
        }
    )

    resp = _Resp()
    resp._hidden_params = {}
    await router.set_response_headers(response=resp, model_group="gpt-3.5-turbo")

    headers = resp._hidden_params["additional_headers"]
    assert headers["x-ratelimit-remaining-tokens"] == 958
    assert headers["x-ratelimit-remaining-requests"] == 99
    assert headers["x-ratelimit-limit-tokens"] == 1000
    assert headers["x-ratelimit-limit-requests"] == 100
    assert headers["x-ratelimit-remaining-input-tokens"] == 1000
    assert headers["x-ratelimit-remaining-output-tokens"] == 500


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
def router_minute_pinned(monkeypatch: pytest.MonkeyPatch) -> datetime:
    pinned: Final = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr("litellm.router.get_utc_datetime", lambda: pinned)
    return pinned


def _ratelimit_headers(response: ModelResponse | CustomStreamWrapper) -> dict[str, int]:
    return {k: v for k, v in response._hidden_params["additional_headers"].items() if k.startswith("x-ratelimit-")}


@pytest.mark.asyncio
@pytest.mark.usefixtures("router_minute_pinned")
async def test_acompletion_wildcard_route_headers_and_counter_use_resolved_deployment_name():
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "sk-fake", "tpm": 1000, "rpm": 100},
                "model_info": {"id": "lit-3058-wildcard"},
            }
        ]
    )

    response = await router.acompletion(
        model="openai/gpt-5-mini", messages=[{"role": "user", "content": "hi"}], mock_response="pong"
    )
    total_tokens = response.usage.total_tokens

    headers = _ratelimit_headers(response)
    assert headers["x-ratelimit-remaining-tokens"] == 1000 - total_tokens
    assert headers["x-ratelimit-remaining-requests"] == 99
    assert await router.get_model_group_usage("openai/gpt-5-mini") == (total_tokens, 1)


@pytest.mark.asyncio
async def test_deployment_callback_on_success_adds_only_uncounted_tokens():
    import time

    router = _rpm_tpm_router("lit-3058-callback")
    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["total_tokens"] = 100
    kwargs = {
        "litellm_params": {
            "metadata": {
                "deployment": "gpt-5-mini",
                "model_group": "gpt-5-mini",
                ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY: 60,
            },
            "model_info": {"id": "lit-3058-callback"},
        },
        "standard_logging_object": standard_logging_payload,
    }

    tpm_key = await router.deployment_callback_on_success(
        kwargs=kwargs,
        completion_response=litellm.ModelResponse(model="gpt-5-mini", usage={"total_tokens": 100}),
        start_time=time.time(),
        end_time=time.time(),
    )

    assert tpm_key is not None
    assert await router.get_model_group_usage("gpt-5-mini") == (40, 0)


@pytest.mark.asyncio
@pytest.mark.usefixtures("router_minute_pinned")
async def test_increment_deployment_usage_for_response_skips_session_wrappers():
    router = _rpm_tpm_router("lit-3058-ws")
    request_kwargs = {
        "model": "gpt-5-mini",
        "litellm_metadata": {"model_group": "gpt-5-mini", "model_info": {"id": "lit-3058-ws"}},
    }

    await router.increment_deployment_usage_for_response(response=None, request_kwargs=request_kwargs)

    assert await router.get_model_group_usage("gpt-5-mini") == (None, None)
    assert ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY not in request_kwargs["litellm_metadata"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("router_minute_pinned")
async def test_increment_deployment_usage_writes_only_positive_deltas_for_limited_deployments():
    router = _rpm_tpm_router("lit-3058-delta")
    unlimited = Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-fake"},
                "model_info": {"id": "lit-3058-unlimited"},
            }
        ]
    )

    tpm_key = await router._increment_deployment_usage(
        deployment_id="lit-3058-delta",
        deployment_name="gpt-5-mini",
        model_group="gpt-5-mini",
        total_tokens=25,
        rpm_increment=1,
        parent_otel_span=None,
    )
    assert tpm_key is not None
    assert await router.get_model_group_usage("gpt-5-mini") == (25, 1)

    assert (
        await router._increment_deployment_usage(
            deployment_id="lit-3058-delta",
            deployment_name="gpt-5-mini",
            model_group="gpt-5-mini",
            total_tokens=0,
            rpm_increment=0,
            parent_otel_span=None,
        )
        is None
    )
    assert await router.get_model_group_usage("gpt-5-mini") == (25, 1)

    assert (
        await unlimited._increment_deployment_usage(
            deployment_id="lit-3058-unlimited",
            deployment_name="gpt-5-mini",
            model_group="gpt-5-mini",
            total_tokens=25,
            rpm_increment=1,
            parent_otel_span=None,
        )
        is None
    )
    assert await unlimited.get_model_group_usage("gpt-5-mini") == (None, None)


def _shared_redis_stub(store: dict) -> MagicMock:
    from litellm.caching.redis_cache import RedisCache

    async def increment_pipeline(increment_list, **kwargs):
        for op in increment_list:
            store[op["key"]] = store.get(op["key"], 0.0) + op["increment_value"]
        return [store[op["key"]] for op in increment_list]

    async def batch_get(keys, **kwargs):
        return {key: store.get(key) for key in keys}

    redis_stub = MagicMock(spec=RedisCache)
    redis_stub.async_increment_pipeline = increment_pipeline
    redis_stub.async_batch_get_cache = batch_get
    return redis_stub


@pytest.mark.asyncio
async def test_headers_on_fresh_worker_reflect_shared_redis_usage():
    store: dict = {}
    worker_a = _rpm_tpm_router("lit-3058-workers")
    worker_b = _rpm_tpm_router("lit-3058-workers")
    worker_a.cache = DualCache(redis_cache=_shared_redis_stub(store), in_memory_cache=InMemoryCache())
    worker_b.cache = DualCache(redis_cache=_shared_redis_stub(store), in_memory_cache=InMemoryCache())

    messages = [{"role": "user", "content": "hi"}]
    tokens_on_a = 0
    for _ in range(3):
        response = await worker_a.acompletion(model="gpt-5-mini", messages=messages, mock_response="pong")
        tokens_on_a += response.usage.total_tokens

    response = await worker_b.acompletion(model="gpt-5-mini", messages=messages, mock_response="pong")
    headers = _ratelimit_headers(response)
    assert headers["x-ratelimit-remaining-requests"] == 96
    assert headers["x-ratelimit-remaining-tokens"] == 1000 - tokens_on_a - response.usage.total_tokens

    counted_tokens = tokens_on_a + response.usage.total_tokens
    for _ in range(2):
        response = await worker_a.acompletion(model="gpt-5-mini", messages=messages, mock_response="pong")
        counted_tokens += response.usage.total_tokens

    stream = await worker_b.acompletion(model="gpt-5-mini", messages=messages, mock_response="pong", stream=True)
    stream_headers = _ratelimit_headers(stream)
    assert stream_headers["x-ratelimit-remaining-requests"] == 93
    assert stream_headers["x-ratelimit-remaining-tokens"] == 1000 - counted_tokens
    assert [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_get_model_group_io_token_usage_sums_across_deployments():
    """
    get_model_group_io_token_usage must sum ITPM/OTPM across every deployment
    in the model group (not just the first), reading the same per-deployment
    cache keys the pre-call reservation writes to.
    """
    from litellm.types.router import RouterCacheEnum
    from litellm.utils import get_utc_datetime

    router = Router(
        model_list=[
            {
                "model_name": "opus",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "itpm": 1000,
                    "otpm": 500,
                },
                "model_info": {"id": "io-usage-dep-1"},
            },
            {
                "model_name": "opus",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "itpm": 1000,
                    "otpm": 500,
                },
                "model_info": {"id": "io-usage-dep-2"},
            },
        ]
    )

    minute = get_utc_datetime().strftime("%H-%M")
    keys_and_values = [
        (
            RouterCacheEnum.ITPM.value.format(
                id="io-usage-dep-1", model="openai/gpt-4o-mini", current_minute=minute
            ),
            30,
        ),
        (
            RouterCacheEnum.OTPM.value.format(
                id="io-usage-dep-1", model="openai/gpt-4o-mini", current_minute=minute
            ),
            10,
        ),
        (
            RouterCacheEnum.ITPM.value.format(
                id="io-usage-dep-2", model="openai/gpt-4o", current_minute=minute
            ),
            70,
        ),
        (
            RouterCacheEnum.OTPM.value.format(
                id="io-usage-dep-2", model="openai/gpt-4o", current_minute=minute
            ),
            20,
        ),
    ]
    for key, value in keys_and_values:
        await router.cache.async_increment_cache(key=key, value=value, ttl=60)

    current_itpm, current_otpm = await router.get_model_group_io_token_usage("opus")

    assert current_itpm == 100
    assert current_otpm == 30


@pytest.mark.asyncio
@pytest.mark.usefixtures("router_minute_pinned")
async def test_get_model_group_io_token_usage_no_deployments_returns_none():
    router = Router(model_list=[])
    current_itpm, current_otpm = await router.get_model_group_io_token_usage("nonexistent-group")
    assert current_itpm is None
    assert current_otpm is None


@pytest.mark.asyncio
async def test_get_remaining_model_group_usage_merges_io_and_tpm_headers(model_list):
    """
    A model group with both itpm/otpm and tpm/rpm limits must expose the
    standard remaining-tokens/requests headers alongside the input/output token
    headers, so clients and prometheus gauges relying on either still get data.
    """
    from unittest.mock import Mock

    from litellm.types.router import ModelGroupInfo

    router = Router(model_list=model_list)
    router._cached_get_model_group_info = Mock(
        return_value=ModelGroupInfo(
            model_group="gpt-3.5-turbo",
            providers=["openai"],
            itpm=2000,
            otpm=1000,
            tpm=5000,
            rpm=50,
        )
    )
    router.get_model_group_io_token_usage = AsyncMock(return_value=(100, 40))
    router.get_model_group_usage = AsyncMock(return_value=(500, 5))

    headers = await router.get_remaining_model_group_usage("gpt-3.5-turbo")

    assert headers["x-ratelimit-remaining-input-tokens"] == 1900
    assert headers["x-ratelimit-remaining-output-tokens"] == 960
    assert headers["x-ratelimit-remaining-tokens"] == 4500
    assert headers["x-ratelimit-remaining-requests"] == 45


@pytest.mark.asyncio
async def test_set_response_headers_native_input_token_header_does_not_suppress_router_headers(model_list):
    """
    A provider that natively returns `x-ratelimit-remaining-input-tokens` must
    not suppress the router's own remaining-tokens/requests headers for a
    non-IO model group.
    """
    from pydantic import BaseModel

    class _Usage(BaseModel):
        total_tokens: int = 42

    class _Resp(BaseModel):
        usage: _Usage = _Usage()
        _hidden_params: dict = {}

    router = Router(model_list=model_list)
    router.get_remaining_model_group_usage = AsyncMock(
        return_value={
            "x-ratelimit-remaining-tokens": 1000,
            "x-ratelimit-remaining-requests": 100,
        }
    )

    resp = _Resp()
    resp._hidden_params = {"additional_headers": {"x-ratelimit-remaining-input-tokens": 5}}
    await router.set_response_headers(response=resp, model_group="gpt-3.5-turbo")

    headers = resp._hidden_params["additional_headers"]
    assert headers["x-ratelimit-remaining-tokens"] == 1000
    assert headers["x-ratelimit-remaining-requests"] == 100

    assert headers["x-ratelimit-remaining-input-tokens"] == 5


@pytest.mark.asyncio
async def test_set_response_headers_native_token_header_does_not_suppress_io_headers(model_list):
    from pydantic import BaseModel

    class _Usage(BaseModel):
        total_tokens: int = 42

    class _Resp(BaseModel):
        usage: _Usage = _Usage()
        _hidden_params: dict = {}

    router = Router(model_list=model_list)
    router.get_remaining_model_group_usage = AsyncMock(
        return_value={
            "x-ratelimit-remaining-tokens": 1000,
            "x-ratelimit-remaining-requests": 100,
            "x-ratelimit-remaining-input-tokens": 900,
            "x-ratelimit-remaining-output-tokens": 450,
        }
    )

    resp = _Resp()
    resp._hidden_params = {"additional_headers": {"x-ratelimit-remaining-tokens": 5}}
    await router.set_response_headers(response=resp, model_group="gpt-3.5-turbo")

    headers = resp._hidden_params["additional_headers"]
    assert headers["x-ratelimit-remaining-tokens"] == 5
    assert headers["x-ratelimit-remaining-requests"] == 100
    assert headers["x-ratelimit-remaining-input-tokens"] == 900
    assert headers["x-ratelimit-remaining-output-tokens"] == 450


@pytest.mark.asyncio
async def test_set_response_headers_handles_missing_usage(model_list):
    """
    Streaming chunks and some response shapes may lack a `usage` attribute or
    populated `total_tokens`. Header composition must not depend on usage and never raise.
    """
    from pydantic import BaseModel

    class _Resp(BaseModel):
        _hidden_params: dict = {}

    router = Router(model_list=model_list)
    router.get_remaining_model_group_usage = AsyncMock(
        return_value={
            "x-ratelimit-remaining-tokens": 1000,
            "x-ratelimit-remaining-requests": 100,
        }
    )

    resp = _Resp()
    resp._hidden_params = {}
    await router.set_response_headers(response=resp, model_group="gpt-3.5-turbo")

    headers = resp._hidden_params["additional_headers"]
    assert headers["x-ratelimit-remaining-tokens"] == 1000
    assert headers["x-ratelimit-remaining-requests"] == 100


@pytest.mark.asyncio
async def test_set_response_headers_dict_anthropic_messages_response(model_list):
    """Anthropic /v1/messages returns a dict; IO rate-limit headers must attach."""
    router = Router(model_list=model_list)
    router.get_remaining_model_group_usage = AsyncMock(
        return_value={
            "x-ratelimit-limit-input-tokens": 25,
            "x-ratelimit-remaining-input-tokens": 20,
            "x-ratelimit-limit-output-tokens": 100,
            "x-ratelimit-remaining-output-tokens": 95,
        }
    )

    resp = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    await router.set_response_headers(response=resp, model_group="io-itpm-strict")

    assert "_hidden_params" in resp
    headers = resp["_hidden_params"]["additional_headers"]
    assert headers["x-litellm-model-group"] == "io-itpm-strict"
    assert headers["x-ratelimit-limit-input-tokens"] == 25
    assert headers["x-ratelimit-remaining-input-tokens"] == 20
    assert headers["x-ratelimit-remaining-output-tokens"] == 95


@pytest.mark.asyncio
async def test_set_response_headers_wraps_bare_async_generator(model_list):
    """
    Streaming responses that never go through Router.make_call's usual
    object-based wrappers (e.g. the Anthropic /v1/messages -> Responses API
    bridge, which yields a raw async generator with no `_hidden_params` slot)
    must still get IO rate-limit headers attached via a thin wrapper.
    """

    async def _raw_generator():
        yield {"type": "message_start"}
        yield {"type": "message_stop"}

    router = Router(model_list=model_list)
    router.get_remaining_model_group_usage = AsyncMock(
        return_value={
            "x-ratelimit-limit-input-tokens": 25,
            "x-ratelimit-remaining-input-tokens": 20,
        }
    )

    wrapped = await router.set_response_headers(response=_raw_generator(), model_group="io-itpm-strict")

    assert hasattr(wrapped, "_hidden_params")
    headers = wrapped._hidden_params["additional_headers"]
    assert headers["x-litellm-model-group"] == "io-itpm-strict"
    assert headers["x-ratelimit-limit-input-tokens"] == 25
    assert headers["x-ratelimit-remaining-input-tokens"] == 20

    from collections.abc import AsyncIterator

    assert isinstance(wrapped, AsyncIterator)
    chunks = [chunk async for chunk in wrapped]
    assert chunks == [{"type": "message_start"}, {"type": "message_stop"}]


def test_get_all_deployments(model_list):
    """Test if the 'get_all_deployments' function is working correctly"""
    router = Router(model_list=model_list)
    deployments = router.get_all_deployments(model_name="gpt-5-mini", model_alias="gpt-5-mini")
    assert len(deployments) > 0


def test_get_model_access_groups(model_list):
    """Test if the 'get_model_access_groups' function is working correctly"""
    router = Router(model_list=model_list)
    access_groups = router.get_model_access_groups()
    assert len(access_groups) == 2


def test_update_settings(model_list):
    """Test if the 'update_settings' function is working correctly"""
    router = Router(model_list=model_list)
    pre_update_allowed_fails = router.allowed_fails
    router.update_settings(**{"allowed_fails": 20})
    assert router.allowed_fails != pre_update_allowed_fails
    assert router.allowed_fails == 20


def test_common_checks_available_deployment(model_list):
    """Test if the 'common_checks_available_deployment' function is working correctly"""
    router = Router(model_list=model_list)
    _, available_deployments = router._common_checks_available_deployment(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        input="hi",
        specific_deployment=False,
    )

    assert len(available_deployments) > 0


def test_filter_cooldown_deployments(model_list):
    """Test if the 'filter_cooldown_deployments' function is working correctly"""
    router = Router(model_list=model_list)
    deployments = router._filter_cooldown_deployments(
        healthy_deployments=router.get_all_deployments(model_name="gpt-5-mini"),
        cooldown_deployments=[],
    )
    assert len(deployments) == len(router.get_all_deployments(model_name="gpt-5-mini"))


@pytest.mark.parametrize(
    "exception_type, exception_name, num_retries",
    [
        (litellm.exceptions.BadRequestError, "BadRequestError", 3),
        (litellm.exceptions.AuthenticationError, "AuthenticationError", 4),
        (litellm.exceptions.RateLimitError, "RateLimitError", 6),
        (
            litellm.exceptions.ContentPolicyViolationError,
            "ContentPolicyViolationError",
            7,
        ),
    ],
)
def test_get_num_retries_from_retry_policy(model_list, exception_type, exception_name, num_retries):
    """Test if the 'get_num_retries_from_retry_policy' function is working correctly"""
    from litellm.router import RetryPolicy

    data = {exception_name + "Retries": num_retries}
    print("data", data)
    router = Router(
        model_list=model_list,
        retry_policy=RetryPolicy(**data),
    )
    print("exception_type", exception_type)
    calc_num_retries = router.get_num_retries_from_retry_policy(
        exception=exception_type(message="test", llm_provider="openai", model="gpt-5-mini")
    )
    assert calc_num_retries == num_retries


@pytest.mark.parametrize(
    "exception_type, exception_name, allowed_fails",
    [
        (litellm.exceptions.BadRequestError, "BadRequestError", 3),
        (litellm.exceptions.AuthenticationError, "AuthenticationError", 4),
        (litellm.exceptions.RateLimitError, "RateLimitError", 6),
        (
            litellm.exceptions.ContentPolicyViolationError,
            "ContentPolicyViolationError",
            7,
        ),
    ],
)
def test_get_allowed_fails_from_policy(model_list, exception_type, exception_name, allowed_fails):
    """Test if the 'get_allowed_fails_from_policy' function is working correctly"""
    from litellm.types.router import AllowedFailsPolicy

    data = {exception_name + "AllowedFails": allowed_fails}
    router = Router(model_list=model_list, allowed_fails_policy=AllowedFailsPolicy(**data))
    calc_allowed_fails = router.get_allowed_fails_from_policy(
        exception=exception_type(message="test", llm_provider="openai", model="gpt-5-mini")
    )
    assert calc_allowed_fails == allowed_fails


def test_initialize_alerting(model_list):
    """Test if the 'initialize_alerting' function is working correctly"""
    from litellm.types.router import AlertingConfig
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting

    router = Router(
        model_list=model_list, alerting_config=AlertingConfig(webhook_url="test")
    )
    router._initialize_alerting()

    callback_added = False
    for callback in litellm.callbacks:
        if isinstance(callback, SlackAlerting):
            callback_added = True
    assert callback_added is True


def test_flush_cache(model_list):
    """Test if the 'flush_cache' function is working correctly"""
    router = Router(model_list=model_list)
    router.cache.set_cache("test", "test")
    assert router.cache.get_cache("test") == "test"
    router.flush_cache()
    assert router.cache.get_cache("test") is None


def test_discard(model_list):
    """
    Test that discard properly removes a Router from the callback lists
    """
    litellm.callbacks = []
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.failure_callback = []
    litellm._async_failure_callback = []
    litellm.input_callback = []
    litellm.service_callback = []

    router = Router(model_list=model_list)
    router.discard()

    # Verify all callback lists are empty
    assert len(litellm.callbacks) == 0
    assert len(litellm.success_callback) == 0
    assert len(litellm.failure_callback) == 0
    assert len(litellm._async_success_callback) == 0
    assert len(litellm._async_failure_callback) == 0
    assert len(litellm.input_callback) == 0
    assert len(litellm.service_callback) == 0


def test_initialize_assistants_endpoint(model_list):
    """Test if the 'initialize_assistants_endpoint' function is working correctly"""
    router = Router(model_list=model_list)
    router.initialize_assistants_endpoint()
    assert router.acreate_assistants is not None
    assert router.adelete_assistant is not None
    assert router.aget_assistants is not None
    assert router.acreate_thread is not None
    assert router.aget_thread is not None
    assert router.arun_thread is not None
    assert router.aget_messages is not None
    assert router.a_add_message is not None


def test_get_model_from_alias(model_list):
    """Test if the 'get_model_from_alias' function is working correctly"""
    router = Router(
        model_list=model_list,
        model_group_alias={"gpt-5.5": "gpt-5-mini"},
    )
    model = router.get_model_from_alias(model="gpt-5.5")
    assert model == "gpt-5-mini"


def test_get_deployment_by_litellm_model(model_list):
    """Test if the 'get_deployment_by_litellm_model' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router._get_deployment_by_litellm_model(model="gpt-5-mini")
    assert deployment is not None


def test_get_pattern(model_list):
    router = Router(model_list=model_list)
    pattern = router.pattern_router.get_pattern(model="claude-3")
    assert pattern is not None


def test_deployments_by_pattern(model_list):
    router = Router(model_list=model_list)
    deployments = router.pattern_router.get_deployments_by_pattern(model="claude-3")
    assert deployments is not None


@pytest.mark.parametrize(
    "user_request_model, model_name, litellm_model, expected_model",
    [
        ("llmengine/foo", "llmengine/*", "openai/foo", "openai/foo"),
        ("llmengine/foo", "llmengine/*", "openai/*", "openai/foo"),
        (
            "fo::hi::static::hello",
            "fo::*::static::*",
            "openai/fo::*:static::*",
            "openai/fo::hi:static::hello",
        ),
        (
            "fo::hi::static::hello",
            "fo::*::static::*",
            "openai/gpt-5-mini",
            "openai/gpt-5-mini",
        ),
        (
            "bedrock/meta.llama3-70b",
            "*meta.llama3*",
            "bedrock/meta.llama3-*",
            "bedrock/meta.llama3-70b",
        ),
        (
            "meta.llama3-70b",
            "*meta.llama3*",
            "bedrock/meta.llama3-*",
            "meta.llama3-70b",
        ),
    ],
)
def test_pattern_match_deployment_set_model_name(user_request_model, model_name, litellm_model, expected_model):
    from re import Match
    from litellm.router_utils.pattern_match_deployments import PatternMatchRouter

    pattern_router = PatternMatchRouter()

    import re

    model_name_regex = pattern_router.pattern_to_regex(model_name)

    match = re.match(model_name_regex, user_request_model)

    if match is None:
        raise ValueError("Match not found")

    updated_model = pattern_router.set_deployment_model_name(match, litellm_model)

    print(updated_model)
    assert updated_model == expected_model

    updated_models = pattern_router._return_pattern_matched_deployments(
        match,
        deployments=[
            {
                "model_name": model_name,
                "litellm_params": {"model": litellm_model},
            }
        ],
    )

    for model in updated_models:
        assert model["litellm_params"]["model"] == expected_model


@pytest.mark.parametrize(
    "has_default_fallbacks, expected_result",
    [(True, True), (False, False)],
)
def test_has_default_fallbacks(model_list, has_default_fallbacks, expected_result):
    router = Router(
        model_list=model_list,
        default_fallbacks=(["my-default-fallback-model"] if has_default_fallbacks else None),
    )
    assert router._has_default_fallbacks() is expected_result


def test_add_optional_pre_call_checks(model_list):
    router = Router(model_list=model_list)

    router.add_optional_pre_call_checks(["prompt_caching"])
    assert len(litellm.callbacks) > 0


@pytest.mark.asyncio
async def test_async_callback_filter_deployments(model_list):
    from litellm.router_strategy.budget_limiter import RouterBudgetLimiting

    router = Router(model_list=model_list)

    healthy_deployments = router.get_model_list(model_name="gpt-5-mini")

    new_healthy_deployments = await router.async_callback_filter_deployments(
        model="gpt-5-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        parent_otel_span=None,
    )

    assert len(new_healthy_deployments) == len(healthy_deployments)


def test_cached_get_model_group_info(model_list):
    """Test if the '_cached_get_model_group_info' function is working correctly with LRU cache"""
    router = Router(model_list=model_list)

    result1 = router._cached_get_model_group_info("gpt-5-mini")

    result2 = router._cached_get_model_group_info("gpt-5-mini")

    assert result1 == result2

    cache_info = router._cached_get_model_group_info.cache_info()
    assert cache_info.hits > 0


def test_init_responses_api_endpoints(model_list):
    """Test if the '_init_responses_api_endpoints' function is working correctly"""
    from typing import Callable

    router = Router(model_list=model_list)

    assert router.aget_responses is not None
    assert isinstance(router.aget_responses, Callable)
    assert router.adelete_responses is not None
    assert isinstance(router.adelete_responses, Callable)


@pytest.mark.parametrize(
    "mock_testing_fallbacks, mock_testing_context_fallbacks, mock_testing_content_policy_fallbacks, expected_fallbacks, expected_context, expected_content_policy",
    [
        ("true", "false", "True", True, False, True),
        ("TRUE", "FALSE", "False", True, False, False),
        ("false", "true", "false", False, True, False),
        (True, False, True, True, False, True),
        (False, True, False, False, True, False),
        (None, None, None, None, None, None),
        ("true", False, None, True, False, None),
    ],
)
def test_mock_router_testing_params_str_to_bool_conversion(
    mock_testing_fallbacks,
    mock_testing_context_fallbacks,
    mock_testing_content_policy_fallbacks,
    expected_fallbacks,
    expected_context,
    expected_content_policy,
):
    """Test if MockRouterTestingParams.from_kwargs correctly converts string values to booleans using str_to_bool"""
    from litellm.types.router import MockRouterTestingParams

    kwargs = {
        "mock_testing_fallbacks": mock_testing_fallbacks,
        "mock_testing_context_fallbacks": mock_testing_context_fallbacks,
        "mock_testing_content_policy_fallbacks": mock_testing_content_policy_fallbacks,
        "other_param": "should_remain",
    }

    original_kwargs = kwargs.copy()

    mock_params = MockRouterTestingParams.from_kwargs(kwargs)

    assert mock_params.mock_testing_fallbacks == expected_fallbacks
    assert mock_params.mock_testing_context_fallbacks == expected_context
    assert mock_params.mock_testing_content_policy_fallbacks == expected_content_policy

    assert "mock_testing_fallbacks" not in kwargs
    assert "mock_testing_context_fallbacks" not in kwargs
    assert "mock_testing_content_policy_fallbacks" not in kwargs

    assert kwargs["other_param"] == "should_remain"


def test_is_auto_router_deployment(model_list):
    """Test if the '_is_auto_router_deployment' function correctly identifies auto-router deployments"""
    router = Router(model_list=model_list)

    litellm_params_auto = LiteLLM_Params(model="auto_router/my-auto-router")
    assert router._is_auto_router_deployment(litellm_params_auto) is True

    litellm_params_regular = LiteLLM_Params(model="gpt-5-mini")
    assert router._is_auto_router_deployment(litellm_params_regular) is False

    litellm_params_empty = LiteLLM_Params(model="")
    assert router._is_auto_router_deployment(litellm_params_empty) is False

    litellm_params_contains = LiteLLM_Params(model="prefix_auto_router/something")
    assert router._is_auto_router_deployment(litellm_params_contains) is False


@patch("litellm.router_strategy.auto_router.auto_router.AutoRouter")
def test_init_auto_router_deployment_success(mock_auto_router, model_list):
    """Test if the 'init_auto_router_deployment' function successfully initializes auto-router when all params provided"""
    router = Router(model_list=model_list)

    mock_auto_router_instance = MagicMock()
    mock_auto_router.return_value = mock_auto_router_instance

    litellm_params = LiteLLM_Params(
        model="auto_router/test",
        auto_router_config_path="/path/to/config",
        auto_router_default_model="gpt-5-mini",
        auto_router_embedding_model="text-embedding-3-small",
    )
    deployment = Deployment(
        model_name="test-auto-router",
        litellm_params=litellm_params,
        model_info={"id": "test-id"},
    )

    router.init_auto_router_deployment(deployment)

    mock_auto_router.assert_called_once_with(
        model_name="test-auto-router",
        auto_router_config_path="/path/to/config",
        auto_router_config=None,
        default_model="gpt-5-mini",
        embedding_model="text-embedding-3-small",
        litellm_router_instance=router,
        max_input_chars=DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS,
    )

    assert "test-auto-router" in router.auto_routers
    assert router.auto_routers["test-auto-router"][0].strategy == mock_auto_router_instance


@patch("litellm.router_strategy.auto_router.auto_router.AutoRouter")
def test_init_auto_router_deployment_duplicate_model_name(mock_auto_router, model_list):
    """Test if the 'init_auto_router_deployment' function raises ValueError when model_name already exists"""
    router = Router(model_list=model_list)

    mock_auto_router_instance = MagicMock()
    mock_auto_router.return_value = mock_auto_router_instance

    from litellm.types.router import TaggedPreRoutingStrategy

    router.auto_routers["test-auto-router"] = [TaggedPreRoutingStrategy(tags=(), strategy=mock_auto_router_instance)]

    litellm_params = LiteLLM_Params(
        model="auto_router/test",
        auto_router_config_path="/path/to/config",
        auto_router_default_model="gpt-5-mini",
        auto_router_embedding_model="text-embedding-3-small",
    )
    deployment = Deployment(
        model_name="test-auto-router",
        litellm_params=litellm_params,
        model_info={"id": "test-id"},
    )

    with pytest.raises(ValueError, match=r"Auto-router deployment test-auto-router with tags .* already exists"):
        router.init_auto_router_deployment(deployment)


def testgenerate_model_id_with_deployment_model_name(model_list):
    """Test that generate_model_id works correctly with deployment model_name and handles None values properly"""
    router = Router(model_list=model_list)

    model_group = "gpt-4.1"
    litellm_params = {
        "model": "gpt-4.1",
        "api_key": "test_key",
        "api_base": "https://api.openai.com/v1",
    }

    try:
        result = router.generate_model_id(model_group=model_group, litellm_params=litellm_params)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ Success with valid model_group: {result}")
    except Exception as e:
        pytest.fail(f"Failed with valid model_group: {e}")

    with pytest.raises(TypeError) as exc_info:
        router.generate_model_id(model_group=None, litellm_params=litellm_params)

    error_str = str(exc_info.value)
    assert "unsupported operand type(s) for +=" in error_str or "expected str instance, NoneType found" in error_str

    litellm_params_with_none_key = {
        "model": "gpt-4.1",
        "api_key": "test_key",
        None: "should_be_skipped",
    }

    try:
        result = router.generate_model_id(model_group=model_group, litellm_params=litellm_params_with_none_key)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ Success with None key in litellm_params: {result}")
    except Exception as e:
        pytest.fail(f"Failed with None key in litellm_params: {e}")

    try:
        result = router.generate_model_id(model_group=model_group, litellm_params={})
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ Success with empty litellm_params: {result}")
    except Exception as e:
        pytest.fail(f"Failed with empty litellm_params: {e}")

    result1 = router.generate_model_id(model_group=model_group, litellm_params=litellm_params)
    result2 = router.generate_model_id(model_group=model_group, litellm_params=litellm_params)
    assert result1 == result2, "Model ID generation should be deterministic"

    print("✓ All generate_model_id tests passed!")


def test_handle_clientside_credential_with_deployment_model_name(model_list):
    """Test that _handle_clientside_credential uses deployment model_name correctly"""
    router = Router(model_list=model_list)

    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test_key"},
    }

    kwargs = {
        "metadata": {},
        "litellm_params": {
            "api_key": "client_side_key",
            "api_base": "https://api.openai.com/v1",
        },
    }

    dynamic_litellm_params = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
    }

    try:
        model_group = deployment["model_name"]
        assert model_group == "gpt-4.1"

        result = router.generate_model_id(model_group=model_group, litellm_params=dynamic_litellm_params)
        assert isinstance(result, str)
        assert len(result) > 0

        print(f"✓ Success with deployment model_name: {result}")
    except Exception as e:
        pytest.fail(f"Failed with deployment model_name: {e}")

    print("✓ _handle_clientside_credential test passed!")


def test_sync_generic_api_call_preserves_requested_model_group_in_logs():
    router = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4-6",
                "litellm_params": {
                    "model": "bedrock/global.anthropic.claude-sonnet-4-6",
                    "aws_access_key_id": "test-access-key",
                    "aws_secret_access_key": "test-secret-key",
                    "aws_region_name": "us-west-2",
                },
            }
        ]
    )

    try:
        captured_kwargs = {}

        def mock_original_function(**kwargs):
            captured_kwargs.update(kwargs)
            return {"status": "ok"}

        response = router._generic_api_call_with_fallbacks(
            model="claude-sonnet-4-6",
            original_function=mock_original_function,
        )

        assert response == {"status": "ok"}
        assert captured_kwargs["model"] == "bedrock/global.anthropic.claude-sonnet-4-6"
        assert captured_kwargs["litellm_metadata"]["model_group"] == "claude-sonnet-4-6"
        assert captured_kwargs["litellm_metadata"]["deployment"] == "bedrock/global.anthropic.claude-sonnet-4-6"
    finally:
        router.discard()


def test_sync_generic_api_call_uses_request_kwargs_for_deployment_selection():
    router = Router(
        model_list=[
            {
                "model_name": "regional-model",
                "litellm_params": {
                    "model": "anthropic/us-model",
                    "api_key": "test-api-key",
                    "region_name": "us",
                },
            },
            {
                "model_name": "regional-model",
                "litellm_params": {
                    "model": "anthropic/eu-model",
                    "api_key": "test-api-key",
                    "region_name": "eu",
                },
            },
        ],
        enable_pre_call_checks=True,
    )

    try:
        captured_kwargs = {}

        def mock_original_function(**kwargs):
            captured_kwargs.update(kwargs)
            return {"status": "ok"}

        response = router._generic_api_call_with_fallbacks(
            model="regional-model",
            original_function=mock_original_function,
            messages=[{"role": "user", "content": "Hello from Europe"}],
            allowed_model_region="eu",
        )

        assert response == {"status": "ok"}
        assert captured_kwargs["model"] == "anthropic/eu-model"
    finally:
        router.discard()


@pytest.mark.parametrize(
    "function_name, expected_metadata_key",
    [
        ("acompletion", "metadata"),
        ("_ageneric_api_call_with_fallbacks", "litellm_metadata"),
        ("batch", "litellm_metadata"),
        ("completion", "metadata"),
        ("acreate_file", "litellm_metadata"),
        ("aget_file", "litellm_metadata"),
    ],
)
def test_handle_clientside_credential_metadata_loading(model_list, function_name, expected_metadata_key):
    """Test that _handle_clientside_credential correctly loads metadata based on function name"""
    router = Router(model_list=model_list)

    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test_key"},
        "model_info": {"id": "original-id-123"},
    }

    kwargs = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
        expected_metadata_key: {"model_group": "gpt-4.1", "custom_field": "test_value"},
    }

    result_deployment = router._handle_clientside_credential(
        deployment=deployment, kwargs=kwargs, function_name=function_name
    )

    assert isinstance(result_deployment, Deployment)

    assert result_deployment.model_name == "gpt-4.1"

    assert result_deployment.litellm_params.api_key == "client_side_key"
    assert result_deployment.litellm_params.api_base == "https://api.openai.com/v1"

    assert result_deployment.model_info.id != "original-id-123"
    assert result_deployment.model_info.original_model_id == "original-id-123"

    assert len(router.model_list) == len(model_list)
    assert router.get_deployment(model_id=result_deployment.model_info.id) is None

    if function_name == "acompletion":
        assert "metadata" in kwargs
        assert "litellm_metadata" not in kwargs
    elif function_name in [
        "_ageneric_api_call_with_fallbacks",
        "batch",
        "acreate_file",
        "aget_file",
    ]:
        assert "litellm_metadata" in kwargs

    print(f"✓ Success with function_name '{function_name}' using '{expected_metadata_key}' metadata key")


@pytest.mark.parametrize(
    "function_name, metadata_key",
    [
        ("acompletion", "metadata"),
        ("_ageneric_api_call_with_fallbacks", "litellm_metadata"),
    ],
)
def test_handle_clientside_credential_metadata_variable_name(model_list, function_name, metadata_key):
    """Test that _handle_clientside_credential uses the correct metadata variable name based on function name"""
    from litellm.router_utils.batch_utils import get_router_metadata_variable_name

    router = Router(model_list=model_list)

    expected_metadata_key = get_router_metadata_variable_name(function_name=function_name)
    assert expected_metadata_key == metadata_key

    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test_key"},
        "model_info": {"id": "original-id-456"},
    }

    kwargs = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
        metadata_key: {"model_group": "gpt-4.1", "test_field": "test_value"},
    }

    result_deployment = router._handle_clientside_credential(
        deployment=deployment, kwargs=kwargs, function_name=function_name
    )

    assert result_deployment.model_name == "gpt-4.1"

    assert result_deployment.litellm_params.api_key == "client_side_key"
    assert result_deployment.litellm_params.api_base == "https://api.openai.com/v1"

    print(f"✓ Success with function_name '{function_name}' correctly using '{metadata_key}' for metadata")


def test_handle_clientside_credential_with_responses_function(model_list):
    """Test that _handle_clientside_credential works correctly with responses function name"""
    router = Router(model_list=model_list)

    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test_key"},
        "model_info": {"id": "original-id-responses"},
    }

    kwargs = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
        "litellm_metadata": {
            "model_group": "gpt-4.1",
            "responses_field": "responses_value",
        },
    }

    result_deployment = router._handle_clientside_credential(
        deployment=deployment,
        kwargs=kwargs,
        function_name="_ageneric_api_call_with_fallbacks",
    )

    assert isinstance(result_deployment, Deployment)
    assert result_deployment.model_name == "gpt-4.1"
    assert result_deployment.litellm_params.api_key == "client_side_key"
    assert result_deployment.litellm_params.api_base == "https://api.openai.com/v1"
    assert result_deployment.model_info.id != "original-id-responses"
    assert result_deployment.model_info.original_model_id == "original-id-responses"

    assert len(router.model_list) == len(model_list)
    assert router.get_deployment(model_id=result_deployment.model_info.id) is None

    print("✓ Success with _ageneric_api_call_with_fallbacks function name and litellm_metadata")


def test_handle_clientside_credential_still_registers_custom_pricing(model_list):
    """A clientside-credential call must still price against the deployment's own
    custom rate, even though the call's ephemeral deployment is never added to the
    router (see LIT-7811): losing that registration would silently fall back to
    public catalog pricing for every clientside-credential call on a deployment
    with a custom rate configured."""
    router = Router(model_list=model_list)
    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {
            "model": "gpt-4.1",
            "api_key": "test_key",
            "input_cost_per_token": 0.0001234,
            "output_cost_per_token": 0.0005678,
        },
        "model_info": {"id": "original-id-pricing"},
    }
    kwargs = {"api_key": "client_side_key", "metadata": {"model_group": "gpt-4.1"}}

    result_deployment = router._handle_clientside_credential(
        deployment=deployment, kwargs=kwargs, function_name="acompletion"
    )

    registered = litellm.model_cost.get(result_deployment.model_info.id)
    assert registered is not None
    assert registered["input_cost_per_token"] == 0.0001234
    assert registered["output_cost_per_token"] == 0.0005678


def test_register_deployment_pricing_direct_call():
    """Direct-call unit test for the pricing-registration helper `_handle_clientside_credential`
    relies on, so it prices a deployment that is deliberately never added to `self.model_list`."""
    deployment = Deployment(
        model_name="gpt-4.1",
        litellm_params=LiteLLM_Params(
            model="gpt-4.1",
            api_key="test_key",
            input_cost_per_token=0.0009999,
        ),
        model_info=ModelInfo(id="direct-call-pricing-id"),
    )

    Router._register_deployment_pricing(deployment=deployment)

    assert litellm.model_cost["direct-call-pricing-id"]["input_cost_per_token"] == 0.0009999


def test_get_metadata_variable_name_from_kwargs(model_list):
    """
    Test _get_metadata_variable_name_from_kwargs method returns correct metadata variable name based on kwargs content.
    """
    router = Router(model_list=model_list)

    kwargs_with_litellm_metadata = {
        "litellm_metadata": {"user": "test"},
        "metadata": {"other": "data"},
    }
    result = router._get_metadata_variable_name_from_kwargs(kwargs_with_litellm_metadata)
    assert result == "litellm_metadata"

    kwargs_with_metadata_only = {"metadata": {"user": "test"}}
    result = router._get_metadata_variable_name_from_kwargs(kwargs_with_metadata_only)
    assert result == "metadata"

    kwargs_empty = {}
    result = router._get_metadata_variable_name_from_kwargs(kwargs_empty)
    assert result == "metadata"

    kwargs_other = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "hello"}],
    }
    result = router._get_metadata_variable_name_from_kwargs(kwargs_other)
    assert result == "metadata"


@pytest.fixture
def search_tools():
    """Fixture for search tools configuration"""
    return [
        {
            "search_tool_name": "test-search-tool",
            "litellm_params": {
                "search_provider": "perplexity",
                "api_key": "test-api-key",
                "api_base": "https://api.perplexity.ai",
                "mode": "turbo",
            },
        },
        {
            "search_tool_name": "test-search-tool",
            "litellm_params": {
                "search_provider": "perplexity",
                "api_key": "test-api-key-2",
                "api_base": "https://api.perplexity.ai",
                "mode": "turbo",
            },
        },
    ]


@pytest.mark.asyncio
async def test_asearch_with_fallbacks(search_tools):
    """
    Test _asearch_with_fallbacks method of Router.

    Tests that the _asearch_with_fallbacks method correctly:
    - Accepts search parameters
    - Calls async_function_with_fallbacks with correct configuration
    - Returns SearchResponse
    """
    from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult

    router = Router(search_tools=search_tools)

    mock_response = SearchResponse(
        object="search",
        results=[
            SearchResult(
                title="Test Result",
                url="https://example.com",
                snippet="Test snippet content",
            )
        ],
    )

    with patch.object(router, "async_function_with_fallbacks", new_callable=AsyncMock) as mock_fallbacks:
        mock_fallbacks.return_value = mock_response

        async def mock_asearch(**kwargs):
            return mock_response

        response = await router._asearch_with_fallbacks(
            original_function=mock_asearch,
            search_tool_name="test-search-tool",
            query="test query",
            max_results=5,
        )

        assert mock_fallbacks.called

        assert isinstance(response, SearchResponse)
        assert response.object == "search"
        assert len(response.results) == 1
        assert response.results[0].title == "Test Result"


@pytest.mark.asyncio
async def test_asearch_with_fallbacks_helper(search_tools):
    """
    Test _asearch_with_fallbacks_helper method of Router.

    Tests that the _asearch_with_fallbacks_helper method correctly:
    - Selects a search tool from available options
    - Calls the original search function with correct provider parameters
    - Returns SearchResponse
    """
    from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult

    router = Router(search_tools=search_tools)

    mock_response = SearchResponse(
        object="search",
        results=[
            SearchResult(
                title="Helper Test Result",
                url="https://example.com/helper",
                snippet="Helper test snippet",
            )
        ],
    )

    async def mock_original_function(**kwargs):

        assert "search_provider" in kwargs
        assert kwargs["search_provider"] == "perplexity"
        assert "api_key" in kwargs
        assert kwargs["mode"] == "turbo"
        assert kwargs["query"] == "helper test query"
        return mock_response

    response = await router._asearch_with_fallbacks_helper(
        model="test-search-tool",
        original_generic_function=mock_original_function,
        query="helper test query",
        max_results=3,
    )

    assert isinstance(response, SearchResponse)
    assert response.object == "search"
    assert len(response.results) == 1
    assert response.results[0].title == "Helper Test Result"
    assert response.results[0].url == "https://example.com/helper"


@pytest.mark.asyncio
async def test_asearch_with_fallbacks_helper_missing_search_tool():
    """
    Test _asearch_with_fallbacks_helper raises error when search tool not found.

    Tests that the helper method raises a ValueError when the requested
    search tool name doesn't exist in the router's search_tools configuration.
    """

    router = Router(model_list=[])

    async def mock_original_function(**kwargs):
        return None

    with pytest.raises(ValueError, match="Search tool 'nonexistent-tool' not found"):
        await router._asearch_with_fallbacks_helper(
            model="nonexistent-tool",
            original_generic_function=mock_original_function,
            query="test query",
        )


@pytest.mark.asyncio
async def test_asearch_with_fallbacks_helper_missing_search_provider():
    """
    Test _asearch_with_fallbacks_helper raises error when search_provider not configured.

    Tests that the helper method raises a ValueError when a search tool
    is found but doesn't have search_provider in its litellm_params.
    """

    search_tools_bad = [
        {
            "search_tool_name": "bad-tool",
            "litellm_params": {"api_key": "test-key"},
        }
    ]

    router = Router(search_tools=search_tools_bad)

    async def mock_original_function(**kwargs):
        return None

    with pytest.raises(ValueError, match="search_provider not found in litellm_params"):
        await router._asearch_with_fallbacks_helper(
            model="bad-tool",
            original_generic_function=mock_original_function,
            query="test query",
        )


def test_get_first_default_fallback():
    """Test _get_first_default_fallback method"""

    model_list = [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {"model": "gpt-5-mini", "api_key": "fake-key"},
        }
    ]

    router = Router(model_list=model_list, fallbacks=[{"*": ["gpt-5-mini"]}])

    result = router._get_first_default_fallback()
    assert result == "gpt-5-mini"

    router_no_fallbacks = Router(model_list=model_list)
    result = router_no_fallbacks._get_first_default_fallback()
    assert result is None

    router_no_default = Router(model_list=model_list, fallbacks=[{"gpt-5.5": ["gpt-5-mini"]}])
    result = router_no_default._get_first_default_fallback()
    assert result is None

    router_empty_list = Router(model_list=model_list, fallbacks=[{"*": []}])
    result = router_empty_list._get_first_default_fallback()
    assert result is None


def test_resolve_model_name_from_model_id():
    """Test resolve_model_name_from_model_id function with various scenarios"""

    router = Router(model_list=[])
    result = router.resolve_model_name_from_model_id(None)
    assert result is None

    model_list = [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)
    result = router.resolve_model_name_from_model_id("gpt-5-mini")
    assert result == "gpt-5-mini"

    model_list = [
        {
            "model_name": "vertex-ai-sora-2",
            "litellm_params": {
                "model": "vertex_ai/veo-2.0-generate-001",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)
    result = router.resolve_model_name_from_model_id("vertex_ai/veo-2.0-generate-001")
    assert result == "vertex-ai-sora-2"

    model_list = [
        {
            "model_name": "vertex-ai-sora-2",
            "litellm_params": {
                "model": "vertex_ai/veo-2.0-generate-001",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)
    result = router.resolve_model_name_from_model_id("veo-2.0-generate-001")
    assert result == "vertex-ai-sora-2"

    model_list = [
        {
            "model_name": "vertex-ai-sora-2",
            "litellm_params": {
                "model": "vertex_ai/veo-2.0-generate-001",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)

    result = router.resolve_model_name_from_model_id("veo-2.0-generate-001")
    assert result == "vertex-ai-sora-2"

    model_list = [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)
    result = router.resolve_model_name_from_model_id("non-existent-model")
    assert result is None

    router = Router(model_list=[])
    result = router.resolve_model_name_from_model_id("some-model")
    assert result is None

    model_list = [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": "test-key",
            },
        },
        {
            "model_name": "vertex-ai-sora-2",
            "litellm_params": {
                "model": "vertex_ai/veo-2.0-generate-001",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)
    result = router.resolve_model_name_from_model_id("veo-2.0-generate-001")
    assert result == "vertex-ai-sora-2"

    model_list = [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)

    result = router.resolve_model_name_from_model_id("gpt-5-mini")
    assert result == "gpt-5-mini"

    model_list = [
        {
            "model_name": "bedrock-batch-model",
            "litellm_params": {
                "model": "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
            },
            "model_info": {"id": "8d0eaa7e6c6f54a425dfd0062cb6b0dc"},
        },
    ]
    router = Router(model_list=model_list)
    result = router.resolve_model_name_from_model_id("8d0eaa7e6c6f54a425dfd0062cb6b0dc")
    assert result == "bedrock-batch-model"


def test_get_valid_args():
    """Test get_valid_args static method returns valid Router.__init__ arguments"""

    valid_args = Router.get_valid_args()

    assert isinstance(valid_args, list)
    assert len(valid_args) > 0

    expected_args = [
        "model_list",
        "routing_strategy",
        "cache_responses",
        "num_retries",
        "timeout",
        "fallbacks",
    ]
    for arg in expected_args:
        assert arg in valid_args, f"Expected argument '{arg}' not found in valid_args"

    assert "self" not in valid_args

    assert "assistants_config" in valid_args or "search_tools" in valid_args


def test_get_router_model_info_with_deployment_object():
    """Test get_router_model_info accepts Deployment object directly and reuses LiteLLM_Params"""
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.5",
                "litellm_params": {"model": "gpt-5.5", "api_key": "test-key"},
                "model_info": {"id": "test-id"},
            }
        ]
    )

    deployment = router.get_deployment(model_id="test-id")
    assert deployment is not None
    assert isinstance(deployment, Deployment)
    assert isinstance(deployment.litellm_params, LiteLLM_Params)

    model_info = router.get_router_model_info(
        deployment=deployment,
        received_model_name="gpt-5.5",
    )

    assert model_info is not None
    assert isinstance(model_info, dict)


def test_deployment_has_budget_limits():
    router = Router(model_list=[])

    with_budget = Deployment(
        model_name="budgeted-model",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4o-mini",
            max_budget=0.001,
            budget_duration="1d",
        ),
        model_info=ModelInfo(id="budget-deployment-id"),
    )
    without_budget = Deployment(
        model_name="unbudgeted-model",
        litellm_params=LiteLLM_Params(model="openai/gpt-4o-mini"),
        model_info=ModelInfo(id="no-budget-deployment-id"),
    )

    assert router._deployment_has_budget_limits(deployment=with_budget) is True
    assert router._deployment_has_budget_limits(deployment=without_budget) is False


def test_sync_deployment_budget_config(monkeypatch):
    import asyncio

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)

    router = Router(model_list=[], optional_pre_call_checks=[])
    deployment = Deployment(
        model_name="dynamic-budget-model",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4o-mini",
            api_key="fake-key",
            max_budget=0.000000000001,
            budget_duration="1d",
        ),
        model_info=ModelInfo(id="runtime-budget-deployment"),
    )

    router._sync_deployment_budget_config(deployment=deployment)

    budget_limiter = router.get_router_deployment_budget_limiter()
    assert budget_limiter is not None
    config = budget_limiter._get_budget_config_for_deployment("runtime-budget-deployment")
    assert config is not None
    assert config.max_budget == 0.000000000001


def test_sync_deployment_budget_config_clears_removed_limits(monkeypatch):
    import asyncio

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)

    router = Router(model_list=[], optional_pre_call_checks=[])
    model_id = "runtime-budget-deployment"
    budgeted = Deployment(
        model_name="dynamic-budget-model",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4o-mini",
            api_key="fake-key",
            max_budget=0.000000000001,
            budget_duration="1d",
        ),
        model_info=ModelInfo(id=model_id),
    )
    unbudgeted = Deployment(
        model_name="dynamic-budget-model",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4o-mini",
            api_key="fake-key",
        ),
        model_info=ModelInfo(id=model_id),
    )

    router._sync_deployment_budget_config(deployment=budgeted)
    budget_limiter = router.get_router_deployment_budget_limiter()
    assert budget_limiter is not None
    assert budget_limiter._get_budget_config_for_deployment(model_id) is not None

    router._sync_deployment_budget_config(deployment=unbudgeted)
    assert budget_limiter._get_budget_config_for_deployment(model_id) is None


def test_upsert_deployment_clears_stale_budget_config(monkeypatch):
    import asyncio

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)

    router = Router(model_list=[], optional_pre_call_checks=[])
    model_id = "upsert-budget-deployment"
    budgeted = Deployment(
        model_name="dynamic-budget-model",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4o-mini",
            api_key="fake-key",
            max_budget=0.000000000001,
            budget_duration="1d",
        ),
        model_info=ModelInfo(id=model_id),
    )
    unbudgeted = Deployment(
        model_name="dynamic-budget-model",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4o-mini",
            api_key="fake-key",
        ),
        model_info=ModelInfo(id=model_id),
    )

    router.upsert_deployment(deployment=budgeted)
    budget_limiter = router.get_router_deployment_budget_limiter()
    assert budget_limiter is not None
    assert budget_limiter._get_budget_config_for_deployment(model_id) is not None

    router.upsert_deployment(deployment=unbudgeted)
    assert budget_limiter._get_budget_config_for_deployment(model_id) is None
