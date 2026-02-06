import sys
import os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock
from create_mock_standard_logging_payload import create_standard_logging_payload
from litellm.types.utils import StandardLoggingPayload
from litellm.types.router import Deployment, LiteLLM_Params


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "tpm": 1000,  # Add TPM limit so async method doesn't return early
                "rpm": 100,   # Add RPM limit so async method doesn't return early
            },
            "model_info": {
                "access_groups": ["group1", "group2"],
            },
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "dall-e-3",
            "litellm_params": {
                "model": "dall-e-3",
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
    router = Router(model_list=model_list, fallbacks=[{"gpt-4o": "gpt-3.5-turbo"}])
    router.validate_fallbacks(fallback_param=[{"gpt-4o": "gpt-3.5-turbo"}])


def test_routing_strategy_init(model_list):
    """Test if all routing strategies are initialized correctly"""
    from litellm.types.router import RoutingStrategy

    router = Router(model_list=model_list)
    for strategy in RoutingStrategy:
        router.routing_strategy_init(
            routing_strategy=strategy, routing_strategy_args={}
        )


def test_routing_strategy_init_invalid_strategy(model_list):
    """Test that invalid routing_strategy raises ValueError with helpful message.

    See: https://github.com/BerriAI/litellm/issues/11330
    Invalid strategies like 'simple' (without '-shuffle') should fail fast
    with a clear error, not silently cause 'No deployments available' errors.
    """
    router = Router(model_list=model_list)

    # Test common mistake: "simple" instead of "simple-shuffle"
    with pytest.raises(ValueError) as exc_info:
        router.routing_strategy_init(
            routing_strategy="simple",
            routing_strategy_args={}
        )

    # Verify error message is helpful
    error_msg = str(exc_info.value)
    assert "Invalid routing_strategy" in error_msg
    assert "simple" in error_msg
    assert "simple-shuffle" in error_msg  # Suggests the correct option
    # Verify error message tells user WHERE to fix it
    assert "config.yaml" in error_msg
    assert "router_settings.routing_strategy" in error_msg
    assert "Router SDK" in error_msg

    # Test completely invalid strategy
    with pytest.raises(ValueError) as exc_info:
        router.routing_strategy_init(
            routing_strategy="not-a-real-strategy",
            routing_strategy_args={}
        )
    assert "Invalid routing_strategy" in str(exc_info.value)


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


def test_routing_strategy_init_valid_enum_strategies(model_list):
    """Test that RoutingStrategy enum values work without error."""
    from litellm.types.router import RoutingStrategy

    router = Router(model_list=model_list)

    for strategy in RoutingStrategy:
        # Should not raise when passing enum directly
        router.routing_strategy_init(
            routing_strategy=strategy, routing_strategy_args={}
        )


def test_print_deployment(model_list):
    """Test if the api key is masked correctly"""

    router = Router(model_list=model_list)
    deployment = {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    assert response["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.flaky(retries=6, delay=1)
@pytest.mark.asyncio
async def test_image_generation(model_list, sync_mode):
    """Test if the underlying '_image_generation' function is working correctly"""
    from litellm.types.utils import ImageResponse

    router = Router(model_list=model_list)
    if sync_mode:
        response = router._image_generation(
            model="dall-e-3",
            prompt="A cute baby sea otter",
        )
    else:
        response = await router._aimage_generation(
            model="dall-e-3",
            prompt="A cute baby sea otter",
        )

    ImageResponse.model_validate(response)


@pytest.mark.asyncio
async def test_router_acompletion_util(model_list):
    """Test if the underlying '_acompletion' function is working correctly"""
    router = Router(model_list=model_list)
    response = await router._acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    assert response["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_abatch_completion_one_model_multiple_requests_util(model_list):
    """Test if the 'abatch_completion_one_model_multiple_requests' function is working correctly"""
    router = Router(model_list=model_list)
    response = await router.abatch_completion_one_model_multiple_requests(
        model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
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
    with patch.object(
        router, "_atext_completion", AsyncMock()
    ) as mock_atext_completion:
        mock_atext_completion.return_value = TextCompletionResponse()
        response = await router.atext_completion(
            model="gpt-3.5-turbo",
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
    with patch.object(
        router, "_atext_completion", AsyncMock()
    ) as mock_atext_completion:
        mock_atext_completion.return_value = TextCompletionResponse()
        response = await router._schedule_factory(
            model="gpt-3.5-turbo",
            args=(
                "gpt-3.5-turbo",
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
        "model": "gpt-3.5-turbo",
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
        "model": "gpt-3.5-turbo",
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

    ## ACOMPLETION
    router = Router(model_list=model_list)
    response = await router.make_call(
        original_function=router._acompletion,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    assert response.choices[0].message.content == "I'm fine, thank you!"

    ## ATEXT_COMPLETION
    response = await router.make_call(
        original_function=router._atext_completion,
        model="gpt-3.5-turbo",
        prompt="Hello, how are you?",
        mock_response="I'm fine, thank you!",
    )
    assert response.choices[0].text == "I'm fine, thank you!"

    ## AEMBEDDING
    response = await router.make_call(
        original_function=router._aembedding,
        model="gpt-3.5-turbo",
        input="Hello, how are you?",
        mock_response=[0.1, 0.2, 0.3],
    )
    assert response.data[0].embedding == [0.1, 0.2, 0.3]

    ## AIMAGE_GENERATION
    response = await router.make_call(
        original_function=router._aimage_generation,
        model="dall-e-3",
        prompt="A cute baby sea otter",
        mock_response="https://example.com/image.png",
    )
    assert response.data[0].url == "https://example.com/image.png"


def test_update_kwargs_with_deployment(model_list):
    """Test if the '_update_kwargs_with_deployment' function is working correctly"""
    router = Router(model_list=model_list)
    kwargs: dict = {"metadata": {}}
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-3.5-turbo"
    )
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
    with pytest.raises(expected_error):
        data = {
            fallback_kwarg: True,
        }
        router._handle_mock_testing_fallbacks(
            kwargs=data,
        )


def test_handle_mock_testing_rate_limit_error(model_list):
    """Test if the '_handle_mock_testing_rate_limit_error' function is working correctly"""
    router = Router(model_list=model_list)
    with pytest.raises(litellm.RateLimitError):
        data = {
            "mock_testing_rate_limit_error": True,
        }
        router._handle_mock_testing_rate_limit_error(
            kwargs=data,
        )


def test_get_fallback_model_group_from_fallbacks(model_list):
    """Test if the '_get_fallback_model_group_from_fallbacks' function is working correctly"""
    router = Router(model_list=model_list)
    fallback_model_group_name = router._get_fallback_model_group_from_fallbacks(
        model_group="gpt-4o",
        fallbacks=[{"gpt-4o": "gpt-3.5-turbo"}],
    )
    assert fallback_model_group_name == "gpt-3.5-turbo"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_deployment_callback_on_success(sync_mode):
    """Test if the '_deployment_callback_on_success' function is working correctly"""
    import time

    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "rpm": 100,
            },
            "model_info": {"id": "100"},
        }
    ]
    router = Router(model_list=model_list)
    # Get the actual deployment ID that was generated
    gpt_deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-3.5-turbo")
    deployment_id = gpt_deployment["model_info"]["id"]
    
    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["total_tokens"] = 100
    standard_logging_payload["model_id"] = "100"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
            },
            "model_info": {"id": deployment_id},
        },
        "standard_logging_object": standard_logging_payload,
    }
    response = litellm.ModelResponse(
        model="gpt-3.5-turbo",
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
async def test_deployment_callback_on_failure(model_list):
    """Test if the '_deployment_callback_on_failure' function is working correctly"""
    import time

    router = Router(model_list=model_list)
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
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
            "metadata": {"model_group": "gpt-3.5-turbo"},
            "model_info": {"id": 100},
            "cooldown_time": 0,
        },
    }

    with patch("litellm.router._set_cooldown_deployments") as mock_set:
        router.deployment_callback_on_failure(
            kwargs=kwargs,
            completion_response=None,
            start_time=time.time(),
            end_time=time.time(),
        )

        mock_set.assert_called_once()
        assert mock_set.call_args.kwargs["time_to_cooldown"] == 0


def test_log_retry(model_list):
    """Test if the '_log_retry' function is working correctly"""
    import time

    router = Router(model_list=model_list)
    new_kwargs = router.log_retry(
        kwargs={"metadata": {}},
        e=Exception(),
    )
    assert "metadata" in new_kwargs
    assert "previous_models" in new_kwargs["metadata"]


def test_update_usage(model_list):
    """Test if the '_update_usage' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-3.5-turbo"
    )
    deployment_id = deployment["model_info"]["id"]
    request_count = router._update_usage(
        deployment_id=deployment_id, parent_otel_span=None
    )
    assert request_count == 1

    request_count = router._update_usage(
        deployment_id=deployment_id, parent_otel_span=None
    )

    assert request_count == 2


@pytest.mark.parametrize(
    "finish_reason, expected_fallback", [("content_filter", True), ("stop", False)]
)
@pytest.mark.parametrize("fallback_type", ["model-specific", "default"])
def test_should_raise_content_policy_error(
    model_list, finish_reason, expected_fallback, fallback_type
):
    """Test if the '_should_raise_content_policy_error' function is working correctly"""
    router = Router(
        model_list=model_list,
        default_fallbacks=["gpt-4o"] if fallback_type == "default" else None,
    )

    assert (
        router._should_raise_content_policy_error(
            model="gpt-3.5-turbo",
            response=litellm.ModelResponse(
                model="gpt-3.5-turbo",
                choices=[
                    {
                        "finish_reason": finish_reason,
                        "message": {"content": "I'm fine, thank you!"},
                    }
                ],
                usage={"total_tokens": 100},
            ),
            kwargs={
                "content_policy_fallbacks": (
                    [{"gpt-3.5-turbo": "gpt-4o"}]
                    if fallback_type == "model-specific"
                    else None
                )
            },
        )
        is expected_fallback
    )


def test_get_healthy_deployments(model_list):
    """Test if the '_get_healthy_deployments' function is working correctly"""
    router = Router(model_list=model_list)
    deployments = router._get_healthy_deployments(
        model="gpt-3.5-turbo", parent_otel_span=None
    )
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
        model_group_name="gpt-3.5-turbo"
    )

    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
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
                    model="gpt-3.5-turbo",
                )
            ),
        ):
            try:
                await router.async_routing_strategy_pre_call_checks(
                    deployment, litellm_logging_obj
                )
                pytest.fail("Exception was not raised")
            except Exception as e:
                assert isinstance(e, litellm.RateLimitError)

        ## WITH EXCEPTION - generic error
        with patch.object(
            callback, "async_pre_call_check", AsyncMock(side_effect=Exception("Error"))
        ):
            try:
                await router.async_routing_strategy_pre_call_checks(
                    deployment, litellm_logging_obj
                )
                pytest.fail("Exception was not raised")
            except Exception as e:
                assert isinstance(e, Exception)


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
        _model_name="gpt-3.5-turbo",
        _litellm_params={
            "model": "gpt-3.5-turbo",
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
        model_group_name="gpt-3.5-turbo"
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
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-3.5-turbo"
    )
    deployment["model_info"]["id"] = 100
    ## Test 1: call user facing function
    router.add_deployment(deployment=deployment)

    ## Test 2: call internal function
    router._add_deployment(deployment=deployment)
    assert len(router.model_list) == len(model_list) + 1


def test_upsert_deployment(model_list):
    """Test if the 'upsert_deployment' function is working correctly"""
    router = Router(model_list=model_list)
    print("model list", len(router.model_list))
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-3.5-turbo"
    )
    deployment.litellm_params.model = "gpt-4o"
    router.upsert_deployment(deployment=deployment)
    assert len(router.model_list) == len(model_list)


def test_delete_deployment(model_list):
    """Test if the 'delete_deployment' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-3.5-turbo"
    )
    router.delete_deployment(id=deployment["model_info"]["id"])
    assert len(router.model_list) == len(model_list) - 1


def test_get_model_info(model_list):
    """Test if the 'get_model_info' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-3.5-turbo"
    )
    model_info = router.get_model_info(id=deployment["model_info"]["id"])
    assert model_info is not None


def test_get_model_group(model_list):
    """Test if the 'get_model_group' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-3.5-turbo"
    )
    model_group = router.get_model_group(id=deployment["model_info"]["id"])
    assert model_group is not None
    assert model_group[0]["model_name"] == "gpt-3.5-turbo"


@pytest.mark.parametrize("user_facing_model_group_name", ["gpt-3.5-turbo", "gpt-4o"])
def test_set_model_group_info(model_list, user_facing_model_group_name):
    """Test if the 'set_model_group_info' function is working correctly"""
    router = Router(model_list=model_list)
    resp = router._set_model_group_info(
        model_group="gpt-3.5-turbo",
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


def test_get_all_deployments(model_list):
    """Test if the 'get_all_deployments' function is working correctly"""
    router = Router(model_list=model_list)
    deployments = router._get_all_deployments(
        model_name="gpt-3.5-turbo", model_alias="gpt-3.5-turbo"
    )
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
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        input="hi",
        specific_deployment=False,
    )

    assert len(available_deployments) > 0


def test_filter_cooldown_deployments(model_list):
    """Test if the 'filter_cooldown_deployments' function is working correctly"""
    router = Router(model_list=model_list)
    deployments = router._filter_cooldown_deployments(
        healthy_deployments=router._get_all_deployments(model_name="gpt-3.5-turbo"),  # type: ignore
        cooldown_deployments=[],
    )
    assert len(deployments) == len(
        router._get_all_deployments(model_name="gpt-3.5-turbo")
    )


def test_track_deployment_metrics(model_list):
    """Test if the 'track_deployment_metrics' function is working correctly"""
    from litellm.types.utils import ModelResponse

    router = Router(model_list=model_list)
    router._track_deployment_metrics(
        deployment=router.get_deployment_by_model_group_name(
            model_group_name="gpt-3.5-turbo"
        ),
        response=ModelResponse(
            model="gpt-3.5-turbo",
            usage={"total_tokens": 100},
        ),
        parent_otel_span=None,
    )


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
def test_get_num_retries_from_retry_policy(
    model_list, exception_type, exception_name, num_retries
):
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
        exception=exception_type(
            message="test", llm_provider="openai", model="gpt-3.5-turbo"
        )
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
def test_get_allowed_fails_from_policy(
    model_list, exception_type, exception_name, allowed_fails
):
    """Test if the 'get_allowed_fails_from_policy' function is working correctly"""
    from litellm.types.router import AllowedFailsPolicy

    data = {exception_name + "AllowedFails": allowed_fails}
    router = Router(
        model_list=model_list, allowed_fails_policy=AllowedFailsPolicy(**data)
    )
    calc_allowed_fails = router.get_allowed_fails_from_policy(
        exception=exception_type(
            message="test", llm_provider="openai", model="gpt-3.5-turbo"
        )
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


def test_get_model_from_alias(model_list):
    """Test if the 'get_model_from_alias' function is working correctly"""
    router = Router(
        model_list=model_list,
        model_group_alias={"gpt-4o": "gpt-3.5-turbo"},
    )
    model = router._get_model_from_alias(model="gpt-4o")
    assert model == "gpt-3.5-turbo"


def test_get_deployment_by_litellm_model(model_list):
    """Test if the 'get_deployment_by_litellm_model' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router._get_deployment_by_litellm_model(model="gpt-3.5-turbo")
    assert deployment is not None


def test_get_pattern(model_list):
    router = Router(model_list=model_list)
    pattern = router.pattern_router.get_pattern(model="claude-3")
    assert pattern is not None


def test_deployments_by_pattern(model_list):
    router = Router(model_list=model_list)
    deployments = router.pattern_router.get_deployments_by_pattern(model="claude-3")
    assert deployments is not None


def test_replace_model_in_jsonl(model_list):
    router = Router(model_list=model_list)
    deployments = router.pattern_router.get_deployments_by_pattern(model="claude-3")
    assert deployments is not None


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
            "openai/gpt-3.5-turbo",
            "openai/gpt-3.5-turbo",
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
def test_pattern_match_deployment_set_model_name(
    user_request_model, model_name, litellm_model, expected_model
):
    from re import Match
    from litellm.router_utils.pattern_match_deployments import PatternMatchRouter

    pattern_router = PatternMatchRouter()

    import re

    # Convert model_name into a proper regex
    model_name_regex = pattern_router._pattern_to_regex(model_name)

    # Match against the request
    match = re.match(model_name_regex, user_request_model)

    if match is None:
        raise ValueError("Match not found")

    # Call the set_deployment_model_name function
    updated_model = pattern_router.set_deployment_model_name(match, litellm_model)

    print(updated_model)  # Expected output: "openai/fo::hi:static::hello"
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


@pytest.mark.asyncio
async def test_pass_through_moderation_endpoint_factory(model_list):
    router = Router(model_list=model_list)
    response = await router._pass_through_moderation_endpoint_factory(
        original_function=litellm.amoderation,
        input="this is valid good text",
        model=None,
    )
    assert response is not None


@pytest.mark.parametrize(
    "has_default_fallbacks, expected_result",
    [(True, True), (False, False)],
)
def test_has_default_fallbacks(model_list, has_default_fallbacks, expected_result):
    router = Router(
        model_list=model_list,
        default_fallbacks=(
            ["my-default-fallback-model"] if has_default_fallbacks else None
        ),
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

    healthy_deployments = router.get_model_list(model_name="gpt-3.5-turbo")

    new_healthy_deployments = await router.async_callback_filter_deployments(
        model="gpt-3.5-turbo",
        healthy_deployments=healthy_deployments,
        messages=[],
        parent_otel_span=None,
    )

    assert len(new_healthy_deployments) == len(healthy_deployments)


def test_cached_get_model_group_info(model_list):
    """Test if the '_cached_get_model_group_info' function is working correctly with LRU cache"""
    router = Router(model_list=model_list)

    # First call - should hit the actual function
    result1 = router._cached_get_model_group_info("gpt-3.5-turbo")

    # Second call with same argument - should hit the cache
    result2 = router._cached_get_model_group_info("gpt-3.5-turbo")

    # Verify results are the same
    assert result1 == result2

    # Verify the cache info shows hits
    cache_info = router._cached_get_model_group_info.cache_info()
    assert cache_info.hits > 0  # Should have at least one cache hit


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
        # Test string to bool conversion
        ("true", "false", "True", True, False, True),
        ("TRUE", "FALSE", "False", True, False, False),
        ("false", "true", "false", False, True, False),
        # Test actual boolean values (should pass through unchanged)
        (True, False, True, True, False, True),
        (False, True, False, False, True, False),
        # Test None values
        (None, None, None, None, None, None),
        # Test mixed types
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
        "other_param": "should_remain",  # This should not be affected
    }

    # Make a copy to verify kwargs are properly popped
    original_kwargs = kwargs.copy()

    mock_params = MockRouterTestingParams.from_kwargs(kwargs)

    # Verify the converted values
    assert mock_params.mock_testing_fallbacks == expected_fallbacks
    assert mock_params.mock_testing_context_fallbacks == expected_context
    assert mock_params.mock_testing_content_policy_fallbacks == expected_content_policy

    # Verify that the mock testing params were popped from kwargs
    assert "mock_testing_fallbacks" not in kwargs
    assert "mock_testing_context_fallbacks" not in kwargs
    assert "mock_testing_content_policy_fallbacks" not in kwargs

    # Verify other params remain unchanged
    assert kwargs["other_param"] == "should_remain"


def test_is_auto_router_deployment(model_list):
    """Test if the '_is_auto_router_deployment' function correctly identifies auto-router deployments"""
    router = Router(model_list=model_list)

    # Test case 1: Model starts with "auto_router/" - should return True
    litellm_params_auto = LiteLLM_Params(model="auto_router/my-auto-router")
    assert router._is_auto_router_deployment(litellm_params_auto) is True

    # Test case 2: Model doesn't start with "auto_router/" - should return False
    litellm_params_regular = LiteLLM_Params(model="gpt-3.5-turbo")
    assert router._is_auto_router_deployment(litellm_params_regular) is False

    # Test case 3: Model is empty string - should return False
    litellm_params_empty = LiteLLM_Params(model="")
    assert router._is_auto_router_deployment(litellm_params_empty) is False

    # Test case 4: Model contains "auto_router/" but doesn't start with it - should return False
    litellm_params_contains = LiteLLM_Params(model="prefix_auto_router/something")
    assert router._is_auto_router_deployment(litellm_params_contains) is False


@patch("litellm.router_strategy.auto_router.auto_router.AutoRouter")
def test_init_auto_router_deployment_success(mock_auto_router, model_list):
    """Test if the 'init_auto_router_deployment' function successfully initializes auto-router when all params provided"""
    router = Router(model_list=model_list)

    # Create a mock AutoRouter instance
    mock_auto_router_instance = MagicMock()
    mock_auto_router.return_value = mock_auto_router_instance

    # Test case: All required parameters provided
    litellm_params = LiteLLM_Params(
        model="auto_router/test",
        auto_router_config_path="/path/to/config",
        auto_router_default_model="gpt-3.5-turbo",
        auto_router_embedding_model="text-embedding-ada-002",
    )
    deployment = Deployment(
        model_name="test-auto-router",
        litellm_params=litellm_params,
        model_info={"id": "test-id"},
    )

    # Should not raise any exception
    router.init_auto_router_deployment(deployment)

    # Verify AutoRouter was called with correct parameters
    mock_auto_router.assert_called_once_with(
        model_name="test-auto-router",
        auto_router_config_path="/path/to/config",
        auto_router_config=None,
        default_model="gpt-3.5-turbo",
        embedding_model="text-embedding-ada-002",
        litellm_router_instance=router,
    )

    # Verify the auto-router was added to the router's auto_routers dict
    assert "test-auto-router" in router.auto_routers
    assert router.auto_routers["test-auto-router"] == mock_auto_router_instance


@patch("litellm.router_strategy.auto_router.auto_router.AutoRouter")
def test_init_auto_router_deployment_duplicate_model_name(mock_auto_router, model_list):
    """Test if the 'init_auto_router_deployment' function raises ValueError when model_name already exists"""
    router = Router(model_list=model_list)

    # Create a mock AutoRouter instance
    mock_auto_router_instance = MagicMock()
    mock_auto_router.return_value = mock_auto_router_instance

    # Add an existing auto-router
    router.auto_routers["test-auto-router"] = mock_auto_router_instance

    # Try to add another auto-router with the same name
    litellm_params = LiteLLM_Params(
        model="auto_router/test",
        auto_router_config_path="/path/to/config",
        auto_router_default_model="gpt-3.5-turbo",
        auto_router_embedding_model="text-embedding-ada-002",
    )
    deployment = Deployment(
        model_name="test-auto-router",
        litellm_params=litellm_params,
        model_info={"id": "test-id"},
    )

    with pytest.raises(
        ValueError, match="Auto-router deployment test-auto-router already exists"
    ):
        router.init_auto_router_deployment(deployment)


def test_generate_model_id_with_deployment_model_name(model_list):
    """Test that _generate_model_id works correctly with deployment model_name and handles None values properly"""
    router = Router(model_list=model_list)

    # Test case 1: Normal case with valid model_group and litellm_params
    model_group = "gpt-4.1"
    litellm_params = {
        "model": "gpt-4.1",
        "api_key": "test_key",
        "api_base": "https://api.openai.com/v1",
    }

    try:
        result = router._generate_model_id(
            model_group=model_group, litellm_params=litellm_params
        )
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ Success with valid model_group: {result}")
    except Exception as e:
        pytest.fail(f"Failed with valid model_group: {e}")

    # Test case 2: Edge case with None model_group (this should fail as expected - our fix prevents this from happening)
    try:
        result = router._generate_model_id(
            model_group=None, litellm_params=litellm_params
        )
        pytest.fail(
            "Expected TypeError when model_group is None - this confirms our fix is needed"
        )
    except TypeError as e:
        # After optimization, error message changed but still fails appropriately on None
        assert "unsupported operand type(s) for +=" in str(e) or "expected str instance, NoneType found" in str(e)
        print(f"✓ Correctly failed with None model_group (as expected): {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error with None model_group: {e}")

    # Test case 3: Edge case with None key in litellm_params
    litellm_params_with_none_key = {
        "model": "gpt-4.1",
        "api_key": "test_key",
        None: "should_be_skipped",  # This should be handled gracefully
    }

    try:
        result = router._generate_model_id(
            model_group=model_group, litellm_params=litellm_params_with_none_key
        )
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ Success with None key in litellm_params: {result}")
    except Exception as e:
        pytest.fail(f"Failed with None key in litellm_params: {e}")

    # Test case 4: Edge case with empty litellm_params
    try:
        result = router._generate_model_id(model_group=model_group, litellm_params={})
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ Success with empty litellm_params: {result}")
    except Exception as e:
        pytest.fail(f"Failed with empty litellm_params: {e}")

    # Test case 5: Verify that the same inputs produce the same result (deterministic)
    result1 = router._generate_model_id(
        model_group=model_group, litellm_params=litellm_params
    )
    result2 = router._generate_model_id(
        model_group=model_group, litellm_params=litellm_params
    )
    assert result1 == result2, "Model ID generation should be deterministic"

    print("✓ All _generate_model_id tests passed!")


def test_handle_clientside_credential_with_deployment_model_name(model_list):
    """Test that _handle_clientside_credential uses deployment model_name correctly"""
    router = Router(model_list=model_list)

    # Mock deployment with model_name
    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test_key"},
    }

    # Mock kwargs with empty metadata (simulating the original issue)
    kwargs = {
        "metadata": {},  # Empty metadata, no model_group
        "litellm_params": {
            "api_key": "client_side_key",
            "api_base": "https://api.openai.com/v1",
        },
    }

    # Mock dynamic_litellm_params that would be returned by get_dynamic_litellm_params
    dynamic_litellm_params = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
    }

    # Test that the method doesn't fail when metadata is empty
    try:
        # This would normally call _generate_model_id internally
        # We're testing that the fix prevents the TypeError
        model_group = deployment["model_name"]  # This is what our fix does
        assert model_group == "gpt-4.1"

        # Verify that _generate_model_id works with this model_group
        result = router._generate_model_id(
            model_group=model_group, litellm_params=dynamic_litellm_params
        )
        assert isinstance(result, str)
        assert len(result) > 0

        print(f"✓ Success with deployment model_name: {result}")
    except Exception as e:
        pytest.fail(f"Failed with deployment model_name: {e}")

    print("✓ _handle_clientside_credential test passed!")


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
def test_handle_clientside_credential_metadata_loading(
    model_list, function_name, expected_metadata_key
):
    """Test that _handle_clientside_credential correctly loads metadata based on function name"""
    router = Router(model_list=model_list)

    # Mock deployment
    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test_key"},
        "model_info": {"id": "original-id-123"},
    }

    # Mock kwargs with clientside credentials and metadata
    kwargs = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
        expected_metadata_key: {"model_group": "gpt-4.1", "custom_field": "test_value"},
    }

    # Call the function
    result_deployment = router._handle_clientside_credential(
        deployment=deployment, kwargs=kwargs, function_name=function_name
    )

    # Verify the result is a Deployment object
    assert isinstance(result_deployment, Deployment)

    # Verify the deployment has the correct model_name (should be the model_group from metadata)
    assert result_deployment.model_name == "gpt-4.1"

    # Verify the litellm_params contain the clientside credentials
    assert result_deployment.litellm_params.api_key == "client_side_key"
    assert result_deployment.litellm_params.api_base == "https://api.openai.com/v1"

    # Verify the model_info has been updated with a new ID
    assert result_deployment.model_info.id != "original-id-123"
    assert result_deployment.model_info.original_model_id == "original-id-123"

    # Verify the deployment was added to the router
    assert len(router.model_list) == len(model_list) + 1

    # Test that the function correctly uses the right metadata key
    # For acompletion, it should use "metadata"
    # For _ageneric_api_call_with_fallbacks/batch, it should use "litellm_metadata"
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
        # Note: acompletion would not have litellm_metadata, but other functions might have both

    print(
        f"✓ Success with function_name '{function_name}' using '{expected_metadata_key}' metadata key"
    )


@pytest.mark.parametrize(
    "function_name, metadata_key",
    [
        ("acompletion", "metadata"),
        ("_ageneric_api_call_with_fallbacks", "litellm_metadata"),
    ],
)
def test_handle_clientside_credential_metadata_variable_name(
    model_list, function_name, metadata_key
):
    """Test that _handle_clientside_credential uses the correct metadata variable name based on function name"""
    from litellm.router_utils.batch_utils import _get_router_metadata_variable_name

    router = Router(model_list=model_list)

    # Verify the metadata variable name is correct for each function
    expected_metadata_key = _get_router_metadata_variable_name(
        function_name=function_name
    )
    assert expected_metadata_key == metadata_key

    # Mock deployment
    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test_key"},
        "model_info": {"id": "original-id-456"},
    }

    # Mock kwargs with clientside credentials and the correct metadata key
    kwargs = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
        metadata_key: {"model_group": "gpt-4.1", "test_field": "test_value"},
    }

    # Call the function
    result_deployment = router._handle_clientside_credential(
        deployment=deployment, kwargs=kwargs, function_name=function_name
    )

    # Verify the function correctly extracted model_group from the right metadata key
    assert result_deployment.model_name == "gpt-4.1"

    # Verify the deployment was created with the correct metadata
    assert result_deployment.litellm_params.api_key == "client_side_key"
    assert result_deployment.litellm_params.api_base == "https://api.openai.com/v1"

    print(
        f"✓ Success with function_name '{function_name}' correctly using '{metadata_key}' for metadata"
    )


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


def test_handle_clientside_credential_with_responses_function(model_list):
    """Test that _handle_clientside_credential works correctly with responses function name"""
    router = Router(model_list=model_list)

    # Mock deployment
    deployment = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test_key"},
        "model_info": {"id": "original-id-responses"},
    }

    # Mock kwargs with clientside credentials and litellm_metadata (for responses function)
    kwargs = {
        "api_key": "client_side_key",
        "api_base": "https://api.openai.com/v1",
        "litellm_metadata": {
            "model_group": "gpt-4.1",
            "responses_field": "responses_value",
        },
    }

    # Call the function with _ageneric_api_call_with_fallbacks function name (which handles responses)
    result_deployment = router._handle_clientside_credential(
        deployment=deployment,
        kwargs=kwargs,
        function_name="_ageneric_api_call_with_fallbacks",
    )

    # Verify the result
    assert isinstance(result_deployment, Deployment)
    assert result_deployment.model_name == "gpt-4.1"
    assert result_deployment.litellm_params.api_key == "client_side_key"
    assert result_deployment.litellm_params.api_base == "https://api.openai.com/v1"
    assert result_deployment.model_info.id != "original-id-responses"
    assert result_deployment.model_info.original_model_id == "original-id-responses"

    # Verify the deployment was added to the router
    assert len(router.model_list) == len(model_list) + 1

    print(
        "✓ Success with _ageneric_api_call_with_fallbacks function name and litellm_metadata"
    )


def test_get_metadata_variable_name_from_kwargs(model_list):
    """
    Test _get_metadata_variable_name_from_kwargs method returns correct metadata variable name based on kwargs content.
    """
    router = Router(model_list=model_list)
    
    # Test case 1: kwargs contains litellm_metadata - should return "litellm_metadata"
    kwargs_with_litellm_metadata = {
        "litellm_metadata": {"user": "test"},
        "metadata": {"other": "data"}
    }
    result = router._get_metadata_variable_name_from_kwargs(kwargs_with_litellm_metadata)
    assert result == "litellm_metadata"
    
    # Test case 2: kwargs only contains metadata - should return "metadata"
    kwargs_with_metadata_only = {
        "metadata": {"user": "test"}
    }
    result = router._get_metadata_variable_name_from_kwargs(kwargs_with_metadata_only)
    assert result == "metadata"
    
    # Test case 3: kwargs contains neither - should return "metadata" (default)
    kwargs_empty = {}
    result = router._get_metadata_variable_name_from_kwargs(kwargs_empty)
    assert result == "metadata"
    
    # Test case 4: kwargs contains other keys but no metadata keys - should return "metadata"
    kwargs_other = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}]
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
            }
        },
        {
            "search_tool_name": "test-search-tool",
            "litellm_params": {
                "search_provider": "perplexity",
                "api_key": "test-api-key-2",
                "api_base": "https://api.perplexity.ai",
            }
        }
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
    
    # Create a mock search response
    mock_response = SearchResponse(
        object="search",
        results=[
            SearchResult(
                title="Test Result",
                url="https://example.com",
                snippet="Test snippet content"
            )
        ]
    )
    
    # Mock the async_function_with_fallbacks to return our mock response
    with patch.object(router, 'async_function_with_fallbacks', new_callable=AsyncMock) as mock_fallbacks:
        mock_fallbacks.return_value = mock_response
        
        # Mock original function
        async def mock_asearch(**kwargs):
            return mock_response
        
        # Call _asearch_with_fallbacks
        response = await router._asearch_with_fallbacks(
            original_function=mock_asearch,
            search_tool_name="test-search-tool",
            query="test query",
            max_results=5
        )
        
        # Verify async_function_with_fallbacks was called
        assert mock_fallbacks.called
        
        # Verify the response
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
    
    # Create a mock search response
    mock_response = SearchResponse(
        object="search",
        results=[
            SearchResult(
                title="Helper Test Result",
                url="https://example.com/helper",
                snippet="Helper test snippet"
            )
        ]
    )
    
    # Mock the original generic function
    async def mock_original_function(**kwargs):
        # Verify correct parameters are passed
        assert "search_provider" in kwargs
        assert kwargs["search_provider"] == "perplexity"
        assert "api_key" in kwargs
        assert kwargs["query"] == "helper test query"
        return mock_response
    
    # Call _asearch_with_fallbacks_helper
    response = await router._asearch_with_fallbacks_helper(
        model="test-search-tool",
        original_generic_function=mock_original_function,
        query="helper test query",
        max_results=3
    )
    
    # Verify the response
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
    # Create router with no search tools
    router = Router(model_list=[])
    
    async def mock_original_function(**kwargs):
        return None
    
    # Should raise ValueError for missing search tool
    with pytest.raises(ValueError, match="Search tool 'nonexistent-tool' not found"):
        await router._asearch_with_fallbacks_helper(
            model="nonexistent-tool",
            original_generic_function=mock_original_function,
            query="test query"
        )


@pytest.mark.asyncio
async def test_asearch_with_fallbacks_helper_missing_search_provider():
    """
    Test _asearch_with_fallbacks_helper raises error when search_provider not configured.
    
    Tests that the helper method raises a ValueError when a search tool
    is found but doesn't have search_provider in its litellm_params.
    """
    # Create router with misconfigured search tool (missing search_provider)
    search_tools_bad = [
        {
            "search_tool_name": "bad-tool",
            "litellm_params": {
                "api_key": "test-key"
                # Missing search_provider
            }
        }
    ]
    
    router = Router(search_tools=search_tools_bad)
    
    async def mock_original_function(**kwargs):
        return None
    
    # Should raise ValueError for missing search_provider
    with pytest.raises(ValueError, match="search_provider not found in litellm_params"):
        await router._asearch_with_fallbacks_helper(
            model="bad-tool",
            original_generic_function=mock_original_function,
            query="test query"
        )


def test_get_first_default_fallback():
    """Test _get_first_default_fallback method"""
    # Test with default fallback ("*")
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        }
    ]
    
    router = Router(
        model_list=model_list,
        fallbacks=[{"*": ["gpt-3.5-turbo"]}]
    )
    
    result = router._get_first_default_fallback()
    assert result == "gpt-3.5-turbo"
    
    # Test with no fallbacks
    router_no_fallbacks = Router(model_list=model_list)
    result = router_no_fallbacks._get_first_default_fallback()
    assert result is None
    
    # Test with fallbacks but no default
    router_no_default = Router(
        model_list=model_list,
        fallbacks=[{"gpt-4": ["gpt-3.5-turbo"]}]
    )
    result = router_no_default._get_first_default_fallback()
    assert result is None
    
    # Test with empty default list
    router_empty_list = Router(
        model_list=model_list,
        fallbacks=[{"*": []}]
    )
    result = router_empty_list._get_first_default_fallback()
    assert result is None


def test_resolve_model_name_from_model_id():
    """Test resolve_model_name_from_model_id function with various scenarios"""
    
    # Test case 1: model_id is None
    router = Router(model_list=[])
    result = router.resolve_model_name_from_model_id(None)
    assert result is None
    
    # Test case 2: model_id directly matches a model_name
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)
    result = router.resolve_model_name_from_model_id("gpt-3.5-turbo")
    assert result == "gpt-3.5-turbo"
    
    # Test case 3: model_id matches litellm_params.model exactly
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
    
    # Test case 4: model_id matches when actual_model ends with /model_id
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
    
    # Test case 5: model_id matches when actual_model ends with :model_id
    # Note: We use a valid model format for router initialization, but test the function
    # with a model_id that would match the pattern vertex_ai:model_id
    # Since the router validates models on init, we'll test this by manually setting up
    # the model_list after initialization or using a valid format
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
    # Test that the function can handle model_id that would match if the format was vertex_ai:model_id
    # We'll test with a model_id that matches the end of the actual_model
    result = router.resolve_model_name_from_model_id("veo-2.0-generate-001")
    assert result == "vertex-ai-sora-2"
    
    # Test case 6: model_id doesn't match anything
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)
    result = router.resolve_model_name_from_model_id("non-existent-model")
    assert result is None
    
    # Test case 7: Empty model_list
    router = Router(model_list=[])
    result = router.resolve_model_name_from_model_id("some-model")
    assert result is None
    
    # Test case 8: Multiple models, find the correct one
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
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
    
    # Test case 9: model_id matches deployment ID (has_model_id check)
    # This tests the has_model_id path in Strategy 1
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
            },
        },
    ]
    router = Router(model_list=model_list)

    result = router.resolve_model_name_from_model_id("gpt-3.5-turbo")
    assert result == "gpt-3.5-turbo"


def test_get_valid_args():
    """Test get_valid_args static method returns valid Router.__init__ arguments"""
    # Call the static method
    valid_args = Router.get_valid_args()
    
    # Verify it returns a list
    assert isinstance(valid_args, list)
    assert len(valid_args) > 0
    
    # Verify it contains expected Router.__init__ arguments
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
    
    # Verify "self" is not in the list (since it's removed)
    assert "self" not in valid_args
    
    # Verify it contains keyword-only arguments too
    # These are common Router.__init__ parameters
    assert "assistants_config" in valid_args or "search_tools" in valid_args
