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


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
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
    for strategy in RoutingStrategy._member_names_:
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
async def test_router_arealtime(model_list):
    """Test if the '_arealtime' function is working correctly"""
    import litellm

    router = Router(model_list=model_list)
    with patch.object(litellm, "_arealtime", AsyncMock()) as mock_arealtime:
        mock_arealtime.return_value = "I'm fine, thank you!"
        await router._arealtime(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        mock_arealtime.assert_awaited_once()


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
    if sync_mode:
        response = router.function_with_retries(
            original_function=router._completion,
            **data,
        )
    else:
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


def test_get_async_openai_model_client(model_list):
    """Test if the '_get_async_openai_model_client' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-3.5-turbo"
    )
    model_client = router._get_async_openai_model_client(
        deployment=deployment, kwargs={}
    )
    assert model_client is not None


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
async def test_deployment_callback_on_success(model_list, sync_mode):
    """Test if the '_deployment_callback_on_success' function is working correctly"""
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


def test_deployment_callback_on_failure(model_list):
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
        original_function=litellm.amoderation, input="this is valid good text"
    )


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
