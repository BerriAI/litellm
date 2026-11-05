import asyncio
import gc
import json
import os
import time
import weakref
from collections.abc import Awaitable, Callable, Iterator, Mapping
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from pydantic import BaseModel

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS, ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook
from litellm.types.litellm_params import RoutingStrategyName
from litellm.types.router import AlertingConfig, Deployment, DeploymentTypedDict, LiteLLM_Params, ModelInfo
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
    gpt_deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")
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
async def test_deployment_callback_on_success_counts_a_responses_websocket_turn():
    import time

    model_list = [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-fake", "rpm": 100},
            "model_info": {"id": "ws-deployment"},
        }
    ]
    router = Router(model_list=model_list)
    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["total_tokens"] = 17

    def turn_kwargs(call_type: str) -> dict:
        return {
            "call_type": call_type,
            "litellm_params": {
                "metadata": {"model_group": "gpt-5-mini"},
                "model_info": {"id": "ws-deployment"},
            },
            "standard_logging_object": standard_logging_payload,
        }

    tpm_key = await router.deployment_callback_on_success(
        kwargs=turn_kwargs("_aresponses_websocket"),
        completion_response=None,
        start_time=time.time(),
        end_time=time.time(),
    )
    assert tpm_key is not None
    assert await router.cache.async_get_cache(key=tpm_key) == 17

    realtime_key = await router.deployment_callback_on_success(
        kwargs=turn_kwargs("_arealtime"),
        completion_response=None,
        start_time=time.time(),
        end_time=time.time(),
    )
    assert realtime_key is None
    assert await router.cache.async_get_cache(key=tpm_key) == 17


def _io_limited_router(kind: str, **litellm_params: object) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "ws-io",
                "litellm_params": {
                    "model": "openai/ws-io-unpriced",
                    "api_key": "sk-fake",
                    kind: 10_000,
                    **litellm_params,
                },
                "model_info": {"id": "ws-io-dep"},
            }
        ],
        optional_pre_call_checks=["enforce_model_rate_limits"],
    )


def _minutes_counter_key(kind: str, minute: datetime, model: str = "openai/ws-io-unpriced") -> str:
    return f"global_router:ws-io-dep:{model}:{kind}:{minute.strftime('%H-%M')}"


_PROVIDER_RESPONSE: Final = {
    "id": "resp_io",
    "object": "response",
    "created_at": 1767268830,
    "status": "completed",
    "model": "ws-io-unpriced",
    "output": [
        {
            "type": "message",
            "id": "msg_io",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_metadata", [None, {"session": "abc", "team": "search"}], ids=["without_metadata", "with_metadata"]
)
async def test_responses_call_sends_the_provider_only_the_client_metadata_and_settles_deployment_io_tokens(
    client_metadata: Mapping[str, str] | None,
    router_minute_pinned: datetime,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
):
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.responses.streaming_iterator import drain_logging_tasks

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    provider: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_PROVIDER_RESPONSE)
    )
    router: Final = _io_limited_router("itpm", otpm=10_000)

    await router.aresponses(
        model="ws-io",
        input="hi",
        max_output_tokens=50,
        **({} if client_metadata is None else {"metadata": dict(client_metadata)}),
    )
    await drain_logging_tasks()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)

    assert [json.loads(call.request.read()).get("metadata") for call in provider.calls] == [client_metadata]
    assert [
        await router.cache.async_get_cache(key=_minutes_counter_key(kind, router_minute_pinned))
        for kind in ("itpm", "otpm")
    ] == [10, 5]


@pytest.mark.parametrize(
    "client_metadata", [None, {"session": "abc", "team": "search"}], ids=["without_metadata", "with_metadata"]
)
def test_sync_responses_call_sends_the_provider_only_the_client_metadata(
    client_metadata: Mapping[str, str] | None, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    litellm.in_memory_llm_clients_cache.flush_cache()
    provider: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_PROVIDER_RESPONSE)
    )

    _io_limited_router("itpm", otpm=10_000).responses(
        model="ws-io",
        input="hi",
        max_output_tokens=50,
        **({} if client_metadata is None else {"metadata": dict(client_metadata)}),
    )

    assert [json.loads(call.request.read()).get("metadata") for call in provider.calls] == [client_metadata]


_ANTHROPIC_PROVIDER_RESPONSE: Final = {
    "id": "msg_io",
    "type": "message",
    "role": "assistant",
    "model": "ws-io-unpriced",
    "content": [{"type": "text", "text": "Done."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


async def _responses_call(router: Router) -> object:
    return await router.aresponses(model="ws-io", input="hi", max_output_tokens=50)


async def _messages_call(router: Router) -> object:
    return await router.aanthropic_messages(model="ws-io", messages=[{"role": "user", "content": "hi"}], max_tokens=50)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "provider_url", "provider_response", "call"),
    [
        pytest.param(
            "openai/ws-io-unpriced",
            "https://api.openai.com/v1/responses",
            _PROVIDER_RESPONSE,
            _responses_call,
            id="responses",
        ),
        pytest.param(
            "anthropic/ws-io-unpriced",
            "https://api.anthropic.com/v1/messages",
            _ANTHROPIC_PROVIDER_RESPONSE,
            _messages_call,
            id="messages",
        ),
    ],
)
async def test_retried_call_settles_deployment_io_tokens_to_the_successful_attempt_s_tokens(
    model: str,
    provider_url: str,
    provider_response: Mapping[str, object],
    call: Callable[[Router], Awaitable[object]],
    router_minute_pinned: datetime,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
):
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.responses.streaming_iterator import drain_logging_tasks

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    provider: Final = respx_mock.post(provider_url).mock(
        side_effect=[
            httpx.Response(500, json={"error": {"message": "upstream failed", "type": "server_error"}}),
            httpx.Response(200, json=provider_response),
        ]
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "ws-io",
                "litellm_params": {"model": model, "api_key": "sk-fake", "itpm": 10_000, "otpm": 10_000},
                "model_info": {"id": "ws-io-dep"},
            }
        ],
        optional_pre_call_checks=["enforce_model_rate_limits"],
        num_retries=1,
        retry_after=0,
    )

    await call(router)
    await drain_logging_tasks()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)

    assert provider.call_count == 2
    assert [
        await router.cache.async_get_cache(key=_minutes_counter_key(kind, router_minute_pinned, model))
        for kind in ("itpm", "otpm")
    ] == [10, 5]


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
            metadata_key: {
                "model_info": {"id": "deployment-1", "base_model": "azure/gpt-3.5-turbo"},
                "deployment": "azure/my-35-deployment",
                "attempted_retries": 2,
                "user_api_key": "sk-proxy",
            },
        },
        e=rate_limit_error,
    )
    assert json.loads(json.dumps(new_kwargs[metadata_key]["previous_models"])) == [
        {
            "model_group": "gpt-3.5-turbo",
            "deployment_id": "deployment-1",
            "model": "azure/my-35-deployment",
            "base_model": "azure/gpt-3.5-turbo",
            "exception_type": "RateLimitError",
            "exception_string": "litellm.RateLimitError: slow down",
            "attempted_retries": 2,
        }
    ]
    assert new_kwargs[metadata_key]["request_retry_count"] == 1
    assert router.log_retry(kwargs=new_kwargs, e=rate_limit_error)[metadata_key]["request_retry_count"] == 2
    planted_kwargs = {"model": "gpt-3.5-turbo", metadata_key: {"request_retry_count": -100}}
    assert router.log_retry(kwargs=planted_kwargs, e=rate_limit_error)[metadata_key]["request_retry_count"] == 1


def test_log_retry_reads_base_model_from_the_deployments_litellm_params() -> None:
    """An Azure named deployment declares ``base_model`` under ``litellm_params``, not ``model_info``, and
    that is the model the failure row is priced at, so the retry record has to carry it too"""
    router = Router(
        model_list=[
            {
                "model_name": "azure-group",
                "litellm_params": {
                    "model": "azure/my-named-deployment",
                    "api_key": "sk-fake",
                    "api_base": "https://example.openai.azure.com",
                    "base_model": "azure/gpt-4o-mini",
                },
                "model_info": {"id": "dep-azure-named"},
            }
        ]
    )
    new_kwargs = router.log_retry(
        kwargs={
            "model": "azure-group",
            "metadata": {
                "model_info": {"id": "dep-azure-named"},
                "deployment": "azure/my-named-deployment",
                "attempted_retries": 0,
            },
        },
        e=litellm.Timeout(message="slow", llm_provider="azure", model="azure/my-named-deployment"),
    )
    record = new_kwargs["metadata"]["previous_models"][0]
    assert (record["model"], record["base_model"]) == ("azure/my-named-deployment", "azure/gpt-4o-mini")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_routing_strategy_pre_call_checks(model_list, sync_mode):
    """Test if the '_routing_strategy_pre_call_checks' function is working correctly"""
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.litellm_logging import Logging

    callback = CustomLogger()
    litellm.callbacks = [callback]

    router = Router(model_list=model_list)

    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")

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
        await router.async_routing_strategy_pre_call_checks(deployment, litellm_logging_obj)

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
                await router.async_routing_strategy_pre_call_checks(deployment, litellm_logging_obj)

        ## WITH EXCEPTION - generic error
        with patch.object(callback, "async_pre_call_check", AsyncMock(side_effect=Exception("Error"))):
            with pytest.raises(Exception, match="Error"):
                await router.async_routing_strategy_pre_call_checks(deployment, litellm_logging_obj)


@pytest.mark.parametrize(
    "set_supported_environments, supported_environments, is_supported",
    [(True, ["staging"], True), (False, None, True), (True, ["development"], False)],
)
def test_deployment_is_active_for_environment(
    model_list, set_supported_environments, supported_environments, is_supported
):
    """Test if the '_deployment_is_active_for_environment' function is working correctly"""
    router = Router(model_list=model_list)
    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-5-mini")
    if set_supported_environments:
        os.environ["LITELLM_ENVIRONMENT"] = "staging"
    deployment["model_info"]["supported_environments"] = supported_environments
    if is_supported:
        assert router.deployment_is_active_for_environment(deployment=deployment) is True
    else:
        assert router.deployment_is_active_for_environment(deployment=deployment) is False


def test_set_model_list(model_list):
    """Test if the '_set_model_list' function is working correctly"""
    router = Router(model_list=model_list)
    router.set_model_list(model_list=model_list)
    assert len(router.model_list) == len(model_list)


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


@pytest.mark.asyncio
async def test_set_response_headers(model_list):
    """Test if the 'set_response_headers' function is working correctly"""
    router = Router(model_list=model_list)
    resp = await router.set_response_headers(response=None, model_group=None)
    assert resp is None


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


def _pin_router_clock(monkeypatch: pytest.MonkeyPatch, pinned: datetime) -> None:
    monkeypatch.setattr(
        "litellm.router_utils.pre_call_checks.io_token_rate_limit_check.get_utc_datetime", lambda: pinned
    )


@pytest.fixture
def router_minute_pinned(monkeypatch: pytest.MonkeyPatch) -> datetime:
    # The router's own usage counters read the real clock: start well inside a minute so a test never spans two
    seconds_left: Final = 60 - datetime.now(timezone.utc).second
    if seconds_left < 10:
        time.sleep(seconds_left + 0.1)
    pinned: Final = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    _pin_router_clock(monkeypatch, pinned)
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
@pytest.mark.usefixtures("router_minute_pinned")
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
            RouterCacheEnum.ITPM.value.format(id="io-usage-dep-1", model="openai/gpt-4o-mini", current_minute=minute),
            30,
        ),
        (
            RouterCacheEnum.OTPM.value.format(id="io-usage-dep-1", model="openai/gpt-4o-mini", current_minute=minute),
            10,
        ),
        (
            RouterCacheEnum.ITPM.value.format(id="io-usage-dep-2", model="openai/gpt-4o", current_minute=minute),
            70,
        ),
        (
            RouterCacheEnum.OTPM.value.format(id="io-usage-dep-2", model="openai/gpt-4o", current_minute=minute),
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


def _limited_router(**io_limits: int) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "limited",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-fake",
                    "tpm": 1000,
                    "rpm": 100,
                    **io_limits,
                },
            }
        ],
        optional_pre_call_checks=["enforce_model_rate_limits"],
    )


@pytest.mark.asyncio
async def test_get_remaining_model_group_usage_merges_io_and_tpm_headers():
    """
    A model group with both itpm/otpm and tpm/rpm limits must expose the
    standard remaining-tokens/requests headers alongside the input/output token
    headers, so clients and prometheus gauges relying on either still get data.
    """
    router = _limited_router(itpm=900, otpm=450)

    headers = await router.get_remaining_model_group_usage("limited")

    assert headers["x-ratelimit-remaining-input-tokens"] == 900
    assert headers["x-ratelimit-remaining-output-tokens"] == 450
    assert headers["x-ratelimit-remaining-tokens"] == 1000
    assert headers["x-ratelimit-remaining-requests"] == 100


class _HeaderlessResponse(BaseModel):
    _hidden_params: dict = {}


@pytest.mark.asyncio
async def test_set_response_headers_native_input_token_header_does_not_suppress_router_headers():
    """
    A provider that natively returns `x-ratelimit-remaining-input-tokens` must
    not suppress the router's own remaining-tokens/requests headers for a
    non-IO model group.
    """
    router = _limited_router()

    resp = _HeaderlessResponse()
    resp._hidden_params = {"additional_headers": {"x-ratelimit-remaining-input-tokens": 5}}
    await router.set_response_headers(response=resp, model_group="limited")

    headers = resp._hidden_params["additional_headers"]
    assert headers["x-ratelimit-remaining-tokens"] == 1000
    assert headers["x-ratelimit-remaining-requests"] == 100

    assert headers["x-ratelimit-remaining-input-tokens"] == 5


@pytest.mark.asyncio
async def test_set_response_headers_native_token_header_does_not_suppress_io_headers():
    router = _limited_router(itpm=900, otpm=450)

    resp = _HeaderlessResponse()
    resp._hidden_params = {"additional_headers": {"x-ratelimit-remaining-tokens": 5}}
    await router.set_response_headers(response=resp, model_group="limited")

    headers = resp._hidden_params["additional_headers"]
    assert headers["x-ratelimit-remaining-tokens"] == 5
    assert headers["x-ratelimit-remaining-requests"] == 100
    assert headers["x-ratelimit-remaining-input-tokens"] == 900
    assert headers["x-ratelimit-remaining-output-tokens"] == 450


@pytest.mark.asyncio
async def test_set_response_headers_handles_missing_usage():
    """
    Streaming chunks and some response shapes may lack a `usage` attribute or
    populated `total_tokens`. Header composition must not depend on usage and never raise.
    """
    router = _limited_router()

    resp = _HeaderlessResponse()
    resp._hidden_params = {}
    await router.set_response_headers(response=resp, model_group="limited")

    headers = resp._hidden_params["additional_headers"]
    assert headers["x-ratelimit-remaining-tokens"] == 1000
    assert headers["x-ratelimit-remaining-requests"] == 100


@pytest.mark.asyncio
async def test_set_response_headers_dict_anthropic_messages_response():
    """Anthropic /v1/messages returns a dict; IO rate-limit headers must attach."""
    router = _limited_router(itpm=25, otpm=100)

    resp = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    await router.set_response_headers(response=resp, model_group="limited")

    assert "_hidden_params" in resp
    headers = resp["_hidden_params"]["additional_headers"]
    assert headers["x-litellm-model-group"] == "limited"
    assert headers["x-ratelimit-limit-input-tokens"] == 25
    assert headers["x-ratelimit-remaining-input-tokens"] == 25
    assert headers["x-ratelimit-remaining-output-tokens"] == 100


@pytest.mark.asyncio
async def test_set_response_headers_wraps_bare_async_generator():
    """
    Streaming responses that never go through Router.make_call's usual
    object-based wrappers (e.g. the Anthropic /v1/messages -> Responses API
    bridge, which yields a raw async generator with no `_hidden_params` slot)
    must still get IO rate-limit headers attached via a thin wrapper.
    """

    async def _raw_generator():
        yield {"type": "message_start"}
        yield {"type": "message_stop"}

    router = _limited_router(itpm=25)

    wrapped = await router.set_response_headers(response=_raw_generator(), model_group="limited")

    assert hasattr(wrapped, "_hidden_params")
    headers = wrapped._hidden_params["additional_headers"]
    assert headers["x-litellm-model-group"] == "limited"
    assert headers["x-ratelimit-limit-input-tokens"] == 25
    assert headers["x-ratelimit-remaining-input-tokens"] == 25

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


_CALLBACK_LIST_NAMES: Final = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
)


def _callback_ids(list_name: str) -> list[int]:
    return [id(callback) for callback in getattr(litellm, list_name)]


@pytest.mark.parametrize(
    "routing_strategy, selector_attr",
    [
        ("least-busy", "leastbusy_logger"),
        ("usage-based-routing", "lowesttpm_logger"),
        ("usage-based-routing-v2", "lowesttpm_logger_v2"),
        ("latency-based-routing", "lowestlatency_logger"),
        ("cost-based-routing", "lowestcost_logger"),
    ],
)
@pytest.mark.asyncio
async def test_discard_releases_strategy_selector_across_router_lifecycles(
    routing_strategy: RoutingStrategyName, selector_attr: str
):
    def retire() -> weakref.ReferenceType[object]:
        router: Final = Router(model_list=[], routing_strategy=routing_strategy)
        selector: Final = getattr(router, selector_attr)
        assert any(callback is selector for callback in litellm.callbacks)
        router.discard()
        return weakref.ref(selector)

    selector_refs: Final = tuple(retire() for _ in range(3))
    await asyncio.sleep(0)
    gc.collect()

    assert all(getattr(litellm, list_name) == [] for list_name in _CALLBACK_LIST_NAMES)
    assert all(selector_ref() is None for selector_ref in selector_refs)


@pytest.mark.parametrize(
    "routing_strategy, selector_attr",
    [
        ("least-busy", "leastbusy_logger"),
        ("usage-based-routing", "lowesttpm_logger"),
        ("usage-based-routing-v2", "lowesttpm_logger_v2"),
        ("latency-based-routing", "lowestlatency_logger"),
        ("cost-based-routing", "lowestcost_logger"),
    ],
)
@pytest.mark.asyncio
async def test_discard_purges_selector_from_event_lists_after_serving_a_request(
    routing_strategy: RoutingStrategyName, selector_attr: str
):
    router: Final = Router(
        model_list=[{"model_name": "gpt-5.5", "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-test"}}],
        routing_strategy=routing_strategy,
    )
    selector: Final = getattr(router, selector_attr)
    await router.acompletion(model="gpt-5.5", messages=[{"role": "user", "content": "hi"}], mock_response="ok")
    assert any(callback is selector for callback in litellm.success_callback)

    router.discard()

    assert all(getattr(litellm, list_name) == [] for list_name in _CALLBACK_LIST_NAMES)


def test_discard_keeps_callbacks_owned_by_other_live_routers():
    survivor: Final = Router(model_list=[], routing_strategy="latency-based-routing")
    unrelated: Final = CustomLogger()
    litellm.logging_callback_manager.add_litellm_callback(unrelated)
    before_ids: Final = {list_name: _callback_ids(list_name) for list_name in _CALLBACK_LIST_NAMES}
    retired: Final = Router(model_list=[], routing_strategy="least-busy")
    assert any(callback is retired.leastbusy_logger for callback in litellm.callbacks)

    retired.discard()

    assert {list_name: _callback_ids(list_name) for list_name in before_ids} == before_ids
    assert any(callback is survivor.lowestlatency_logger for callback in litellm.callbacks)
    survivor.discard()


@pytest.mark.parametrize(
    "routing_strategy, selector_attr",
    [
        ("usage-based-routing", "lowesttpm_logger"),
        ("usage-based-routing-v2", "lowesttpm_logger_v2"),
        ("latency-based-routing", "lowestlatency_logger"),
        ("cost-based-routing", "lowestcost_logger"),
    ],
)
@pytest.mark.asyncio
async def test_discard_leaves_same_strategy_survivor_receiving_events(
    routing_strategy: RoutingStrategyName, selector_attr: str
):
    model_list: Final[list[DeploymentTypedDict]] = [
        {"model_name": "gpt-5.5", "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-test"}}
    ]
    retired: Final = Router(model_list=model_list, routing_strategy=routing_strategy)
    survivor: Final = Router(model_list=model_list, routing_strategy=routing_strategy)

    retired.discard()
    await survivor.acompletion(model="gpt-5.5", messages=[{"role": "user", "content": "hi"}], mock_response="ok")

    assert any(callback is getattr(survivor, selector_attr) for callback in litellm.success_callback)
    survivor.discard()


def test_discard_drops_routing_group_selectors():
    router: Final = Router(
        model_list=[
            {"model_name": "grouped-model", "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-test"}}
        ],
        routing_groups=[
            {"group_name": "fast", "models": ["grouped-model"], "routing_strategy": "latency-based-routing"}
        ],
    )
    router.discard()

    assert all(getattr(litellm, list_name) == [] for list_name in _CALLBACK_LIST_NAMES)


_GROUPED_MODEL_LIST: Final[list[DeploymentTypedDict]] = [
    {"model_name": "grouped-model", "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-test"}}
]


class _LabelledHook(CustomLogger):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label


def _visit_hooks_reloading_routing_groups_mid_loop(
    router: Router, hooks: tuple[_LabelledHook, ...]
) -> Iterator[CustomLogger]:
    for callback in litellm.callbacks:
        if callback is hooks[0]:
            router.update_settings(
                routing_groups=[{"group_name": "fast", "models": ["grouped-model"], "routing_strategy": "least-busy"}]
            )
        if callback in hooks:
            yield callback


def test_routing_group_reload_keeps_every_hook_of_an_in_flight_callback_loop():
    router: Final = Router(
        model_list=_GROUPED_MODEL_LIST,
        routing_groups=[{"group_name": "fast", "models": ["grouped-model"], "routing_strategy": "least-busy"}],
    )
    hooks: Final = tuple(_LabelledHook(label) for label in ("guardrail_a", "guardrail_b", "guardrail_c"))
    for hook in hooks:
        litellm.logging_callback_manager.add_litellm_callback(hook)

    visited: Final = tuple(_visit_hooks_reloading_routing_groups_mid_loop(router, hooks))

    assert visited == hooks
    router.discard()


@pytest.mark.asyncio
async def test_discard_purges_optional_pre_call_checks_from_event_lists_after_serving_a_request():
    router: Final = Router(
        model_list=_GROUPED_MODEL_LIST,
        optional_pre_call_checks=["enforce_model_rate_limits", "prompt_caching", "deployment_affinity"],
    )
    await router.acompletion(model="grouped-model", messages=[{"role": "user", "content": "hi"}], mock_response="ok")

    router.discard()

    assert all(getattr(litellm, list_name) == [] for list_name in _CALLBACK_LIST_NAMES)


def test_discard_registers_same_config_survivor_optional_pre_call_checks():
    retired: Final = Router(model_list=_GROUPED_MODEL_LIST, optional_pre_call_checks=["enforce_model_rate_limits"])
    survivor: Final = Router(model_list=_GROUPED_MODEL_LIST, optional_pre_call_checks=["enforce_model_rate_limits"])

    retired.discard()

    assert survivor.optional_callbacks
    assert all(_is_registered(check) for check in survivor.optional_callbacks)
    survivor.discard()


def _is_registered(callback: object) -> bool:
    return any(registered is callback for registered in litellm.callbacks)


def test_discard_registers_the_selector_of_a_same_strategy_survivor_built_without_a_model_list():
    retired: Final = Router(model_list=[], routing_strategy="latency-based-routing")
    survivor: Final = Router(routing_strategy="latency-based-routing")
    assert not _is_registered(survivor.lowestlatency_logger)

    retired.discard()

    assert _is_registered(survivor.lowestlatency_logger)
    survivor.discard()


def test_discard_drops_its_slack_alerting_callbacks():
    router: Final = Router(model_list=_GROUPED_MODEL_LIST, alerting_config=AlertingConfig(webhook_url="test"))
    assert any(isinstance(callback, SlackAlerting) for callback in litellm.callbacks)

    router.discard()

    assert all(getattr(litellm, list_name) == [] for list_name in _CALLBACK_LIST_NAMES)


def _adaptive_router_model_list(name: str) -> list[DeploymentTypedDict]:
    return [
        {"model_name": "gpt-5.5", "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-test"}},
        {
            "model_name": name,
            "litellm_params": {
                "model": "auto_router/adaptive_router",
                "adaptive_router_config": {"available_models": ["gpt-5.5"]},
            },
        },
    ]


def _registered_adaptive_router_strategies() -> tuple[object, ...]:
    return tuple(
        callback.adaptive_router for callback in litellm.callbacks if isinstance(callback, AdaptiveRouterPostCallHook)
    )


def test_discard_drops_its_adaptive_router_hook():
    router: Final = Router(model_list=_adaptive_router_model_list("adaptive"), ignore_invalid_deployments=True)
    assert _registered_adaptive_router_strategies()

    router.discard()

    assert _registered_adaptive_router_strategies() == ()


def test_discard_registers_the_adaptive_router_hook_of_a_survivor():
    survivor: Final = Router(model_list=_adaptive_router_model_list("adaptive"), ignore_invalid_deployments=True)
    retired: Final = Router(model_list=_adaptive_router_model_list("adaptive"), ignore_invalid_deployments=True)

    retired.discard()

    survivor_strategies: Final = tuple(tagged.strategy for tagged in survivor.adaptive_routers["adaptive"])
    assert _registered_adaptive_router_strategies() == survivor_strategies
    survivor.discard()


@pytest.mark.asyncio
async def test_discard_releases_its_router_budget_limiter():
    def build_budget_limiter_then_discard() -> weakref.ReferenceType[object]:
        router: Final = Router(
            model_list=_GROUPED_MODEL_LIST,
            provider_budget_config={"openai": {"budget_limit": 10.0, "time_period": "1d"}},
        )
        budget_limiter: Final = router.get_router_deployment_budget_limiter()
        assert budget_limiter is not None
        router.discard()
        return weakref.ref(budget_limiter)

    budget_limiter_ref: Final = build_budget_limiter_then_discard()
    await asyncio.sleep(0)
    gc.collect()

    assert budget_limiter_ref() is None


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

    from litellm.litellm_core_utils.pattern_match_deployments import PatternMatchRouter

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


def test_add_optional_pre_call_checks(model_list):
    router = Router(model_list=model_list)

    router.add_optional_pre_call_checks(["prompt_caching"])
    assert len(litellm.callbacks) > 0


@pytest.mark.asyncio
async def test_async_callback_filter_deployments(model_list):

    router = Router(model_list=model_list)

    healthy_deployments = router.get_model_list(model_name="gpt-5-mini")

    new_healthy_deployments = await router.async_callback_filter_deployments(
        model="gpt-5-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        parent_otel_span=None,
    )

    assert len(new_healthy_deployments) == len(healthy_deployments)


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
        ctx=router._core_ctx,
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


@pytest.mark.parametrize(
    ("model_id", "custom_llm_provider", "expected"),
    [
        ("sora-2", "openai", "sora-2"),
        ("fal-ai/veo3", "fal_ai", None),
        ("veo-3.0-generate-001", "gemini", None),
        ("sora-2", None, None),
    ],
)
def test_resolve_model_name_from_model_id_keeps_a_wildcard_name_only_for_the_ids_provider(
    model_id, custom_llm_provider, expected
):
    router = Router(model_list=[{"model_name": "*", "litellm_params": {"model": "openai/*", "api_key": "sk-fake"}}])
    assert router.resolve_model_name_from_model_id(model_id, custom_llm_provider) == expected


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
