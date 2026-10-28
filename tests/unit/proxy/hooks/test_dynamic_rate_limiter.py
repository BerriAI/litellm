from datetime import datetime, timezone
from typing import Final

import asyncio, importlib, litellm, os, pytest

from litellm.caching.caching import DualCache
from litellm.proxy.hooks.dynamic_rate_limiter import(
    PROXY_DynamicRateLimitHandler as DynamicRateLimitHandler,
    DynamicRateLimiterCache,
    PROXY_DynamicRateLimitHandler,
)
from litellm.types.utils import HiddenParams, ModelResponse
from litellm import DualCache as DualCache_dynamic_rate, Router
from litellm._uuid import uuid
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy._types import UserAPIKeyAuth
from litellm.utils import _invalidate_model_cost_lowercase_map
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from typing import Optional


@pytest.mark.asyncio
async def test_sadd_and_get_share_injected_clock_window():
    dual_cache = DualCache()
    cache = DynamicRateLimiterCache(
        cache=dual_cache,
        time_fn=lambda: datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc),
    )
    await cache.async_set_cache_sadd(model="my-fake-model", value=["p1", "p2", "p3"])
    assert await cache.async_get_cache(model="my-fake-model") == 3
    assert await dual_cache.async_get_cache(key="10-30:my-fake-model") is not None


@pytest.mark.asyncio
async def test_minute_rollover_between_sadd_and_get_reads_empty_window():
    ticks = iter(
        (
            datetime(2024, 1, 1, 10, 30, 59, 999999, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 10, 31, 0, 0, tzinfo=timezone.utc),
        )
    )
    cache = DynamicRateLimiterCache(cache=DualCache(), time_fn=lambda: next(ticks))
    await cache.async_set_cache_sadd(model="my-fake-model", value=["p1"])
    assert await cache.async_get_cache(model="my-fake-model") is None


@pytest.mark.asyncio
async def test_handler_threads_time_fn_to_internal_cache():
    handler = PROXY_DynamicRateLimitHandler(
        internal_usage_cache=DualCache(),
        time_fn=lambda: datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc),
    )
    await handler.internal_usage_cache.async_set_cache_sadd(model="my-fake-model", value=["p1", "p2"])
    assert await handler.internal_usage_cache.async_get_cache(model="my-fake-model") == 2


@pytest.mark.asyncio
async def test_success_hook_updates_existing_hidden_params_storage() -> None:
    model_id: Final = "rate-limit-deployment"
    router: Final = Router(
        model_list=[
            {
                "model_name": "my-fake-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test-key", "tpm": 100, "rpm": 10},
                "model_info": {"id": model_id},
            }
        ]
    )
    handler: Final = PROXY_DynamicRateLimitHandler(internal_usage_cache=DualCache())
    handler.update_variables(llm_router=router)
    response: Final = ModelResponse()
    hidden_params: Final = HiddenParams(model_id=model_id)
    response._hidden_params = hidden_params

    result: Final = await handler.async_post_call_success_hook(
        data={},
        user_api_key_dict=UserAPIKeyAuth(metadata={}),
        response=response,
    )

    assert result is response
    assert response._hidden_params is hidden_params
    assert response.hidden_params["additional_headers"]["x-litellm-model_group"] == "my-fake-model"


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}

@pytest.fixture(scope="module")
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception as e:
            print(f"Error reloading litellm.proxy.proxy_server: {e}")
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield

@pytest.fixture
def _pr4_dynamic_rate_limit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LICENSE", "pr4-test-license")

"""
Basic test cases:

- If 1 'active' project => give all tpm
- If 2 'active' projects => divide tpm in 2
"""

@pytest.fixture
def dynamic_rate_limit_handler() -> DynamicRateLimitHandler:
    internal_cache = DualCache_dynamic_rate()
    frozen_now = datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
    return DynamicRateLimitHandler(internal_usage_cache=internal_cache, time_fn=lambda: frozen_now)

@pytest.fixture
def mock_response() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        **{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1699896916,
            "model": "gpt-3.5-turbo-0125",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{\n"location": "Boston, MA"\n}',
                                },
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
    )

@pytest.fixture
def user_api_key_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth()

@pytest.mark.usefixtures(
    "_pr4_dynamic_rate_limit_env",
    "_vcr_outcome_gate",
    "isolate_litellm_state",
    "setup_and_teardown",
)
@pytest.mark.parametrize("num_projects", [1, 2, 100])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_available_tpm(num_projects, dynamic_rate_limit_handler):
    model = "my-fake-model"
    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4()) for _ in range(num_projects)]

    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(model=model, value=projects)

    model_tpm = 100
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "tpm": model_tpm,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    ## CHECK AVAILABLE TPM PER PROJECT

    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    availability = resp[0]

    expected_availability = int(model_tpm / num_projects)

    assert availability == expected_availability

@pytest.mark.usefixtures(
    "_pr4_dynamic_rate_limit_env",
    "_vcr_outcome_gate",
    "isolate_litellm_state",
    "setup_and_teardown",
)
@pytest.mark.parametrize("num_projects", [1, 2, 100])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_available_rpm(num_projects, dynamic_rate_limit_handler):
    model = "my-fake-model"
    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4()) for _ in range(num_projects)]

    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(model=model, value=projects)

    model_rpm = 100
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "rpm": model_rpm,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    ## CHECK AVAILABLE rpm PER PROJECT

    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    availability = resp[1]

    expected_availability = int(model_rpm / num_projects)

    assert availability == expected_availability

@pytest.mark.usefixtures(
    "_pr4_dynamic_rate_limit_env",
    "_vcr_outcome_gate",
    "isolate_litellm_state",
    "setup_and_teardown",
)
@pytest.mark.parametrize("usage", ["rpm", "tpm"])
@pytest.mark.asyncio
async def test_rate_limit_raised(dynamic_rate_limit_handler, user_api_key_auth, usage):
    """
    Unit test. Tests if rate limit error raised when quota exhausted.
    """
    from fastapi import HTTPException

    model = "my-fake-model"
    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4())]

    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(model=model, value=projects)

    model_usage = 0
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    usage: model_usage,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    ## CHECK AVAILABLE TPM PER PROJECT

    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    if usage == "tpm":
        availability = resp[0]
    else:
        availability = resp[1]

    expected_availability = 0

    assert availability == expected_availability

    ## CHECK if exception raised

    with pytest.raises(HTTPException) as exc_info:
        await dynamic_rate_limit_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_auth,
            cache=DualCache_dynamic_rate(),
            data={"model": model},
            call_type="completion",
        )
    e = exc_info.value
    assert e.status_code == 429  # check if rate limit error raised

@pytest.mark.usefixtures(
    "_pr4_dynamic_rate_limit_env",
    "_vcr_outcome_gate",
    "isolate_litellm_state",
    "setup_and_teardown",
)
@pytest.mark.asyncio
async def test_base_case(dynamic_rate_limit_handler, mock_response):
    """
    If just 1 active project

    it should get all the quota

    = allow request to go through
    - update token usage
    - exhaust all tpm with just 1 project
    - assert ratelimiterror raised at 100%+1 tpm
    """
    model = "my-fake-model"
    ## model tpm - 50
    model_tpm = 50
    ## tpm per request - 10
    setattr(
        mock_response,
        "usage",
        litellm.Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "tpm": model_tpm,
                    "mock_response": mock_response,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    prev_availability: Optional[int] = None
    allowed_fails = 1
    for _ in range(2):
        try:
            # check availability
            resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

            availability = resp[0]

            print("prev_availability={}, availability={}".format(prev_availability, availability))

            ## assert availability updated
            if prev_availability is not None and availability is not None:
                assert availability == prev_availability - 10

            prev_availability = availability

            # make call
            await llm_router.acompletion(model=model, messages=[{"role": "user", "content": "hey!"}])

            await asyncio.sleep(3)
        except Exception:
            if allowed_fails > 0:
                allowed_fails -= 1
            else:
                raise

@pytest.mark.usefixtures(
    "_pr4_dynamic_rate_limit_env",
    "_vcr_outcome_gate",
    "isolate_litellm_state",
    "setup_and_teardown",
)
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_update_cache(dynamic_rate_limit_handler, mock_response, user_api_key_auth):
    """
    Check if active project correctly updated
    """
    model = "my-fake-model"
    model_tpm = 50

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "tpm": model_tpm,
                    "mock_response": mock_response,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    ## INITIAL ACTIVE PROJECTS - ASSERT NONE
    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    active_projects = resp[-1]

    assert active_projects is None

    ## MAKE CALL
    await dynamic_rate_limit_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_auth,
        cache=DualCache_dynamic_rate(),
        data={"model": model},
        call_type="completion",
    )

    await asyncio.sleep(2)
    ## INITIAL ACTIVE PROJECTS - ASSERT 1
    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    active_projects = resp[-1]

    assert active_projects == 1

@pytest.mark.usefixtures(
    "_pr4_dynamic_rate_limit_env",
    "_vcr_outcome_gate",
    "isolate_litellm_state",
    "setup_and_teardown",
)
@pytest.mark.parametrize("num_projects", [1, 2, 100])
@pytest.mark.asyncio
async def test_priority_reservation(num_projects, dynamic_rate_limit_handler):
    """
    If reservation is set + `mock_testing_reservation` passed in

    assert correct rpm is reserved
    """
    model = "my-fake-model"
    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4()) for _ in range(num_projects)]

    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(model=model, value=projects)

    litellm.priority_reservation = {"dev": 0.1, "prod": 0.9}

    model_usage = 100

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "rpm": model_usage,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    ## CHECK AVAILABLE TPM PER PROJECT

    resp = await dynamic_rate_limit_handler.check_available_usage(model=model, priority="prod")

    availability = resp[1]

    expected_availability = int(model_usage * litellm.priority_reservation["prod"] / num_projects)

    assert availability == expected_availability
