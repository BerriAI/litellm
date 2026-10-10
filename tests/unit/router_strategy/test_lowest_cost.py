import asyncio, copy, importlib, itertools, os
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.utils import _invalidate_model_cost_lowercase_map
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome

DEPLOYMENT_ID = "9876"
FROZEN_INSTANT: Final = datetime(2026, 1, 31, 23, 59, 30)
LAST_MICROSECOND_OF_JANUARY: Final = datetime(2026, 1, 31, 23, 59, 59, 999999)
FIRST_MICROSECOND_OF_FEBRUARY: Final = datetime(2026, 2, 1, 0, 0, 0, 1)
COST_KEY = "cost_map:gpt-5.5-pool"
LATENCY_KEYS = ("gpt-5.5-pool_map", "gpt-5.5-pool_cost_map")
KWARGS = {
    "litellm_params": {
        "metadata": {"model_group": "gpt-5.5-pool"},
        "model_info": {"id": DEPLOYMENT_ID},
    }
}


def _chat_response_with_no_completion_tokens() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model="gpt-5.5",
        choices=[{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
        usage=litellm.Usage(prompt_tokens=12, completion_tokens=0, total_tokens=12),
    )


def _frozen_clock() -> datetime:
    return FROZEN_INSTANT


def _clock_reading(instants: Iterator[datetime]) -> Callable[[], datetime]:
    return lambda: next(instants)


def _clock_rolling_over_after_first_read() -> Callable[[], datetime]:
    return _clock_reading(
        itertools.chain([LAST_MICROSECOND_OF_JANUARY], itertools.repeat(FIRST_MICROSECOND_OF_FEBRUARY))
    )


def _recorded_minute_counters(cache: DualCache) -> dict[str, int]:
    cached = cache.get_cache(key=COST_KEY) or {}
    minute_buckets = cached.get(DEPLOYMENT_ID, {})
    assert len(minute_buckets) == 1, f"expected one minute bucket, got {minute_buckets}"
    return next(iter(minute_buckets.values()))


def test_log_success_event_counts_a_response_with_no_completion_tokens():
    cache = DualCache()
    handler = LowestCostLoggingHandler(router_cache=cache)

    handler.log_success_event(
        kwargs=KWARGS,
        response_obj=_chat_response_with_no_completion_tokens(),
        start_time=datetime(2026, 1, 1, 12, 0, 0),
        end_time=datetime(2026, 1, 1, 12, 0, 2),
    )

    assert _recorded_minute_counters(cache) == {"tpm": 12, "rpm": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_log_success_event_keeps_cost_bookkeeping_out_of_the_latency_routing_entry(use_async: bool):
    cache = DualCache()
    latency_entry = {DEPLOYMENT_ID: {"latency": [0.5], "time_to_first_token": [0.1]}}
    for latency_key in LATENCY_KEYS:
        cache.set_cache(key=latency_key, value=copy.deepcopy(latency_entry))
    handler = LowestCostLoggingHandler(router_cache=cache)
    call_args = {
        "kwargs": KWARGS,
        "response_obj": _chat_response_with_no_completion_tokens(),
        "start_time": datetime(2026, 1, 1, 12, 0, 0),
        "end_time": datetime(2026, 1, 1, 12, 0, 2),
    }

    if use_async:
        await handler.async_log_success_event(**call_args)
    else:
        handler.log_success_event(**call_args)

    assert [cache.get_cache(key=latency_key) for latency_key in LATENCY_KEYS] == [latency_entry, latency_entry]
    assert _recorded_minute_counters(cache) == {"tpm": 12, "rpm": 1}


@pytest.mark.asyncio
async def test_async_get_available_deployments_applies_rpm_limit_from_the_cost_entry():
    cache: Final = DualCache()
    handler: Final = LowestCostLoggingHandler(router_cache=cache, clock=_frozen_clock)
    precise_minute: Final = FROZEN_INSTANT.strftime("%Y-%m-%d-%H-%M")
    cache.set_cache(key=COST_KEY, value={DEPLOYMENT_ID: {precise_minute: {"tpm": 12, "rpm": 1}}})
    healthy_deployments: Final = [
        {"model_info": {"id": DEPLOYMENT_ID}, "litellm_params": {"model": "gpt-5.5", "rpm": 1}}
    ]

    picked: Final = await handler.async_get_available_deployments(
        model_group="gpt-5.5-pool",
        healthy_deployments=healthy_deployments,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert picked is None


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_success_event_across_a_day_boundary_lands_in_the_bucket_of_its_first_clock_read(use_async: bool):
    cache: Final = DualCache()
    handler: Final = LowestCostLoggingHandler(router_cache=cache, clock=_clock_rolling_over_after_first_read())
    call_args: Final = {
        "kwargs": KWARGS,
        "response_obj": _chat_response_with_no_completion_tokens(),
        "start_time": datetime(2026, 1, 31, 23, 59, 58),
        "end_time": datetime(2026, 1, 31, 23, 59, 59),
    }

    if use_async:
        await handler.async_log_success_event(**call_args)
    else:
        handler.log_success_event(**call_args)

    assert cache.get_cache(key=COST_KEY) == {DEPLOYMENT_ID: {"2026-01-31-23-59": {"tpm": 12, "rpm": 1}}}


@pytest.mark.asyncio
async def test_a_read_across_a_day_boundary_uses_the_bucket_of_its_first_clock_read():
    cache: Final = DualCache()
    handler: Final = LowestCostLoggingHandler(router_cache=cache, clock=_clock_rolling_over_after_first_read())
    cache.set_cache(key=COST_KEY, value={DEPLOYMENT_ID: {"2026-01-31-23-59": {"tpm": 12, "rpm": 1}}})
    healthy_deployments: Final = [
        {"model_info": {"id": DEPLOYMENT_ID}, "litellm_params": {"model": "gpt-5.5", "rpm": 1}}
    ]

    picked: Final = await handler.async_get_available_deployments(
        model_group="gpt-5.5-pool",
        healthy_deployments=healthy_deployments,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert picked is None


@pytest.mark.asyncio
async def test_a_write_in_one_minute_is_not_counted_by_a_read_in_the_next_minute():
    cache: Final = DualCache()
    handler: Final = LowestCostLoggingHandler(
        router_cache=cache,
        clock=_clock_reading(
            iter([datetime(2026, 1, 31, 12, 0, 0), datetime(2026, 1, 31, 12, 0, 59), datetime(2026, 1, 31, 12, 1, 0)])
        ),
    )
    healthy_deployments: Final = [
        {
            "model_info": {"id": DEPLOYMENT_ID},
            "litellm_params": {"model": "gpt-5.5", "rpm": 1, "input_cost_per_token": 1, "output_cost_per_token": 1},
        },
        {
            "model_info": {"id": "pricier"},
            "litellm_params": {"model": "gpt-5.5", "input_cost_per_token": 2, "output_cost_per_token": 2},
        },
    ]
    await handler.async_log_success_event(
        kwargs=KWARGS,
        response_obj=_chat_response_with_no_completion_tokens(),
        start_time=datetime(2026, 1, 31, 11, 59, 59),
        end_time=datetime(2026, 1, 31, 12, 0, 0),
    )

    picked_in_the_same_minute: Final = await handler.async_get_available_deployments(
        model_group="gpt-5.5-pool", healthy_deployments=healthy_deployments
    )
    picked_in_the_next_minute: Final = await handler.async_get_available_deployments(
        model_group="gpt-5.5-pool", healthy_deployments=healthy_deployments
    )

    assert [picked_in_the_same_minute["model_info"]["id"], picked_in_the_next_minute["model_info"]["id"]] == [
        "pricier",
        DEPLOYMENT_ID,
    ]


@pytest.mark.parametrize(
    "instant",
    [datetime(2026, 1, 2, 3, 4, 5), datetime(2026, 12, 31, 23, 59, 59, 999999), datetime(2027, 1, 1, 0, 0, 0)],
    ids=["single-digit-fields", "last-microsecond-of-a-year", "first-instant-of-a-year"],
)
def test_bucket_keys_use_the_minute_format_of_the_clock_reading(instant: datetime):
    cache: Final = DualCache()
    handler: Final = LowestCostLoggingHandler(router_cache=cache, clock=lambda: instant)

    handler.log_success_event(
        kwargs=KWARGS,
        response_obj=_chat_response_with_no_completion_tokens(),
        start_time=instant,
        end_time=instant,
    )

    assert list(cache.get_cache(key=COST_KEY)[DEPLOYMENT_ID]) == [instant.strftime("%Y-%m-%d-%H-%M")]


@pytest.mark.asyncio
async def test_async_log_success_event_counts_a_response_with_no_completion_tokens():
    cache = DualCache()
    handler = LowestCostLoggingHandler(router_cache=cache)

    await handler.async_log_success_event(
        kwargs=KWARGS,
        response_obj=_chat_response_with_no_completion_tokens(),
        start_time=datetime(2026, 1, 1, 12, 0, 0),
        end_time=datetime(2026, 1, 1, 12, 0, 2),
    )

    assert _recorded_minute_counters(cache) == {"tpm": 12, "rpm": 1}


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
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_get_available_deployments():
    test_cache = DualCache()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "openai-gpt-4"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "groq/openai/gpt-oss-20b"},
            "model_info": {"id": "groq-llama"},
        },
    ]
    lowest_cost_logger = LowestCostLoggingHandler(
        router_cache=test_cache,
    )
    model_group = "gpt-3.5-turbo"

    ## CHECK WHAT'S SELECTED ##
    selected_model = await lowest_cost_logger.async_get_available_deployments(
        model_group=model_group, healthy_deployments=model_list
    )
    print("selected model: ", selected_model)

    assert selected_model["model_info"]["id"] == "groq-llama"

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_get_available_deployments_custom_price():
    import logging

    from litellm._logging import verbose_router_logger

    verbose_router_logger.setLevel(logging.DEBUG)
    test_cache = DualCache()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/gpt-4.1-mini",
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00003,
            },
            "model_info": {"id": "chatgpt-v-experimental"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-1",
                "input_cost_per_token": 0.000000001,
                "output_cost_per_token": 0.00000001,
            },
            "model_info": {"id": "chatgpt-v-1"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-5",
                "input_cost_per_token": 10,
                "output_cost_per_token": 12,
            },
            "model_info": {"id": "chatgpt-v-5"},
        },
    ]
    lowest_cost_logger = LowestCostLoggingHandler(
        router_cache=test_cache,
    )
    model_group = "gpt-3.5-turbo"

    ## CHECK WHAT'S SELECTED ##
    selected_model = await lowest_cost_logger.async_get_available_deployments(
        model_group=model_group, healthy_deployments=model_list
    )
    print("selected model: ", selected_model)

    assert selected_model["model_info"]["id"] == "chatgpt-v-1"

async def _deploy(lowest_cost_logger, deployment_id, tokens_used):
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "gpt-4",
            },
            "model_info": {"id": deployment_id},
        }
    }
    response_obj = {"usage": {"total_tokens": tokens_used}}
    await lowest_cost_logger.async_log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=FROZEN_INSTANT,
        end_time=FROZEN_INSTANT,
    )

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("ans_rpm", [1, 5])  # 1 should produce nothing, 10 should select first
@pytest.mark.asyncio
async def test_get_available_endpoints_tpm_rpm_check_async(ans_rpm):
    """
    Pass in list of 2 valid models

    Update cache with 1 model clearly being at tpm/rpm limit

    assert that only the valid model is returned
    """
    import logging

    from litellm._logging import verbose_router_logger

    verbose_router_logger.setLevel(logging.DEBUG)
    test_cache = DualCache()
    ans = "1234"
    non_ans_rpm = 3
    assert ans_rpm != non_ans_rpm, "invalid test"
    if ans_rpm < non_ans_rpm:
        ans = None
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "1234", "rpm": ans_rpm},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "groq/llama-3.1-8b-instant"},
            "model_info": {"id": "5678", "rpm": non_ans_rpm},
        },
    ]
    lowest_cost_logger = LowestCostLoggingHandler(router_cache=test_cache, clock=_frozen_clock)
    model_group = "gpt-3.5-turbo"
    d1 = [(lowest_cost_logger, "1234", 50)] * non_ans_rpm
    d2 = [(lowest_cost_logger, "5678", 50)] * non_ans_rpm

    await asyncio.gather(*[_deploy(*t) for t in [*d1, *d2]])

    ## CHECK WHAT'S SELECTED ##
    d_ans = await lowest_cost_logger.async_get_available_deployments(
        model_group=model_group, healthy_deployments=model_list
    )
    assert (d_ans and d_ans["model_info"]["id"]) == ans

    print("selected deployment:", d_ans)
