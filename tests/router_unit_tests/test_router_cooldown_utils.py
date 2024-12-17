import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params, ModelInfo
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.integrations.prometheus import PrometheusLogger
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback
from litellm.router_utils.cooldown_handlers import (
    _should_run_cooldown_logic,
    _should_cooldown_deployment,
    cast_exception_status_to_int,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    increment_deployment_failures_for_current_minute,
    increment_deployment_successes_for_current_minute,
)

load_dotenv()


class CustomPrometheusLogger(PrometheusLogger):
    def __init__(self):
        super().__init__()
        self.deployment_complete_outages = []
        self.deployment_cooled_downs = []

    def set_deployment_complete_outage(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
    ):
        self.deployment_complete_outages.append(
            [litellm_model_name, model_id, api_base, api_provider]
        )

    def increment_deployment_cooled_down(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
        exception_status: str,
    ):
        self.deployment_cooled_downs.append(
            [litellm_model_name, model_id, api_base, api_provider, exception_status]
        )


@pytest.mark.asyncio
async def test_router_cooldown_event_callback():
    """
    Test the router_cooldown_event_callback function

    Ensures that the router_cooldown_event_callback function correctly logs the cooldown event to the PrometheusLogger
    """
    # Mock Router instance
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {"model": "gpt-3.5-turbo"},
        "model_name": "gpt-3.5-turbo",
        "model_info": ModelInfo(id="test-model-id"),
    }
    mock_router.get_deployment.return_value = mock_deployment

    # Create a real PrometheusLogger instance
    prometheus_logger = CustomPrometheusLogger()
    litellm.callbacks = [prometheus_logger]

    await router_cooldown_event_callback(
        litellm_router_instance=mock_router,
        deployment_id="test-deployment",
        exception_status="429",
        cooldown_time=60.0,
    )

    await asyncio.sleep(0.5)

    # Assert that the router's get_deployment method was called
    mock_router.get_deployment.assert_called_once_with(model_id="test-deployment")

    print(
        "prometheus_logger.deployment_complete_outages",
        prometheus_logger.deployment_complete_outages,
    )
    print(
        "prometheus_logger.deployment_cooled_downs",
        prometheus_logger.deployment_cooled_downs,
    )

    # Assert that PrometheusLogger methods were called
    assert len(prometheus_logger.deployment_complete_outages) == 1
    assert len(prometheus_logger.deployment_cooled_downs) == 1

    assert prometheus_logger.deployment_complete_outages[0] == [
        "gpt-3.5-turbo",
        "test-model-id",
        "https://api.openai.com",
        "openai",
    ]
    assert prometheus_logger.deployment_cooled_downs[0] == [
        "gpt-3.5-turbo",
        "test-model-id",
        "https://api.openai.com",
        "openai",
        "429",
    ]


@pytest.mark.asyncio
async def test_router_cooldown_event_callback_no_prometheus():
    """
    Test the router_cooldown_event_callback function

    Ensures that the router_cooldown_event_callback function does not raise an error when no PrometheusLogger is found
    """
    # Mock Router instance
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {"model": "gpt-3.5-turbo"},
        "model_name": "gpt-3.5-turbo",
        "model_info": ModelInfo(id="test-model-id"),
    }
    mock_router.get_deployment.return_value = mock_deployment

    await router_cooldown_event_callback(
        litellm_router_instance=mock_router,
        deployment_id="test-deployment",
        exception_status="429",
        cooldown_time=60.0,
    )

    # Assert that the router's get_deployment method was called
    mock_router.get_deployment.assert_called_once_with(model_id="test-deployment")


@pytest.mark.asyncio
async def test_router_cooldown_event_callback_no_deployment():
    """
    Test the router_cooldown_event_callback function

    Ensures that the router_cooldown_event_callback function does not raise an error when no deployment is found

    In this scenario it should do nothing
    """
    # Mock Router instance
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None

    await router_cooldown_event_callback(
        litellm_router_instance=mock_router,
        deployment_id="test-deployment",
        exception_status="429",
        cooldown_time=60.0,
    )

    # Assert that the router's get_deployment method was called
    mock_router.get_deployment.assert_called_once_with(model_id="test-deployment")


@pytest.fixture
def testing_litellm_router():
    return Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_id": "test_deployment",
            },
            {
                "model_name": "test_deployment",
                "litellm_params": {"model": "openai/test_deployment"},
                "model_id": "test_deployment_2",
            },
        ]
    )


def test_should_run_cooldown_logic(testing_litellm_router):
    testing_litellm_router.disable_cooldowns = True
    # don't run cooldown logic if disable_cooldowns is True
    assert (
        _should_run_cooldown_logic(
            testing_litellm_router, "test_deployment", 500, Exception("Test")
        )
        is False
    )

    # don't cooldown if deployment is None
    testing_litellm_router.disable_cooldowns = False
    assert (
        _should_run_cooldown_logic(testing_litellm_router, None, 500, Exception("Test"))
        is False
    )

    # don't cooldown if it's a provider default deployment
    testing_litellm_router.provider_default_deployment_ids = ["test_deployment"]
    assert (
        _should_run_cooldown_logic(
            testing_litellm_router, "test_deployment", 500, Exception("Test")
        )
        is False
    )


def test_should_cooldown_deployment_rate_limit_error(testing_litellm_router):
    """
    Test the _should_cooldown_deployment function when a rate limit error occurs
    """
    # Test 429 error (rate limit) -> always cooldown a deployment returning 429s
    _exception = litellm.exceptions.RateLimitError(
        "Rate limit", "openai", "gpt-3.5-turbo"
    )
    assert (
        _should_cooldown_deployment(
            testing_litellm_router, "test_deployment", 429, _exception
        )
        is True
    )


def test_should_cooldown_deployment_auth_limit_error(testing_litellm_router):
    """
    Test the _should_cooldown_deployment function when an auth limit error occurs
    """
    # Test 401 error (auth limit) -> always cooldown a deployment returning 401s
    _exception = litellm.exceptions.AuthenticationError(
        "Unauthorized", "openai", "gpt-3.5-turbo"
    )
    assert (
        _should_cooldown_deployment(
            testing_litellm_router, "test_deployment", 401, _exception
        )
        is True
    )


@pytest.mark.asyncio
async def test_should_cooldown_deployment(testing_litellm_router):
    """
    Cooldown a deployment if it fails 60% of requests in 1 minute - DEFAULT threshold is 50%
    """
    from litellm._logging import verbose_router_logger
    import logging

    verbose_router_logger.setLevel(logging.DEBUG)

    # Test 429 error (rate limit) -> always cooldown a deployment returning 429s
    _exception = litellm.exceptions.RateLimitError(
        "Rate limit", "openai", "gpt-3.5-turbo"
    )
    assert (
        _should_cooldown_deployment(
            testing_litellm_router, "test_deployment", 429, _exception
        )
        is True
    )

    available_deployment = testing_litellm_router.get_available_deployment(
        model="test_deployment"
    )
    print("available_deployment", available_deployment)
    assert available_deployment is not None

    deployment_id = available_deployment["model_info"]["id"]
    print("deployment_id", deployment_id)

    # set current success for deployment to 40
    for _ in range(40):
        increment_deployment_successes_for_current_minute(
            litellm_router_instance=testing_litellm_router, deployment_id=deployment_id
        )

    # now we fail 40 requests in a row
    tasks = []
    for _ in range(41):
        tasks.append(
            testing_litellm_router.acompletion(
                model=deployment_id,
                messages=[{"role": "user", "content": "Hello, world!"}],
                max_tokens=100,
                mock_response="litellm.InternalServerError",
            )
        )
    try:
        await asyncio.gather(*tasks)
    except Exception:
        pass

    await asyncio.sleep(1)

    # expect this to fail since it's now 51% of requests are failing
    assert (
        _should_cooldown_deployment(
            testing_litellm_router, deployment_id, 500, Exception("Test")
        )
        is True
    )


@pytest.mark.asyncio
async def test_should_cooldown_deployment_allowed_fails_set_on_router():
    """
    Test the _should_cooldown_deployment function when Router.allowed_fails is set
    """
    # Create a Router instance with a test deployment
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_id": "test_deployment",
            },
        ]
    )

    # Set up allowed_fails for the test deployment
    router.allowed_fails = 100

    # should not cooldown when fails are below the allowed limit
    for _ in range(100):
        assert (
            _should_cooldown_deployment(
                router, "test_deployment", 500, Exception("Test")
            )
            is False
        )

    assert (
        _should_cooldown_deployment(router, "test_deployment", 500, Exception("Test"))
        is True
    )


def test_increment_deployment_successes_for_current_minute_does_not_write_to_redis(
    testing_litellm_router,
):
    """
    Ensure tracking deployment metrics does not write to redis

    Important - If it writes to redis on every request it will seriously impact performance / latency
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.redis_cache import RedisCache
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.router_utils.router_callbacks.track_deployment_metrics import (
        increment_deployment_successes_for_current_minute,
    )

    # Mock RedisCache
    mock_redis_cache = MagicMock(spec=RedisCache)

    testing_litellm_router.cache = DualCache(
        redis_cache=mock_redis_cache, in_memory_cache=InMemoryCache()
    )

    # Call the function we're testing
    increment_deployment_successes_for_current_minute(
        litellm_router_instance=testing_litellm_router, deployment_id="test_deployment"
    )

    increment_deployment_failures_for_current_minute(
        litellm_router_instance=testing_litellm_router, deployment_id="test_deployment"
    )

    time.sleep(1)

    # Assert that no methods were called on the mock_redis_cache
    assert not mock_redis_cache.method_calls, "RedisCache methods should not be called"

    print(
        "in memory cache values=",
        testing_litellm_router.cache.in_memory_cache.cache_dict,
    )
    assert (
        testing_litellm_router.cache.in_memory_cache.get_cache(
            "test_deployment:successes"
        )
        is not None
    )


def test_cast_exception_status_to_int():
    assert cast_exception_status_to_int(200) == 200
    assert cast_exception_status_to_int("404") == 404
    assert cast_exception_status_to_int("invalid") == 500
