import sys, os, time
import traceback, asyncio
import pytest

import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from litellm.types.router import ModelInfo
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback
from litellm.router_utils.cooldown_handlers import (
    _should_run_cooldown_logic,
    _should_cooldown_deployment,
    cast_exception_status_to_int,
    _is_cooldown_required,
    _has_explicit_allowed_fails_policy_for_exception,
)
from litellm.types.router import AllowedFailsPolicy
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    increment_deployment_failures_for_current_minute,
    increment_deployment_successes_for_current_minute,
)


load_dotenv()


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
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini"},
                "model_id": "test_deployment",
            },
            {
                "model_name": "test_deployment",
                "litellm_params": {"model": "openai/test_deployment"},
                "model_id": "test_deployment_2",
            },
            {
                "model_name": "test_deployment",
                "litellm_params": {"model": "openai/test_deployment-2"},
                "model_id": "test_deployment_3",
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


@pytest.fixture
def single_deployment_router():
    """A router with one deployment whose model_info.id is the lookup-able
    "dep-1" (unlike `testing_litellm_router`'s top-level "model_id" key, which
    is not absorbed into model_info.id and so never resolves via
    get_model_info/get_model_group)."""
    return Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini"},
                "model_info": {"id": "dep-1"},
            },
        ]
    )


def test_should_run_cooldown_logic_generic_bad_request_excluded_by_default(
    single_deployment_router,
):
    """A generic BadRequestError/ContentPolicyViolationError (400) is excluded from
    cooldown evaluation by _is_cooldown_required when no allowed_fails_policy is
    configured for that exception type. This is the pre-existing, intentional
    default: a client error is usually not the deployment's fault."""
    exc = litellm.BadRequestError("bad request", "openai", "gpt-5-mini")
    assert (
        _should_run_cooldown_logic(single_deployment_router, "dep-1", 400, exc) is False
    )


def test_should_run_cooldown_logic_router_level_policy_does_not_override_bad_request_exclusion(
    single_deployment_router,
):
    """A router-level allowed_fails_policy is a pre-existing, router-wide setting that
    predates the per-deployment override feature, so it must keep its existing behavior
    and stay subject to the generic 4XX exclusion. Only an explicit deployment-level
    policy (an unambiguous per-exception opt-in for that one deployment) overrides it;
    see test_should_run_cooldown_logic_explicit_deployment_level_policy_overrides_content_policy_exclusion."""
    exc = litellm.BadRequestError("bad request", "openai", "gpt-5-mini")
    single_deployment_router.allowed_fails_policy = AllowedFailsPolicy(
        BadRequestErrorAllowedFails=5
    )
    assert (
        _should_run_cooldown_logic(single_deployment_router, "dep-1", 400, exc) is False
    )


def test_should_run_cooldown_logic_explicit_deployment_level_policy_overrides_content_policy_exclusion(
    single_deployment_router,
):
    """Same as the router-level case, but for a deployment-level allowed_fails_policy
    entry (this PR's per-deployment feature) targeting ContentPolicyViolationError."""
    exc = litellm.ContentPolicyViolationError("flagged content", "openai", "gpt-5-mini")
    deployment_dict = single_deployment_router.get_model_info(id="dep-1")
    deployment_dict["model_info"]["allowed_fails_policy"] = {
        "ContentPolicyViolationErrorAllowedFails": 0
    }
    assert (
        _should_run_cooldown_logic(single_deployment_router, "dep-1", 400, exc) is True
    )


class TestHasExplicitAllowedFailsPolicyForException:
    def test_no_policy_anywhere_returns_false(self, single_deployment_router):
        exc = litellm.BadRequestError("bad request", "openai", "gpt-5-mini")
        assert (
            _has_explicit_allowed_fails_policy_for_exception(
                single_deployment_router, "dep-1", exc
            )
            is False
        )

    def test_router_level_policy_for_matching_exception_returns_false(
        self, single_deployment_router
    ):
        """Deliberately scoped to deployment-level only: a router-level policy
        predates this feature and must not be treated as an explicit per-exception
        opt-in for cooldown-gate purposes."""
        exc = litellm.RateLimitError("rate limited", "openai", "gpt-5-mini")
        single_deployment_router.allowed_fails_policy = AllowedFailsPolicy(
            RateLimitErrorAllowedFails=3
        )
        assert (
            _has_explicit_allowed_fails_policy_for_exception(
                single_deployment_router, "dep-1", exc
            )
            is False
        )

    def test_router_level_policy_for_different_exception_returns_false(
        self, single_deployment_router
    ):
        exc = litellm.BadRequestError("bad request", "openai", "gpt-5-mini")
        single_deployment_router.allowed_fails_policy = AllowedFailsPolicy(
            RateLimitErrorAllowedFails=3
        )
        assert (
            _has_explicit_allowed_fails_policy_for_exception(
                single_deployment_router, "dep-1", exc
            )
            is False
        )

    def test_deployment_level_policy_for_matching_exception_returns_true(
        self, single_deployment_router
    ):
        exc = litellm.ContentPolicyViolationError("flagged", "openai", "gpt-5-mini")
        deployment_dict = single_deployment_router.get_model_info(id="dep-1")
        deployment_dict["model_info"]["allowed_fails_policy"] = {
            "ContentPolicyViolationErrorAllowedFails": 0
        }
        assert (
            _has_explicit_allowed_fails_policy_for_exception(
                single_deployment_router, "dep-1", exc
            )
            is True
        )

    def test_none_deployment_returns_false(self, single_deployment_router):
        exc = litellm.RateLimitError("rate limited", "openai", "gpt-5-mini")
        single_deployment_router.allowed_fails_policy = AllowedFailsPolicy(
            RateLimitErrorAllowedFails=3
        )
        assert (
            _has_explicit_allowed_fails_policy_for_exception(
                single_deployment_router, None, exc
            )
            is False
        )


def test_should_cooldown_deployment_rate_limit_error(testing_litellm_router):
    """
    Test the _should_cooldown_deployment function when a rate limit error occurs
    """
    # Test 429 error (rate limit) -> always cooldown a deployment returning 429s
    _exception = litellm.exceptions.RateLimitError(
        "Rate limit", "openai", "gpt-5-mini"
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
        "Unauthorized", "openai", "gpt-5-mini"
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
        "Rate limit", "openai", "gpt-5-mini"
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
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini"},
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


@pytest.fixture
def router():
    return Router(
        model_list=[
            {
                "model_name": "gpt-5.5",
                "litellm_params": {"model": "gpt-5.5"},
                "model_info": {
                    "id": "gpt-4--0",
                },
            }
        ]
    )


@patch(
    "litellm.router_utils.cooldown_handlers.get_deployment_successes_for_current_minute"
)
@patch(
    "litellm.router_utils.cooldown_handlers.get_deployment_failures_for_current_minute"
)
def test_should_cooldown_high_traffic_all_fails(mock_failures, mock_successes, router):
    # Simulate 10 failures, 0 successes
    from litellm.constants import SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD

    mock_failures.return_value = SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD + 1
    mock_successes.return_value = 0

    should_cooldown = _should_cooldown_deployment(
        litellm_router_instance=router,
        deployment="gpt-4--0",
        exception_status=500,
        original_exception=Exception("Test error"),
    )

    assert (
        should_cooldown is True
    ), "Should cooldown when all requests fail with sufficient traffic"


@patch(
    "litellm.router_utils.cooldown_handlers.get_deployment_successes_for_current_minute"
)
@patch(
    "litellm.router_utils.cooldown_handlers.get_deployment_failures_for_current_minute"
)
def test_no_cooldown_low_traffic(mock_failures, mock_successes, router):
    # Simulate 3 failures (below MIN_TRAFFIC_THRESHOLD)
    mock_failures.return_value = 3
    mock_successes.return_value = 0

    should_cooldown = _should_cooldown_deployment(
        litellm_router_instance=router,
        deployment="gpt-4--0",
        exception_status=500,
        original_exception=Exception("Test error"),
    )

    assert (
        should_cooldown is False
    ), "Should not cooldown when traffic is below threshold"


@patch(
    "litellm.router_utils.cooldown_handlers.get_deployment_successes_for_current_minute"
)
@patch(
    "litellm.router_utils.cooldown_handlers.get_deployment_failures_for_current_minute"
)
def test_cooldown_rate_limit(mock_failures, mock_successes, router):
    """
    Don't cooldown single deployment models, for anything besides traffic
    """
    mock_failures.return_value = 1
    mock_successes.return_value = 0

    should_cooldown = _should_cooldown_deployment(
        litellm_router_instance=router,
        deployment="gpt-4--0",
        exception_status=429,  # Rate limit error
        original_exception=Exception("Rate limit exceeded"),
    )

    assert (
        should_cooldown is False
    ), "Should not cooldown on rate limit error for single deployment models"


@patch(
    "litellm.router_utils.cooldown_handlers.get_deployment_successes_for_current_minute"
)
@patch(
    "litellm.router_utils.cooldown_handlers.get_deployment_failures_for_current_minute"
)
def test_mixed_success_failure(mock_failures, mock_successes, router):
    # Simulate 3 failures, 7 successes
    mock_failures.return_value = 3
    mock_successes.return_value = 7

    should_cooldown = _should_cooldown_deployment(
        litellm_router_instance=router,
        deployment="gpt-4--0",
        exception_status=500,
        original_exception=Exception("Test error"),
    )

    assert (
        should_cooldown is False
    ), "Should not cooldown when failure rate is below threshold"


def test_is_cooldown_required_empty_string_exception_status(testing_litellm_router):
    """
    Test that _is_cooldown_required returns False when exception_status is an empty string
    """
    result = _is_cooldown_required(
        litellm_router_instance=testing_litellm_router,
        model_id="test_deployment",
        exception_status="",
    )

    assert (
        result is False
    ), "Should not require cooldown when exception_status is empty string"


def test_should_cooldown_deployment_minimum_request_threshold(testing_litellm_router):
    """
    Test that error rate cooldown does NOT trigger on first failure.

    Fixes GitHub issue #17418: Error Rate Cooldown Triggers on First Failed Request

    The problem: With DEFAULT_FAILURE_THRESHOLD_PERCENT=0.5 (50%), a deployment
    gets cooled down after just 1 failed request because 1/1 = 100% > 50%.

    The fix: Add a minimum request threshold (DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS)
    before applying error rate cooldown.
    """
    from litellm.constants import DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS

    # Get a deployment that's not a single-deployment model group
    # (test_deployment_2 and test_deployment_3 are both for "test_deployment" model)
    available_deployment = testing_litellm_router.get_available_deployment(
        model="test_deployment"
    )
    assert available_deployment is not None
    deployment_id = available_deployment["model_info"]["id"]

    # Simulate only 1 failure (below minimum threshold)
    # This should NOT trigger cooldown even though 100% > 50%
    increment_deployment_failures_for_current_minute(
        litellm_router_instance=testing_litellm_router, deployment_id=deployment_id
    )

    _exception = litellm.exceptions.InternalServerError(
        "Internal error", "openai", "gpt-5-mini"
    )

    # With only 1 request, should NOT cooldown (below minimum threshold)
    should_cooldown = _should_cooldown_deployment(
        testing_litellm_router, deployment_id, 500, _exception
    )
    assert (
        should_cooldown is False
    ), f"Should NOT cooldown with only 1 failed request (below minimum threshold of {DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS})"

    # Now add more failures to reach the minimum threshold
    for _ in range(DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS - 1):
        increment_deployment_failures_for_current_minute(
            litellm_router_instance=testing_litellm_router, deployment_id=deployment_id
        )

    # Now with enough requests (all failures), it SHOULD trigger cooldown
    should_cooldown = _should_cooldown_deployment(
        testing_litellm_router, deployment_id, 500, _exception
    )
    assert (
        should_cooldown is True
    ), f"Should cooldown when we have {DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS} failed requests (100% failure rate)"
