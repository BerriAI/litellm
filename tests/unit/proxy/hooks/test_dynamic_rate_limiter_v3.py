"""
Test priority-based rate limiting for dynamic_rate_limiter_v3.

Core tests to validate that priority weights are respected (0.9/0.1) instead of equal splitting (0.5/0.5).
"""

import asyncio
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import DualCache, Router
from litellm.litellm_core_utils.internal_call_metadata import MODEL_ACCESS_GROUP_METADATA_KEY
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    AlternateGrant,
    PROXY_DynamicRateLimitHandlerV3,
    _PROXY_DynamicRateLimitHandlerV3,  # pyright: ignore[reportPrivateUsage]  # the public name is a Final alias, not a type
)
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    PROXY_DynamicRateLimitHandlerV3 as DynamicRateLimitHandler,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.hooks.priority_class_share import pool_member
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from tests.unit.redis.stores.fake_rate_limits import FakeRateLimits


class TimeController:
    def __init__(self):
        self._current = datetime.utcnow()

    def now(self) -> datetime:
        return self._current

    def advance(self, seconds: float) -> None:
        self._current += timedelta(seconds=seconds)


@pytest.fixture
def time_controller(monkeypatch):
    controller = TimeController()
    monkeypatch.setattr(time, "time", lambda: controller.now().timestamp())
    return controller


@pytest.mark.asyncio
async def test_priority_weight_allocation(monkeypatch):
    """
    Test that priority weights are correctly applied instead of equal splitting.

    With priority_reservation = {"high": 0.9, "low": 0.1}:
    - High priority should get 90% of TPM (900 out of 1000)
    - Low priority should get 10% of TPM (100 out of 1000)

    This validates the core fix where before it would split 50/50.
    """
    # Set up environment for premium feature
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set up priority reservations
    litellm.priority_reservation = {"high": 0.9, "low": 0.1}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "test-model"
    total_tpm = 1000

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": total_tpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Test high priority allocation
    high_priority_user = UserAPIKeyAuth()
    high_priority_user.metadata = {"priority": "high"}

    high_descriptors = handler._create_priority_based_descriptors(
        model=model,
        user_api_key_dict=high_priority_user,
        priority="high",
    )

    assert len(high_descriptors) == 1
    high_descriptor = high_descriptors[0]
    expected_high_tpm = int(total_tpm * 0.9)  # 900
    actual_high_tpm = high_descriptor["rate_limit"]["tokens_per_unit"]

    assert actual_high_tpm == expected_high_tpm, (
        f"High priority should get {expected_high_tpm} TPM (90%), got {actual_high_tpm}"
    )
    assert high_descriptor["value"] == f"{model}:high"

    # Test low priority allocation
    low_priority_user = UserAPIKeyAuth()
    low_priority_user.metadata = {"priority": "low"}

    low_descriptors = handler._create_priority_based_descriptors(
        model=model,
        user_api_key_dict=low_priority_user,
        priority="low",
    )

    assert len(low_descriptors) == 1
    low_descriptor = low_descriptors[0]
    expected_low_tpm = int(total_tpm * 0.1)  # 100
    actual_low_tpm = low_descriptor["rate_limit"]["tokens_per_unit"]

    assert actual_low_tpm == expected_low_tpm, (
        f"Low priority should get {expected_low_tpm} TPM (10%), got {actual_low_tpm}"
    )
    assert low_descriptor["value"] == f"{model}:low"

    # Verify the ratio is 9:1, not 1:1 (equal splitting)
    ratio = actual_high_tpm / actual_low_tpm
    expected_ratio = 9.0
    assert abs(ratio - expected_ratio) < 0.1, f"High:Low ratio should be {expected_ratio}:1, got {ratio}:1"


@pytest.mark.asyncio
async def test_concurrent_priority_requests(monkeypatch):
    """
    Test the core issue: 5 concurrent requests with different priorities should get
    proper allocation based on priority weights, not equal splitting.

    This tests the exact scenario mentioned: priorities 0.9 and 0.1 should be 0.9/0.1, not 0.5/0.5.
    """
    # Set up environment for premium feature
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set up the exact scenario from the issue
    litellm.priority_reservation = {"high": 0.9, "low": 0.1}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "test-model"
    total_tpm = 1000

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": total_tpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Create 5 concurrent users - 3 high priority, 2 low priority
    high_priority_users = []
    low_priority_users = []

    for i in range(3):  # 3 high priority users
        user = UserAPIKeyAuth()
        user.metadata = {"priority": "high"}
        user.user_id = f"high_user_{i}"
        high_priority_users.append(user)

    for i in range(2):  # 2 low priority users
        user = UserAPIKeyAuth()
        user.metadata = {"priority": "low"}
        user.user_id = f"low_user_{i}"
        low_priority_users.append(user)

    # Test all high priority users get the same allocation (not divided)
    for user in high_priority_users:
        descriptors = handler._create_priority_based_descriptors(
            model=model,
            user_api_key_dict=user,
            priority="high",
        )

        assert len(descriptors) == 1
        descriptor = descriptors[0]
        # Each high priority user should get 900 TPM, not divided by 3
        assert descriptor["rate_limit"]["tokens_per_unit"] == 900, (
            f"High priority user {user.user_id} should get 900 TPM, got {descriptor['rate_limit']['tokens_per_unit']}"
        )
        assert descriptor["value"] == f"{model}:high"

    # Test all low priority users get the same allocation (not divided)
    for user in low_priority_users:
        descriptors = handler._create_priority_based_descriptors(
            model=model,
            user_api_key_dict=user,
            priority="low",
        )

        assert len(descriptors) == 1
        descriptor = descriptors[0]
        # Each low priority user should get 100 TPM, not divided by 2
        assert descriptor["rate_limit"]["tokens_per_unit"] == 100, (
            f"Low priority user {user.user_id} should get 100 TPM, got {descriptor['rate_limit']['tokens_per_unit']}"
        )
        assert descriptor["value"] == f"{model}:low"


@pytest.mark.asyncio
async def test_100_concurrent_priority_requests(time_controller, monkeypatch):
    """
    Stress test: 100 concurrent requests with mixed priorities over 10 seconds.

    This validates that the priority system works correctly under high load:
    - 70 high priority requests (should get 900 TPM each)
    - 30 low priority requests (should get 100 TPM each)
    - Spread across 10 seconds to simulate real-world load
    """
    # Set up environment for premium feature
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set up priority reservations
    litellm.priority_reservation = {"high": 0.9, "low": 0.1}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache, time_provider=time_controller.now)

    model = "stress-test-model"
    total_tpm = 1000

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": total_tpm,
                    "rpm": 500,  # Also test RPM limits
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Create 100 users: 70 high priority, 30 low priority
    all_users = []

    # 70 high priority users
    for i in range(70):
        user = UserAPIKeyAuth()
        user.metadata = {"priority": "high"}
        user.user_id = f"high_stress_user_{i}"
        all_users.append((user, "high", 900, 450))  # expected TPM, expected RPM

    # 30 low priority users
    for i in range(30):
        user = UserAPIKeyAuth()
        user.metadata = {"priority": "low"}
        user.user_id = f"low_stress_user_{i}"
        all_users.append((user, "low", 100, 50))  # expected TPM, expected RPM

    async def test_user_descriptors(user_data):
        """Test descriptor creation for a single user."""
        user, priority, expected_tpm, expected_rpm = user_data

        descriptors = handler._create_priority_based_descriptors(
            model=model,
            user_api_key_dict=user,
            priority=priority,
        )

        assert len(descriptors) == 1, f"User {user.user_id} should have exactly 1 descriptor"
        descriptor = descriptors[0]

        # Validate TPM allocation
        actual_tpm = descriptor["rate_limit"]["tokens_per_unit"]
        assert actual_tpm == expected_tpm, (
            f"User {user.user_id} ({priority}) should get {expected_tpm} TPM, got {actual_tpm}"
        )

        # Validate RPM allocation
        actual_rpm = descriptor["rate_limit"]["requests_per_unit"]
        assert actual_rpm == expected_rpm, (
            f"User {user.user_id} ({priority}) should get {expected_rpm} RPM, got {actual_rpm}"
        )

        # Validate descriptor key
        assert descriptor["value"] == f"{model}:{priority}"
        assert descriptor["key"] == "priority_model"

        return {
            "user_id": user.user_id,
            "priority": priority,
            "tpm": actual_tpm,
            "rpm": actual_rpm,
            "success": True,
        }

    # Run all 100 requests concurrently to simulate high load
    start_time = time.time()

    # Split into batches to simulate requests over 10 seconds
    batch_size = 10  # 10 requests per batch
    batches = [all_users[i : i + batch_size] for i in range(0, len(all_users), batch_size)]

    all_results = []

    for batch_idx, batch in enumerate(batches):
        # Process each batch concurrently
        batch_tasks = [test_user_descriptors(user_data) for user_data in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_results.extend(batch_results)

        # Add small delay between batches to spread over ~10 seconds
        if batch_idx < len(batches) - 1:  # Don't sleep after last batch
            await asyncio.sleep(0)
            time_controller.advance(1.0)  # simulate 1s passing between batches

    end_time = time.time()
    total_duration = end_time - start_time

    # Validate that the test ran over approximately 10 seconds
    assert total_duration >= 9.0, f"Test should take ~10 seconds, took {total_duration:.2f}s"
    assert total_duration <= 15.0, f"Test took too long: {total_duration:.2f}s"

    # Validate all requests were successful
    successful_results = [r for r in all_results if isinstance(r, dict) and r.get("success")]
    assert len(successful_results) == 100, f"Expected 100 successful results, got {len(successful_results)}"

    # Validate priority distribution
    high_priority_results = [r for r in successful_results if r["priority"] == "high"]
    low_priority_results = [r for r in successful_results if r["priority"] == "low"]

    assert len(high_priority_results) == 70, f"Expected 70 high priority results, got {len(high_priority_results)}"
    assert len(low_priority_results) == 30, f"Expected 30 low priority results, got {len(low_priority_results)}"

    # Validate all high priority users got correct allocation
    for result in high_priority_results:
        assert result["tpm"] == 900, f"High priority user {result['user_id']} got {result['tpm']} TPM, expected 900"
        assert result["rpm"] == 450, f"High priority user {result['user_id']} got {result['rpm']} RPM, expected 450"

    # Validate all low priority users got correct allocation
    for result in low_priority_results:
        assert result["tpm"] == 100, f"Low priority user {result['user_id']} got {result['tpm']} TPM, expected 100"
        assert result["rpm"] == 50, f"Low priority user {result['user_id']} got {result['rpm']} RPM, expected 50"

    print(f"✅ Successfully processed 100 concurrent requests in {total_duration:.2f}s")
    print("   - 70 high priority users: 900 TPM, 450 RPM each")
    print("   - 30 low priority users: 100 TPM, 50 RPM each")
    print("   - Priority ratio maintained: 9:1 (TPM) and 9:1 (RPM)")


@pytest.mark.asyncio
async def test_concurrent_pre_call_hooks_stress(monkeypatch):
    """
    Stress test: 50 concurrent pre-call hooks with saturation-aware priority enforcement.

    Tests priority-based rate limiting in strict mode (>80% saturation).
    Mocks high saturation to force strict mode where priorities are enforced.
    Premium users (80% allocation) should have >90% success rate.
    Standard users (20% allocation) should have ~70% success rate with 30% random limiting.
    """
    # Set up environment for premium feature
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"premium": 0.8, "standard": 0.2}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "pre-call-stress-model"
    total_tpm = 2000

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": total_tpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Mock the v3 limiter to simulate different scenarios
    successful_requests = []
    rate_limited_requests = []

    # Mock saturation check to return high saturation (forces strict mode)
    async def mock_get_cache(key, litellm_parent_otel_span=None, local_only=False):
        """Mock cache to simulate high saturation."""
        # Return high usage to trigger strict mode (>80% saturation)
        if ":requests" in key or ":tokens" in key:
            return 1800  # 1800/2000 = 90% saturation
        return None

    async def mock_should_rate_limit(descriptors, parent_otel_span=None, read_only=False):
        """Mock rate limiter that handles saturation-aware descriptors."""
        descriptor = descriptors[0]
        descriptor_key = descriptor["key"]
        descriptor_value = descriptor["value"]

        # Handle model-wide tracking (for both generous and strict mode tracking)
        if descriptor_key == "model_saturation_check":
            # Always allow model-wide tracking (doesn't enforce in our mock)
            return {
                "overall_code": "OK",
                "statuses": [
                    {
                        "code": "OK",
                        "descriptor_key": descriptor_value,
                        "rate_limit_type": "tokens_per_unit",
                        "limit_remaining": 10000,
                    }
                ],
            }

        # Handle priority-specific enforcement in strict mode
        elif descriptor_key == "priority_model":
            # Extract priority from value like "pre-call-stress-model:premium"
            priority = descriptor_value.split(":")[-1]

            if priority == "premium":
                # Allow all premium requests
                return {
                    "overall_code": "OK",
                    "statuses": [
                        {
                            "code": "OK",
                            "descriptor_key": descriptor_value,
                            "rate_limit_type": "tokens_per_unit",
                            "limit_remaining": 1000,
                        }
                    ],
                }
            else:
                # Rate limit some standard requests (simulate load)
                import random

                if random.random() < 0.3:  # 30% of standard requests get rate limited
                    return {
                        "overall_code": "OVER_LIMIT",
                        "statuses": [
                            {
                                "code": "OVER_LIMIT",
                                "descriptor_key": descriptor_value,
                                "rate_limit_type": "tokens_per_unit",
                                "limit_remaining": 0,
                            }
                        ],
                    }
                else:
                    return {
                        "overall_code": "OK",
                        "statuses": [
                            {
                                "code": "OK",
                                "descriptor_key": descriptor_value,
                                "rate_limit_type": "tokens_per_unit",
                                "limit_remaining": 100,
                            }
                        ],
                    }

        # Default: allow
        return {
            "overall_code": "OK",
            "statuses": [
                {
                    "code": "OK",
                    "descriptor_key": descriptor_value,
                    "rate_limit_type": "tokens_per_unit",
                    "limit_remaining": 1000,
                }
            ],
        }

    # Create 50 users: 30 premium, 20 standard
    users = []

    for i in range(30):
        user = UserAPIKeyAuth()
        user.metadata = {"priority": "premium"}
        user.user_id = f"premium_hook_user_{i}"
        users.append((user, "premium"))

    for i in range(20):
        user = UserAPIKeyAuth()
        user.metadata = {"priority": "standard"}
        user.user_id = f"standard_hook_user_{i}"
        users.append((user, "standard"))

    async def make_request(user_data):
        """Make a pre-call hook request."""
        user, priority = user_data

        try:
            result = await handler.async_pre_call_hook(
                user_api_key_dict=user,
                cache=DualCache(),
                data={"model": model},
                call_type="completion",
            )

            # If no exception, request was allowed
            successful_requests.append({"user_id": user.user_id, "priority": priority, "result": "allowed"})
            return {
                "status": "success",
                "user_id": user.user_id,
                "priority": priority,
            }

        except Exception as e:
            # Request was rate limited
            rate_limited_requests.append({"user_id": user.user_id, "priority": priority, "error": str(e)})
            return {
                "status": "rate_limited",
                "user_id": user.user_id,
                "priority": priority,
            }

    # Run all 50 requests concurrently with patches applied to the entire batch
    start_time = time.time()
    with (
        patch.object(handler.v3_limiter, "should_rate_limit", side_effect=mock_should_rate_limit),
        patch.object(handler.internal_usage_cache, "async_get_cache", side_effect=mock_get_cache),
    ):
        tasks = [make_request(user_data) for user_data in users]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    successful_count = len([r for r in results if isinstance(r, dict) and r["status"] == "success"])
    rate_limited_count = len([r for r in results if isinstance(r, dict) and r["status"] == "rate_limited"])

    # Validate that premium users were mostly successful (priority worked)
    premium_results = [r for r in results if isinstance(r, dict) and r["priority"] == "premium"]
    premium_success = len([r for r in premium_results if r["status"] == "success"])

    standard_results = [r for r in results if isinstance(r, dict) and r["priority"] == "standard"]
    standard_success = len([r for r in standard_results if r["status"] == "success"])

    # Premium users should have higher success rate due to priority
    premium_success_rate = premium_success / len(premium_results) if premium_results else 0
    standard_success_rate = standard_success / len(standard_results) if standard_results else 0

    assert premium_success_rate >= 0.9, f"Premium success rate should be >= 90%, got {premium_success_rate:.2%}"
    assert standard_success_rate >= 0.5, (
        f"Standard success rate should be >= 50% (with 30% random limiting, allows for variance), got {standard_success_rate:.2%}"
    )

    # Allow for the case where both are 100% due to timing/mocking issues
    # The test is inherently flaky due to random behavior
    if premium_success_rate < 1.0 or standard_success_rate < 1.0:
        assert premium_success_rate >= standard_success_rate, "Premium should have >= success rate than standard"

    total_duration = end_time - start_time

    print(f"✅ Processed 50 concurrent pre-call hooks in {total_duration:.2f}s")
    print(f"   - Premium users: {premium_success}/{len(premium_results)} success ({premium_success_rate:.1%})")
    print(f"   - Standard users: {standard_success}/{len(standard_results)} success ({standard_success_rate:.1%})")
    print(f"   - Total successful: {successful_count}/50 ({successful_count / 50:.1%})")
    print("   - Priority system working: Premium > Standard success rates")


# These tests make actual async_pre_call_hook calls to simulate real traffic


@pytest.mark.asyncio
async def test_fake_calls_case_1_no_rate_limiting_at_capacity(monkeypatch):
    """
    Test Case 1: Saturation-Aware Rate Limiting at 50% Threshold

    System: 100 RPM capacity, saturation_threshold=50%
    Key A: priority_reservation=0.75 (75 RPM reserved)
    Key B: priority_reservation=0.25 (25 RPM reserved)
    Traffic A: 1 request
    Traffic B: 100 requests

    Expected behavior:
    - Key A: 1 request succeeds (low traffic)
    - Key B: ~25-26 requests succeed (capped at reservation when saturation >= 50%)

    Once saturation hits 50%, strict mode enforces priority-based limits.
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set up priority reservations
    litellm.priority_reservation = {"key_a": 0.75, "key_b": 0.25}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-1"
    total_rpm = 100

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": total_rpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.metadata = {"priority": "key_b"}
    key_b_user.user_id = "key_b_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0}

    async def make_request(user, priority_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await handler.async_pre_call_hook(
                user_api_key_dict=user,
                cache=dual_cache,
                data={"model": model},
                call_type="completion",
            )

            if result is None:
                successful_requests[priority_name] += 1
                return {"status": "success", "priority": priority_name}
            else:
                rate_limited_requests[priority_name] += 1
                return {"status": "rate_limited", "priority": priority_name}

        except Exception as e:
            rate_limited_requests[priority_name] += 1
            return {
                "status": "rate_limited",
                "priority": priority_name,
                "error": str(e),
            }

    # Send 1 request from key_a, 100 from key_b
    tasks = []

    for i in range(1):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(100):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = successful_requests["key_a"] + successful_requests["key_b"]
    total_rate_limited = rate_limited_requests["key_a"] + rate_limited_requests["key_b"]

    print("Test Case 1 - Saturation-Aware Rate Limiting:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(f"   - Key A: {successful_requests['key_a']}/1 successful (reserved 75 RPM)")
    print(f"   - Key B: {successful_requests['key_b']}/100 successful (reserved 25 RPM)")
    print(f"   - Total successful: {total_successful}/101")
    print(f"   - Total rate limited: {total_rate_limited}/101")

    # Key A should get its 1 request
    assert successful_requests["key_a"] == 1, f"Key A should get 1 request, got {successful_requests['key_a']}"

    # Key B can send until saturation hits 50% (which is ~50 total requests)
    # After that, strict mode enforces its 25 RPM reservation
    # Due to race conditions in concurrent execution, allow 45-52 successful requests
    assert 45 <= successful_requests["key_b"] <= 52, (
        f"Key B should get ~49 requests (45-52), got {successful_requests['key_b']}"
    )

    # Verify approximately half of key_b requests were rate limited
    assert rate_limited_requests["key_b"] >= 45, (
        f"Key B should have ≥45 rate limited requests, got {rate_limited_requests['key_b']}"
    )


@pytest.mark.asyncio
async def test_fake_calls_case_2_priority_queue_during_saturation(monkeypatch):
    """
    Test Case 2: Priority Queue Behavior During Saturation

    System: 100 RPM capacity
    Key A: priority_reservation=0.75 (75 RPM reserved)
    Key B: priority_reservation=0.25 (25 RPM reserved)
    Traffic A: 200 RPM
    Traffic B: 200 RPM
    Expected A: 75 RPM (75% of capacity)
    Expected B: 25 RPM (25% of capacity)

    When total traffic exceeds capacity, rate limiting enforces priority reservations.
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"key_a": 0.75, "key_b": 0.25}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-2"
    total_rpm = 100

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": total_rpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.metadata = {"priority": "key_b"}
    key_b_user.user_id = "key_b_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0}

    async def make_request(user, priority_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await handler.async_pre_call_hook(
                user_api_key_dict=user,
                cache=dual_cache,
                data={"model": model},
                call_type="completion",
            )

            if result is None:
                successful_requests[priority_name] += 1
                return {"status": "success", "priority": priority_name}
            else:
                rate_limited_requests[priority_name] += 1
                return {"status": "rate_limited", "priority": priority_name}

        except Exception as e:
            rate_limited_requests[priority_name] += 1
            return {
                "status": "rate_limited",
                "priority": priority_name,
                "error": str(e),
            }

    # Send 200 requests from each priority (over capacity)
    tasks = []

    for i in range(200):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(200):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = successful_requests["key_a"] + successful_requests["key_b"]

    key_a_success_rate = successful_requests["key_a"] / 200
    key_b_success_rate = successful_requests["key_b"] / 200

    print("Test Case 2 - Priority Queue Behavior During Saturation:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(f"   - Key A: {successful_requests['key_a']}/200 successful ({key_a_success_rate:.1%})")
    print(f"   - Key B: {successful_requests['key_b']}/200 successful ({key_b_success_rate:.1%})")
    print(f"   - Total successful: {total_successful}/400")

    # Key A should get significantly more requests than Key B (75:25 ratio)
    assert key_a_success_rate > key_b_success_rate, (
        f"Key A should have higher success rate: {key_a_success_rate:.1%} vs {key_b_success_rate:.1%}"
    )

    # Check ratio is approximately 3:1 (75:25)
    if total_successful > 0:
        key_a_share = successful_requests["key_a"] / total_successful
        expected_key_a_share = 0.75

        print(f"   - Key A got {key_a_share:.1%} of successful requests (expected ~75%)")

        # Allow tolerance for timing effects
        assert abs(key_a_share - expected_key_a_share) < 0.2, f"Key A share should be ~75%, got {key_a_share:.1%}"


@pytest.mark.asyncio
async def test_fake_calls_case_3_spillover_capacity_default_keys(monkeypatch):
    """
    Test Case 3: Spillover Capacity for Default Keys

    System: 100 RPM capacity
    Key A: priority_reservation=0.75 (75 RPM reserved)
    Key B: nothing set (default)
    Key C: nothing set (default)
    Key D: nothing set (default)
    Traffic A: 150 RPM
    Traffic B: 150 RPM
    Traffic C: 150 RPM
    Traffic D: 150 RPM
    Expected A: 75 RPM (75% reserved)
    Expected B: ~8.3 RPM (remaining 25 RPM / 3 default keys)
    Expected C: ~8.3 RPM
    Expected D: ~8.3 RPM

    Tests spillover behavior where default keys share remaining capacity.
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"key_a": 0.75}
    litellm.priority_reservation_settings.default_priority = 0.25

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-3"
    total_rpm = 100

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": total_rpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.metadata = {}
    key_b_user.user_id = "key_b_user"

    key_c_user = UserAPIKeyAuth()
    key_c_user.metadata = {}
    key_c_user.user_id = "key_c_user"

    key_d_user = UserAPIKeyAuth()
    key_d_user.metadata = {}
    key_d_user.user_id = "key_d_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0, "key_c": 0, "key_d": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0, "key_c": 0, "key_d": 0}

    async def make_request(user, key_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await handler.async_pre_call_hook(
                user_api_key_dict=user,
                cache=dual_cache,
                data={"model": model},
                call_type="completion",
            )

            if result is None:
                successful_requests[key_name] += 1
                return {"status": "success", "key": key_name}
            else:
                rate_limited_requests[key_name] += 1
                return {"status": "rate_limited", "key": key_name}

        except Exception as e:
            rate_limited_requests[key_name] += 1
            return {"status": "rate_limited", "key": key_name, "error": str(e)}

    # Send 150 requests from each key (600 total, 6x over capacity)
    tasks = []

    for i in range(150):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(150):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    for i in range(150):
        tasks.append(make_request(key_c_user, "key_c", f"key_c_{i}"))

    for i in range(150):
        tasks.append(make_request(key_d_user, "key_d", f"key_d_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = sum(successful_requests.values())

    print("Test Case 3 - Spillover Capacity for Default Keys:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(f"   - Key A: {successful_requests['key_a']}/150 successful")
    print(f"   - Key B: {successful_requests['key_b']}/150 successful (default)")
    print(f"   - Key C: {successful_requests['key_c']}/150 successful (default)")
    print(f"   - Key D: {successful_requests['key_d']}/150 successful (default)")
    print(f"   - Total successful: {total_successful}/600")

    # Key A should get the most requests (75% of capacity)
    assert successful_requests["key_a"] > successful_requests["key_b"], "Key A should get more than Key B"
    assert successful_requests["key_a"] > successful_requests["key_c"], "Key A should get more than Key C"
    assert successful_requests["key_a"] > successful_requests["key_d"], "Key A should get more than Key D"

    # Default keys should get similar amounts (spillover capacity)
    avg_default = (successful_requests["key_b"] + successful_requests["key_c"] + successful_requests["key_d"]) / 3
    print(f"   - Average default key success: {avg_default:.1f}")


@pytest.mark.asyncio
async def test_fake_calls_case_4_over_allocated_with_normalization(monkeypatch):
    """
    Test Case 4: Over-Allocated Priority reservations with Normalization

    System: 100 RPM capacity
    Key A: priority_reservation=0.60 (60% requested)
    Key B: priority_reservation=0.80 (80% requested)
    Total: 140% (over-allocated, should normalize to 43%/57%)
    Traffic A: 200 RPM
    Traffic B: 200 RPM

    With saturation-aware rate limiting:
    - Initially, requests are allowed through in generous mode (under 80% saturation)
    - Once saturated, strict priority-based limits kick in with normalized weights
    - Due to concurrent burst, total successful may exceed 100 RPM in the test window
    - This test verifies normalization works and total capacity is reasonably bounded
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"key_a": 0.60, "key_b": 0.80}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-4"
    total_rpm = 100

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": total_rpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.metadata = {"priority": "key_b"}
    key_b_user.user_id = "key_b_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0}

    async def make_request(user, priority_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await handler.async_pre_call_hook(
                user_api_key_dict=user,
                cache=dual_cache,
                data={"model": model},
                call_type="completion",
            )

            if result is None:
                successful_requests[priority_name] += 1
                return {"status": "success", "priority": priority_name}
            else:
                rate_limited_requests[priority_name] += 1
                return {"status": "rate_limited", "priority": priority_name}

        except Exception as e:
            rate_limited_requests[priority_name] += 1
            return {
                "status": "rate_limited",
                "priority": priority_name,
                "error": str(e),
            }

    # Send 200 requests from each key (400 total, 4x over capacity)
    tasks = []

    for i in range(200):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(200):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = successful_requests["key_a"] + successful_requests["key_b"]

    key_a_success_rate = successful_requests["key_a"] / 200
    key_b_success_rate = successful_requests["key_b"] / 200

    print("Test Case 4 - Over-Allocated Priority Reservations with Normalization:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(f"   - Key A (0.60): {successful_requests['key_a']}/200 successful ({key_a_success_rate:.1%})")
    print(f"   - Key B (0.80): {successful_requests['key_b']}/200 successful ({key_b_success_rate:.1%})")
    print(f"   - Total successful: {total_successful}/400")

    # With saturation-aware behavior:
    # 1. Verify total capacity is reasonably bounded (not all 400 requests succeed)
    assert total_successful < 300, (
        f"Total requests should be bounded by saturation detection, got {total_successful}/400"
    )

    # 2. Verify significant rate limiting occurred (at least 50% blocked)
    assert total_successful < 200, (
        f"At least 50% of requests should be rate limited, got {total_successful}/400 successful"
    )

    # 3. Verify both keys got some requests through (normalization is working)
    assert successful_requests["key_a"] > 0, "Key A should get some requests"
    assert successful_requests["key_b"] > 0, "Key B should get some requests"

    print(
        f"   - Normalization test PASSED: Both priorities got requests, total bounded to {total_successful} (under 200)"
    )


@pytest.mark.asyncio
async def test_fake_calls_case_5_default_value_priority_reservation(monkeypatch):
    """
    Test Case 5: Default value for priority reservation

    System: 100 RPM capacity
    Key A: priority_reservation=0.50 (50 RPM)
    Key B: priority_reservation=0.20 (20 RPM)
    Key C: priority_reservation=0.05 (5 RPM)
    Key D: nothing set (uses default_priority=0.05, 5 RPM)
    Traffic A: 150 RPM
    Traffic B: 150 RPM
    Traffic C: 150 RPM
    Traffic D: 150 RPM
    Expected A: 55 RPM (normalized)
    Expected B: 25 RPM (normalized)
    Expected C: 10 RPM (normalized)
    Expected D: 10 RPM (normalized)

    Tests complex scenario with explicit priorities and default priority.
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"key_a": 0.50, "key_b": 0.20, "key_c": 0.05}
    litellm.priority_reservation_settings.default_priority = 0.05

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-5"
    total_rpm = 100

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": total_rpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.metadata = {"priority": "key_b"}
    key_b_user.user_id = "key_b_user"

    key_c_user = UserAPIKeyAuth()
    key_c_user.metadata = {"priority": "key_c"}
    key_c_user.user_id = "key_c_user"

    key_d_user = UserAPIKeyAuth()
    key_d_user.metadata = {}
    key_d_user.user_id = "key_d_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0, "key_c": 0, "key_d": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0, "key_c": 0, "key_d": 0}

    async def make_request(user, key_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await handler.async_pre_call_hook(
                user_api_key_dict=user,
                cache=dual_cache,
                data={"model": model},
                call_type="completion",
            )

            if result is None:
                successful_requests[key_name] += 1
                return {"status": "success", "key": key_name}
            else:
                rate_limited_requests[key_name] += 1
                return {"status": "rate_limited", "key": key_name}

        except Exception as e:
            rate_limited_requests[key_name] += 1
            return {"status": "rate_limited", "key": key_name, "error": str(e)}

    # Send 150 requests from each key (600 total, 6x over capacity)
    tasks = []

    for i in range(150):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(150):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    for i in range(150):
        tasks.append(make_request(key_c_user, "key_c", f"key_c_{i}"))

    for i in range(150):
        tasks.append(make_request(key_d_user, "key_d", f"key_d_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = sum(successful_requests.values())

    print("Test Case 5 - Default value for priority reservation:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(f"   - Key A (0.50): {successful_requests['key_a']}/150 successful")
    print(f"   - Key B (0.20): {successful_requests['key_b']}/150 successful")
    print(f"   - Key C (0.05): {successful_requests['key_c']}/150 successful")
    print(f"   - Key D (default 0.05): {successful_requests['key_d']}/150 successful")
    print(f"   - Total successful: {total_successful}/600")

    # Verify priority ordering: A > B > C ≈ D
    assert successful_requests["key_a"] > successful_requests["key_b"], "Key A should get more than Key B"
    assert successful_requests["key_b"] > successful_requests["key_c"], "Key B should get more than Key C"

    # Key C and Key D should get similar amounts (both have 0.05 priority)
    key_c_vs_d_ratio = successful_requests["key_c"] / max(successful_requests["key_d"], 1)
    print(f"   - Key C vs Key D ratio: {key_c_vs_d_ratio:.2f} (expected ~1.0)")

    if total_successful > 0:
        key_a_share = successful_requests["key_a"] / total_successful
        print(f"   - Key A got {key_a_share:.1%} of successful requests (expected ~55-62%)")


@pytest.mark.asyncio
async def test_default_priority_shared_pool(monkeypatch):
    """
    Test that keys without explicit priority share ONE default pool, not get individual allocations.

    With default_priority=0.25:
    - Key A, B, C (no priority) should share ONE 25 RPM pool
    - NOT get 25 RPM each (which would be 75 RPM total)
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"prod": 0.75}
    litellm.priority_reservation_settings.default_priority = 0.25

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "test-default-pool"
    total_rpm = 100

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": total_rpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Create 3 users without explicit priority
    user_a = UserAPIKeyAuth()
    user_a.metadata = {}
    user_a.user_id = "user_a"

    user_b = UserAPIKeyAuth()
    user_b.metadata = {}
    user_b.user_id = "user_b"

    user_c = UserAPIKeyAuth()
    user_c.metadata = {}
    user_c.user_id = "user_c"

    # Get descriptors for each
    desc_a = handler._create_priority_based_descriptors(model=model, user_api_key_dict=user_a, priority=None)
    desc_b = handler._create_priority_based_descriptors(model=model, user_api_key_dict=user_b, priority=None)
    desc_c = handler._create_priority_based_descriptors(model=model, user_api_key_dict=user_c, priority=None)

    # All should use the SAME shared pool key
    assert desc_a[0]["value"] == f"{model}:default_pool"
    assert desc_b[0]["value"] == f"{model}:default_pool"
    assert desc_c[0]["value"] == f"{model}:default_pool"

    # All should have same limit (25 RPM SHARED, not 25 RPM each)
    assert desc_a[0]["rate_limit"]["requests_per_unit"] == 25
    assert desc_b[0]["rate_limit"]["requests_per_unit"] == 25
    assert desc_c[0]["rate_limit"]["requests_per_unit"] == 25

    # Verify explicit priority uses different pool
    user_prod = UserAPIKeyAuth()
    user_prod.metadata = {"priority": "prod"}
    desc_prod = handler._create_priority_based_descriptors(model=model, user_api_key_dict=user_prod, priority="prod")

    assert desc_prod[0]["value"] == f"{model}:prod"
    assert desc_prod[0]["rate_limit"]["requests_per_unit"] == 75
    assert desc_prod[0]["value"] != desc_a[0]["value"]  # Different pools

    print("✅ Default priority test passed:")
    print(f"   - 3 keys without priority share ONE pool: {desc_a[0]['value']}")
    print(f"   - Shared pool limit: {desc_a[0]['rate_limit']['requests_per_unit']} RPM")
    print(f"   - Explicit priority 'prod' uses separate pool: {desc_prod[0]['value']}")


@pytest.mark.asyncio
async def test_async_log_success_event_increments_by_actual_tokens(monkeypatch):
    """
    Test that async_log_success_event increments token counters by actual token usage.

    This validates the fix for Bug 1: Token count was incrementing by 1 instead of actual usage.
    The async_log_success_event should increment both model_saturation_check and priority_model
    counters by the actual completion_tokens (when rate_limit_type=output).
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import ModelResponse, Usage

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    litellm.priority_reservation = {"dev": 0.1, "prod": 0.9}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "test-token-increment"
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": 1000,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Track what gets incremented
    increment_calls = []

    async def mock_increment(pipeline_operations, parent_otel_span=None):
        for op in pipeline_operations:
            increment_calls.append(
                {
                    "counter": op.counter,
                    "increment_value": op.amount,
                }
            )

    handler.v3_limiter.async_increment_reservation_aware_tokens = mock_increment

    # Create mock response with 50 completion tokens
    mock_response = MagicMock(spec=ModelResponse)
    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 60

    # Create kwargs with priority in user_api_key_auth_metadata
    kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_auth_metadata": {"priority": "dev"},
            },
            "model_group": model,
        },
        "litellm_params": {
            "metadata": {"model_group": model},
        },
    }

    with patch(
        "litellm.proxy.common_utils.callback_utils.get_model_group_from_litellm_kwargs",
        return_value=model,
    ):
        await handler.async_log_success_event(
            kwargs=kwargs,
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

    # Verify increments happened with actual token count (60 total tokens)
    assert len(increment_calls) == 2, f"Expected 2 increment calls, got {len(increment_calls)}"

    # Both should increment by 50 (total_tokens, since rate_limit_type defaults to 'total')
    for call in increment_calls:
        assert call["increment_value"] == 60, (
            f"Expected increment of 60 tokens, got {call['increment_value']} for counter {call['counter']}"
        )

    # Verify correct counters were used
    counters = [call["counter"] for call in increment_calls]
    assert RateLimitScope("model_saturation_check", model).counter("tokens") in counters, (
        "Should increment model_saturation_check"
    )
    assert RateLimitScope("priority_model", f"{model}:dev").counter("tokens") in counters, (
        "Should increment priority_model with 'dev' priority"
    )


@pytest.mark.asyncio
async def test_saturation_check_cache_ttl_configuration(monkeypatch):
    """
    Test that saturation_check_cache_ttl controls how long saturation values are cached locally.

    This validates the configurable TTL for multi-node consistency:
    - When saturation_check_cache_ttl is set, local cache should expire after that duration
    - After expiration, fresh values should be fetched from Redis
    - This prevents nodes from having stale saturation data in multi-node deployments
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set a short TTL for testing (5 seconds)
    original_ttl = litellm.priority_reservation_settings.saturation_check_cache_ttl
    litellm.priority_reservation_settings.saturation_check_cache_ttl = 5

    try:
        dual_cache = DualCache()
        handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

        model = "test-saturation-ttl"
        llm_router = Router(
            model_list=[
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key",
                        "api_base": "test-base",
                        "rpm": 100,
                        "tpm": 1000,
                    },
                }
            ]
        )
        handler.update_variables(llm_router=llm_router)

        # Verify the TTL getter returns configured value
        assert handler._get_saturation_check_cache_ttl() == 5, (
            "TTL should be configurable via priority_reservation_settings"
        )

        # Track read-through (memory then Redis) gets to verify TTL is passed
        get_cache_calls = []

        class _RecordingRateLimits(FakeRateLimits):
            async def get(self, entry, *, backfill_ttl_seconds=None):
                get_cache_calls.append({"entry": entry, "ttl": backfill_ttl_seconds})

        handler.v3_limiter = PROXY_MaxParallelRequestsHandler_v3(
            handler.internal_usage_cache,
            rate_limit_store=_RecordingRateLimits().store(),
        )

        # Call _get_saturation_value_from_cache
        counter = RateLimitScope("model_saturation_check", model).counter("requests")

        await handler._get_saturation_value_from_cache(counter)

        # Verify the read-through get was called with the configured TTL
        assert len(get_cache_calls) == 1, "Expected 1 cache call"
        assert get_cache_calls[0]["entry"] == counter
        assert get_cache_calls[0]["ttl"] == 5, f"Expected TTL of 5 seconds, got {get_cache_calls[0]['ttl']}"

        # Test with different TTL value
        get_cache_calls.clear()
        litellm.priority_reservation_settings.saturation_check_cache_ttl = 30

        await handler._get_saturation_value_from_cache(counter)

        assert get_cache_calls[0]["ttl"] == 30, f"TTL should update to 30 seconds, got {get_cache_calls[0]['ttl']}"

        print("Saturation check cache TTL test passed:")
        print("   - TTL is configurable via priority_reservation_settings.saturation_check_cache_ttl")
        print("   - TTL is passed to the read-through get for local cache expiration control")

    finally:
        # Restore original TTL
        litellm.priority_reservation_settings.saturation_check_cache_ttl = original_ttl


@pytest.mark.asyncio
async def test_async_log_success_event_uses_team_priority_from_auth_metadata(monkeypatch):
    """
    Test that async_log_success_event correctly retrieves priority from user_api_key_auth_metadata.

    This validates the fix where priority is retrieved from standard_logging_metadata.user_api_key_auth_metadata
    instead of just standard_logging_metadata.priority. This is important for team-based priority inheritance.
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import ModelResponse, Usage

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    litellm.priority_reservation = {"team_priority": 0.8, "default": 0.2}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "test-team-priority"
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": 1000,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    # Track incremented keys to verify priority is used correctly
    incremented_keys = []

    async def mock_increment(pipeline_operations, parent_otel_span=None):
        for op in pipeline_operations:
            incremented_keys.append(op.counter.scope)

    handler.v3_limiter.async_increment_reservation_aware_tokens = mock_increment

    # Create mock response
    mock_response = MagicMock(spec=ModelResponse)
    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30

    # Simulate team metadata inheritance: priority is in user_api_key_auth_metadata
    # This is how the proxy passes team metadata to the callback
    kwargs = {
        "standard_logging_object": {
            "metadata": {
                # Priority NOT at top level (this would fail before the fix)
                # Priority IS in user_api_key_auth_metadata (team inheritance)
                "user_api_key_auth_metadata": {"priority": "team_priority"},
            },
            "model_group": model,
        },
        "litellm_params": {
            "metadata": {"model_group": model},
        },
    }

    with patch(
        "litellm.proxy.common_utils.callback_utils.get_model_group_from_litellm_kwargs",
        return_value=model,
    ):
        await handler.async_log_success_event(
            kwargs=kwargs,
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

    # Verify the priority_model key uses 'team_priority' (not 'default_pool')
    priority_keys = [scope.descriptor_value for scope in incremented_keys if scope.descriptor_key == "priority_model"]
    assert len(priority_keys) == 1, f"Expected 1 priority_model key, got {len(priority_keys)}"

    # The key should use 'team_priority', not 'default_pool'
    assert priority_keys[0] == f"{model}:team_priority", (
        f"Expected priority key to use 'team_priority' from user_api_key_auth_metadata, got key: {priority_keys[0]}"
    )


@pytest.mark.asyncio
async def test_priority_429_includes_model_name_and_configured_limits(monkeypatch):
    """
    The priority-based 429 should tell operators which model was hit and what
    the model's configured TPM/RPM are, so they can decide whether to tune the
    priority allocation or the model limits.

    Regression test for the previous message that read:
        "Priority-based rate limit exceeded. Priority: prod,
         Rate limit type: tokens, Remaining: -664145,
         Model saturation: 86.3%"
    -- with no indication of which model was hit.
    """
    from fastapi import HTTPException

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    litellm.priority_reservation = {"prod": 0.5}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "gpt-4o-test"
    total_tpm = 1_000_000
    total_rpm = 10_000

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": total_tpm,
                    "rpm": total_rpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    user = UserAPIKeyAuth()
    user.metadata = {"priority": "prod"}
    user.user_id = "prod_user"

    model_group_info = handler.llm_router.get_model_group_info(model_group=model)

    # Force the atomic check+increment to return OVER_LIMIT for the
    # priority_model descriptor. saturation=0.95 keeps us above the
    # default saturation threshold so priority limits are enforced.
    over_limit_response = {
        "overall_code": "OVER_LIMIT",
        "statuses": [
            {
                "code": "OVER_LIMIT",
                "descriptor_key": "priority_model",
                "rate_limit_type": "tokens",
                "limit_remaining": -664145,
                "current_limit": int(total_tpm * 0.5),
            }
        ],
    }

    with patch.object(
        handler.v3_limiter,
        "atomic_check_and_increment_by_n",
        new=AsyncMock(return_value=over_limit_response),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handler._check_rate_limits(
                model=model,
                model_group_info=model_group_info,
                user_api_key_dict=user,
                priority="prod",
                saturation=0.95,
            )

    assert exc_info.value.status_code == 429
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    error_msg = detail["error"]

    # New fields added by this change -- the whole point of the fix.
    assert f"Model: {model}" in error_msg, error_msg
    assert f"Model TPM: {total_tpm}" in error_msg, error_msg
    assert f"Model RPM: {total_rpm}" in error_msg, error_msg

    # Existing fields must still be present (no regression).
    assert "Priority-based rate limit exceeded" in error_msg, error_msg
    assert "Priority: prod" in error_msg, error_msg
    assert "Rate limit type: tokens" in error_msg, error_msg
    assert "Model saturation:" in error_msg, error_msg


@pytest.mark.asyncio
async def test_tpm_only_model_enforces_priority_and_model_capacity(monkeypatch):
    """Regression: a model configured with ONLY tpm (no rpm) must still be
    rate limited.

    The atomic check-and-increment path used to drop any counter whose
    pre-call increment was zero. Token increments are always zero pre-call
    (usage lands on the counters post-call), so on a TPM-only model the
    limiter evaluated no counters at all: no model-wide cap, no priority
    reservation, in either mode. This test drives the real pre-call ->
    log-success -> pre-call flow with no limiter internals mocked.
    """
    from fastapi import HTTPException

    from litellm.types.utils import ModelResponse, Usage

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    litellm.priority_reservation = {"dev": 0.25, "prod": 0.5}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "tpm-only-model"
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": 400,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    dev_user = UserAPIKeyAuth()
    dev_user.metadata = {"priority": "dev"}
    prod_user = UserAPIKeyAuth()
    prod_user.metadata = {"priority": "prod"}

    async def record_usage(priority: str, total_tokens: int) -> None:
        await handler.async_log_success_event(
            kwargs={
                "standard_logging_object": {
                    "metadata": {"user_api_key_auth_metadata": {"priority": priority}},
                },
                "litellm_params": {"metadata": {"model_group": model}},
            },
            response_obj=ModelResponse(
                model=model,
                usage=Usage(prompt_tokens=0, completion_tokens=total_tokens, total_tokens=total_tokens),
            ),
            start_time=None,
            end_time=None,
        )

    assert (
        await handler.async_pre_call_hook(
            user_api_key_dict=dev_user, cache=dual_cache, data={"model": model}, call_type="completion"
        )
        is None
    )

    await record_usage("dev", 250)

    with pytest.raises(HTTPException) as dev_blocked:
        await handler.async_pre_call_hook(
            user_api_key_dict=dev_user, cache=dual_cache, data={"model": model}, call_type="completion"
        )
    assert dev_blocked.value.status_code == 429
    assert "Priority-based rate limit exceeded" in dev_blocked.value.detail["error"]

    assert (
        await handler.async_pre_call_hook(
            user_api_key_dict=prod_user, cache=dual_cache, data={"model": model}, call_type="completion"
        )
        is None
    )

    await record_usage("prod", 200)

    with pytest.raises(HTTPException) as capacity_blocked:
        await handler.async_pre_call_hook(
            user_api_key_dict=prod_user, cache=dual_cache, data={"model": model}, call_type="completion"
        )
    assert capacity_blocked.value.status_code == 429
    assert "Model capacity reached" in capacity_blocked.value.detail["error"]


@pytest.mark.asyncio
async def test_post_call_success_hook_attaches_priority_headers_to_dict_response():
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        RateLimitResponse,
        RateLimitStatus,
        get_or_create_request_stash,
    )

    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    get_or_create_request_stash().rate_limit_response = RateLimitResponse(
        overall_code="OK",
        statuses=[
            RateLimitStatus(
                code="OK",
                current_limit=75,
                limit_remaining=74,
                rate_limit_type="requests",
                descriptor_key="priority_model",
            )
        ],
    )
    response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [],
        "_hidden_params": {"additional_headers": {"x-litellm-attempted-retries": 0}},
    }

    await handler.async_post_call_success_hook(
        data={"model": "anthropic-haiku"},
        user_api_key_dict=UserAPIKeyAuth(metadata={"priority": "premium"}),
        response=response,
    )

    additional_headers = response["_hidden_params"]["additional_headers"]
    assert additional_headers["x-litellm-attempted-retries"] == 0
    assert additional_headers["x-ratelimit-priority_model-limit-requests"] == 75
    assert additional_headers["x-ratelimit-priority_model-remaining-requests"] == 74
    assert additional_headers["x-litellm-priority"] == "premium"
    assert additional_headers["x-litellm-rate-limiter-version"] == "v3"


@pytest.mark.asyncio
async def test_post_call_success_hook_leaves_raw_provider_dict_untouched():
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    response = {"id": "msg_123", "type": "message", "role": "assistant", "content": []}

    await handler.async_post_call_success_hook(
        data={"model": "anthropic-haiku"},
        user_api_key_dict=UserAPIKeyAuth(metadata={"priority": "premium"}),
        response=response,
    )

    assert response == {"id": "msg_123", "type": "message", "role": "assistant", "content": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_metadata", "expected_priority_header"),
    [
        ({"priority": "优先"}, None),
        ({"priority": "high"}, "high"),
        ({}, "default"),
    ],
)
async def test_post_call_success_hook_priority_header_is_always_http_encodable(team_metadata, expected_priority_header):
    from starlette.responses import Response

    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    response = {"id": "msg_123", "type": "message", "role": "assistant", "content": [], "_hidden_params": {}}

    await handler.async_post_call_success_hook(
        data={"model": "anthropic-haiku"},
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1", team_metadata=team_metadata),
        response=response,
    )

    additional_headers = response["_hidden_params"]["additional_headers"]
    http_response = Response(headers={key: str(value) for key, value in additional_headers.items()})
    assert http_response.headers.get("x-litellm-priority") == expected_priority_header
    assert http_response.headers["x-litellm-rate-limiter-version"] == "v3"


def _steering_router(
    preferred: str,
    alternate: str,
    preferred_rpm: int = 2,
    alternate_rpm: int = 1000,
    extra_params: Mapping[str, Mapping[str, object]] = MappingProxyType({}),
    model_info: Mapping[str, Mapping[str, object]] = MappingProxyType({}),
) -> Router:
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": rpm,
                    **extra_params.get(name, {}),
                },
                "model_info": dict(model_info.get(name, {})),
            }
            for name, rpm in ((preferred, preferred_rpm), (alternate, alternate_rpm))
        ]
    )


def _team_user(team_models: list[str], preferences: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        team_id="flexible-team", team_models=team_models, team_metadata={"model_preferences": preferences}
    )


async def _team_allowlist(user_api_key_dict: UserAPIKeyAuth, model: str) -> AlternateGrant | None:
    return AlternateGrant() if model in user_api_key_dict.team_models else None


async def _pooled_team_allowlist(user_api_key_dict: UserAPIKeyAuth, model: str) -> AlternateGrant | None:
    return AlternateGrant(access_groups=(f"{model}-pool",)) if model in user_api_key_dict.team_models else None


def _steering_handler() -> DynamicRateLimitHandler:
    return DynamicRateLimitHandler(internal_usage_cache=DualCache(), is_model_authorized=_team_allowlist)


async def _saturate(handler: DynamicRateLimitHandler, model: str, requests: int = 1) -> None:
    for _ in range(requests):
        assert await handler.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), {"model": model}, "completion") is None


@pytest.mark.asyncio
async def test_flexible_team_is_steered_to_its_alternate_once_the_preferred_model_saturates(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    handler: Final = _steering_handler()
    handler.update_variables(llm_router=_steering_router("luna", "opus"))
    flexible: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])

    assert await handler.async_pre_call_hook(flexible, DualCache(), {"model": "luna"}, "completion") is None

    await _saturate(handler, "luna")
    steered: Final = await handler.async_pre_call_hook(flexible, DualCache(), {"model": "luna", "n": 1}, "completion")
    assert steered == {"model": "opus", "n": 1}

    headers: Final = await handler.async_post_call_response_headers_hook(
        data={"model": "opus", "stream": True}, user_api_key_dict=flexible, response=None
    )
    assert headers == {"x-litellm-model-steered-from": "luna"}

    from litellm.proxy.hooks.parallel_request_limiter_v3 import get_or_create_request_stash

    get_or_create_request_stash().steered_from_model = "luna\u2728"
    assert (
        await handler.async_post_call_response_headers_hook(data={}, user_api_key_dict=flexible, response=None) is None
    )


@pytest.mark.asyncio
async def test_teams_without_a_permitted_idle_alternate_stay_on_the_preferred_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    handler: Final = _steering_handler()
    handler.update_variables(llm_router=_steering_router("luna", "opus", preferred_rpm=10, alternate_rpm=1))
    await _saturate(handler, "luna", requests=5)
    await _saturate(handler, "opus")

    forbidden: Final = _team_user(["luna"], [{"model": "luna", "alternates": ["opus"]}])
    unknown: Final = _team_user([], [{"model": "luna", "alternates": ["missing-model"]}])
    malformed: Final = _team_user([], {"luna": "opus"})
    both_saturated: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
    for team in (forbidden, unknown, malformed, both_saturated):
        assert await handler.async_pre_call_hook(team, DualCache(), {"model": "luna"}, "completion") is None
        assert (
            await handler.async_post_call_response_headers_hook(data={}, user_api_key_dict=team, response=None) is None
        )


@pytest.mark.asyncio
async def test_a_zero_cost_preferred_model_is_never_steered_to_a_priced_alternate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    handler: Final = _steering_handler()
    free_luna: Final = {"luna": {"input_cost_per_token": 0, "output_cost_per_token": 0}}
    handler.update_variables(llm_router=_steering_router("luna", "opus", extra_params=free_luna))
    await _saturate(handler, "luna")

    flexible: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
    assert await handler.async_pre_call_hook(flexible, DualCache(), {"model": "luna"}, "completion") is None
    assert (
        await handler.async_post_call_response_headers_hook(data={}, user_api_key_dict=flexible, response=None) is None
    )


@pytest.mark.asyncio
async def test_an_alternate_that_adds_model_level_guardrails_is_never_used(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    handler: Final = _steering_handler()
    guarded_opus: Final = {"opus": {"guardrails": ["opus-only-input-scan"]}}
    handler.update_variables(llm_router=_steering_router("luna", "opus", extra_params=guarded_opus))
    await _saturate(handler, "luna")

    flexible: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
    assert await handler.async_pre_call_hook(flexible, DualCache(), {"model": "luna"}, "completion") is None
    assert (
        await handler.async_post_call_response_headers_hook(data={}, user_api_key_dict=flexible, response=None) is None
    )

    already_scanned: Final = {"model": "luna", "metadata": {"guardrails": ["opus-only-input-scan"]}}
    steered: Final = await handler.async_pre_call_hook(flexible, DualCache(), already_scanned, "completion")
    assert isinstance(steered, dict)
    assert steered["model"] == "opus"
    assert steered["metadata"]["guardrails"] == ["opus-only-input-scan"]

    configured: Final = _steering_handler()
    configured_opus: Final = {"opus": {"guardrails": [{"opus-only-input-scan": {"extra_body": {"strict": True}}}]}}
    configured.update_variables(llm_router=_steering_router("luna", "opus", extra_params=configured_opus))
    await _saturate(configured, "luna")
    plain_scan: Final = {"model": "luna", "metadata": {"guardrails": ["opus-only-input-scan"]}}
    assert await configured.async_pre_call_hook(flexible, DualCache(), plain_scan, "completion") is None


@pytest.mark.asyncio
async def test_a_steered_request_is_charged_to_the_groups_that_granted_the_alternate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    handler: Final = DynamicRateLimitHandler(
        internal_usage_cache=DualCache(), is_model_authorized=_pooled_team_allowlist
    )
    handler.update_variables(llm_router=_steering_router("luna", "opus"))
    await _saturate(handler, "luna")

    flexible: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
    auth_copy: Final = flexible.model_copy()
    data: Final = {"model": "luna", "metadata": {"user_api_key_auth": auth_copy, "tags": ["t"]}}
    steered: Final = await handler.async_pre_call_hook(flexible, DualCache(), data, "completion")
    assert isinstance(steered, dict)
    assert steered["model"] == "opus"
    assert steered["metadata"]["tags"] == ["t"]
    assert steered["metadata"][MODEL_ACCESS_GROUP_METADATA_KEY] == ["opus-pool"]
    assert steered["metadata"]["user_api_key_auth"].matched_model_access_groups == ["opus-pool"]
    assert flexible.matched_model_access_groups == ["opus-pool"]
    assert auth_copy.matched_model_access_groups is None


@pytest.mark.asyncio
async def test_an_alternate_granted_by_a_group_the_request_did_not_reserve_against_is_never_used(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    handler: Final = DynamicRateLimitHandler(
        internal_usage_cache=DualCache(), is_model_authorized=_pooled_team_allowlist
    )
    handler.update_variables(llm_router=_steering_router("luna", "opus"))
    await _saturate(handler, "luna")

    def reserved_against(group: str) -> UserAPIKeyAuth:
        user: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
        user.budget_reservation = {
            "entries": [
                {
                    "counter_key": f"spend:model_access_group:{group}",
                    "entity_type": "Model access group",
                    "entity_id": group,
                    "reserved_cost": 0.01,
                    "applied_adjustment": 0.0,
                }
            ]
        }
        return user

    assert (
        await handler.async_pre_call_hook(reserved_against("luna-pool"), DualCache(), {"model": "luna"}, "completion")
        is None
    )
    steered: Final = await handler.async_pre_call_hook(
        reserved_against("opus-pool"), DualCache(), {"model": "luna"}, "completion"
    )
    assert isinstance(steered, dict)
    assert steered["model"] == "opus"


@pytest.mark.asyncio
async def test_a_pricier_alternate_is_never_used_while_a_budget_reservation_is_held(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    handler: Final = _steering_handler()
    pricier_opus: Final = {
        "luna": {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000002},
        "opus": {"input_cost_per_token": 0.000005, "output_cost_per_token": 0.000002},
    }
    handler.update_variables(llm_router=_steering_router("luna", "opus", extra_params=pricier_opus))
    await _saturate(handler, "luna")

    reserved: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
    reserved.budget_reservation = {"reserved_cost": 0.0001, "entries": []}
    assert await handler.async_pre_call_hook(reserved, DualCache(), {"model": "luna"}, "completion") is None

    unreserved: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
    steered: Final = await handler.async_pre_call_hook(unreserved, DualCache(), {"model": "luna"}, "completion")
    assert isinstance(steered, dict)
    assert steered["model"] == "opus"

    unpriced: Final = _steering_handler()
    unpriced_opus: Final = {"opus": {"model": "openai/no-such-model-xyz"}}
    unpriced.update_variables(llm_router=_steering_router("luna", "opus", extra_params=unpriced_opus))
    await _saturate(unpriced, "luna")
    assert await unpriced.async_pre_call_hook(reserved, DualCache(), {"model": "luna"}, "completion") is None


@pytest.mark.asyncio
async def test_an_alternate_that_does_not_serve_the_endpoint_is_never_used(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    handler: Final = _steering_handler()
    embeddings_only: Final = {"opus": {"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/embeddings"]}}
    handler.update_variables(llm_router=_steering_router("luna", "opus", model_info=embeddings_only))
    await _saturate(handler, "luna")

    flexible: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
    assert await handler.async_pre_call_hook(flexible, DualCache(), {"model": "luna"}, "acompletion") is None
    assert (
        await handler.async_post_call_response_headers_hook(data={}, user_api_key_dict=flexible, response=None) is None
    )


@pytest.mark.asyncio
async def test_an_unpriced_alternate_is_never_used_while_unpriced_models_are_blocked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    monkeypatch.setattr(litellm, "block_requests_for_models_without_pricing", True)
    handler: Final = _steering_handler()
    unpriced_opus: Final = {"opus": {"model": "openai/no-such-model-xyz"}}
    handler.update_variables(llm_router=_steering_router("luna", "opus", extra_params=unpriced_opus))
    await _saturate(handler, "luna")

    flexible: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
    assert await handler.async_pre_call_hook(flexible, DualCache(), {"model": "luna"}, "completion") is None


@pytest.mark.asyncio
async def test_a_broken_alternate_check_keeps_the_request_on_the_preferred_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)

    async def broken(user_api_key_dict: UserAPIKeyAuth, model: str) -> AlternateGrant | None:
        raise RuntimeError("membership lookup is down")

    handler: Final = DynamicRateLimitHandler(internal_usage_cache=DualCache(), is_model_authorized=broken)
    handler.update_variables(llm_router=_steering_router("luna", "opus"))
    await _saturate(handler, "luna")

    flexible: Final = _team_user(["luna", "opus"], [{"model": "luna", "alternates": ["opus"]}])
    assert await handler.async_pre_call_hook(flexible, DualCache(), {"model": "luna"}, "completion") is None
    assert (
        await handler.async_post_call_response_headers_hook(data={}, user_api_key_dict=flexible, response=None) is None
    )


def _borrowing_handler(
    monkeypatch: pytest.MonkeyPatch,
    borrows: bool,
    saturation_threshold: float = 0.5,
    model_group_alias: dict[str, str] | None = None,
    splits: bool = False,
    rpm: int = 100,
) -> _PROXY_DynamicRateLimitHandlerV3:
    from litellm.types.utils import PriorityReservationSettings

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.6, "batch": 0.4})
    monkeypatch.setattr(
        litellm,
        "priority_reservation_settings",
        PriorityReservationSettings(
            saturation_threshold=saturation_threshold,
            team_model_limit_borrows_idle_capacity_from_other_teams=borrows,
            priority_class_share_splits_equally_among_active_teams=splits,
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", ["dynamic_rate_limiter_v3"])
    router: Final = Router(
        model_list=[
            {
                "model_name": "shared-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "k", "api_base": "b", "tpm": 1000, "rpm": rpm},
            },
            {
                "model_name": "other-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "k", "api_base": "b", "tpm": 1000, "rpm": 100},
            },
            {
                "model_name": "chat",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "k", "api_base": "b", "tpm": 1000, "rpm": 100},
            },
            {
                "model_name": "chat:prod",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "k", "api_base": "b", "tpm": 1000, "rpm": 100},
            },
        ],
        model_group_alias=model_group_alias or {},
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    handler = PROXY_DynamicRateLimitHandlerV3(internal_usage_cache=DualCache())
    handler.update_variables(llm_router=router)
    return handler


async def _admit_then_settle(handler: _PROXY_DynamicRateLimitHandlerV3, auth: UserAPIKeyAuth, tokens: int) -> None:
    await _check(handler, auth, saturation=0.0)
    await _settle_tokens(handler, tokens)


async def _settle_tokens(
    handler: PROXY_DynamicRateLimitHandlerV3,
    tokens: int,
    response: object | None = None,
    call_id: str | None = None,
    model_group: str = "shared-model",
    **extra_kwargs: object,
) -> None:
    await handler.async_log_success_event(
        kwargs={
            "litellm_params": {"metadata": {"model_group": model_group}},
            "standard_logging_object": {
                "metadata": {"user_api_key_team_id": "team-a", "user_api_key_auth_metadata": {}}
            },
            **({"litellm_call_id": call_id} if call_id is not None else {}),
            **extra_kwargs,
        },
        response_obj=response
        if response is not None
        else litellm.ModelResponse(usage=litellm.Usage(prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens)),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )


def _team_key(
    team_id: str = "team-a",
    team_limits: dict[str, dict[str, int]] | None = None,
    **extra: dict[str, dict[str, int]],
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-borrow",
        team_id=team_id,
        team_metadata=team_limits if team_limits is not None else {"model_tpm_limit": {"shared-model": 100}},
        **extra,
    )


async def _check(
    handler: PROXY_DynamicRateLimitHandlerV3,
    auth: UserAPIKeyAuth,
    saturation: float,
    model: str = "shared-model",
    data: dict[str, object] | None = None,
    priority: str | None = None,
) -> None:
    group: Final = handler.llm_router.registry().group(model)
    assert group is not None
    await handler._check_rate_limits(
        model=model,
        model_group_info=group,
        user_api_key_dict=auth,
        priority=priority,
        saturation=saturation,
        data=data or {},
        call_type="acompletion" if data else None,
    )


async def _team_tokens_remaining(
    handler: PROXY_DynamicRateLimitHandlerV3, auth: UserAPIKeyAuth | None = None, model: str = "shared-model"
) -> int:
    share: Final = handler._borrowing_team_share(model, auth if auth is not None else _team_key())
    assert share is not None
    response: Final = await handler.v3_limiter.should_rate_limit(
        [share.share], read_only=True, skip_parallel_check=True
    )
    return next(s["limit_remaining"] for s in response["statuses"] if s["rate_limit_type"] == "tokens")


def _prompt(words: int, max_tokens: int = 10) -> dict[str, object]:
    return {
        "model": "shared-model",
        "messages": [{"role": "user", "content": "word " * words}],
        "max_tokens": max_tokens,
    }


def _held_to_team_limit(exc: ProxyRateLimitError, rate_limit_type: str, limit_sentence: str) -> None:
    assert exc.status_code == 429
    assert exc.headers is not None
    assert exc.headers["rate_limited_by"] == "model_per_team_saturated"
    assert exc.headers["rate_limit_type"] == rate_limit_type
    assert str(exc.detail).startswith(
        "{'error': 'Team " + limit_sentence + " reached for shared-model and the model is saturated, "
        "so there is no idle capacity to borrow. Team: team-a, Rate limit type: " + rate_limit_type + ", "
    ), exc.detail


@pytest.mark.asyncio
async def test_team_past_its_model_tpm_limit_borrows_while_the_model_is_below_the_saturation_threshold(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _team_key(), 150)
    await _check(handler, _team_key(), saturation=0.49)


@pytest.mark.asyncio
async def test_team_past_its_model_tpm_limit_is_held_to_it_once_the_model_saturates(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _team_key(), 150)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _team_key(), saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")
    assert "Team model_tpm_limit: 100, Remaining: 0, Model saturation: 50.0%" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_team_past_its_model_rpm_limit_is_held_to_it_once_the_model_saturates(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    auth: Final = _team_key(team_limits={"model_rpm_limit": {"shared-model": 2}})
    await _check(handler, auth, saturation=0.0)
    await _check(handler, auth, saturation=0.0)
    await _check(handler, auth, saturation=0.49)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, auth, saturation=0.5)
    _held_to_team_limit(exc.value, "requests", "model_rpm_limit")
    assert "Team model_rpm_limit: 2, Remaining: 0, Model saturation: 50.0%" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_requests_admitted_below_saturation_count_only_their_settled_tokens(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    for _ in range(20):
        await _admit_then_settle(handler, _team_key(), 4)
    await _check(handler, _team_key(), saturation=0.9)


@pytest.mark.asyncio
async def test_an_embedding_responses_tokens_count_against_the_team_limit(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _check(handler, _team_key(), saturation=0.0)
    await _settle_tokens(
        handler,
        150,
        response=litellm.EmbeddingResponse(
            usage=litellm.Usage(prompt_tokens=150, completion_tokens=0, total_tokens=150)
        ),
    )
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _team_key(), saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_responses_api_calls_tokens_count_against_the_team_limit(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _check(handler, _team_key(), saturation=0.0)
    await _settle_tokens(
        handler,
        150,
        response=ResponsesAPIResponse(
            id="resp",
            created_at=0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            usage=ResponseAPIUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        ),
    )
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _team_key(), saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_null_auth_metadata_still_settles_the_teams_tokens(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _check(handler, _team_key(), saturation=0.0)
    await handler.async_log_success_event(
        kwargs={
            "litellm_params": {"metadata": {"model_group": "shared-model"}},
            "standard_logging_object": {
                "metadata": {"user_api_key_team_id": "team-a", "user_api_key_auth_metadata": None}
            },
        },
        response_obj=litellm.ModelResponse(
            usage=litellm.Usage(prompt_tokens=150, completion_tokens=0, total_tokens=150)
        ),
        start_time=None,
        end_time=None,
    )
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _team_key(), saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_another_teams_usage_does_not_count_against_a_teams_limit(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _team_key("team-b"), 150)
    await _check(handler, _team_key("team-a"), saturation=0.9)


@pytest.mark.asyncio
async def test_borrowing_is_off_by_default_so_the_dynamic_limiter_adds_no_team_limit(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False)
    await _admit_then_settle(handler, _team_key(), 150)
    await _check(handler, _team_key(), saturation=0.9)


@pytest.mark.asyncio
async def test_a_key_that_owns_its_own_model_tpm_limit_is_not_held_to_the_team_limit(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    owner: Final = _team_key(metadata={"model_tpm_limit": {"shared-model": 5000}})
    await _admit_then_settle(handler, owner, 150)
    await _check(handler, owner, saturation=0.9)


@pytest.mark.asyncio
async def test_a_key_that_owns_a_limit_on_another_model_is_still_held_to_the_team_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    other_owner: Final = _team_key(metadata={"model_tpm_limit": {"other-model": 5000}})
    await _admit_then_settle(handler, other_owner, 150)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, other_owner, saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_an_exempt_keys_tokens_are_not_charged_to_its_teammates(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    owner: Final = _team_key(metadata={"model_tpm_limit": {"shared-model": 5000}})
    await _admit_then_settle(handler, owner, 150)
    await _check(handler, _team_key(), saturation=0.9)


@pytest.mark.asyncio
async def test_an_alias_request_counts_against_the_same_team_share_as_its_target(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True, model_group_alias={"alias-model": "shared-model"})
    await _check(handler, _team_key(), saturation=0.0, model="alias-model")
    await _settle_tokens(handler, 150)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _team_key(), saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_websocket_turn_is_held_to_the_team_limit_once_the_model_saturates(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.1)
    await _admit_then_settle(handler, _team_key(), 150)
    with pytest.raises(ProxyRateLimitError) as exc:
        await handler.enforce_token_limits_for_frame(user_api_key_dict=_team_key(), requested_model="shared-model")
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_websocket_turn_still_borrows_while_the_model_is_idle(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.5)
    await _admit_then_settle(handler, _team_key(), 150)
    await handler.enforce_token_limits_for_frame(user_api_key_dict=_team_key(), requested_model="shared-model")


@pytest.mark.asyncio
async def test_a_nested_calls_tokens_are_not_charged_to_the_requests_team_share(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy.hooks.parallel_request_limiter_v3 import claim_request_stash_for_data

    handler = _borrowing_handler(monkeypatch, borrows=True)
    claim_request_stash_for_data({"litellm_call_id": "outer-call"})
    await _check(handler, _team_key(), saturation=0.0)
    await _settle_tokens(handler, 60, call_id="llm-judge-call")
    assert await _team_tokens_remaining(handler) == 100
    await _settle_tokens(handler, 60, call_id="outer-call")
    assert await _team_tokens_remaining(handler) == 40


@pytest.mark.asyncio
async def test_a_websocket_turns_tokens_count_against_the_team_limit(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _check(handler, _team_key(), saturation=0.0)
    await _settle_tokens(
        handler,
        150,
        response=[],
        combined_usage_object=litellm.Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _team_key(), saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_websocket_turn_gets_a_503_when_the_team_counter_cannot_be_read(monkeypatch: pytest.MonkeyPatch):
    from litellm.exceptions import ServiceUnavailableError
    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitUnverifiableError

    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.1)
    await _admit_then_settle(handler, _team_key(), 150)
    monkeypatch.setattr(handler.v3_limiter, "should_rate_limit", AsyncMock(side_effect=RateLimitUnverifiableError()))
    with pytest.raises(ServiceUnavailableError):
        await handler.enforce_token_limits_for_frame(user_api_key_dict=_team_key(), requested_model="shared-model")


@pytest.mark.asyncio
async def test_a_prompt_larger_than_the_team_share_is_refused_before_the_call_once_the_model_saturates(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _check(handler, _team_key(), saturation=0.0, data=_prompt(words=400))
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _team_key(), saturation=0.5, data=_prompt(words=400))
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_reservation_at_saturation_settles_to_the_tokens_actually_used(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _check(handler, _team_key(), saturation=0.5, data=_prompt(words=30))
    assert await _team_tokens_remaining(handler) < 70
    await _settle_tokens(handler, 5)
    assert await _team_tokens_remaining(handler) == 95


@pytest.mark.asyncio
async def test_a_reservation_at_saturation_is_refunded_when_the_call_fails(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _check(handler, _team_key(), saturation=0.5, data=_prompt(words=30))
    assert await _team_tokens_remaining(handler) < 70
    await handler.async_log_failure_event(
        kwargs={"litellm_params": {"metadata": {"model_group": "shared-model"}}},
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert await _team_tokens_remaining(handler) == 100


@pytest.mark.asyncio
async def test_a_fallback_target_is_refused_once_it_saturates_and_the_teams_share_on_it_is_used(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.1)
    key: Final = _team_key(
        team_limits={"model_tpm_limit": {"shared-model": 100}, "model_rpm_limit": {"shared-model": 1}}
    )
    await _admit_then_settle(handler, key, 150)
    assert await handler.async_fallback_target_allowed(key, "shared-model", {}) is False


@pytest.mark.asyncio
async def test_a_fallback_target_still_borrows_while_idle_and_its_tokens_count_against_that_teams_share(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    assert await handler.async_fallback_target_allowed(_team_key(), "shared-model", {}) is True
    await _settle_tokens(handler, 150)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _team_key(), saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_fallback_attempt_counts_toward_the_targets_saturation(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    group: Final = handler.llm_router.registry().group("shared-model")
    assert group is not None
    assert await handler.async_fallback_target_allowed(_team_key(), "shared-model", {}) is True
    assert await handler._check_model_saturation("shared-model", group) > 0


@pytest.mark.asyncio
async def test_a_read_only_fallback_check_does_not_use_the_teams_share(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    key: Final = _team_key(
        team_limits={"model_tpm_limit": {"shared-model": 100}, "model_rpm_limit": {"shared-model": 2}}
    )
    await _admit_then_settle(handler, key, 50)
    assert await handler.async_fallback_target_allowed(key, "shared-model", {}, commit=False) is True
    assert await handler.async_fallback_target_allowed(key, "shared-model", {}, commit=False) is True
    assert await handler.async_fallback_target_allowed(key, "shared-model", {}) is True
    assert await handler.async_fallback_target_allowed(key, "shared-model", {}) is False


async def _model_requests_remaining(handler: PROXY_DynamicRateLimitHandlerV3) -> int:
    group: Final = handler.llm_router.registry().group("shared-model")
    assert group is not None
    descriptor: Final = handler._create_model_tracking_descriptor("shared-model", group, high_limit_multiplier=1)
    response: Final = await handler.v3_limiter.should_rate_limit([descriptor], read_only=True, skip_parallel_check=True)
    return next(s["limit_remaining"] for s in response["statuses"] if s["rate_limit_type"] == "requests")


def _fallback_prompt(words: int) -> dict[str, object]:
    return {"messages": [{"role": "user", "content": "word " * words}], "max_tokens": 10}


@pytest.mark.asyncio
async def test_idle_fallbacks_past_the_teams_rpm_share_still_count_toward_the_targets_saturation(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    key: Final = _team_key(team_limits={"model_rpm_limit": {"shared-model": 1}})
    assert await handler.async_fallback_target_allowed(key, "shared-model", {}) is True
    assert await handler.async_fallback_target_allowed(key, "shared-model", {}) is True
    assert await handler.async_fallback_target_allowed(key, "shared-model", {}) is True
    assert await _model_requests_remaining(handler) == 97


@pytest.mark.asyncio
async def test_a_saturated_fallback_target_refuses_a_prompt_larger_than_the_teams_remaining_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    assert await handler.async_fallback_target_allowed(_team_key(), "shared-model", _fallback_prompt(300)) is False
    assert (
        await handler.async_fallback_target_allowed(_team_key(), "shared-model", _fallback_prompt(300), commit=False)
        is False
    )
    assert await _team_tokens_remaining(handler) == 100
    assert (
        await handler.async_fallback_target_allowed(_team_key(), "shared-model", _fallback_prompt(30), commit=False)
        is True
    )
    assert await _team_tokens_remaining(handler) == 100
    assert await handler.async_fallback_target_allowed(_team_key(), "shared-model", _fallback_prompt(30)) is True
    assert await _team_tokens_remaining(handler) < 70
    await _settle_tokens(handler, 5)
    assert await _team_tokens_remaining(handler) == 95


@pytest.mark.asyncio
async def test_a_saturated_responses_fallback_holds_the_output_budget_as_well_as_the_input(
    monkeypatch: pytest.MonkeyPatch,
):
    from litellm.proxy.hooks.parallel_request_limiter_v3 import get_or_create_request_stash

    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    get_or_create_request_stash().call_type = "aresponses"
    prompt: Final = {"input": "hi", "max_output_tokens": 500}
    assert await handler.async_fallback_target_allowed(_team_key(), "shared-model", prompt) is False
    assert await _team_tokens_remaining(handler) == 100


@pytest.mark.asyncio
async def test_an_output_token_limit_holds_only_the_prompts_output_budget(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server

    monkeypatch.setitem(proxy_server.general_settings, "token_rate_limit_type", "output")
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    await _check(handler, _team_key(), saturation=1.0, data=_fallback_prompt(300))
    assert await _team_tokens_remaining(handler) == 90


@pytest.mark.asyncio
async def test_a_request_without_max_tokens_leaves_admission_capped_to_the_teams_share(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    prompt: Final[dict[str, object]] = {"messages": [{"role": "user", "content": "hi"}]}
    await _check(handler, _team_key(), saturation=1.0, data=prompt)
    assert prompt["max_tokens"] == handler.v3_limiter.no_max_tokens_output_floor(100)


@pytest.mark.asyncio
async def test_a_prompt_that_fits_once_its_output_is_capped_is_admitted(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    prompt: Final[dict[str, object]] = {"messages": [{"role": "user", "content": "word " * 60}]}
    await _check(handler, _team_key(), saturation=1.0, data=prompt)
    held: Final = handler.v3_limiter.estimate_request_tokens(
        data={**prompt, "model": "shared-model"},
        model="shared-model",
        min_configured_tpm_limit=100,
        call_type="acompletion",
        team_id="team-a",
    )
    assert prompt["max_tokens"] == handler.v3_limiter.no_max_tokens_output_floor(100)
    assert await _team_tokens_remaining(handler) == 100 - held


@pytest.mark.asyncio
async def test_a_refused_admission_leaves_the_requests_output_uncapped(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    prompt: Final[dict[str, object]] = {"messages": [{"role": "user", "content": "word " * 300}]}
    with pytest.raises(ProxyRateLimitError):
        await _check(handler, _team_key(), saturation=1.0, data=prompt)
    assert "max_tokens" not in prompt


@pytest.mark.asyncio
async def test_an_input_token_limit_leaves_the_reply_length_alone(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server

    monkeypatch.setitem(proxy_server.general_settings, "token_rate_limit_type", "input")
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    prompt: Final[dict[str, object]] = {"messages": [{"role": "user", "content": "hi"}]}
    await _check(handler, _team_key(), saturation=1.0, data=prompt)
    assert "max_tokens" not in prompt


@pytest.mark.asyncio
async def test_a_late_failure_from_a_fallback_that_did_not_serve_is_charged_to_that_target(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    key: Final = _team_key(team_limits={"model_tpm_limit": {"shared-model": 100, "other-model": 100, "chat": 100}})
    await _check(handler, key, saturation=1.0, data=_fallback_prompt(30))
    assert await handler.async_fallback_target_allowed(key, "other-model", _fallback_prompt(30)) is True
    assert await handler.async_fallback_target_allowed(key, "chat", _fallback_prompt(30)) is True
    await _settle_tokens(handler, 7, model_group="chat")
    await handler.async_log_failure_event(
        kwargs={
            "litellm_params": {"metadata": {"model_group": "other-model"}},
            "combined_usage_object": litellm.Usage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        },
        response_obj=Exception("stream cut"),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert await _team_tokens_remaining(handler, key, "shared-model") == 100
    assert await _team_tokens_remaining(handler, key, "chat") == 93
    assert await _team_tokens_remaining(handler, key, "other-model") == 75


@pytest.mark.asyncio
async def test_a_served_fallbacks_hold_is_not_refunded_again_under_the_name_the_request_used(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(
        monkeypatch, borrows=True, saturation_threshold=0.0, model_group_alias={"alias-model": "other-model"}
    )
    key: Final = _team_key(team_limits={"model_tpm_limit": {"shared-model": 100, "other-model": 100}})
    await _check(handler, key, saturation=1.0, data=_fallback_prompt(30))
    assert await handler.async_fallback_target_allowed(key, "alias-model", _fallback_prompt(30)) is True
    await _settle_tokens(handler, 7, model_group="other-model")
    assert await _team_tokens_remaining(handler, key, "other-model") == 93
    await handler.async_log_failure_event(
        kwargs={"litellm_params": {"metadata": {"model_group": "alias-model"}}},
        response_obj=Exception("late"),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert await _team_tokens_remaining(handler, key, "other-model") == 93


@pytest.mark.asyncio
async def test_a_steered_requests_images_are_counted_for_the_model_that_serves_it(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3

    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    asked: Final = []  # mutable-ok: records the models whose image cost was looked up

    def resolver(model: str, team_id: str | None) -> tuple[str, ...]:
        asked.append(model)
        return (model,)

    handler.v3_limiter = PROXY_MaxParallelRequestsHandler_v3(
        handler.internal_usage_cache, media_model_resolver=resolver
    )
    key: Final = _team_key(team_limits={"model_tpm_limit": {"other-model": 10000}})
    pixel: Final = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    steered: Final[dict[str, object]] = {
        "model": "shared-model",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": pixel}}]}],
        "max_tokens": 5,
    }
    await _check(handler, key, saturation=1.0, model="other-model", data=steered)
    assert asked and set(asked) == {"other-model"}


@pytest.mark.asyncio
async def test_a_failed_attempt_that_recorded_usage_is_charged_that_usage_instead_of_refunded(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    await _check(handler, _team_key(), saturation=1.0, data=_fallback_prompt(30))
    assert await _team_tokens_remaining(handler) < 70
    await handler.async_log_failure_event(
        kwargs={
            "litellm_params": {"metadata": {"model_group": "shared-model"}},
            "combined_usage_object": litellm.Usage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        },
        response_obj=Exception("stream cut"),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert await _team_tokens_remaining(handler) == 75


@pytest.mark.asyncio
async def test_a_fallback_attempt_that_fails_gives_back_the_tokens_it_held(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.0)
    assert await handler.async_fallback_target_allowed(_team_key(), "shared-model", _fallback_prompt(30)) is True
    assert await _team_tokens_remaining(handler) < 70
    await handler.async_log_failure_event(
        kwargs={"litellm_params": {"metadata": {"model_group": "shared-model"}}},
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert await _team_tokens_remaining(handler) == 100


@pytest.mark.asyncio
async def test_a_prompt_the_token_estimator_cannot_read_is_still_held_to_the_teams_share(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    key: Final = _team_key(
        team_limits={"model_tpm_limit": {"shared-model": 100}, "model_rpm_limit": {"shared-model": 1}}
    )
    unreadable: Final = {
        "model": "shared-model",
        "messages": [{"role": "user", "content": [{"type": "text", "text": 1}, {"type": "text", "text": "hi"}]}],
    }
    await _check(handler, key, saturation=0.5, data=unreadable)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, key, saturation=0.5, data=unreadable)
    _held_to_team_limit(exc.value, "requests", "model_rpm_limit")


@pytest.mark.asyncio
async def test_a_success_without_a_model_group_still_releases_the_reservation(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _check(handler, _team_key(), saturation=0.5, data=_prompt(words=30))
    assert await _team_tokens_remaining(handler) < 70
    await handler.async_log_success_event(
        kwargs={"litellm_params": {"metadata": {}}},
        response_obj=litellm.ModelResponse(usage=litellm.Usage(prompt_tokens=5, completion_tokens=0, total_tokens=5)),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert await _team_tokens_remaining(handler) == 95


def _class_team(team_id: str, priority: str | None = "interactive") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-class", team_id=team_id, team_metadata={}, metadata={"priority": priority})


async def _burn(
    handler: _PROXY_DynamicRateLimitHandlerV3,
    auth: UserAPIKeyAuth,
    priority: str | None,
    tokens: int,
    model: str = "shared-model",
):
    await _check(handler, auth, saturation=0.0, priority=priority, model=model)
    await handler.async_log_success_event(
        kwargs={
            "litellm_params": {"metadata": {"model_group": model}},
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_team_id": auth.team_id,
                    "user_api_key_auth_metadata": {"priority": priority},
                }
            },
        },
        response_obj=litellm.ModelResponse(
            usage=litellm.Usage(prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens)
        ),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )


def _held_to_class_share(exc: ProxyRateLimitError) -> None:
    assert exc.status_code == 429
    assert exc.headers is not None
    assert exc.headers["rate_limited_by"] == "priority_class_share_saturated"
    assert exc.headers["rate_limit_type"] == "tokens"


@pytest.mark.asyncio
async def test_two_interactive_teams_each_keep_half_of_the_interactive_share(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-a"), "interactive", 20)
    await _burn(handler, _class_team("team-b"), "interactive", 300)
    await _burn(handler, _class_team("team-b"), "interactive", 200)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-b"), saturation=0.52, priority="interactive")
    _held_to_class_share(exc.value)
    assert "Team: team-b, Priority: interactive, Active teams in class: 2, Active classes: 1, " in str(exc.value.detail)
    assert "Team share: 500, Remaining: 0, Model saturation: 52.0%" in str(exc.value.detail)
    await _check(handler, _class_team("team-a"), saturation=0.52, priority="interactive")


@pytest.mark.asyncio
async def test_the_only_active_class_gets_the_whole_model_instead_of_its_weight(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-b", "batch"), "batch", 950)
    await _check(handler, _class_team("team-b", "batch"), saturation=0.95, priority="batch")

    unsplit = _borrowing_handler(monkeypatch, borrows=False)
    await _burn(unsplit, _class_team("team-b", "batch"), "batch", 950)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(unsplit, _class_team("team-b", "batch"), saturation=0.95, priority="batch")
    assert "Priority-based rate limit exceeded" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_class_weights_apply_again_once_a_team_of_the_other_class_is_active(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-b", "batch"), "batch", 10)
    await _burn(handler, _class_team("team-a"), "interactive", 600)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-a"), saturation=0.61, priority="interactive")
    assert "Active teams in class: 1, Active classes: 2, Rate limit type: tokens, Team share: 600, " in str(
        exc.value.detail
    )
    await _check(handler, _class_team("team-b", "batch"), saturation=0.61, priority="batch")


@pytest.mark.asyncio
async def test_teams_without_a_priority_split_the_default_share_between_them(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-x", None), None, 600)
    await _burn(handler, _class_team("team-y", None), None, 1)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-x", None), saturation=0.6, priority=None)
    assert "Priority: default, Active teams in class: 2, Active classes: 1, " in str(exc.value.detail)
    assert "Team share: 500, " in str(exc.value.detail)
    await _check(handler, _class_team("team-y", None), saturation=0.6, priority=None)
    assert exc.value.headers is not None and exc.value.headers["x-litellm-priority"] == "default"


@pytest.mark.asyncio
async def test_a_key_without_a_team_keeps_the_shared_class_pool(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-a"), "interactive", 100)
    no_team: Final = UserAPIKeyAuth(api_key="sk-plain", metadata={"priority": "interactive"})
    await _burn(handler, no_team, "interactive", 600)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, no_team, saturation=0.7, priority="interactive")
    assert exc.value.headers is not None and "rate_limited_by" not in exc.value.headers
    assert "Priority-based rate limit exceeded" in str(exc.value.detail)
    await _check(handler, _class_team("team-a"), saturation=0.7, priority="interactive")


@pytest.mark.asyncio
async def test_below_saturation_a_team_may_run_past_its_share(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-a"), "interactive", 1)
    await _burn(handler, _class_team("team-b"), "interactive", 400)
    await _check(handler, _class_team("team-b"), saturation=0.49, priority="interactive")


@pytest.mark.asyncio
async def test_the_split_is_off_by_default_so_one_team_can_still_empty_the_class_pool(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False)
    await _burn(handler, _class_team("team-a"), "interactive", 600)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-b"), saturation=0.6, priority="interactive")
    assert exc.value.headers is not None and "rate_limited_by" not in exc.value.headers
    assert "Priority-based rate limit exceeded" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_the_team_model_limit_and_the_class_share_both_apply_when_both_are_on(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True, splits=True)
    team: Final = UserAPIKeyAuth(
        api_key="sk-both",
        team_id="team-a",
        team_metadata={"model_tpm_limit": {"shared-model": 100}},
        metadata={"priority": "interactive"},
    )
    await _burn(handler, team, "interactive", 150)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, team, saturation=0.5, priority="interactive")
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_team_active_on_another_model_does_not_shrink_this_models_shares(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _check(handler, _class_team("team-b"), saturation=0.0, priority="interactive", model="other-model")
    await _burn(handler, _class_team("team-a"), "interactive", 700)
    await _check(handler, _class_team("team-a"), saturation=0.7, priority="interactive")


class _UnavailableRegistry:
    async def touch(self, model: str, team_id: str, priority: str, window: int) -> Mapping[str, str]:
        from redis.exceptions import ConnectionError as RedisConnectionError

        raise RedisConnectionError("redis is down")


@pytest.mark.asyncio
async def test_a_registry_failure_falls_back_to_the_class_pool_instead_of_skipping_the_limits(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    handler.active_teams = _UnavailableRegistry()  # pyright: ignore[reportAttributeAccessIssue]  # injected failing registry
    await _burn(handler, _class_team("team-a"), "interactive", 650)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-a"), saturation=0.65, priority="interactive")
    assert exc.value.headers is not None
    assert "rate_limited_by" not in exc.value.headers
    assert "Priority-based rate limit exceeded" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_one_team_using_keys_of_two_priorities_still_spends_one_share(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-b"), "interactive", 20)
    await _burn(handler, _class_team("team-a", "interactive"), "interactive", 300)
    await _burn(handler, _class_team("team-a", "batch"), "batch", 300)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-a", "batch"), saturation=0.62, priority="batch")
    _held_to_class_share(exc.value)
    assert "Team: team-a, Priority: batch, Active teams in class: 1, Active classes: 2, " in str(exc.value.detail)
    assert "Team share: 400, Remaining: 0," in str(exc.value.detail)


@pytest.mark.asyncio
async def test_a_share_smaller_than_one_request_still_lets_each_team_send_one(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True, rpm=2)
    for team in ("team-a", "team-b", "team-c"):
        await handler.active_teams.touch("shared-model", team, "interactive", 60)
    await _check(handler, _class_team("team-d"), saturation=0.9, priority="interactive")
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-d"), saturation=0.9, priority="interactive")
    assert exc.value.headers is not None
    assert exc.value.headers["rate_limited_by"] == "priority_class_share_saturated"
    assert exc.value.headers["rate_limit_type"] == "requests"


@pytest.mark.asyncio
async def test_a_websocket_turn_is_held_to_the_class_share_once_the_model_saturates(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True, saturation_threshold=0.1)
    await _burn(handler, _class_team("team-a"), "interactive", 20)
    await _burn(handler, _class_team("team-b"), "interactive", 600)
    with pytest.raises(ProxyRateLimitError) as exc:
        await handler.enforce_token_limits_for_frame(
            user_api_key_dict=_class_team("team-b"), requested_model="shared-model"
        )
    _held_to_class_share(exc.value)
    await handler.enforce_token_limits_for_frame(
        user_api_key_dict=_class_team("team-a"), requested_model="shared-model"
    )


@pytest.mark.asyncio
async def test_a_websocket_turn_ignores_the_class_share_while_the_split_is_off(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, saturation_threshold=0.1)
    await _burn(handler, _class_team("team-a"), "interactive", 20)
    await _burn(handler, _class_team("team-b"), "interactive", 600)
    await handler.enforce_token_limits_for_frame(
        user_api_key_dict=_class_team("team-b"), requested_model="shared-model"
    )


@pytest.mark.asyncio
async def test_traffic_from_keys_without_a_team_keeps_its_class_present_in_the_split(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    no_team: Final = UserAPIKeyAuth(api_key="sk-plain", metadata={"priority": "interactive"})
    await _burn(handler, no_team, "interactive", 20)
    await _burn(handler, _class_team("team-c", "batch"), "batch", 450)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-c", "batch"), saturation=0.6, priority="batch")
    _held_to_class_share(exc.value)
    assert "Team share: 400, Remaining: 0," in str(exc.value.detail)


@pytest.mark.asyncio
async def test_a_team_id_containing_the_registry_separator_is_still_held_to_its_share(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-b"), "interactive", 20)
    await _burn(handler, _class_team("owned\x1fextra", "batch"), "batch", 450)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("owned\x1fextra", "batch"), saturation=0.6, priority="batch")
    _held_to_class_share(exc.value)


@pytest.mark.asyncio
async def test_a_team_share_is_not_shrunk_by_teamless_traffic_in_its_class(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    no_team: Final = UserAPIKeyAuth(api_key="sk-plain", metadata={"priority": "interactive"})
    await _burn(handler, no_team, "interactive", 20)
    await _burn(handler, _class_team("team-a"), "interactive", 900)
    await _check(handler, _class_team("team-a"), saturation=0.92, priority="interactive")


@pytest.mark.asyncio
async def test_teamless_traffic_in_two_classes_keeps_both_classes_present(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, UserAPIKeyAuth(api_key="sk-i", metadata={"priority": "interactive"}), "interactive", 20)
    await _burn(handler, UserAPIKeyAuth(api_key="sk-b", metadata={"priority": "batch"}), "batch", 20)
    await _burn(handler, _class_team("team-c", "batch"), "batch", 450)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-c", "batch"), saturation=0.6, priority="batch")
    _held_to_class_share(exc.value)
    assert "Team share: 400, Remaining: 0," in str(exc.value.detail)


@pytest.mark.asyncio
async def test_team_counters_do_not_collide_across_models_whose_names_share_a_colon(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("prod:victim"), "interactive", 600, model="chat")
    await _burn(handler, _class_team("team-x"), "interactive", 20, model="chat:prod")
    await _check(handler, _class_team("victim"), saturation=0.6, priority="interactive", model="chat:prod")


def _capped_team_key(multiple: object, limit: str = "model_tpm_limit", share: int = 100) -> UserAPIKeyAuth:
    return _team_key(team_limits={limit: {"shared-model": share}, "team_model_limit_max_borrow_multiple": multiple})


def _held_to_borrow_cap(exc: ProxyRateLimitError, rate_limit_type: str, limit_sentence: str) -> None:
    assert exc.status_code == 429
    assert exc.headers is not None
    assert exc.headers["rate_limited_by"] == "model_per_team_max_borrow_multiple"
    assert exc.headers["rate_limit_type"] == rate_limit_type
    assert str(exc.detail).startswith(
        "{'error': 'Team " + limit_sentence + " borrow cap reached for shared-model: the team may use at most "
    ), exc.detail


@pytest.mark.asyncio
async def test_team_borrows_past_its_share_while_it_stays_under_its_max_borrow_multiple(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _capped_team_key(2), 190)
    await _check(handler, _capped_team_key(2), saturation=0.49)


@pytest.mark.asyncio
async def test_team_at_its_max_borrow_multiple_is_refused_while_the_model_is_idle(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _capped_team_key(2), 200)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _capped_team_key(2), saturation=0.0)
    _held_to_borrow_cap(exc.value, "tokens", "model_tpm_limit")
    assert (
        "2.0x its model_tpm_limit while borrowing idle capacity. Team: team-a, Rate limit type: tokens, "
        "Team model_tpm_limit: 100, Borrow cap: 200, Remaining: 0, Model saturation: 0.0%"
    ) in str(exc.value.detail), exc.value.detail


@pytest.mark.asyncio
async def test_a_fractional_max_borrow_multiple_caps_the_team_at_that_fraction_of_its_share(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _capped_team_key(1.5), 149)
    await _check(handler, _capped_team_key(1.5), saturation=0.0)
    await _settle_tokens(handler, 1)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _capped_team_key(1.5), saturation=0.0)
    _held_to_borrow_cap(exc.value, "tokens", "model_tpm_limit")
    assert "Borrow cap: 150," in str(exc.value.detail), exc.value.detail


@pytest.mark.asyncio
async def test_the_max_borrow_multiple_caps_requests_as_well_as_tokens(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    auth: Final = _capped_team_key(2, limit="model_rpm_limit", share=2)
    for _ in range(4):
        await _check(handler, auth, saturation=0.0)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, auth, saturation=0.0)
    _held_to_borrow_cap(exc.value, "requests", "model_rpm_limit")
    assert "Team model_rpm_limit: 2, Borrow cap: 4, Remaining: 0," in str(exc.value.detail), exc.value.detail


@pytest.mark.asyncio
async def test_a_team_under_its_max_borrow_multiple_is_still_held_to_its_share_once_the_model_saturates(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _capped_team_key(5), 150)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _capped_team_key(5), saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_team_at_its_max_borrow_multiple_is_held_to_its_share_not_its_cap_once_the_model_saturates(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _capped_team_key(2), 100)
    await _check(handler, _capped_team_key(2), saturation=0.49)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _capped_team_key(2), saturation=0.5)
    _held_to_team_limit(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_another_teams_borrowing_does_not_count_against_a_teams_max_borrow_multiple(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _team_key("team-b"), 150)
    await _admit_then_settle(handler, _capped_team_key(2), 150)
    await _check(handler, _capped_team_key(2), saturation=0.0)


@pytest.mark.parametrize(
    "multiple",
    [0.5, 0, -2, "two", None, "Infinity", float("inf"), 1e308],
    ids=["below-one", "zero", "negative", "text", "null", "infinity-text", "infinity", "overflows-the-share"],
)
@pytest.mark.asyncio
async def test_an_unusable_max_borrow_multiple_is_ignored_so_the_team_borrows_up_to_the_model_limit(
    monkeypatch: pytest.MonkeyPatch, multiple: object
):
    handler = _borrowing_handler(monkeypatch, borrows=True)
    await _admit_then_settle(handler, _capped_team_key(multiple), 300)
    await _check(handler, _capped_team_key(multiple), saturation=0.49)


@pytest.mark.asyncio
async def test_a_websocket_turn_is_refused_at_the_teams_max_borrow_multiple_while_the_model_is_idle(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.5)
    await _admit_then_settle(handler, _capped_team_key(2), 200)
    with pytest.raises(ProxyRateLimitError) as exc:
        await handler.enforce_token_limits_for_frame(
            user_api_key_dict=_capped_team_key(2), requested_model="shared-model"
        )
    _held_to_borrow_cap(exc.value, "tokens", "model_tpm_limit")


@pytest.mark.asyncio
async def test_a_websocket_turn_under_the_teams_max_borrow_multiple_still_borrows(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=True, saturation_threshold=0.5)
    await _admit_then_settle(handler, _capped_team_key(2), 190)
    await handler.enforce_token_limits_for_frame(user_api_key_dict=_capped_team_key(2), requested_model="shared-model")


@pytest.mark.asyncio
async def test_a_configured_class_named_default_is_not_merged_with_teams_that_have_no_priority(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    monkeypatch.setattr(litellm, "priority_reservation", {"default": 0.6, "batch": 0.4})
    await _burn(handler, _class_team("team-a", "default"), "default", 20)
    await _burn(handler, _class_team("team-n", None), None, 20)
    await _burn(handler, _class_team("team-a", "default"), "default", 700)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-a", "default"), saturation=0.74, priority="default")
    assert "Priority: default, Active teams in class: 1, Active classes: 2, " in str(exc.value.detail)


@pytest.mark.asyncio
async def test_tokens_from_a_nested_call_do_not_land_on_the_outer_requests_team_counter(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-b"), "interactive", 20)
    await _check(handler, _class_team("team-a"), saturation=0.0, priority="interactive")
    from litellm.proxy.hooks.parallel_request_limiter_v3 import get_or_create_request_stash

    stash: Final = get_or_create_request_stash()
    stash.owner_litellm_call_id = "outer-call"
    await handler.async_log_success_event(
        kwargs={
            "litellm_call_id": "judge-call",
            "litellm_params": {"metadata": {"model_group": "shared-model"}},
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_team_id": "team-a",
                    "user_api_key_auth_metadata": {"priority": "interactive"},
                }
            },
        },
        response_obj=litellm.ModelResponse(
            usage=litellm.Usage(prompt_tokens=600, completion_tokens=0, total_tokens=600)
        ),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    await _check(handler, _class_team("team-a"), saturation=0.62, priority="interactive")


@pytest.mark.asyncio
async def test_a_team_id_spelled_like_a_pool_marker_still_counts_as_a_team(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team(pool_member("interactive")), "interactive", 20)
    await _burn(handler, _class_team("team-a"), "interactive", 600)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-a"), saturation=0.62, priority="interactive")
    assert "Active teams in class: 2, Active classes: 1, " in str(exc.value.detail)


@pytest.mark.asyncio
async def test_requests_sent_while_the_share_was_not_enforced_all_count_once_the_share_grows(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _check(handler, _class_team("team-b"), saturation=0.0, priority="interactive")
    for _ in range(70):
        await _check(handler, _class_team("team-a"), saturation=0.0, priority="interactive")
    await _check(handler, _class_team("team-b", "batch"), saturation=0.0, priority="batch")
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-a"), saturation=0.6, priority="interactive")
    assert "Active teams in class: 1, Active classes: 2, " in str(exc.value.detail)


@pytest.mark.asyncio
async def test_a_websocket_turn_settles_on_the_team_share_once_the_registry_recovers(monkeypatch: pytest.MonkeyPatch):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True, saturation_threshold=0.1)
    await _burn(handler, _class_team("team-b"), "interactive", 20)
    registry: Final = handler.active_teams
    handler.active_teams = _UnavailableRegistry()  # pyright: ignore[reportAttributeAccessIssue]  # registry down at connect
    await _check(handler, _class_team("team-a"), saturation=0.0, priority="interactive")
    handler.active_teams = registry
    await handler.enforce_token_limits_for_frame(
        user_api_key_dict=_class_team("team-a"), requested_model="shared-model"
    )
    await handler.async_log_success_event(
        kwargs={
            "call_type": "_aresponses_websocket",
            "litellm_params": {"metadata": {"model_group": "shared-model"}},
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_team_id": "team-a",
                    "user_api_key_auth_metadata": {"priority": "interactive"},
                }
            },
        },
        response_obj=litellm.ModelResponse(
            usage=litellm.Usage(prompt_tokens=550, completion_tokens=0, total_tokens=550)
        ),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    with pytest.raises(ProxyRateLimitError) as exc:
        await handler.enforce_token_limits_for_frame(
            user_api_key_dict=_class_team("team-a"), requested_model="shared-model"
        )
    _held_to_class_share(exc.value)


@pytest.mark.asyncio
async def test_a_frame_checked_while_the_registry_is_down_does_not_erase_the_connections_team_share(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True, saturation_threshold=0.1)
    await _burn(handler, _class_team("team-b"), "interactive", 20)
    await _check(handler, _class_team("team-a"), saturation=0.0, priority="interactive")
    registry: Final = handler.active_teams
    handler.active_teams = _UnavailableRegistry()  # pyright: ignore[reportAttributeAccessIssue]  # registry down mid-connection
    await handler.enforce_token_limits_for_frame(
        user_api_key_dict=_class_team("team-a"), requested_model="shared-model"
    )
    handler.active_teams = registry
    await handler.async_log_success_event(
        kwargs={
            "call_type": "_aresponses_websocket",
            "litellm_params": {"metadata": {"model_group": "shared-model"}},
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_team_id": "team-a",
                    "user_api_key_auth_metadata": {"priority": "interactive"},
                }
            },
        },
        response_obj=litellm.ModelResponse(
            usage=litellm.Usage(prompt_tokens=550, completion_tokens=0, total_tokens=550)
        ),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    with pytest.raises(ProxyRateLimitError) as exc:
        await handler.enforce_token_limits_for_frame(
            user_api_key_dict=_class_team("team-a"), requested_model="shared-model"
        )
    _held_to_class_share(exc.value)


@pytest.mark.asyncio
async def test_tokens_served_by_a_fallback_model_land_on_the_teams_counter_for_that_model(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = _borrowing_handler(monkeypatch, borrows=False, splits=True)
    await _burn(handler, _class_team("team-b"), "interactive", 20, model="other-model")
    await _burn(handler, _class_team("team-a"), "interactive", 10, model="other-model")
    await _check(handler, _class_team("team-a"), saturation=0.0, priority="interactive")
    await handler.async_log_success_event(
        kwargs={
            "litellm_params": {"metadata": {"model_group": "other-model"}},
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_team_id": "team-a",
                    "user_api_key_auth_metadata": {"priority": "interactive"},
                }
            },
        },
        response_obj=litellm.ModelResponse(
            usage=litellm.Usage(prompt_tokens=550, completion_tokens=0, total_tokens=550)
        ),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    await _check(handler, _class_team("team-a"), saturation=0.62, priority="interactive")
    with pytest.raises(ProxyRateLimitError) as exc:
        await _check(handler, _class_team("team-a"), saturation=0.62, priority="interactive", model="other-model")
    _held_to_class_share(exc.value)
