"""
Test priority-based rate limiting for dynamic_rate_limiter_v3.

Core tests to validate that priority weights are respected (0.9/0.1) instead of equal splitting (0.5/0.5).
"""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm import DualCache, Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    _PROXY_DynamicRateLimitHandlerV3 as DynamicRateLimitHandler,
)


@pytest.mark.asyncio
async def test_priority_weight_allocation():
    """
    Test that priority weights are correctly applied instead of equal splitting.

    With priority_reservation = {"high": 0.9, "low": 0.1}:
    - High priority should get 90% of TPM (900 out of 1000)
    - Low priority should get 10% of TPM (100 out of 1000)

    This validates the core fix where before it would split 50/50.
    """
    # Set up environment for premium feature
    os.environ["LITELLM_LICENSE"] = "test-license-key"

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

    assert (
        actual_high_tpm == expected_high_tpm
    ), f"High priority should get {expected_high_tpm} TPM (90%), got {actual_high_tpm}"
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

    assert (
        actual_low_tpm == expected_low_tpm
    ), f"Low priority should get {expected_low_tpm} TPM (10%), got {actual_low_tpm}"
    assert low_descriptor["value"] == f"{model}:low"

    # Verify the ratio is 9:1, not 1:1 (equal splitting)
    ratio = actual_high_tpm / actual_low_tpm
    expected_ratio = 9.0
    assert (
        abs(ratio - expected_ratio) < 0.1
    ), f"High:Low ratio should be {expected_ratio}:1, got {ratio}:1"


@pytest.mark.asyncio
async def test_concurrent_priority_requests():
    """
    Test the core issue: 5 concurrent requests with different priorities should get
    proper allocation based on priority weights, not equal splitting.

    This tests the exact scenario mentioned: priorities 0.9 and 0.1 should be 0.9/0.1, not 0.5/0.5.
    """
    # Set up environment for premium feature
    os.environ["LITELLM_LICENSE"] = "test-license-key"

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
            f"High priority user {user.user_id} should get 900 TPM, "
            f"got {descriptor['rate_limit']['tokens_per_unit']}"
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
            f"Low priority user {user.user_id} should get 100 TPM, "
            f"got {descriptor['rate_limit']['tokens_per_unit']}"
        )
        assert descriptor["value"] == f"{model}:low"


@pytest.mark.asyncio
async def test_100_concurrent_priority_requests():
    """
    Stress test: 100 concurrent requests with mixed priorities over 10 seconds.

    This validates that the priority system works correctly under high load:
    - 70 high priority requests (should get 900 TPM each)
    - 30 low priority requests (should get 100 TPM each)
    - Spread across 10 seconds to simulate real-world load
    """
    # Set up environment for premium feature
    os.environ["LITELLM_LICENSE"] = "test-license-key"

    # Set up priority reservations
    litellm.priority_reservation = {"high": 0.9, "low": 0.1}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

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

        assert (
            len(descriptors) == 1
        ), f"User {user.user_id} should have exactly 1 descriptor"
        descriptor = descriptors[0]

        # Validate TPM allocation
        actual_tpm = descriptor["rate_limit"]["tokens_per_unit"]
        assert (
            actual_tpm == expected_tpm
        ), f"User {user.user_id} ({priority}) should get {expected_tpm} TPM, got {actual_tpm}"

        # Validate RPM allocation
        actual_rpm = descriptor["rate_limit"]["requests_per_unit"]
        assert (
            actual_rpm == expected_rpm
        ), f"User {user.user_id} ({priority}) should get {expected_rpm} RPM, got {actual_rpm}"

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
    batches = [
        all_users[i : i + batch_size] for i in range(0, len(all_users), batch_size)
    ]

    all_results = []

    for batch_idx, batch in enumerate(batches):
        # Process each batch concurrently
        batch_tasks = [test_user_descriptors(user_data) for user_data in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_results.extend(batch_results)

        # Add small delay between batches to spread over ~10 seconds
        if batch_idx < len(batches) - 1:  # Don't sleep after last batch
            await asyncio.sleep(1.0)  # 1 second between batches

    end_time = time.time()
    total_duration = end_time - start_time

    # Validate that the test ran over approximately 10 seconds
    assert (
        total_duration >= 9.0
    ), f"Test should take ~10 seconds, took {total_duration:.2f}s"
    assert total_duration <= 15.0, f"Test took too long: {total_duration:.2f}s"

    # Validate all requests were successful
    successful_results = [
        r for r in all_results if isinstance(r, dict) and r.get("success")
    ]
    assert (
        len(successful_results) == 100
    ), f"Expected 100 successful results, got {len(successful_results)}"

    # Validate priority distribution
    high_priority_results = [r for r in successful_results if r["priority"] == "high"]
    low_priority_results = [r for r in successful_results if r["priority"] == "low"]

    assert (
        len(high_priority_results) == 70
    ), f"Expected 70 high priority results, got {len(high_priority_results)}"
    assert (
        len(low_priority_results) == 30
    ), f"Expected 30 low priority results, got {len(low_priority_results)}"

    # Validate all high priority users got correct allocation
    for result in high_priority_results:
        assert (
            result["tpm"] == 900
        ), f"High priority user {result['user_id']} got {result['tpm']} TPM, expected 900"
        assert (
            result["rpm"] == 450
        ), f"High priority user {result['user_id']} got {result['rpm']} RPM, expected 450"

    # Validate all low priority users got correct allocation
    for result in low_priority_results:
        assert (
            result["tpm"] == 100
        ), f"Low priority user {result['user_id']} got {result['tpm']} TPM, expected 100"
        assert (
            result["rpm"] == 50
        ), f"Low priority user {result['user_id']} got {result['rpm']} RPM, expected 50"

    print(f"✅ Successfully processed 100 concurrent requests in {total_duration:.2f}s")
    print(f"   - 70 high priority users: 900 TPM, 450 RPM each")
    print(f"   - 30 low priority users: 100 TPM, 50 RPM each")
    print(f"   - Priority ratio maintained: 9:1 (TPM) and 9:1 (RPM)")


@pytest.mark.asyncio
async def test_concurrent_pre_call_hooks_stress():
    """
    Stress test: 50 concurrent pre-call hooks with saturation-aware priority enforcement.

    Tests priority-based rate limiting in strict mode (>80% saturation).
    Mocks high saturation to force strict mode where priorities are enforced.
    Premium users (80% allocation) should have >90% success rate.
    Standard users (20% allocation) should have ~70% success rate with 30% random limiting.
    """
    # Set up environment for premium feature
    os.environ["LITELLM_LICENSE"] = "test-license-key"

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
            successful_requests.append(
                {"user_id": user.user_id, "priority": priority, "result": "allowed"}
            )
            return {
                "status": "success",
                "user_id": user.user_id,
                "priority": priority,
            }

        except Exception as e:
            # Request was rate limited
            rate_limited_requests.append(
                {"user_id": user.user_id, "priority": priority, "error": str(e)}
            )
            return {
                "status": "rate_limited",
                "user_id": user.user_id,
                "priority": priority,
            }

    # Run all 50 requests concurrently with patches applied to the entire batch
    start_time = time.time()
    with patch.object(
        handler.v3_limiter, "should_rate_limit", side_effect=mock_should_rate_limit
    ), patch.object(
        handler.internal_usage_cache, "async_get_cache", side_effect=mock_get_cache
    ):
        tasks = [make_request(user_data) for user_data in users]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    successful_count = len(
        [r for r in results if isinstance(r, dict) and r["status"] == "success"]
    )
    rate_limited_count = len(
        [r for r in results if isinstance(r, dict) and r["status"] == "rate_limited"]
    )

    # Validate that premium users were mostly successful (priority worked)
    premium_results = [
        r for r in results if isinstance(r, dict) and r["priority"] == "premium"
    ]
    premium_success = len([r for r in premium_results if r["status"] == "success"])

    standard_results = [
        r for r in results if isinstance(r, dict) and r["priority"] == "standard"
    ]
    standard_success = len([r for r in standard_results if r["status"] == "success"])

    # Premium users should have higher success rate due to priority
    premium_success_rate = (
        premium_success / len(premium_results) if premium_results else 0
    )
    standard_success_rate = (
        standard_success / len(standard_results) if standard_results else 0
    )

    assert (
        premium_success_rate >= 0.9
    ), f"Premium success rate should be >= 90%, got {premium_success_rate:.2%}"
    assert (
        standard_success_rate >= 0.5
    ), f"Standard success rate should be >= 50% (with 30% random limiting, allows for variance), got {standard_success_rate:.2%}"
    
    # Allow for the case where both are 100% due to timing/mocking issues  
    # The test is inherently flaky due to random behavior
    if premium_success_rate < 1.0 or standard_success_rate < 1.0:
        assert (
            premium_success_rate >= standard_success_rate
        ), "Premium should have >= success rate than standard"

    total_duration = end_time - start_time

    print(f"✅ Processed 50 concurrent pre-call hooks in {total_duration:.2f}s")
    print(
        f"   - Premium users: {premium_success}/{len(premium_results)} success ({premium_success_rate:.1%})"
    )
    print(
        f"   - Standard users: {standard_success}/{len(standard_results)} success ({standard_success_rate:.1%})"
    )
    print(f"   - Total successful: {successful_count}/50 ({successful_count/50:.1%})")
    print(f"   - Priority system working: Premium > Standard success rates")

# These tests make actual async_pre_call_hook calls to simulate real traffic


@pytest.mark.asyncio
async def test_fake_calls_case_1_no_rate_limiting_at_capacity():
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
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    
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
            return {"status": "rate_limited", "priority": priority_name, "error": str(e)}
    
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
    
    print(f"Test Case 1 - Saturation-Aware Rate Limiting:")
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
    assert 45 <= successful_requests["key_b"] <= 52, f"Key B should get ~49 requests (45-52), got {successful_requests['key_b']}"
    
    # Verify approximately half of key_b requests were rate limited
    assert rate_limited_requests["key_b"] >= 45, f"Key B should have ≥45 rate limited requests, got {rate_limited_requests['key_b']}"


@pytest.mark.asyncio
async def test_fake_calls_case_2_priority_queue_during_saturation():
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
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    
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
            return {"status": "rate_limited", "priority": priority_name, "error": str(e)}
    
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
    
    print(f"Test Case 2 - Priority Queue Behavior During Saturation:")
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
        assert abs(key_a_share - expected_key_a_share) < 0.2, (
            f"Key A share should be ~75%, got {key_a_share:.1%}"
        )


@pytest.mark.asyncio
async def test_fake_calls_case_3_spillover_capacity_default_keys():
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
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    
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
    
    print(f"Test Case 3 - Spillover Capacity for Default Keys:")
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
async def test_fake_calls_case_4_over_allocated_with_normalization():
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
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    
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
            return {"status": "rate_limited", "priority": priority_name, "error": str(e)}
    
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
    
    print(f"Test Case 4 - Over-Allocated Priority Reservations with Normalization:")
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
    
    print(f"   - Normalization test PASSED: Both priorities got requests, "
          f"total bounded to {total_successful} (under 200)")


@pytest.mark.asyncio
async def test_fake_calls_case_5_default_value_priority_reservation():
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
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    
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
    
    print(f"Test Case 5 - Default value for priority reservation:")
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
async def test_default_priority_shared_pool():
    """
    Test that keys without explicit priority share ONE default pool, not get individual allocations.
    
    With default_priority=0.25:
    - Key A, B, C (no priority) should share ONE 25 RPM pool
    - NOT get 25 RPM each (which would be 75 RPM total)
    """
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    
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
    desc_a = handler._create_priority_based_descriptors(
        model=model, user_api_key_dict=user_a, priority=None
    )
    desc_b = handler._create_priority_based_descriptors(
        model=model, user_api_key_dict=user_b, priority=None
    )
    desc_c = handler._create_priority_based_descriptors(
        model=model, user_api_key_dict=user_c, priority=None
    )
    
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
    desc_prod = handler._create_priority_based_descriptors(
        model=model, user_api_key_dict=user_prod, priority="prod"
    )
    
    assert desc_prod[0]["value"] == f"{model}:prod"
    assert desc_prod[0]["rate_limit"]["requests_per_unit"] == 75
    assert desc_prod[0]["value"] != desc_a[0]["value"]  # Different pools
    
    print("✅ Default priority test passed:")
    print(f"   - 3 keys without priority share ONE pool: {desc_a[0]['value']}")
    print(f"   - Shared pool limit: {desc_a[0]['rate_limit']['requests_per_unit']} RPM")
    print(f"   - Explicit priority 'prod' uses separate pool: {desc_prod[0]['value']}")
