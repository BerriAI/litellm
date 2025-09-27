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
    Stress test: 50 concurrent pre-call hooks with priority enforcement.

    This tests the actual rate limiting logic under concurrent load.
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

    async def mock_should_rate_limit(descriptors, parent_otel_span=None):
        """Mock rate limiter that allows premium users, limits some standard users."""
        descriptor = descriptors[0]
        priority = descriptor["value"].split(":")[-1]

        if priority == "premium":
            # Allow all premium requests
            return {
                "overall_code": "OK",
                "statuses": [
                    {
                        "code": "OK",
                        "descriptor_key": descriptor["value"],
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
                            "descriptor_key": descriptor["value"],
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
                            "descriptor_key": descriptor["value"],
                            "rate_limit_type": "tokens_per_unit",
                            "limit_remaining": 100,
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

        with patch.object(
            handler.v3_limiter, "should_rate_limit", side_effect=mock_should_rate_limit
        ):
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

    # Run all 50 requests concurrently
    start_time = time.time()
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
    ), f"Standard success rate should be >= 50%, got {standard_success_rate:.2%}"
    assert (
        premium_success_rate > standard_success_rate
    ), "Premium should have higher success rate than standard"

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
