"""
Test percentage-based rate limiting for models without known rpm/tpm limits.

Tests the new PercentageBasedRateLimitHandler that uses failure-based saturation
and percentage splits instead of absolute rate limits.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm import DualCache, Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.percentage_based_rate_limiter import (
    _PROXY_PercentageBasedRateLimitHandler as PercentageBasedRateLimitHandler,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    increment_deployment_failures_for_current_minute,
)


@pytest.mark.asyncio
async def test_percentage_enforcement_after_saturation():
    """
    Test that percentage-based rate limiting only enforces after saturation.
    
    Setup:
    - No rpm/tpm configured on model
    - saturation_policy.RateLimitErrorSaturationThreshold = 5
    - priority_reservation = {"prod": 0.9, "dev": 0.1}
    
    Expected behavior:
    1. Before 5 failures: No rate limiting (both priorities can send)
    2. After 5 failures (saturation = 100%): Percentage enforcement kicks in
       - Prod gets 90% of traffic
       - Dev gets 10% of traffic
    """
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    
    # Set up priority reservations
    litellm.priority_reservation = {"prod": 0.9, "dev": 0.1}
    
    # Set up saturation policy - trigger saturation after 5 failures
    from litellm.types.router import SaturationPolicy
    litellm.priority_reservation_settings.saturation_policy = SaturationPolicy(
        RateLimitErrorSaturationThreshold=5
    )
    
    dual_cache = DualCache()
    handler = PercentageBasedRateLimitHandler(internal_usage_cache=dual_cache)
    
    model = "percentage-test-model"
    # NO rpm/tpm configured - testing failure-triggered percentage limiting
    
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    # No rpm/tpm - testing percentage-based rate limiting
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)
    
    # Create users
    prod_user = UserAPIKeyAuth()
    prod_user.metadata = {"priority": "prod"}
    prod_user.user_id = "prod_user"
    
    dev_user = UserAPIKeyAuth()
    dev_user.metadata = {"priority": "dev"}
    dev_user.user_id = "dev_user"
    
    # Get deployment ID for mocking failure counts
    deployment_ids = llm_router.get_model_ids(model_name=model)
    assert len(deployment_ids) > 0, "Should have at least one deployment"
    deployment_id = deployment_ids[0]
    
    # Track results across both phases
    phase_1_results = {"prod": 0, "dev": 0}  # Before saturation
    phase_2_results = {"prod": 0, "dev": 0}  # After saturation
    
    async def make_request(user, priority_name):
        """Make a single request and track the result."""
        try:
            result = await handler.async_pre_call_hook(
                user_api_key_dict=user,
                cache=dual_cache,
                data={"model": model},
                call_type="completion",
            )
            return {"status": "success", "priority": priority_name}
        except Exception as e:
            return {"status": "rate_limited", "priority": priority_name, "error": str(e)}
    
    # ============================================================
    # PHASE 1: Before saturation (< 5 failures)
    # Both prod and dev should have 100% success (no limiting)
    # ============================================================
    print("\n=== PHASE 1: Before Saturation (4 failures) ===")
    
    # Simulate 4 failures in the router's cache (below threshold)
    for _ in range(4):
        increment_deployment_failures_for_current_minute(
            litellm_router_instance=llm_router,
            deployment_id=deployment_id,
        )
    
    # Check saturation level
    saturation_level = handler._check_error_saturation(model)
    print(f"Saturation level with 4 failures: {saturation_level:.1%} (threshold: 5 failures = 100%)")
    assert saturation_level < 1.0, f"Should not be saturated yet, got {saturation_level:.1%}"
    
    # Send 20 requests from each priority (40 total)
    phase_1_tasks = []
    for _ in range(20):
        phase_1_tasks.append(make_request(prod_user, "prod"))
    for _ in range(20):
        phase_1_tasks.append(make_request(dev_user, "dev"))
    
    phase_1_raw_results = await asyncio.gather(*phase_1_tasks)
    
    for result in phase_1_raw_results:
        if result["status"] == "success":
            phase_1_results[result["priority"]] += 1
    
    phase_1_prod_rate = phase_1_results["prod"] / 20
    phase_1_dev_rate = phase_1_results["dev"] / 20
    
    print(f"Prod: {phase_1_results['prod']}/20 success ({phase_1_prod_rate:.1%})")
    print(f"Dev: {phase_1_results['dev']}/20 success ({phase_1_dev_rate:.1%})")
    print(f"Both priorities should have 100% success (no limiting yet)")
    
    # In non-saturated mode, both should have 100% success
    assert phase_1_prod_rate == 1.0, f"Prod should have 100% success before saturation, got {phase_1_prod_rate:.1%}"
    assert phase_1_dev_rate == 1.0, f"Dev should have 100% success before saturation, got {phase_1_dev_rate:.1%}"
    
    # ============================================================
    # PHASE 2: After saturation (>= 5 failures)
    # Percentage enforcement should kick in: prod=90%, dev=10%
    # ============================================================
    print("\n=== PHASE 2: After Saturation (6 failures total) ===")
    
    # Add 2 more failures to push over threshold (4 + 2 = 6 > 5)
    for _ in range(2):
        increment_deployment_failures_for_current_minute(
            litellm_router_instance=llm_router,
            deployment_id=deployment_id,
        )
    
    # Check saturation level again
    saturation_level = handler._check_error_saturation(model)
    print(f"Saturation level with 6 failures: {saturation_level:.1%} (threshold: 5)")
    assert saturation_level >= 1.0, f"Should be saturated now, got {saturation_level:.1%}"
    
    # Send requests in smaller batches to reduce race conditions
    # With 200 concurrent requests, all read counters before any increment
    # Batching gives more realistic testing
    phase_2_raw_results = []
    batch_size = 10
    
    for batch_num in range(10):  # 10 batches of 20 requests each
        batch_tasks = []
        for _ in range(10):  # 10 prod per batch
            batch_tasks.append(make_request(prod_user, "prod"))
        for _ in range(10):  # 10 dev per batch
            batch_tasks.append(make_request(dev_user, "dev"))
        
        batch_results = await asyncio.gather(*batch_tasks)
        phase_2_raw_results.extend(batch_results)
        
        # Small delay to let counters update
        await asyncio.sleep(0.01)
    
    for result in phase_2_raw_results:
        if result["status"] == "success":
            phase_2_results[result["priority"]] += 1
    
    phase_2_prod_rate = phase_2_results["prod"] / 100
    phase_2_dev_rate = phase_2_results["dev"] / 100
    
    total_phase_2_success = phase_2_results["prod"] + phase_2_results["dev"]
    prod_share = phase_2_results["prod"] / total_phase_2_success if total_phase_2_success > 0 else 0
    dev_share = phase_2_results["dev"] / total_phase_2_success if total_phase_2_success > 0 else 0
    
    print(f"Prod (90% reservation): {phase_2_results['prod']}/100 success ({phase_2_prod_rate:.1%})")
    print(f"Dev (10% reservation): {phase_2_results['dev']}/100 success ({phase_2_dev_rate:.1%})")
    print(f"Total successful: {total_phase_2_success}/200")
    print(f"Prod share of traffic: {prod_share:.1%} (expected ~90%)")
    print(f"Dev share of traffic: {dev_share:.1%} (expected ~10%)")
    
    # After saturation, percentage enforcement should work
    # Prod should get significantly more than dev
    assert phase_2_prod_rate > phase_2_dev_rate, (
        f"Prod should have higher success rate than Dev after saturation: "
        f"{phase_2_prod_rate:.1%} vs {phase_2_dev_rate:.1%}"
    )
    
    # Check that the share is approximately 90/10
    if total_phase_2_success > 10:  # Need enough samples
        assert prod_share >= 0.8, f"Prod should get ~90% of traffic, got {prod_share:.1%}"
        assert dev_share <= 0.2, f"Dev should get ~10% of traffic, got {dev_share:.1%}"
    
    print("\n✅ Test passed!")
    print("   - Phase 1: No limiting before saturation")
    print("   - Phase 2: Percentage split enforced after saturation")

