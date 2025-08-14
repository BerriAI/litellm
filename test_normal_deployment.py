#!/usr/bin/env python3

"""
Test script to verify that normal deployments still work after the memory leak fix.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litellm import Router
from litellm.types.router import Deployment

def test_normal_deployment():
    """Test that normal deployments still work correctly."""
    
    # Create router
    router = Router()
    
    # Create a normal deployment that should work
    normal_deployment = Deployment(
        model_name="openai-test",
        litellm_params={
            "model": "gpt-3.5-turbo",
            "api_key": "fake-api-key-for-testing",
        },
        model_info={
            "id": "test-openai-deployment-normal",
        }
    )
    
    print("Testing normal deployment functionality...")
    
    # First attempt - should succeed
    print("\nFirst attempt with normal deployment (should succeed):")
    result1 = router.upsert_deployment(normal_deployment)
    print(f"Result: {result1}")
    print(f"Model list size: {len(router.model_list)}")
    
    # Second attempt with same deployment - should return None (no change needed)
    print("\nSecond attempt with same deployment (should return None - no change):")
    result2 = router.upsert_deployment(normal_deployment)
    print(f"Result: {result2}")
    print(f"Model list size: {len(router.model_list)}")
    
    # Check that deployment is in the model list
    deployment_found = False
    for model in router.model_list:
        if model.get("model_info", {}).get("id") == "test-openai-deployment-normal":
            deployment_found = True
            break
    
    print(f"\nDeployment found in model list: {deployment_found}")
    
    # Test successful deployment is not in failed cache
    failed_cache_key = "failed_deployment_test-openai-deployment-normal"
    cached_error = router.failed_deployments.get_cache(failed_cache_key)
    print(f"Normal deployment in failed cache: {cached_error is not None}")
    
    if deployment_found and cached_error is None:
        print("\n✅ Normal deployment functionality working correctly!")
        return True
    else:
        print("\n❌ Normal deployment functionality has issues!")
        return False

if __name__ == "__main__":
    test_normal_deployment()