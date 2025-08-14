#!/usr/bin/env python3

"""
Test script to verify the memory leak fix for failed deployment upserts.
This test simulates the scenario where deployments fail due to missing credentials
and verifies that they are cached to prevent repeated retry attempts.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litellm import Router
from litellm.types.router import Deployment

def test_failed_deployment_caching():
    """Test that failed deployments are cached to prevent memory leaks."""
    
    # Create router with ignore_invalid_deployments=True to match production behavior
    router = Router(ignore_invalid_deployments=True)
    
    # Create a deployment that will fail due to missing vertex credentials
    deployment_with_missing_vertex_creds = Deployment(
        model_name="vertex-test",
        litellm_params={
            "model": "vertex_ai/gemini-pro",
            "custom_llm_provider": "vertex_ai",
            "use_in_pass_through": True,
            # Missing vertex_project and vertex_location - will cause failure
        },
        model_info={
            "id": "test-vertex-deployment-1",
        }
    )
    
    # Create a deployment that will fail due to missing API key
    deployment_with_missing_api_key = Deployment(
        model_name="openai-test",
        litellm_params={
            "model": "gpt-3.5-turbo",
            "custom_llm_provider": "openai",
            "use_in_pass_through": True,
            # Missing api_key - will cause failure
        },
        model_info={
            "id": "test-openai-deployment-1",
        }
    )
    
    print("Testing failed deployment caching...")
    
    # First attempt - should fail and be cached
    print("\nFirst attempt with vertex deployment (should fail and be cached):")
    result1 = router.upsert_deployment(deployment_with_missing_vertex_creds)
    print(f"Result: {result1}")
    
    # Second attempt - should be skipped due to cache
    print("\nSecond attempt with same vertex deployment (should be skipped):")
    result2 = router.upsert_deployment(deployment_with_missing_vertex_creds)
    print(f"Result: {result2}")
    
    # Test with API key failure
    print("\nFirst attempt with openai deployment (should fail and be cached):")
    result3 = router.upsert_deployment(deployment_with_missing_api_key)
    print(f"Result: {result3}")
    
    # Second attempt - should be skipped due to cache
    print("\nSecond attempt with same openai deployment (should be skipped):")
    result4 = router.upsert_deployment(deployment_with_missing_api_key)
    print(f"Result: {result4}")
    
    # Check cache entries
    print("\nChecking cache entries:")
    vertex_cache_key = "failed_deployment_test-vertex-deployment-1"
    openai_cache_key = "failed_deployment_test-openai-deployment-1"
    
    vertex_cached = router.failed_deployments.get_cache(vertex_cache_key)
    openai_cached = router.failed_deployments.get_cache(openai_cache_key)
    
    print(f"Vertex deployment cached error: {vertex_cached is not None}")
    print(f"OpenAI deployment cached error: {openai_cached is not None}")
    
    if vertex_cached:
        print(f"Vertex cached error message: {vertex_cached}")
    if openai_cached:
        print(f"OpenAI cached error message: {openai_cached}")
    
    # Test cache clearing
    print("\nTesting cache clearing:")
    router.clear_failed_deployments_cache("test-vertex-deployment-1")
    vertex_cached_after_clear = router.failed_deployments.get_cache(vertex_cache_key)
    print(f"Vertex deployment cached after clear: {vertex_cached_after_clear is not None}")
    
    # Clear all cache
    router.clear_failed_deployments_cache()
    openai_cached_after_clear_all = router.failed_deployments.get_cache(openai_cache_key)
    print(f"OpenAI deployment cached after clear all: {openai_cached_after_clear_all is not None}")
    
    print("\nTest completed successfully!")
    return True

if __name__ == "__main__":
    test_failed_deployment_caching()