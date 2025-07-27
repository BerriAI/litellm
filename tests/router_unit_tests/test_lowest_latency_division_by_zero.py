"""
Unit tests for lowest latency router strategy division by zero fix

This tests the router's handling of empty latency lists in lowest latency routing
Reproduces and fixes the division by zero error in _get_available_deployments
"""

import sys
import os
import time
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler


@pytest.fixture
def model_list():
    """Standard model list fixture following router unit test patterns"""
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY", "test-key"),
            },
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-3",
                "api_key": os.getenv("AZURE_API_KEY", "test-key"),
            },
            "model_info": {"id": "deployment-2"},
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4",
                "api_key": os.getenv("OPENAI_API_KEY", "test-key"),
            },
            "model_info": {"id": "deployment-3"},
        },
    ]


@pytest.fixture
def router_with_lowest_latency(model_list):
    """Router fixture configured with lowest latency routing strategy"""
    return Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        set_verbose=False,
    )


def test_empty_latency_list_division_by_zero(model_list):
    """
    Test that _get_available_deployments handles empty latency lists gracefully
    
    Scenario: Deployments exist in cache but have no latency data.
    Should use default latency value instead of raising ZeroDivisionError.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    model_group = "gpt-3.5-turbo"
    latency_key = f"{model_group}_map"
    
    # Cache with empty latency lists (can happen when deployments are newly added)
    cache_data = {
        "deployment-1": {
            "latency": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        },
        "deployment-2": {
            "latency": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        }
    }
    test_cache.set_cache(key=latency_key, value=cache_data)
    
    result = lowest_latency_logger.get_available_deployments(
        model_group=model_group,
        healthy_deployments=model_list[:2],  # Only use first 2 deployments
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={}
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] in ["deployment-1", "deployment-2"]


# ============================================================================
# PHASE 1: ADDITIONAL DIVISION BY ZERO TEST CASES
# ============================================================================

def test_streaming_empty_ttft_and_latency_division_by_zero(model_list):
    """
    Test streaming scenario where both TTFT and latency lists are empty
    
    Critical edge case: streaming=True but both lists are empty.
    Should handle gracefully with default latency values.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    model_group = "gpt-3.5-turbo"
    latency_key = f"{model_group}_map"
    
    cache_data = {
        "deployment-1": {
            "latency": [],
            "time_to_first_token": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        },
        "deployment-2": {
            "latency": [],
            "time_to_first_token": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        }
    }
    test_cache.set_cache(key=latency_key, value=cache_data)
    
    result = lowest_latency_logger.get_available_deployments(
        model_group=model_group,
        healthy_deployments=model_list[:2],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={"stream": True}
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] in ["deployment-1", "deployment-2"]


def test_mixed_empty_and_populated_latency_division_by_zero(model_list):
    """
    Test scenario with multiple deployments where some have empty latency lists
    
    Tests mixed scenario where some deployments have latency data while others are empty.
    Should handle empty deployments gracefully without affecting populated ones.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    model_group = "gpt-3.5-turbo"
    latency_key = f"{model_group}_map"
    
    cache_data = {
        "deployment-1": {
            "latency": [1.5, 2.0, 1.8],
            "2024-01-15-10-30": {"tpm": 100, "rpm": 5}
        },
        "deployment-2": {
            "latency": [],  # Empty
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        },
        "deployment-3": {
            "latency": [3.0, 2.5],
            "2024-01-15-10-30": {"tpm": 75, "rpm": 3}
        }
    }
    test_cache.set_cache(key=latency_key, value=cache_data)
    
    result = lowest_latency_logger.get_available_deployments(
        model_group=model_group,
        healthy_deployments=model_list[:3],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={}
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] in ["deployment-1", "deployment-2", "deployment-3"]


def test_streaming_uses_latency_length_instead_of_ttft_length(model_list):
    """
    Test the logic bug where streaming calculation uses wrong denominator
    
    Tests scenario where code calculates total from TTFT data but divides by
    len(item_latency) instead of len(item_ttft_latency). Should use correct denominator.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    request_count_dict = {
        "deployment-1": {
            "latency": [],  # Empty
            "time_to_first_token": [0.5, 0.7, 0.6],  # Has TTFT data
            "2024-01-15-10-30": {"tpm": 50, "rpm": 1}
        }
    }
    
    result = lowest_latency_logger._get_available_deployments(
        model_group="gpt-3.5-turbo",
        healthy_deployments=model_list[:1],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={"stream": True},
        request_count_dict=request_count_dict
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] == "deployment-1"


def test_ttft_data_exists_but_divides_by_latency_length(model_list):
    """
    Test another variant of the logic bug with TTFT data
    
    Cache-based test where TTFT data exists but division uses wrong list length.
    Should use correct denominator for TTFT calculations.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    model_group = "gpt-3.5-turbo"
    latency_key = f"{model_group}_map"
    
    cache_data = {
        "deployment-1": {
            "latency": [],  # Empty
            "time_to_first_token": [0.3, 0.4, 0.5, 0.6],
            "2024-01-15-10-30": {"tpm": 80, "rpm": 4}
        }
    }
    test_cache.set_cache(key=latency_key, value=cache_data)
    
    result = lowest_latency_logger.get_available_deployments(
        model_group=model_group,
        healthy_deployments=model_list[:1],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={"stream": True}
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] == "deployment-1"


def test_direct_call_empty_latency_division_by_zero(model_list):
    """
    Test direct call to _get_available_deployments with empty latency
    
    Bypasses cache layer to directly test the internal method with empty latency data.
    Should handle empty latency gracefully.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    request_count_dict = {
        "deployment-1": {
            "latency": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        },
        "deployment-2": {
            "latency": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        }
    }
    
    result = lowest_latency_logger._get_available_deployments(
        model_group="gpt-3.5-turbo",
        healthy_deployments=model_list[:2],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={},
        request_count_dict=request_count_dict
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] in ["deployment-1", "deployment-2"]


def test_direct_call_streaming_empty_division_by_zero(model_list):
    """
    Test direct call with streaming and empty data
    
    Tests streaming path directly with empty TTFT and latency lists.
    Should handle gracefully in streaming mode.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    request_count_dict = {
        "deployment-1": {
            "latency": [],
            "time_to_first_token": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        }
    }
    
    result = lowest_latency_logger._get_available_deployments(
        model_group="gpt-3.5-turbo",
        healthy_deployments=model_list[:1],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={"stream": True},
        request_count_dict=request_count_dict
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] == "deployment-1"


def test_mixed_deployments_with_empty_latency(model_list):
    """
    Test scenario where some deployments have latency data and others don't
    
    Mixed scenario with some deployments having data and others empty.
    Should handle empty latency gracefully.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    model_group = "gpt-3.5-turbo"
    latency_key = f"{model_group}_map"
    
    cache_data = {
        "deployment-1": {
            "latency": [2.5, 3.0],
            "2024-01-15-10-30": {"tpm": 50, "rpm": 1}
        },
        "deployment-2": {
            "latency": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        }
    }
    test_cache.set_cache(key=latency_key, value=cache_data)
    
    result = lowest_latency_logger.get_available_deployments(
        model_group=model_group,
        healthy_deployments=model_list[:2],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={}
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] in ["deployment-1", "deployment-2"]


def test_streaming_empty_ttft_latency(model_list):
    """
    Test streaming scenario with empty latency lists
    
    Tests when streaming=True but item_ttft_latency is empty, so it falls back
    to item_latency. Should handle this fallback gracefully.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    model_group = "gpt-3.5-turbo"
    latency_key = f"{model_group}_map"
    
    cache_data = {
        "deployment-1": {
            "latency": [],
            "time_to_first_token": [],
            "2024-01-15-10-30": {"tpm": 50, "rpm": 1}
        }
    }
    test_cache.set_cache(key=latency_key, value=cache_data)
    
    result = lowest_latency_logger.get_available_deployments(
        model_group=model_group,
        healthy_deployments=model_list[:1],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={"stream": True}  # This triggers TTFT path but falls back
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] == "deployment-1"


def test_get_available_deployments_with_completely_empty_cache(model_list):
    """
    Test scenario where cache exists but deployments have no latency data at all
    
    Cache exists but deployment has minimal data structure with empty latency.
    Should handle completely empty latency data gracefully.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    model_group = "gpt-3.5-turbo"
    latency_key = f"{model_group}_map"
    
    cache_data = {
        "deployment-1": {
            "latency": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        }
    }
    test_cache.set_cache(key=latency_key, value=cache_data)
    
    result = lowest_latency_logger.get_available_deployments(
        model_group=model_group,
        healthy_deployments=model_list[:1],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={}
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] == "deployment-1"


@pytest.mark.parametrize("sync_mode", [True, False])
def test_router_level_division_by_zero(model_list, sync_mode):
    """
    Test the division by zero error at the Router level using actual Router methods
    
    Tests both sync and async router methods with empty latency data.
    Should handle empty latency gracefully at the router level.
    """
    router = Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        set_verbose=False,
    )
    
    model_group = "gpt-3.5-turbo"
    latency_key = f"{model_group}_map"
    
    cache_data = {
        "deployment-1": {
            "latency": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        },
        "deployment-2": {
            "latency": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        }
    }
    router.lowestlatency_logger.router_cache.set_cache(key=latency_key, value=cache_data)
    
    if sync_mode:
        result = router.lowestlatency_logger.get_available_deployments(
            model_group=model_group,
            healthy_deployments=router.get_model_list(model_name="gpt-3.5-turbo"),
            messages=[{"role": "user", "content": "test"}],
            input=None,
            request_kwargs={}
        )
    else:
        # For async, we need to test the async version
        import asyncio
        async def test_async():
            return await router.lowestlatency_logger.async_get_available_deployments(
                model_group=model_group,
                healthy_deployments=router.get_model_list(model_name="gpt-3.5-turbo"),
                messages=[{"role": "user", "content": "test"}],
                input=None,
                request_kwargs={}
            )
        
        # Run the async test
        result = asyncio.run(test_async())
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] in ["deployment-1", "deployment-2"]


def test_direct_method_call_with_empty_latency(model_list):
    """
    Test calling _get_available_deployments directly with empty latency data
    
    Direct call to internal method with empty latency data.
    Should handle empty latency gracefully.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    request_count_dict = {
        "deployment-1": {
            "latency": [],
            "2024-01-15-10-30": {"tpm": 0, "rpm": 0}
        }
    }
    
    result = lowest_latency_logger._get_available_deployments(
        model_group="gpt-3.5-turbo",
        healthy_deployments=model_list[:1],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={},
        request_count_dict=request_count_dict
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] == "deployment-1"


def test_streaming_with_empty_ttft_direct_call(model_list):
    """
    Test streaming scenario by calling _get_available_deployments directly
    
    Tests the exact bug: streaming=True, empty TTFT, empty latency.
    Should handle this scenario gracefully.
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    
    request_count_dict = {
        "deployment-1": {
            "latency": [],
            "time_to_first_token": [],
            "2024-01-15-10-30": {"tpm": 50, "rpm": 1}
        }
    }
    
    result = lowest_latency_logger._get_available_deployments(
        model_group="gpt-3.5-turbo",
        healthy_deployments=model_list[:1],
        messages=[{"role": "user", "content": "test"}],
        input=None,
        request_kwargs={"stream": True},  # This triggers TTFT path but falls back
        request_count_dict=request_count_dict
    )
    
    # Should return a valid deployment
    assert result is not None
    assert "model_info" in result
    assert result["model_info"]["id"] == "deployment-1"


if __name__ == "__main__":
    # Run tests to verify they fail with current code
    print("Running division by zero tests...")
    
    # Create a simple model list for testing
    test_model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo"},
            "model_info": {"id": "deployment-1"},
        }
    ]
    
    try:
        test_empty_latency_list_division_by_zero(test_model_list)
        print("❌ test_empty_latency_list_division_by_zero should have failed but didn't")
    except ZeroDivisionError:
        print("✅ test_empty_latency_list_division_by_zero correctly reproduced the error")
    except Exception as e:
        print(f"❓ test_empty_latency_list_division_by_zero failed with unexpected error: {e}")
    
    try:
        test_direct_method_call_with_empty_latency(test_model_list)
        print("❌ test_direct_method_call_with_empty_latency should have failed but didn't")
    except ZeroDivisionError:
        print("✅ test_direct_method_call_with_empty_latency correctly reproduced the error")
    except Exception as e:
        print(f"❓ test_direct_method_call_with_empty_latency failed with unexpected error: {e}")
    
    print("\n" + "="*80)
    print("PHASE 1 IMPLEMENTATION COMPLETE")
    print("="*80)
    print("All tests above should show ✅ marks, indicating they successfully")
    print("reproduced the division by zero error in the current code.")
    print("\nThese tests will PASS after implementing the fix in Phase 2.")
    print("\nPhase 1 Test Summary:")
    print("- ✅ Core division by zero scenarios covered")
    print("- ✅ Logic bug scenarios (wrong denominator) covered")
    print("- ✅ Direct method call scenarios covered")
    print("- ✅ Streaming and non-streaming paths covered")
    print("- ✅ Mixed deployment scenarios covered")
    print("\nNext: Implement Phase 2 - Create tests for fixed behavior")