#### What this tests ####
#    This tests the router's handling of zero completion tokens in lowest latency routing

import os
import sys
import time
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler


def test_zero_completion_tokens_no_division_error():
    """
    Test that log_success_event handles zero completion tokens without ZeroDivisionError
    
    This tests the fix for issue #12641 where responses with zero completion tokens
    (e.g., from Gemini with long contexts) caused ZeroDivisionError
    """
    test_cache = DualCache()

    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gemini-2.5-flash",
                "deployment": "gemini/gemini-2.5-flash",
            },
            "model_info": {"id": deployment_id},
        }
    }
    
    # Create a ModelResponse with zero completion tokens (as reported in issue)
    response_obj = litellm.ModelResponse(
        id='9p13aIGDDNmPmLAP5-23mQQ',
        created=1752669685,
        model='gemini-2.5-flash',
        object='chat.completion',
        choices=[
            litellm.Choices(
                finish_reason='stop',
                index=0,
                message=litellm.Message(
                    content=None,
                    role='assistant',
                    tool_calls=None
                )
            )
        ],
        usage=litellm.Usage(
            completion_tokens=0,  # This causes the ZeroDivisionError
            prompt_tokens=245537,
            total_tokens=245537
        )
    )
    
    start_time = time.time()
    time.sleep(0.1)  # Simulate some response time
    end_time = time.time()
    
    # This should not raise ZeroDivisionError
    try:
        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    except ZeroDivisionError:
        pytest.fail("log_success_event raised ZeroDivisionError with zero completion tokens")
    
    # Verify the deployment was logged (even with zero completion tokens)
    cached_value = test_cache.get_cache(
        key=f"{kwargs['litellm_params']['metadata']['model_group']}_map"
    )
    assert cached_value is not None
    assert deployment_id in cached_value


def test_zero_completion_tokens_with_time_to_first_token():
    """
    Test that time_to_first_token calculation also handles zero completion tokens
    """
    test_cache = DualCache()
    
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gemini-2.5-flash",
                "deployment": "gemini/gemini-2.5-flash",
            },
            "model_info": {"id": deployment_id},
            "stream": True,
        },
        "completion_start_time": time.time() + 0.05,  # Simulate time to first token
    }
    
    # Create a ModelResponse with zero completion tokens
    response_obj = litellm.ModelResponse(
        usage=litellm.Usage(
            completion_tokens=0,
            prompt_tokens=100000,
            total_tokens=100000
        )
    )
    
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    
    # This should not raise ZeroDivisionError
    try:
        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    except ZeroDivisionError:
        pytest.fail("log_success_event raised ZeroDivisionError with zero completion tokens in streaming")


if __name__ == "__main__":
    test_zero_completion_tokens_no_division_error()
    test_zero_completion_tokens_with_time_to_first_token()
    print("All tests passed!")