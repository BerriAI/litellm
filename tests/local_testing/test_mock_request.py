#### What this tests ####
#    This tests mock request calls to litellm

import os

import pytest

import litellm
import time


def test_mock_request_with_mock_timeout():
    """
    Allow user to set 'mock_timeout = True', this allows for testing if fallbacks/retries are working on timeouts.
    """
    start_time = time.time()
    with pytest.raises(litellm.Timeout):
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, I'm a mock request"}],
            timeout=3,
            mock_timeout=True,
        )
    end_time = time.time()
    assert end_time - start_time >= 3, f"Time taken: {end_time - start_time}"


def test_router_mock_request_with_mock_timeout():
    """
    Allow user to set 'mock_timeout = True', this allows for testing if fallbacks/retries are working on timeouts.
    """
    start_time = time.time()
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
    )
    with pytest.raises(litellm.Timeout):
        router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, I'm a mock request"}],
            timeout=3,
            mock_timeout=True,
        )
    end_time = time.time()
    assert end_time - start_time >= 3, f"Time taken: {end_time - start_time}"


def test_router_mock_request_with_mock_timeout_with_fallbacks():
    """
    Allow user to set 'mock_timeout = True', this allows for testing if fallbacks/retries are working on timeouts.
    """
    litellm.set_verbose = True
    start_time = time.time()
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "gpt-4.1-nano",
                "litellm_params": {
                    "model": "gpt-4.1-nano",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
        fallbacks=[{"gpt-3.5-turbo": ["gpt-4.1-nano"]}],
    )
    response = router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, I'm a mock request"}],
        timeout=3,
        num_retries=1,
        mock_timeout=True,
    )
    print(response)
    end_time = time.time()
    assert end_time - start_time >= 3, f"Time taken: {end_time - start_time}"
    assert "gpt-4.1-nano" in response.model, "Model should be gpt-4.1-nano"
