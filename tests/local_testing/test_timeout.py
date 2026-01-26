#### What this tests ####
#    This tests the timeout decorator

import os
import sys
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
from litellm._uuid import uuid

import httpx
import openai
import pytest

import litellm


@pytest.mark.parametrize(
    "model, provider",
    [
        ("gpt-3.5-turbo", "openai"),
        ("azure/gpt-4.1-mini", "azure"),
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_httpx_timeout(model, provider, sync_mode):
    """
    Test if setting httpx.timeout works for completion calls
    """
    timeout_val = httpx.Timeout(10.0, connect=60.0)

    messages = [{"role": "user", "content": "Hey, how's it going?"}]

    if sync_mode:
        response = litellm.completion(
            model=model, messages=messages, timeout=timeout_val
        )
    else:
        response = await litellm.acompletion(
            model=model, messages=messages, timeout=timeout_val
        )

    print(f"response: {response}")


def test_timeout():
    # this Will Raise a timeout
    litellm.set_verbose = False
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            timeout=0.01,
            messages=[{"role": "user", "content": "hello, write a 20 pg essay"}],
        )
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


# test_timeout()


def test_bedrock_timeout():
    # this Will Raise a timeout
    litellm.set_verbose = True
    try:
        response = litellm.completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            timeout=0.01,
            messages=[{"role": "user", "content": "hello, write a 20 pg essay"}],
        )
        pytest.fail("Did not raise error `openai.APITimeoutError`")
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


def test_hanging_request_azure():
    litellm.set_verbose = True
    import asyncio
    from unittest.mock import AsyncMock, patch, MagicMock

    try:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "azure-gpt",
                    "litellm_params": {
                        "model": "azure/gpt-4o-new-test",
                        "api_base": os.environ.get("AZURE_API_BASE", "https://test.openai.azure.com"),
                        "api_key": os.environ.get("AZURE_API_KEY", "test-key"),
                    },
                },
                {
                    "model_name": "openai-gpt",
                    "litellm_params": {"model": "gpt-3.5-turbo"},
                },
            ],
            num_retries=0,
        )

        encoded = litellm.utils.encode(model="gpt-3.5-turbo", text="blue")[0]

        async def _test():
            # Mock the Azure OpenAI client's create method to simulate a hanging request
            with patch("openai.resources.chat.completions.AsyncCompletions.create") as mock_create:
                # Simulate a hanging request that takes longer than the timeout
                async def hanging_request(*args, **kwargs):
                    await asyncio.sleep(10)  # Sleep much longer than the 0.01s timeout
                    return MagicMock()
                
                mock_create.side_effect = hanging_request
                
                response = await router.acompletion(
                    model="azure-gpt",
                    messages=[
                        {"role": "user", "content": f"what color is red {uuid.uuid4()}"}
                    ],
                    logit_bias={encoded: 100},
                    timeout=0.01,
                )
                print(response)
                return response

        response = asyncio.run(_test())

        if response.choices[0].message.content is not None:
            pytest.fail("Got a response, expected a timeout")
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except litellm.exceptions.APIError as e:
        # Azure may convert CancelledError to APIError - this is also acceptable for timeout scenarios
        print(
            "Passed: Raised APIError due to timeout (CancelledError). This is acceptable.", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError` or `litellm.exceptions.APIError`. Instead raised error type: {type(e)}, Error: {e}"
        )


# test_hanging_request_azure()


def test_hanging_request_openai():
    litellm.set_verbose = True
    try:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "azure-gpt",
                    "litellm_params": {
                        "model": "azure/gpt-4.1-mini",
                        "api_base": os.environ["AZURE_API_BASE"],
                        "api_key": os.environ["AZURE_API_KEY"],
                    },
                },
                {
                    "model_name": "openai-gpt",
                    "litellm_params": {"model": "gpt-3.5-turbo"},
                },
            ],
            num_retries=0,
        )

        encoded = litellm.utils.encode(model="gpt-3.5-turbo", text="blue")[0]
        response = router.completion(
            model="openai-gpt",
            messages=[{"role": "user", "content": "what color is red"}],
            logit_bias={encoded: 100},
            timeout=0.01,
        )
        print(response)

        if response.choices[0].message.content is not None:
            pytest.fail("Got a response, expected a timeout")
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


# test_hanging_request_openai()

# test_timeout()


def test_timeout_streaming():
    # this Will Raise a timeout
    litellm.set_verbose = False
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello, write a 20 pg essay"}],
            timeout=0.0001,
            stream=True,
        )
        for chunk in response:
            print(chunk)
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


# test_timeout_streaming()


@pytest.mark.skip(reason="local test")
def test_timeout_ollama():
    # this Will Raise a timeout
    import litellm

    litellm.set_verbose = True
    try:
        litellm.request_timeout = 0.1
        litellm.set_verbose = True
        response = litellm.completion(
            model="ollama/phi",
            messages=[{"role": "user", "content": "hello, what llm are u"}],
            max_tokens=1,
            api_base="https://test-ollama-endpoint.onrender.com",
        )
        # Add any assertions here to check the response
        litellm.request_timeout = None
        print(response)
    except openai.APITimeoutError as e:
        print("got a timeout error! Passed ! ")
        pass


# test_timeout_ollama()


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_anthropic_timeout(streaming, sync_mode):
    litellm.set_verbose = False

    try:
        if sync_mode:
            response = litellm.completion(
                model="claude-3-5-sonnet-20240620",
                timeout=0.01,
                messages=[{"role": "user", "content": "hello, write a 20 pg essay"}],
                stream=streaming,
            )
            if isinstance(response, litellm.CustomStreamWrapper):
                for chunk in response:
                    pass
        else:
            response = await litellm.acompletion(
                model="claude-3-5-sonnet-20240620",
                timeout=0.01,
                messages=[{"role": "user", "content": "hello, write a 20 pg essay"}],
                stream=streaming,
            )
            if isinstance(response, litellm.CustomStreamWrapper):
                async for chunk in response:
                    pass
        pytest.fail("Did not raise error `openai.APITimeoutError`")
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass


@pytest.mark.asyncio
async def test_timeout_respects_total_time_not_per_retry():
    """
    Test that timeout applies to the TOTAL operation time, not per-retry.
    
    This test ensures that when a user sets timeout=2, the entire operation
    (including all retries) times out at ~2 seconds, not at 2s * num_retries.
    
    This is a regression test for the issue where timeout was being applied
    per-retry attempt, causing the total time to be much longer than expected.
    """
    litellm.set_verbose = False
    
    timeout_value = 2.0
    # Allow for some overhead (network, processing, etc.)
    # but ensure we don't wait for multiple retries
    max_allowed_time = timeout_value + 1.0  # 3 seconds max
    
    start_time = time.time()
    
    try:
        # This should timeout because we're asking for a long response
        # with a very short timeout
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            timeout=timeout_value,
            messages=[{"role": "user", "content": "Write a very long detailed essay about the history of computing, at least 5000 words."}],
        )
        pytest.fail("Expected timeout error but got a response")
    except (openai.APITimeoutError, litellm.exceptions.Timeout) as e:
        elapsed_time = time.time() - start_time
        
        print(f"Timeout occurred after {elapsed_time:.2f} seconds")
        print(f"Expected timeout: {timeout_value} seconds")
        print(f"Max allowed time: {max_allowed_time} seconds")
        
        # Verify that the timeout happened within the expected time window
        # It should be close to timeout_value, not timeout_value * num_retries
        assert elapsed_time < max_allowed_time, (
            f"Timeout took too long! Expected ~{timeout_value}s, "
            f"got {elapsed_time:.2f}s. This suggests timeout is being "
            f"applied per-retry instead of to the total operation."
        )
        
        # Also verify it's not TOO fast (sanity check)
        assert elapsed_time >= timeout_value * 0.5, (
            f"Timeout happened too quickly: {elapsed_time:.2f}s. "
            f"Expected at least {timeout_value * 0.5}s"
        )
        
        print("✓ Timeout correctly applied to total operation time, not per-retry")
    except Exception as e:
        pytest.fail(
            f"Expected timeout error but got different error: {type(e).__name__}: {e}"
        )


@pytest.mark.asyncio  
async def test_timeout_with_retries_disabled():
    """
    Test that timeout works correctly when retries are explicitly disabled.
    This should timeout even faster since there are no retry attempts.
    """
    litellm.set_verbose = False
    
    timeout_value = 2.0
    max_allowed_time = timeout_value + 0.5  # Even tighter bound with no retries
    
    start_time = time.time()
    
    try:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            timeout=timeout_value,
            max_retries=0,  # Disable retries
            messages=[{"role": "user", "content": "Write a very long detailed essay about the history of computing, at least 5000 words."}],
        )
        pytest.fail("Expected timeout error but got a response")
    except (openai.APITimeoutError, litellm.exceptions.Timeout) as e:
        elapsed_time = time.time() - start_time
        
        print(f"Timeout with no retries occurred after {elapsed_time:.2f} seconds")
        
        assert elapsed_time < max_allowed_time, (
            f"Timeout took too long even with retries disabled! "
            f"Expected ~{timeout_value}s, got {elapsed_time:.2f}s"
        )
        
        print("✓ Timeout works correctly with retries disabled")
