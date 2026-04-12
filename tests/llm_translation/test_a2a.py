"""
Minimal E2E tests for A2A (Agent-to-Agent) Protocol provider.

Tests validate that the endpoint is reachable and can handle both
streaming and non-streaming requests.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm


@pytest.mark.asyncio
async def test_a2a_completion_async_non_streaming():
    """
    Test A2A provider with async non-streaming request.
    
    Minimal test to validate endpoint reachability.
    
    Note: Requires an A2A agent running at http://0.0.0.0:9999
    Set A2A_API_BASE environment variable to use a different endpoint.
    """
    api_base = os.environ.get("A2A_API_BASE", "http://0.0.0.0:9999")
    
    try:
        response = await litellm.acompletion(
            model="a2a/test-agent",
            messages=[{"role": "user", "content": "Hello"}],
            api_base=api_base,
            stream=False,
        )
        
        print(f"Response: {response}")
        assert response is not None, "Expected non-None response"
        print(f"✅ Async non-streaming test passed")
        
    except litellm.exceptions.APIConnectionError as e:
        pytest.skip(f"A2A agent not reachable at {api_base}: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_a2a_completion_async_streaming():
    """
    Test A2A provider with async streaming request.
    
    Minimal test to validate streaming endpoint reachability.
    """
    api_base = os.environ.get("A2A_API_BASE", "http://0.0.0.0:9999")
    
    try:
        response = await litellm.acompletion(
            model="a2a/test-agent",
            messages=[{"role": "user", "content": "Hello"}],
            api_base=api_base,
            stream=True,
        )
        
        chunks = []
        async for chunk in response:  # type: ignore
            chunks.append(chunk)
            print(f"Chunk: {chunk}")
        
        assert len(chunks) > 0, "Expected at least one chunk in streaming response"
        print(f"✅ Async streaming test passed: received {len(chunks)} chunks")
        
    except litellm.exceptions.APIConnectionError as e:
        pytest.skip(f"A2A agent not reachable at {api_base}: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_a2a_completion_sync():
    """
    Test A2A provider with synchronous non-streaming request.
    
    Minimal test to validate sync endpoint reachability.
    """
    api_base = os.environ.get("A2A_API_BASE", "http://0.0.0.0:9999")
    
    try:
        response = litellm.completion(
            model="a2a/test-agent",
            messages=[{"role": "user", "content": "Hello"}],
            api_base=api_base,
            stream=False,
        )
        
        print(f"Response: {response}")
        assert response is not None, "Expected non-None response"
        print(f"✅ Sync non-streaming test passed")
        
    except litellm.exceptions.APIConnectionError as e:
        pytest.skip(f"A2A agent not reachable at {api_base}: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_a2a_completion_sync_streaming():
    """
    Test A2A provider with synchronous streaming request.
    
    Minimal test to validate sync streaming endpoint reachability.
    """
    api_base = os.environ.get("A2A_API_BASE", "http://0.0.0.0:9999")
    
    try:
        response = litellm.completion(
            model="a2a/test-agent",
            messages=[{"role": "user", "content": "Hello"}],
            api_base=api_base,
            stream=True,
        )
        
        chunks = []
        for chunk in response:  # type: ignore
            chunks.append(chunk)
            print(f"Chunk: {chunk}")
        
        assert len(chunks) > 0, "Expected at least one chunk in streaming response"
        print(f"✅ Sync streaming test passed: received {len(chunks)} chunks")
        
    except litellm.exceptions.APIConnectionError as e:
        pytest.skip(f"A2A agent not reachable at {api_base}: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

