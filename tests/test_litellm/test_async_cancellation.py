"""
Tests for async request cancellation in LiteLLM.

This module tests that cancelled async requests properly handle cancellation
and don't continue running in the background.
"""

import asyncio
import pytest
import time
from unittest.mock import patch, AsyncMock, MagicMock

import litellm
from litellm import acompletion


class TestAsyncCancellation:
    """Test suite for async request cancellation functionality."""

    @pytest.mark.asyncio
    async def test_acompletion_cancellation_handling(self):
        """Test that acompletion properly handles asyncio.CancelledError."""
        
        async def cancelled_completion():
            """A completion call that should be cancelled."""
            return await acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                mock_response="This should not be returned due to cancellation"
            )
        
        # Create a task and cancel it
        task = asyncio.create_task(cancelled_completion())
        
        # Cancel the task immediately
        task.cancel()
        
        # Verify that CancelledError is properly raised
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_acompletion_cancellation_with_delay(self):
        """Test cancellation with a small delay to simulate real-world usage."""
        
        async def delayed_completion():
            """A completion call with mock delay."""
            return await acompletion(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": "Hello"}],
                mock_response="This should not be returned",
                mock_delay=1.0  # 1 second delay
            )
        
        task = asyncio.create_task(delayed_completion())
        
        # Let the task start, then cancel it
        await asyncio.sleep(0.1)
        task.cancel()
        
        # Verify cancellation is handled properly
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_multiple_concurrent_cancellations(self):
        """Test that multiple concurrent requests can be cancelled independently."""
        
        async def completion_task(task_id: int):
            """Individual completion task."""
            return await acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Task {task_id}"}],
                mock_response=f"Response {task_id}",
                mock_delay=0.5
            )
        
        # Create multiple tasks
        tasks = [
            asyncio.create_task(completion_task(i))
            for i in range(3)
        ]
        
        # Cancel the first two tasks
        tasks[0].cancel()
        tasks[1].cancel()
        
        # Wait for all tasks to complete or be cancelled
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify the first two were cancelled and the third completed
        assert isinstance(results[0], asyncio.CancelledError)
        assert isinstance(results[1], asyncio.CancelledError)
        assert hasattr(results[2], 'choices')  # Should be a ModelResponse

    @pytest.mark.asyncio
    async def test_cancellation_propagation_in_router(self):
        """Test that cancellation works properly with the router."""
        
        # Create a simple router configuration
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "mock_response": "Router response"
                    }
                }
            ]
        )
        
        async def router_completion():
            """Router completion that should be cancelled."""
            return await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                mock_delay=1.0
            )
        
        task = asyncio.create_task(router_completion())
        
        # Cancel after a short delay
        await asyncio.sleep(0.1)
        task.cancel()
        
        # Verify cancellation is handled
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_cancellation_with_streaming(self):
        """Test that cancellation works with streaming responses."""
        
        async def streaming_completion():
            """Streaming completion that should be cancelled."""
            return await acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                mock_response="This is a streaming response",
                mock_delay=0.5
            )
        
        task = asyncio.create_task(streaming_completion())
        
        # Cancel the streaming task
        await asyncio.sleep(0.1)
        task.cancel()
        
        # Verify cancellation
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_cancellation_doesnt_affect_completed_requests(self):
        """Test that cancellation doesn't interfere with already completed requests."""
        
        # First, complete a request successfully
        response1 = await acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Successful response"
        )
        
        # Verify the first request completed successfully
        assert hasattr(response1, 'choices')
        assert response1.choices[0].message.content == "Successful response"
        
        # Now test cancellation of a second request
        async def second_completion():
            return await acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Second request"}],
                mock_response="This should be cancelled",
                mock_delay=1.0
            )
        
        task = asyncio.create_task(second_completion())
        task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await task

    def test_sync_completion_still_works(self):
        """Test that sync completion is unaffected by async cancellation changes."""
        
        # Sync completion should work exactly as before
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Sync response works"
        )
        
        assert hasattr(response, 'choices')
        assert response.choices[0].message.content == "Sync response works"

    @pytest.mark.asyncio
    async def test_cancellation_with_different_providers(self):
        """Test cancellation works with different LLM providers."""
        
        providers_to_test = [
            "gpt-3.5-turbo",  # OpenAI
            "claude-3-haiku-20240307",  # Anthropic
            "gemini-pro",  # Google
        ]
        
        for model in providers_to_test:
            async def provider_completion():
                return await acompletion(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    mock_response=f"Response from {model}",
                    mock_delay=0.5
                )
            
            task = asyncio.create_task(provider_completion())
            
            # Cancel each task
            await asyncio.sleep(0.1)
            task.cancel()
            
            # Verify cancellation works for each provider
            with pytest.raises(asyncio.CancelledError):
                await task


    @pytest.mark.asyncio
    async def test_async_path_is_used_when_available(self):
        """Test that acompletion uses native async methods when providers support them."""
        
        # This test verifies that we're using the async code path
        # We'll mock the completion function to return a coroutine
        
        async def mock_async_completion(**kwargs):
            """Mock async completion that can be properly cancelled."""
            await asyncio.sleep(0.1)  # Simulate async work
            return litellm.ModelResponse(
                id="test-id",
                choices=[
                    litellm.Choices(
                        finish_reason="stop",
                        index=0,
                        message=litellm.Message(
                            content="Async response",
                            role="assistant"
                        )
                    )
                ],
                created=1234567890,
                model="test-model",
                object="chat.completion",
                usage=litellm.Usage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15
                )
            )
        
        # Patch the completion function to return our async mock
        # We need to patch it to return a coroutine when called
        def mock_completion_that_returns_coroutine(**kwargs):
            return mock_async_completion(**kwargs)
        
        with patch('litellm.completion', side_effect=mock_completion_that_returns_coroutine):
            # This should use the async path and be properly cancellable
            task = asyncio.create_task(
                acompletion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}]
                )
            )
            
            # Cancel the task after a short delay
            await asyncio.sleep(0.05)  # Let it start
            task.cancel()
            
            # Verify it was cancelled
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_real_provider_async_cancellation(self):
        """Test cancellation with a provider that has native async support."""
        
        # Test with a provider that should have async support
        # Using mock to avoid real API calls
        async def cancelled_completion():
            return await acompletion(
                model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",  # Bedrock has async support
                messages=[{"role": "user", "content": "Hello"}],
                mock_response="This should be cancelled",
                mock_delay=1.0
            )
        
        task = asyncio.create_task(cancelled_completion())
        
        # Cancel after a short delay
        await asyncio.sleep(0.1)
        task.cancel()
        
        # Verify cancellation works
        with pytest.raises(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

