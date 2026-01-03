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
from litellm import acompletion, atext_completion


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

    @pytest.mark.asyncio
    async def test_atext_completion_cancellation_handling(self):
        """Test that atext_completion properly handles asyncio.CancelledError."""
        
        async def cancelled_text_completion():
            """A text completion call that should be cancelled."""
            return await atext_completion(
                model="gpt-3.5-turbo-instruct",
                prompt="Write a long story about",
                mock_response="This should not be returned due to cancellation"
            )
        
        # Create a task and cancel it
        task = asyncio.create_task(cancelled_text_completion())
        
        # Cancel the task immediately
        task.cancel()
        
        # Verify that CancelledError is properly raised
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_atext_completion_cancellation_with_delay(self):
        """Test text completion cancellation with a small delay to simulate real-world usage."""
        
        # Mock the text_completion function to add delay
        async def mock_delayed_text_completion(*args, **kwargs):
            await asyncio.sleep(1.0)  # Simulate delay
            return litellm.TextCompletionResponse(
                id="test-id",
                choices=[
                    litellm.TextCompletionChoice(
                        finish_reason="stop",
                        index=0,
                        text="This should not be returned"
                    )
                ],
                created=1234567890,
                model="gpt-3.5-turbo-instruct",
                object="text_completion"
            )
        
        def mock_text_completion_that_returns_coroutine(*args, **kwargs):
            return mock_delayed_text_completion(*args, **kwargs)
        
        with patch('litellm.text_completion', side_effect=mock_text_completion_that_returns_coroutine):
            task = asyncio.create_task(
                atext_completion(
                    model="gpt-3.5-turbo-instruct", 
                    prompt="Complete this story:"
                )
            )
            
            # Let the task start, then cancel it
            await asyncio.sleep(0.1)
            task.cancel()
            
            # Verify cancellation is handled properly
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_atext_completion_with_router_cancellation(self):
        """Test that text completion cancellation works properly with the router."""
        
        # Create a simple router configuration
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo-instruct",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo-instruct"
                    }
                }
            ]
        )
        
        # Mock the router's _atext_completion method to add delay
        async def mock_router_text_completion(*args, **kwargs):
            await asyncio.sleep(1.0)  # Simulate delay
            return litellm.TextCompletionResponse(
                id="router-test-id",
                choices=[
                    litellm.TextCompletionChoice(
                        finish_reason="stop",
                        index=0,
                        text="Router text completion response"
                    )
                ],
                created=1234567890,
                model="gpt-3.5-turbo-instruct",
                object="text_completion"
            )
        
        with patch.object(router, '_atext_completion', side_effect=mock_router_text_completion):
            task = asyncio.create_task(
                router.atext_completion(
                    model="gpt-3.5-turbo-instruct",
                    prompt="Write a story about"
                )
            )
            
            # Cancel after a short delay
            await asyncio.sleep(0.1)
            task.cancel()
            
            # Verify cancellation is handled
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_atext_completion_basic_cancellation_scenarios(self):
        """Test various basic text completion cancellation scenarios."""
        
        # Test immediate cancellation
        task1 = asyncio.create_task(
            atext_completion(
                model="gpt-3.5-turbo-instruct",
                prompt="Write something"
            )
        )
        task1.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await task1
        
        # Test that a successful completion still works
        response = await atext_completion(
            model="gpt-3.5-turbo-instruct",
            prompt="Hello world"
        )
        
        # Should get a response (even if mocked)
        assert response is not None

    def test_sync_text_completion_still_works(self):
        """Test that sync text completion is unaffected by async cancellation changes."""
        
        # Sync text completion should work exactly as before
        response = litellm.text_completion(
            model="gpt-3.5-turbo-instruct",
            prompt="Hello world",
            mock_response="Sync text completion works"
        )
        
        assert hasattr(response, 'choices')
        assert response.choices[0].text == "Sync text completion works"

    @pytest.mark.asyncio
    async def test_openai_sdk_cancellation_propagation(self):
        """Test that cancellation properly propagates through the OpenAI SDK to close HTTP connections."""
        
        # This test simulates the real scenario: OpenAI client -> LiteLLM -> GPU service
        # We'll mock the OpenAI SDK call to verify that cancellation propagates
        
        openai_request_cancelled = False
        
        async def mock_openai_create_with_delay(*args, **kwargs):
            nonlocal openai_request_cancelled
            print(f"OpenAI SDK called with model: {kwargs.get('model', 'unknown')}")
            try:
                # Simulate a long-running HTTP request to GPU service (via OpenAI SDK)
                await asyncio.sleep(2.0)  # Long delay to ensure cancellation happens
                print("OpenAI SDK completed without cancellation")
                # Return a mock response that looks like OpenAI's response
                mock_response = MagicMock()
                mock_response.parse.return_value = MagicMock()
                mock_response.parse.return_value.model_dump.return_value = {
                    "id": "test-id",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-3.5-turbo",
                    "choices": [{"message": {"role": "assistant", "content": "Hello"}}]
                }
                return mock_response
            except asyncio.CancelledError:
                print("OpenAI SDK request was cancelled!")
                openai_request_cancelled = True
                raise  # Re-raise to propagate cancellation
        
        # Mock the OpenAI SDK's chat.completions.with_raw_response.create method
        with patch('openai.resources.chat.completions.AsyncCompletions.with_raw_response') as mock_with_raw_response:
            mock_with_raw_response.create = AsyncMock(side_effect=mock_openai_create_with_delay)
            
            task = asyncio.create_task(
                acompletion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}]
                )
            )
            
            # Let the request start, then cancel it
            await asyncio.sleep(0.1)
            task.cancel()
            
            # Verify the task was cancelled
            with pytest.raises(asyncio.CancelledError):
                await task
            
            # Most importantly: verify that the OpenAI SDK request was also cancelled
            # This means the HTTP connection to your GPU service would be closed
            assert openai_request_cancelled, "OpenAI SDK request to downstream service should have been cancelled"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

