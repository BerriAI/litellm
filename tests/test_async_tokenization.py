"""
Tests for async tokenization feature

Verifies that large inputs use threadpool tokenization to prevent
event loop blocking, while small inputs use fast inline tokenization.
"""

import asyncio
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.litellm_core_utils.token_counter import (
    async_token_counter,
    _calculate_input_size_bytes,
    _should_use_async_tokenization,
)


def test_calculate_input_size_bytes():
    """Test input size calculation"""
    # Text input
    text = "Hello world"
    size = _calculate_input_size_bytes(text=text, messages=None)
    assert size == len(text.encode('utf-8'))
    
    # Messages input
    messages = [
        {"role": "user", "content": "Test message"}
    ]
    size = _calculate_input_size_bytes(text=None, messages=messages)
    assert size > 0
    
    # List of text
    text_list = ["Hello", "World"]
    size = _calculate_input_size_bytes(text=text_list, messages=None)
    assert size > 0


def test_should_use_async_tokenization():
    """Test async tokenization threshold logic"""
    # Save original setting
    original_threshold = litellm.async_tokenizer_threshold_bytes
    
    try:
        # Disabled by default
        litellm.async_tokenizer_threshold_bytes = None
        assert _should_use_async_tokenization(1_000_000) is False
        
        # Below threshold
        litellm.async_tokenizer_threshold_bytes = 500_000
        assert _should_use_async_tokenization(100_000) is False
        
        # Above threshold
        assert _should_use_async_tokenization(600_000) is True
        
    finally:
        litellm.async_tokenizer_threshold_bytes = original_threshold


def test_token_counter_with_disabled_flag():
    """Test that disable_token_counter still works"""
    original_flag = litellm.disable_token_counter
    
    try:
        litellm.disable_token_counter = True
        
        result = litellm.token_counter(
            model="gpt-4o-mini",
            text="This should return 0"
        )
        
        assert result == 0, "Should return 0 when disabled"
        
    finally:
        litellm.disable_token_counter = original_flag


@pytest.mark.asyncio
async def test_async_token_counter_small_input():
    """Test that small inputs use inline tokenization (fast path)"""
    original_threshold = litellm.async_tokenizer_threshold_bytes
    
    try:
        # Set high threshold so this input stays inline
        litellm.async_tokenizer_threshold_bytes = 1_000_000
        
        text = "Hello, this is a small input"
        result = await async_token_counter(
            model="gpt-4o-mini",
            text=text
        )
        
        # Should get a valid token count
        assert result > 0
        assert isinstance(result, int)
        
    finally:
        litellm.async_tokenizer_threshold_bytes = original_threshold


@pytest.mark.asyncio
async def test_async_token_counter_large_input():
    """Test that large inputs use threadpool (non-blocking path)"""
    original_threshold = litellm.async_tokenizer_threshold_bytes
    
    try:
        # Set low threshold to force threadpool
        litellm.async_tokenizer_threshold_bytes = 1000  # 1KB
        
        # Large input (100KB)
        large_text = "x" * 100_000
        result = await async_token_counter(
            model="gpt-4o-mini",
            text=large_text
        )
        
        # Should still get valid token count
        assert result > 0
        assert isinstance(result, int)
        
    finally:
        litellm.async_tokenizer_threshold_bytes = original_threshold


@pytest.mark.asyncio
async def test_async_tokenization_timeout():
    """Test that tokenization timeout works"""
    original_threshold = litellm.async_tokenizer_threshold_bytes
    original_timeout = litellm.tokenizer_timeout_seconds
    
    try:
        # Enable async tokenization with very short timeout
        litellm.async_tokenizer_threshold_bytes = 1000
        litellm.tokenizer_timeout_seconds = 0.001  # 1ms - will timeout for large inputs
        
        # Very large input that will take longer than 1ms to tokenize
        large_text = "x" * 1_000_000  # 1MB
        result = await async_token_counter(
            model="gpt-4o-mini",
            text=large_text
        )
        
        # Should return 0 on timeout (graceful degradation)
        assert result == 0, "Should return 0 on timeout"
        
    finally:
        litellm.async_tokenizer_threshold_bytes = original_threshold
        litellm.tokenizer_timeout_seconds = original_timeout


@pytest.mark.asyncio
async def test_event_loop_remains_responsive():
    """
    Test that event loop stays responsive during tokenization of large inputs.
    
    """
    original_threshold = litellm.async_tokenizer_threshold_bytes
    
    try:
        # Enable async tokenization
        litellm.async_tokenizer_threshold_bytes = 10_000  # 10KB threshold
        
        # Create a heartbeat task to monitor event loop
        heartbeat_count = 0
        stop_event = asyncio.Event()
        
        async def heartbeat():
            nonlocal heartbeat_count
            while not stop_event.is_set():
                heartbeat_count += 1
                await asyncio.sleep(0.01)  # 10ms heartbeat
        
        heartbeat_task = asyncio.create_task(heartbeat())
        
        # Tokenize large input (should use threadpool, not block event loop)
        large_text = "Test message. " * 50_000  # ~700KB
        
        result = await async_token_counter(
            model="gpt-4o-mini",
            text=large_text
        )
        
        # Stop heartbeat
        stop_event.set()
        await heartbeat_task
        
        # Verify tokenization worked
        assert result > 0, "Tokenization should succeed"
        
        # Verify event loop was responsive
        # If event loop was blocked, heartbeat_count would be 0 or very low
        # If responsive, should be >= 5 (tokenization takes some time)
        assert heartbeat_count >= 5, f"Event loop should remain responsive, got {heartbeat_count} heartbeats"
        
    finally:
        litellm.async_tokenizer_threshold_bytes = original_threshold


@pytest.mark.asyncio
async def test_concurrent_tokenization():
    """
    Test multiple concurrent tokenizations don't block each other.
    
    This simulates the real-world scenario from Issue #9145.
    """
    original_threshold = litellm.async_tokenizer_threshold_bytes
    
    try:
        # Enable async tokenization
        litellm.async_tokenizer_threshold_bytes = 10_000
        
        # Create multiple large inputs
        large_texts = [
            f"Request {i}: " + ("test " * 10_000)  # ~50KB each
            for i in range(5)
        ]
        
        import time
        start = time.time()
        
        tasks = [
            async_token_counter(model="gpt-4o-mini", text=text)
            for text in large_texts
        ]
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        
        assert all(r > 0 for r in results), "All tokenizations should succeed"
        
        # Should complete reasonably fast (concurrent, not sequential)
        # If they ran sequentially, would take much longer
        assert duration < 10, f"Should complete concurrently, took {duration:.2f}s"
        
    finally:
        litellm.async_tokenizer_threshold_bytes = original_threshold


def test_threadpool_max_workers_respected():
    """Test that threadpool max_workers setting is respected"""
    import litellm.litellm_core_utils.token_counter as tc_module
    
    original_threshold = litellm.async_tokenizer_threshold_bytes
    original_max_workers = litellm.tokenizer_threadpool_max_workers
    original_pool = tc_module._tokenizer_threadpool
    
    try:
        # Clean up any existing threadpool
        if tc_module._tokenizer_threadpool is not None:
            tc_module._tokenizer_threadpool.shutdown(wait=False)
            tc_module._tokenizer_threadpool = None
        
        # Set new configuration
        litellm.async_tokenizer_threshold_bytes = 1000
        litellm.tokenizer_threadpool_max_workers = 8
        
        # Get threadpool (should create new one with 8 workers)
        pool = tc_module._get_tokenizer_threadpool()
        
        assert pool is not None, "Threadpool should be created"
        assert pool._max_workers == 8, f"Expected 8 workers, got {pool._max_workers}"
        
    finally:
        # Cleanup
        if tc_module._tokenizer_threadpool is not None:
            tc_module._tokenizer_threadpool.shutdown(wait=False)
        tc_module._tokenizer_threadpool = original_pool
        litellm.async_tokenizer_threshold_bytes = original_threshold
        litellm.tokenizer_threadpool_max_workers = original_max_workers


if __name__ == "__main__":
    test_calculate_input_size_bytes()
    test_should_use_async_tokenization()
    test_token_counter_with_disabled_flag()
    asyncio.run(test_async_token_counter_small_input())
    asyncio.run(test_async_token_counter_large_input())
    asyncio.run(test_async_tokenization_timeout())
    asyncio.run(test_event_loop_remains_responsive())
    asyncio.run(test_concurrent_tokenization())
    test_threadpool_max_workers_respected()
