"""
Memory leak tests for LiteLLM SDK and Router.

This module contains actual test cases that use the memory leak testing framework.

## Current Test Files

**test_sdk_completion.py**
Tests for litellm SDK completion methods:
- litellm.completion() - sync, non-streaming
- litellm.completion() - sync, streaming
- litellm.acompletion() - async, non-streaming
- litellm.acompletion() - async, streaming

**test_router_completion.py**
Tests for Router completion methods:
- Router.completion() - sync, non-streaming
- Router.completion() - sync, streaming
- Router.acompletion() - async, non-streaming
- Router.acompletion() - async, streaming

## Test Coverage

Each test validates that the component does not leak memory across multiple
request batches, with detection of:
- Continuous memory growth patterns
- Error-induced memory spikes
- Measurement noise and instability
"""

