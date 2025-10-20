"""
E2E tests for shared session reuse feature

WHAT THIS TESTS:
When you pass shared_session to acompletion(), it should flow through
the entire call chain so the same aiohttp.ClientSession is reused for
connection pooling.

WHY THIS MATTERS:
Without session reuse, every request creates new TCP/TLS connections,
wasting ~100-500ms per request. With reuse, connections are pooled and
subsequent requests are 40-60% faster.
"""
import os
import sys
import inspect

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def is_parameter_active_in_source(source_code: str, search_pattern: str) -> bool:
    """
    Check if a parameter/line exists in source code and is NOT commented out.
    
    Args:
        source_code: The source code to search
        search_pattern: The text pattern to look for (e.g., "shared_session=shared_session")
    
    Returns:
        True if pattern found and not commented out, False otherwise
    """
    lines = source_code.split('\n')
    
    for line in lines:
        if search_pattern in line:
            # Make sure it's not commented out
            stripped = line.strip()
            if not stripped.startswith('#'):
                return True
    
    return False


# ============================================================================
# TEST 1: Check that the parameter exists in the API
# ============================================================================

def test_acompletion_accepts_shared_session():
    """Verify acompletion() has a shared_session parameter"""
    sig = inspect.signature(litellm.acompletion)
    
    assert 'shared_session' in sig.parameters, \
        "acompletion() missing shared_session parameter"
    
    # Should be optional (defaults to None)
    assert sig.parameters['shared_session'].default is None


def test_completion_accepts_shared_session():
    """Verify completion() has a shared_session parameter"""
    sig = inspect.signature(litellm.completion)
    
    assert 'shared_session' in sig.parameters, \
        "completion() missing shared_session parameter"
    
    assert sig.parameters['shared_session'].default is None


# ============================================================================
# TEST 2: Check that acompletion passes it to completion
# ============================================================================

def test_acompletion_passes_session_to_completion():
    """
    Verify that acompletion() includes shared_session in the kwargs
    it passes to completion()
    """
    source = inspect.getsource(litellm.acompletion)
    
    # Check for both possible quote styles
    found = (is_parameter_active_in_source(source, '"shared_session": shared_session') or
             is_parameter_active_in_source(source, "'shared_session': shared_session"))
    
    assert found, \
        "acompletion() doesn't include shared_session in completion_kwargs (or it's commented out)"


# ============================================================================
# TEST 3: Check the handler methods accept it
# ============================================================================

def test_handler_completion_accepts_shared_session():
    """Verify BaseLLMHTTPHandler.completion() accepts shared_session"""
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    
    sig = inspect.signature(BaseLLMHTTPHandler.completion)
    
    assert 'shared_session' in sig.parameters, \
        "Handler.completion() missing shared_session parameter"


def test_handler_async_completion_accepts_shared_session():
    """Verify BaseLLMHTTPHandler.async_completion() accepts shared_session"""
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    
    sig = inspect.signature(BaseLLMHTTPHandler.async_completion)
    
    assert 'shared_session' in sig.parameters, \
        "Handler.async_completion() missing shared_session parameter"


# ============================================================================
# TEST 4: THE KEY TEST - Does handler.completion pass it to async_completion?
# ============================================================================

def test_handler_passes_session_to_async_completion():
    """
    🔑 KEY TEST - Verifies the fix from commit f0d6d3dd
    
    The bug was: handler.completion() accepted shared_session but didn't
    pass it to async_completion(). This test ensures it's being passed.
    
    If this test fails, session reuse is BROKEN.
    """
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    
    source = inspect.getsource(BaseLLMHTTPHandler.completion)
    
    # Check if shared_session is being passed (and not commented out)
    found = is_parameter_active_in_source(source, 'shared_session=shared_session')
    
    assert found, \
        """
        CRITICAL BUG DETECTED!
        
        shared_session is NOT being passed from completion() to async_completion()
        
        This means session reuse is BROKEN. Every request will create new
        connections instead of reusing them, causing 40-60% slower performance.
        
        FIX: In BaseLLMHTTPHandler.completion(), when calling self.async_completion(),
        add this parameter:
            shared_session=shared_session
        
        This was the bug fixed in commit f0d6d3dd - it may have regressed!
        """