"""
Unit tests for CoroutineChecker utility class.

Focused test suite covering core functionality and main edge cases.
"""

import pytest
from unittest.mock import patch

from litellm.litellm_core_utils.coroutine_checker import CoroutineChecker, coroutine_checker


class TestCoroutineChecker:
    """Test cases for CoroutineChecker class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.checker = CoroutineChecker()

    def test_init(self):
        """Test CoroutineChecker initialization."""
        checker = CoroutineChecker()
        assert isinstance(checker, CoroutineChecker)

    @pytest.mark.parametrize("obj,expected,description", [
        # Basic function types
        (lambda: "sync", False, "sync lambda"),
        (len, False, "built-in function"),
        # Non-callable objects
        ("string", False, "string"),
        (123, False, "integer"),
        ([], False, "list"),
        ({}, False, "dict"),
        (None, False, "None"),
    ])
    def test_is_async_callable_basic_and_non_callable(self, obj, expected, description):
        """Test is_async_callable with basic types and non-callable objects."""
        assert self.checker.is_async_callable(obj) is expected, f"Failed for {description}: {obj}"

    def test_is_async_callable_async_and_sync_callables(self):
        """Test is_async_callable with various async and sync callable types."""
        # Async and sync functions
        async def async_func():
            return "async"
        
        def sync_func():
            return "sync"
        
        # Class methods
        class TestClass:
            def sync_method(self):
                return "sync"
            
            async def async_method(self):
                return "async"
        
        obj = TestClass()
        
        # Callable objects
        class SyncCallable:
            def __call__(self):
                return "sync"
        
        class AsyncCallable:
            async def __call__(self):
                return "async"
        
        # Test all async callables
        assert self.checker.is_async_callable(async_func) is True
        assert self.checker.is_async_callable(obj.async_method) is True
        assert self.checker.is_async_callable(AsyncCallable()) is True
        
        # Test all sync callables
        assert self.checker.is_async_callable(sync_func) is False
        assert self.checker.is_async_callable(obj.sync_method) is False
        assert self.checker.is_async_callable(SyncCallable()) is False

    def test_is_async_callable_caching(self):
        """Test that is_async_callable caches callable objects."""
        async def async_func():
            return "async"
        
        # Test that it works correctly
        result1 = self.checker.is_async_callable(async_func)
        assert result1 is True
        
        # Test that callable objects are cached
        assert async_func in self.checker._cache
        assert self.checker._cache[async_func] is True
        
        # Test that it works consistently
        result2 = self.checker.is_async_callable(async_func)
        assert result2 is True

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        from functools import partial
        
        # Error handling cases
        class ProblematicCallable:
            def __getattr__(self, name):
                if name == "__call__":
                    raise Exception("Cannot access __call__")
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        class UnstringableCallable:
            def __str__(self):
                raise Exception("Cannot convert to string")
            
            async def __call__(self):
                return "async"
        
        # Generator functions
        def sync_generator():
            yield "sync"
        
        async def async_generator():
            yield "async"
        
        # Partial functions
        def sync_func(x, y):
            return x + y
        
        async def async_func(x, y):
            return x + y
        
        sync_partial = partial(sync_func, 1)
        async_partial = partial(async_func, 1)
        
        # Test error handling
        assert self.checker.is_async_callable(ProblematicCallable()) is False
        assert self.checker.is_async_callable(UnstringableCallable()) is True
        
        # Test generators (both sync and async generators are not coroutine functions)
        assert self.checker.is_async_callable(sync_generator) is False
        assert self.checker.is_async_callable(async_generator) is False
        
        # Test partial functions (don't preserve coroutine nature)
        assert self.checker.is_async_callable(sync_partial) is False
        assert self.checker.is_async_callable(async_partial) is False

    def test_error_handling_in_inspect(self):
        """Test error handling when inspect.iscoroutinefunction raises exception."""
        with patch('inspect.iscoroutinefunction', side_effect=Exception("Inspect error")):
            async def async_func():
                return "async"
            
            # Should return False when inspect raises exception
            assert self.checker.is_async_callable(async_func) is False

    def test_global_coroutine_checker_instance(self):
        """Test the global coroutine_checker instance."""
        assert isinstance(coroutine_checker, CoroutineChecker)
        
        async def async_func():
            return "async"
        
        def sync_func():
            return "sync"
        
        assert coroutine_checker.is_async_callable(async_func) is True
        assert coroutine_checker.is_async_callable(sync_func) is False