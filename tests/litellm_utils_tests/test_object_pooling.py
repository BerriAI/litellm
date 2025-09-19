"""
Tests for the object pooling utilities in litellm.litellm_core_utils.object_pooling.
"""

import pytest
import threading
import time

from litellm.litellm_core_utils.object_pooling import (
    GenericPooledObjectFactory,
    get_object_pool,
    _pools
)


class TestObject:
    """Test object with reset method."""
    def __init__(self):
        self.value = 0
        self.reset_called = False
    
    def reset(self):
        self.value = 0
        self.reset_called = True
    
    def increment(self):
        self.value += 1


class TestObjectWithoutReset:
    """Test object without reset method."""
    def __init__(self):
        self.value = 0


class TestObjectPooling:
    """Test suite for object pooling functionality."""
    
    def setup_method(self):
        """Clear global pools before each test."""
        _pools.clear()
    
    def test_factory_creates_and_resets_objects(self):
        """Test factory object creation and reset functionality."""
        factory = GenericPooledObjectFactory(TestObject)
        pooled_obj = factory.createInstance()
        
        # Verify creation
        assert isinstance(pooled_obj.keeped_object, TestObject)
        assert pooled_obj.keeped_object.value == 0
        
        # Test reset
        pooled_obj.keeped_object.value = 42
        factory.reset(pooled_obj)
        assert pooled_obj.keeped_object.reset_called is True
        assert pooled_obj.keeped_object.value == 0
    
    def test_factory_reset_fallback(self):
        """Test fallback reset when no reset method exists."""
        factory = GenericPooledObjectFactory(TestObjectWithoutReset)
        pooled_obj = factory.createInstance()
        
        pooled_obj.keeped_object.value = 42
        factory.reset(pooled_obj)
        assert pooled_obj.keeped_object.__dict__ == {}
    
    def test_factory_validation(self):
        """Test object validation."""
        factory = GenericPooledObjectFactory(TestObject)
        pooled_obj = factory.createInstance()
        
        assert factory.validate(pooled_obj) is True
        
        pooled_obj.keeped_object = None
        assert factory.validate(pooled_obj) is False
    
    def test_pool_creation_and_reuse(self):
        """Test basic pool creation and object reuse."""
        pool_name = "test_pool"
        pool = get_object_pool(pool_name, TestObject, prewarm_count=1)
        
        assert pool is not None
        assert pool_name in _pools
        
        # Borrow, use, and return
        obj = pool.borrow(name=f"{pool_name}Factory")
        assert isinstance(obj.keeped_object, TestObject)
        pool.recycle(obj, name=f"{pool_name}Factory")
        
        # Verify reuse
        obj2 = pool.borrow(name=f"{pool_name}Factory")
        assert obj2 is not None
    
    def test_pool_maxsize_limits(self):
        """Test pool size limits."""
        pool_name = "limited_pool"
        pool = get_object_pool(pool_name, TestObject, pooled_maxsize=2, prewarm_count=0)
        
        # Borrow up to limit
        obj1 = pool.borrow(name=f"{pool_name}Factory")
        obj2 = pool.borrow(name=f"{pool_name}Factory")
        
        assert obj1 is not None
        assert obj2 is not None
        
        # Return and borrow again
        pool.recycle(obj1, name=f"{pool_name}Factory")
        obj3 = pool.borrow(name=f"{pool_name}Factory")
        assert obj3 is not None
    
    def test_pool_concurrent_access(self):
        """Test thread-safe pool access."""
        pool_name = "concurrent_pool"
        pool = get_object_pool(pool_name, TestObject, prewarm_count=3)
        
        results = []
        errors = []
        
        def worker():
            try:
                obj = pool.borrow(name=f"{pool_name}Factory")
                results.append(obj)
                time.sleep(0.01)
                pool.recycle(obj, name=f"{pool_name}Factory")
            except Exception as e:
                errors.append(e)
        
        # Run 5 concurrent workers
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 5
    
    def test_pool_isolation(self):
        """Test that different pools are isolated."""
        pool1 = get_object_pool("pool1", TestObject, prewarm_count=1)
        pool2 = get_object_pool("pool2", TestObjectWithoutReset, prewarm_count=1)
        
        assert pool1 is not pool2
        assert "pool1" in _pools
        assert "pool2" in _pools
        
        # Test different object types
        obj1 = pool1.borrow(name="pool1Factory")
        obj2 = pool2.borrow(name="pool2Factory")
        
        assert isinstance(obj1.keeped_object, TestObject)
        assert isinstance(obj2.keeped_object, TestObjectWithoutReset)
        
        pool1.recycle(obj1, name="pool1Factory")
        pool2.recycle(obj2, name="pool2Factory")


if __name__ == "__main__":
    pytest.main([__file__])