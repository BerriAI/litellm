"""
Simplified tests for object pooling utilities in litellm.
"""

import pytest

from litellm.litellm_core_utils.object_pooling import (
    get_object_pool,
    _pools
)


class SimpleObject:
    """Simple test object with internal reset tracking."""
    def __init__(self):
        self.data = {}
        self.reset_count = 0
        self.creation_id = id(self)
    
    def reset(self):
        """Reset method that tracks how many times it's called."""
        self.data.clear()
        self.reset_count += 1
    
    def set_data(self, key, value):
        """Set data to verify reset works."""
        self.data[key] = value


class SimpleObjectNoReset:
    """Test object without reset method."""
    def __init__(self):
        self.data = {}
        self.creation_id = id(self)


class TestObjectPooling:
    """Simplified test suite for object pooling."""
    
    def setup_method(self):
        """Clear pools before each test."""
        _pools.clear()
    
    def test_reset_method_works(self):
        """Test that reset method is called when recycling objects."""
        pool_name = "reset_test"
        pool = get_object_pool(pool_name, SimpleObject, pooled_maxsize=1, prewarm_count=0)
        
        # Get an object and modify it
        obj = pool.borrow(name=f"{pool_name}Factory")
        obj.keeped_object.set_data("test", "value")
        initial_reset_count = obj.keeped_object.reset_count
        
        # Return to pool (should trigger reset)
        pool.recycle(obj, name=f"{pool_name}Factory")
        
        # Get the same object back
        obj2 = pool.borrow(name=f"{pool_name}Factory")
        
        # Verify reset was called
        assert obj2.keeped_object.reset_count == initial_reset_count + 1
        assert obj2.keeped_object.data == {}  # Data should be cleared
        assert obj.keeped_object.creation_id == obj2.keeped_object.creation_id  # Same object
    
    def test_fallback_reset_works(self):
        """Test fallback reset when no reset method exists."""
        pool_name = "fallback_test"
        pool = get_object_pool(pool_name, SimpleObjectNoReset, pooled_maxsize=1, prewarm_count=0)
        
        # Get an object and modify it
        obj = pool.borrow(name=f"{pool_name}Factory")
        obj.keeped_object.data["test"] = "value"
        
        # Return to pool (should trigger fallback reset)
        pool.recycle(obj, name=f"{pool_name}Factory")
        
        # Get the same object back
        obj2 = pool.borrow(name=f"{pool_name}Factory")
        
        # Verify fallback reset worked - all attributes should be cleared by __dict__.clear()
        assert obj2.keeped_object.__dict__ == {}, "All attributes should be cleared by fallback reset"
        assert obj is obj2, "Should be the same pooled object instance"
    
    def test_pool_reuses_objects(self):
        """Test that pool actually reuses objects instead of creating new ones."""
        pool_name = "reuse_test"
        pool = get_object_pool(pool_name, SimpleObject, pooled_maxsize=1, prewarm_count=0)
        
        # Get first object
        obj1 = pool.borrow(name=f"{pool_name}Factory")
        creation_id1 = obj1.keeped_object.creation_id
        
        # Return it
        pool.recycle(obj1, name=f"{pool_name}Factory")
        
        # Get second object
        obj2 = pool.borrow(name=f"{pool_name}Factory")
        creation_id2 = obj2.keeped_object.creation_id
        
        # Should be the same object (reused)
        assert creation_id1 == creation_id2, "Pool should reuse objects"
        assert obj1.keeped_object is obj2.keeped_object, "Should be same object instance"


if __name__ == "__main__":
    pytest.main([__file__])