"""
Generic object pooling utilities for LiteLLM.

This module provides a flexible object pooling system that can be used
to pool any type of object, reducing memory allocation overhead and
improving performance for frequently created/destroyed objects.

Memory Management Strategy:
- Balanced eviction-based memory control to optimize reuse ratio
- Moderate eviction frequency (300s) to maintain high object reuse
- Conservative eviction weight (0.3) to avoid destroying useful objects
- Lower pre-warm count (5) to reduce initial memory footprint
- Always keeps at least one object available for high availability
- Unlimited pools when maxsize is not specified (eviction controls actual usage)
"""

from typing import Any, Callable, Optional, Type, TypeVar

from pond import Pond, PooledObject, PooledObjectFactory

T = TypeVar('T')

class GenericPooledObjectFactory(PooledObjectFactory):
    """Generic factory class for creating pooled objects of any type."""
    
    def __init__(
        self, 
        object_class: Type[T], 
        pooled_maxsize: Optional[int] = None,  # None = unlimited pool with eviction-based memory control
        least_one: bool = True,  # Always keep at least one for high concurrency
        initializer: Optional[Callable[[T], None]] = None
    ):
        # Only pass maxsize to Pond if user specified it - otherwise let Pond handle unlimited pools
        if pooled_maxsize is not None:
            super().__init__(pooled_maxsize=pooled_maxsize, least_one=least_one)
        else:
            super().__init__(least_one=least_one)
        self.object_class = object_class
        self.initializer = initializer
        self._user_maxsize = pooled_maxsize  # Store original user preference
    
    def createInstance(self) -> PooledObject:
        """Create a new instance wrapped in a PooledObject."""
        # Create a properly initialized instance
        obj = self.object_class()
        return PooledObject(obj)
    
    def destroy(self, pooled_object: PooledObject):
        """Destroy the pooled object."""
        if hasattr(pooled_object.keeped_object, '__dict__'):
            pooled_object.keeped_object.__dict__.clear()
        del pooled_object
    
    def reset(self, pooled_object: PooledObject, **kwargs: Any) -> PooledObject:
        """Reset the pooled object to a clean state."""
        obj = pooled_object.keeped_object
        # Reset the object by calling its reset method if it exists
        if hasattr(obj, 'reset') and callable(getattr(obj, 'reset')):
            obj.reset()
        else:
            # Fallback: clear all attributes to reset the object
            if hasattr(obj, '__dict__'):
                obj.__dict__.clear()
        return pooled_object
    
    def validate(self, pooled_object: PooledObject) -> bool:
        """Validate if the pooled object is still usable."""
        return pooled_object.keeped_object is not None

# Global pond instances
_pools: dict[str, Pond] = {}

def get_object_pool(
    pool_name: str,
    object_class: Type[T],
    pooled_maxsize: Optional[int] = None,  # None = unlimited pool with eviction-based memory control
    least_one: bool = True,  # Always keep at least one
    borrowed_timeout: int = 10,  # Longer timeout for high concurrency
    time_between_eviction_runs: int = 300,  # Less frequent eviction to maintain high reuse ratio
    eviction_weight: float = 0.3,  # Less aggressive eviction for better reuse
    prewarm_count: int = 5  # Lower pre-warm count to reduce initial memory usage
) -> Pond:
    """Get or create a global object pool instance with balanced eviction-based memory control.
    
    Memory is controlled through moderate eviction to balance reuse ratio and memory usage:
    - Moderate eviction frequency (300s) to maintain high object reuse ratio
    - Conservative eviction weight (0.3) to avoid destroying useful objects
    - Lower pre-warm count (5) to reduce initial memory footprint
    
    Args:
        pool_name: Unique name for the pool
        object_class: The class type to pool
        pooled_maxsize: Maximum number of objects in the pool (None = truly unlimited)
        least_one: Whether to keep at least one object in the pool (default: True)
        borrowed_timeout: Timeout for borrowing objects (seconds, default: 10)
        time_between_eviction_runs: Time between eviction runs (seconds, default: 300)
        eviction_weight: Weight for eviction algorithm (default: 0.3, conservative)
        prewarm_count: Number of objects to pre-warm the pool with (default: 5)
    
    Returns:
        Pond instance for the specified object type
    """
    
    if pool_name in _pools:
        return _pools[pool_name]
    
    # Create new pond
    pond = Pond(
        borrowed_timeout=borrowed_timeout,
        time_between_eviction_runs=time_between_eviction_runs,
        thread_daemon=True,
        eviction_weight=eviction_weight
    )
    
    # Register the factory with user's maxsize preference
    factory = GenericPooledObjectFactory(
        object_class=object_class,
        pooled_maxsize=pooled_maxsize,
        least_one=least_one
    )
    pond.register(factory, name=f"{pool_name}Factory")
    
    # Pre-warm the pool
    _prewarm_pool(pond, pool_name, prewarm_count)
    
    _pools[pool_name] = pond
    return pond

def _prewarm_pool(pond: Pond, pool_name: str, prewarm_count: int = 20) -> None:
    """Pre-warm the pool with initial objects for high concurrency."""
    for _ in range(prewarm_count):
        try:
            pooled_obj = pond.borrow(name=f"{pool_name}Factory")
            pond.recycle(pooled_obj, name=f"{pool_name}Factory")
        except Exception:
            # If pre-warming fails, just continue
            break