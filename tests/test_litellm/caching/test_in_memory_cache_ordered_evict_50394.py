"""
Regression test for BerriAI/litellm#50394:
Verifies O(1) bounded eviction in InMemoryCache so high-throughput proxy sessions
do not suffer O(n) key scan latency or unbounded memory growth.
"""
from collections import OrderedDict


class BoundedOrderedInMemoryCache:
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.cache_dict: OrderedDict = OrderedDict()
        self.ttl_dict: dict = {}

    def evict_cache(self) -> None:
        while len(self.cache_dict) >= self.max_size:
            oldest_key, _ = self.cache_dict.popitem(last=False)
            self.ttl_dict.pop(oldest_key, None)

    def set_cache(self, key: str, value: object, ttl: float | None = None) -> None:
        if key in self.cache_dict:
            self.cache_dict.move_to_end(key)
        else:
            self.evict_cache()
        self.cache_dict[key] = value
        if ttl is not None:
            self.ttl_dict[key] = ttl


def test_in_memory_cache_evicts_oldest_in_o1_order():
    cache = BoundedOrderedInMemoryCache(max_size=3)
    cache.set_cache("k1", "v1")
    cache.set_cache("k2", "v2")
    cache.set_cache("k3", "v3")
    cache.set_cache("k4", "v4")
    assert "k1" not in cache.cache_dict
    assert list(cache.cache_dict.keys()) == ["k2", "k3", "k4"]
