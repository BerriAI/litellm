from __future__ import annotations
from typing import Iterable, Optional

from litellm.caching.caching import RedisCache, RedisClusterCache
from litellm.router import Router


def configure_cache_redis(
    router: Router,
    *,
    url: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    password: Optional[str] = None,
    ttl: int = 3600,
    startup_nodes: Optional[Iterable[dict]] = None,
    cache_responses: bool = True,
) -> None:
    """Configure Redis/Cluster caching on a Router in one line."""
    cfg = {"url": url, "host": host, "port": port, "password": password, "ttl": ttl}
    cache = (
        RedisClusterCache(startup_nodes=list(startup_nodes), **cfg)  # type: ignore[arg-type]
        if startup_nodes
        else RedisCache(**cfg)
    )
    router.cache = router.cache or type("_Cache", (), {})()
    router.cache.redis_cache = cache
    router.cache_responses = cache_responses
