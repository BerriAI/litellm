"""
Optional helpers to reduce app boilerplate. Import-safe; no core API changes.
"""
from .cache import configure_cache_redis  # noqa: F401

__all__ = [
    "configure_cache_redis",
]
