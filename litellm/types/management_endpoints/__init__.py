"""
Types for management endpoints
"""

from .cache_settings_endpoints import (
    CACHE_SETTINGS_FIELDS,
    REDIS_TYPE_DESCRIPTIONS,
    CacheSettingsField,
)
from .coordination_redis_endpoints import (
    COORDINATION_REDIS_SETTINGS_FIELDS,
    CoordinationRedisSection,
    CoordinationRedisSettingsField,
    CoordinationRedisSource,
)
from .router_settings_endpoints import (
    ROUTER_SETTINGS_FIELDS,
    ROUTING_STRATEGY_DESCRIPTIONS,
    RouterSettingsField,
)

__all__ = [
    "ROUTER_SETTINGS_FIELDS",
    "ROUTING_STRATEGY_DESCRIPTIONS",
    "RouterSettingsField",
    "CACHE_SETTINGS_FIELDS",
    "REDIS_TYPE_DESCRIPTIONS",
    "CacheSettingsField",
    "COORDINATION_REDIS_SETTINGS_FIELDS",
    "CoordinationRedisSection",
    "CoordinationRedisSettingsField",
    "CoordinationRedisSource",
]
