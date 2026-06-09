"""Governor v2: budget, rate-limit, and quota enforcement.

Inert until ``LITELLM_GOVERNOR_V2`` is truthy. This top-level module re-exports
only the pure config gate and the SDK-free runtime entrypoints; the engine,
policies, and cache tiers are reached via their submodule paths so importing the
package never pulls in redis or prisma.
"""

from litellm.integrations.governor.model.config import (
    GOVERNOR_V2_ENV,
    GovernorV2Config,
    is_governor_v2_enabled,
)
from litellm.integrations.governor.runtime import get_engine, is_enabled, set_engine

__all__ = [
    "GOVERNOR_V2_ENV",
    "GovernorV2Config",
    "is_governor_v2_enabled",
    "get_engine",
    "is_enabled",
    "set_engine",
]
