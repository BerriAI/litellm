"""Re-exported from ``litellm.litellm_core_utils.ptu_pricing``.

The flag lives in core because the router reads it while registering a deployment, and
router code cannot import from the proxy.
"""

from litellm.litellm_core_utils.ptu_pricing import (
    PTU_COST_ATTRIBUTION_ENV_VAR,
    is_ptu_cost_attribution_enabled,
)

__all__ = ("PTU_COST_ATTRIBUTION_ENV_VAR", "is_ptu_cost_attribution_enabled")
