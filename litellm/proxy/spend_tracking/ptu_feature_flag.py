"""Opt-in flag for PTU (provisioned throughput unit) flat-cost attribution.

The whole feature is inert unless an operator sets
``LITELLM_ENABLE_PTU_COST_ATTRIBUTION``: the daily rollup is not scheduled, the
model endpoints reject PTU config, the daily activity read path reports zero flat
cost, and the model form hides the PTU inputs.
"""

from typing import Final

from litellm.secret_managers.main import get_secret_bool

PTU_COST_ATTRIBUTION_ENV_VAR: Final = "LITELLM_ENABLE_PTU_COST_ATTRIBUTION"


def is_ptu_cost_attribution_enabled() -> bool:
    """Report whether this deployment opted into PTU flat-cost attribution."""
    return get_secret_bool(PTU_COST_ATTRIBUTION_ENV_VAR, False) is True
