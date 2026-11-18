"""
Helper util for handling bedrock-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING

import litellm
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import BranchFacts, Elapsed, branch_cost

if TYPE_CHECKING:
    from litellm.types.utils import Usage


def cost_per_token(
    model: str, usage: "Usage", service_tier: str | None = None, elapsed: Elapsed | None = None
) -> tuple[float, float]:
    """
    Prompt and completion cost of ``usage`` on a Bedrock model, with the geo uplift its row carries. An unreadable geo
    value raises, as on main. ``elapsed`` prices a row billed by the second.
    """
    return branch_cost(
        model,
        "bedrock",
        usage,
        litellm.get_model_info(model=model, custom_llm_provider="bedrock"),
        service_tier=service_tier,
        branch=BranchFacts(elapsed=elapsed),
    )
