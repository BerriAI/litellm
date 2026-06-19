"""
Helper util for handling bedrock-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Optional, Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

if TYPE_CHECKING:
        from litellm.types.utils import Usage

# AWS charges a ~10% surcharge for cross-region inference profiles (us./eu./ap. prefixes)
_CROSS_REGION_INFERENCE_SURCHARGE = 1.1


def _is_cross_region_inference_model(model: str) -> bool:
        """Return True if *model* uses a Bedrock cross-region inference prefix.

            Cross-region inference profile IDs begin with a geographic abbreviation
                followed by a dot, e.g. ``us.anthropic.claude-sonnet-4-6``.  AWS bills
                    these at a ~10 % premium over the equivalent base-model price.
                        """
        from litellm.llms.bedrock.common_utils import (
            get_bedrock_cross_region_inference_regions,
    )

    stripped = model
    for prefix in ("bedrock/", "converse/"):
                if stripped.startswith(prefix):
                                stripped = stripped[len(prefix):]
                                break

            potential_region = stripped.split(".", 1)[0]
    return potential_region in get_bedrock_cross_region_inference_regions()


def cost_per_token(
        model: str, usage: "Usage", service_tier: Optional[str] = None
) -> Tuple[float, float]:
        """
            Calculates the cost per token for a given model, prompt tokens, and completion tokens.

                Follows the same logic as Anthropic's cost per token calculation.

                    For cross-region inference profiles (us./eu./ap. prefixes), applies the AWS
                        10 % surcharge on top of the base-model token prices.
                            """
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider="bedrock",
            service_tier=service_tier,
        )

    if _is_cross_region_inference_model(model):
                prompt_cost *= _CROSS_REGION_INFERENCE_SURCHARGE
                completion_cost *= _CROSS_REGION_INFERENCE_SURCHARGE

    return prompt_cost, completion_cost
