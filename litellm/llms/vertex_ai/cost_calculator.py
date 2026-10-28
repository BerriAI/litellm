# What is this?
## Cost calculation for Google AI Studio / Vertex AI models

from litellm.litellm_core_utils.llm_cost_calc.utils import cost_per_text
from litellm.types.utils import Usage

"""
Gemini pricing covers: 
- token
- image
- audio
- video
"""


def cost_per_token(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
    service_tier: str | None = None,
    vertex_location: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, either "vertex_ai-*" or "gemini"
        - prompt_tokens: float, the number of input tokens
        - completion_tokens: float, the number of output tokens
        - service_tier: optional tier derived from Gemini trafficType
          ("priority" for ON_DEMAND_PRIORITY, "flex" for FLEX/batch).
        - vertex_location: the Vertex AI location serving the request; non-global
          locations apply the model's regional-endpoint uplift multiplier

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """

    return cost_per_text(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
        service_tier=service_tier,
        vertex_location=vertex_location,
    )
