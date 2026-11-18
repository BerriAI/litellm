# What is this?
## Cost calculation for Google AI Studio / Vertex AI models
import litellm
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import BranchFacts, Characters, branch_cost
from litellm.types.utils import Usage


def cost_per_character(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
    prompt_characters: float | None = None,
    completion_characters: float | None = None,
    service_tier: str | None = None,
    vertex_location: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per character for a given VertexAI model, input messages, and response object.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, "vertex_ai-*"
        - prompt_characters: float, the number of input characters
        - completion_characters: float, the number of output characters
        - service_tier: optional tier derived from Gemini trafficType
          ("priority" for ON_DEMAND_PRIORITY, "flex" for FLEX/batch).
        - vertex_location: the Vertex AI location serving the request; non-global
          locations apply the model's regional-endpoint uplift multiplier

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return branch_cost(
        model,
        custom_llm_provider,
        usage,
        litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider),
        service_tier=service_tier,
        vertex_location=vertex_location,
        branch=BranchFacts(
            characters=Characters(
                prompt=None if prompt_characters is None else int(prompt_characters),
                completion=None if completion_characters is None else int(completion_characters),
            )
        ),
    )


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
        - service_tier: optional tier derived from Gemini trafficType
          ("priority" for ON_DEMAND_PRIORITY, "flex" for FLEX/batch).
        - vertex_location: the Vertex AI location serving the request; non-global
          locations apply the model's regional-endpoint uplift multiplier

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return branch_cost(
        model,
        custom_llm_provider,
        usage,
        litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider),
        service_tier=service_tier,
        vertex_location=vertex_location,
    )
