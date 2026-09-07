from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    return generic_cost_per_token(model=model, usage=usage, custom_llm_provider="tencent")
