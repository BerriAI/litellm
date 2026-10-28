from litellm.litellm_core_utils.llm_cost_calc.utils import cost_per_text
from litellm.types.utils import Usage


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    return cost_per_text(model=model, usage=usage, custom_llm_provider="tencent")
