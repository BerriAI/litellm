from typing import Final

from litellm.llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig
from litellm.types.rerank import RerankBilledUnits
from litellm.types.utils import ModelInfo


class BitdeerRerankConfig(HostedVLLMRerankConfig):
    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: str | None = None,
        billed_units: RerankBilledUnits | None = None,
        model_info: ModelInfo | None = None,
    ) -> tuple[float, float]:
        if model_info is None or billed_units is None:
            return 0.0, 0.0
        cost_per_token: Final = model_info.get("input_cost_per_token")
        total_tokens: Final = billed_units.get("total_tokens")
        if cost_per_token is None or total_tokens is None:
            return 0.0, 0.0
        return cost_per_token * total_tokens, 0.0
