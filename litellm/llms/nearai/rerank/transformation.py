from litellm.llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankTokens,
)


class NearAIRerankConfig(HostedVLLMRerankConfig):
    def _transform_response(self, response: dict) -> RerankResponse:
        transformed_response = super()._transform_response(response)
        total_tokens = (response.get("usage") or {}).get("total_tokens", 0)
        return RerankResponse(
            id=transformed_response.id,
            results=transformed_response.results,
            meta=RerankResponseMeta(
                billed_units=RerankBilledUnits(search_units=1, total_tokens=total_tokens),
                tokens=RerankTokens(input_tokens=total_tokens),
            ),
        )
