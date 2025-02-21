"""
Transformation logic from Cohere's /v1/rerank format to Together AI's  `/v1/rerank` format. 

Why separate file? Make it easy to see how transformation works
"""

import uuid
from typing import List, Optional, cast

from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class TogetherAIRerankConfig:
    def _transform_response(self, response: dict) -> RerankResponse:

        _billed_units = RerankBilledUnits(**response.get("usage", {}))
        _tokens = RerankTokens(**response.get("usage", {}))
        rerank_meta = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: Optional[List[dict]] = response.get("results")

        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        rerank_results: List[RerankResponseResult] = []
        for result in _results:
            initial_result = {}
            if result.get("index"):
                initial_result["index"] = cast(int, result.get("index"))
            if result.get("relevance_score"):
                initial_result["relevance_score"] = cast(
                    float, result.get("relevance_score")
                )
            if result.get("document"):
                initial_result["document"] = RerankResponseDocument(
                    text=cast(str, result.get("document", {}).get("text"))
                )
            initial_result_typed = RerankResponseResult(**initial_result)
            rerank_results.append(initial_result_typed)

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=rerank_results,
            meta=rerank_meta,
        )  # Return response
