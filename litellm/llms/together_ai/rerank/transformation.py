"""
Transformation logic from Cohere's /v1/rerank format to Together AI's  `/v1/rerank` format. 

Why separate file? Make it easy to see how transformation works
"""

import uuid
from typing import List, Optional

from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
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

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=_results,  # type: ignore
            meta=rerank_meta,
        )  # Return response
