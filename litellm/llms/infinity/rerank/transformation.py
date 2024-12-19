"""
Transformation logic from Cohere's /v1/rerank format to Infinity's  `/v1/rerank` format. 

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


class InfinityRerankConfig:
    def _transform_response(self, response: dict) -> RerankResponse:

        _billed_units = RerankBilledUnits(**response.get("usage", {}))
        _tokens = RerankTokens(
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=(response.get("usage", {}).get("total_tokens", 0) - response.get("usage", {}).get("prompt_tokens", 0))
        )
        rerank_meta = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: Optional[List[dict]] = response.get("results")

        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=_results,  # type: ignore
            meta=rerank_meta,
        )  # Return response
