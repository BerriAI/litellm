"""
Transformation logic from Cohere's /v1/rerank format to Together AI's  `/v1/rerank` format. 

Why separate file? Make it easy to see how transformation works
"""

import uuid
from typing import List, Optional

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
            # Validate required fields exist
            if not all(key in result for key in ["index", "relevance_score"]):
                raise ValueError(f"Missing required fields in the result={result}")

            # Get document data if it exists
            document_data = result.get("document", {})
            document = (
                RerankResponseDocument(text=str(document_data.get("text", "")))
                if document_data
                else None
            )

            # Create typed result
            rerank_result = RerankResponseResult(
                index=int(result["index"]),
                relevance_score=float(result["relevance_score"]),
            )

            # Only add document if it exists
            if document:
                rerank_result["document"] = document

            rerank_results.append(rerank_result)

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=rerank_results,
            meta=rerank_meta,
        )  # Return response
