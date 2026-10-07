"""
Transformation logic from Cohere's /v1/rerank format to Together AI's  `/v1/rerank` format.

Why separate file? Make it easy to see how transformation works
"""

from typing import Final

from pydantic import TypeAdapter

from litellm._uuid import uuid
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class TogetherAIRerankConfig:
    def _transform_response(self, response: dict[str, object]) -> RerankResponse:
        _billed_units: Final = TypeAdapter(RerankBilledUnits).validate_python(response.get("usage", {}))
        _tokens: Final = TypeAdapter(RerankTokens).validate_python(response.get("usage", {}))
        rerank_meta: Final = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        results_value: Final = response.get("results")
        _results: Final[list[dict[str, object]] | None] = (
            TypeAdapter(list[dict[str, object]]).validate_python(results_value) if results_value is not None else None
        )

        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        rerank_results: Final[list[RerankResponseResult]] = []

        for result in _results:
            # Validate required fields exist
            if not all(key in result for key in ["index", "relevance_score"]):
                raise ValueError(f"Missing required fields in the result={result}")

            # Get document data if it exists
            document_data = TypeAdapter(dict[str, object]).validate_python(result.get("document", {}))
            document = RerankResponseDocument(text=str(document_data.get("text", ""))) if document_data else None

            # Create typed result
            rerank_result = RerankResponseResult(
                index=TypeAdapter(int).validate_python(result["index"]),
                relevance_score=TypeAdapter(float).validate_python(result["relevance_score"]),
            )

            # Only add document if it exists
            if document:
                rerank_result["document"] = document

            rerank_results.append(rerank_result)

        response_id: Final = response.get("id")
        return RerankResponse(
            id=TypeAdapter(str | None).validate_python(response_id) or str(uuid.uuid4()),
            results=rerank_results,
            meta=rerank_meta,
        )

    def transform_response(
        self,
        response: dict[str, object],  # mutable-ok: preserves extension signature
    ) -> RerankResponse:
        return self._transform_response(response)
