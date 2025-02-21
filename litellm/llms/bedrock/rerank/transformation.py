"""
Translates from Cohere's `/v1/rerank` input format to Bedrock's `/rerank` input format.

Why separate file? Make it easy to see how transformation works
"""

import uuid
from typing import List, Optional, Union

from litellm.types.llms.bedrock import (
    BedrockRerankBedrockRerankingConfiguration,
    BedrockRerankConfiguration,
    BedrockRerankInlineDocumentSource,
    BedrockRerankModelConfiguration,
    BedrockRerankQuery,
    BedrockRerankRequest,
    BedrockRerankSource,
    BedrockRerankTextDocument,
    BedrockRerankTextQuery,
)
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankRequest,
    RerankResponse,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class BedrockRerankConfig:
    def count_total_queries(
        self,
        query_tokens: int,
        document_tokens: list[int],
        cost_per_1000_queries: float = 1.0,
    ) -> float:
        """
        Calculate the cost of a request based on token counts and document chunks.

        Args:
            query_tokens (int): Number of tokens in the query
            document_tokens (list[int]): List of token counts for each document
            cost_per_1000_queries (float): Cost per 1000 queries in dollars (default: $1.00)

        Returns:
            float: Total cost in dollars
        """
        TOKENS_PER_DOCUMENT = 512
        CHUNKS_PER_QUERY = 100

        # Validate query length
        if query_tokens >= TOKENS_PER_DOCUMENT:
            raise ValueError("Query tokens exceed maximum allowed tokens per document")

        # Calculate total chunks needed
        total_chunks = 0
        available_tokens = TOKENS_PER_DOCUMENT - query_tokens

        for doc_tokens in document_tokens:
            # Calculate chunks needed for this document
            chunks_needed = (doc_tokens + available_tokens - 1) // available_tokens
            total_chunks += max(1, chunks_needed)

        # Calculate total queries needed (rounded up to nearest multiple of CHUNKS_PER_QUERY)
        total_queries = (total_chunks + CHUNKS_PER_QUERY - 1) // CHUNKS_PER_QUERY
        return total_queries

    def _transform_sources(
        self, documents: List[Union[str, dict]]
    ) -> List[BedrockRerankSource]:
        """
        Transform the sources from RerankRequest format to Bedrock format.
        """
        _sources = []
        for document in documents:
            if isinstance(document, str):
                _sources.append(
                    BedrockRerankSource(
                        inlineDocumentSource=BedrockRerankInlineDocumentSource(
                            textDocument=BedrockRerankTextDocument(text=document),
                            type="TEXT",
                        ),
                        type="INLINE",
                    )
                )
            else:
                _sources.append(
                    BedrockRerankSource(
                        inlineDocumentSource=BedrockRerankInlineDocumentSource(
                            jsonDocument=document, type="JSON"
                        ),
                        type="INLINE",
                    )
                )
        return _sources

    def _transform_request(self, request_data: RerankRequest) -> BedrockRerankRequest:
        """
        Transform the request from RerankRequest format to Bedrock format.
        """
        _sources = self._transform_sources(request_data.documents)

        return BedrockRerankRequest(
            queries=[
                BedrockRerankQuery(
                    textQuery=BedrockRerankTextQuery(text=request_data.query),
                    type="TEXT",
                )
            ],
            rerankingConfiguration=BedrockRerankConfiguration(
                bedrockRerankingConfiguration=BedrockRerankBedrockRerankingConfiguration(
                    modelConfiguration=BedrockRerankModelConfiguration(
                        modelArn=request_data.model
                    ),
                    numberOfResults=request_data.top_n or len(request_data.documents),
                ),
                type="BEDROCK_RERANKING_MODEL",
            ),
            sources=_sources,
        )

    def _transform_response(self, response: dict) -> RerankResponse:
        """
        Transform the response from Bedrock into the RerankResponse format.

        example input:
        {"results":[{"index":0,"relevanceScore":0.6847912669181824},{"index":1,"relevanceScore":0.5980774760246277}]}
        """
        _billed_units = RerankBilledUnits(
            **response.get("usage", {"search_units": 1})
        )  # by default 1 search unit
        _tokens = RerankTokens(**response.get("usage", {}))
        rerank_meta = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: Optional[List[RerankResponseResult]] = None

        bedrock_results = response.get("results")
        if bedrock_results:
            _results = [
                RerankResponseResult(
                    index=result.get("index"),
                    relevance_score=result.get("relevanceScore"),
                )
                for result in bedrock_results
            ]

        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=_results,
            meta=rerank_meta,
        )  # Return response
