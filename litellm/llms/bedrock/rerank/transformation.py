"""
Translates from Cohere's `/v1/rerank` input format to Bedrock's `/rerank` input format.

Why separate file? Make it easy to see how transformation works
"""

from typing import Final

from litellm._uuid import uuid
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
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class BedrockRerankConfig:
    def _transform_sources(self, documents: list[str | dict]) -> list[BedrockRerankSource]:
        """
        Transform the sources from RerankRequest format to Bedrock format.
        """
        _sources: Final = []
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
                        inlineDocumentSource=BedrockRerankInlineDocumentSource(jsonDocument=document, type="JSON"),
                        type="INLINE",
                    )
                )
        return _sources

    def _transform_request(self, request_data: RerankRequest) -> BedrockRerankRequest:
        """
        Transform the request from RerankRequest format to Bedrock format.
        """
        _sources: Final = self._transform_sources(request_data.documents)

        return BedrockRerankRequest(
            queries=[
                BedrockRerankQuery(
                    textQuery=BedrockRerankTextQuery(text=request_data.query),
                    type="TEXT",
                )
            ],
            rerankingConfiguration=BedrockRerankConfiguration(
                bedrockRerankingConfiguration=BedrockRerankBedrockRerankingConfiguration(
                    modelConfiguration=BedrockRerankModelConfiguration(modelArn=request_data.model),
                    numberOfResults=request_data.top_n or len(request_data.documents),
                ),
                type="BEDROCK_RERANKING_MODEL",
            ),
            sources=_sources,
        )

    @staticmethod
    def _document_text(document: str | dict) -> str | None:
        """Text of an input document, in either shape the API accepts."""
        if isinstance(document, str):
            return document
        if isinstance(document, dict):
            text = document.get("text")
            if isinstance(text, str):
                return text
        return None

    def _transform_response(self, response: dict, request_data: RerankRequest | None = None) -> RerankResponse:
        """
        Transform the response from Bedrock into the RerankResponse format.

        example input:
        {"results":[{"index":0,"relevanceScore":0.6847912669181824},{"index":1,"relevanceScore":0.5980774760246277}]}

        Bedrock never echoes document text, so `document` is back-filled from
        the request's documents by index — the same thing the HuggingFace
        rerank config does for the identical provider limitation. Without
        `request_data` there is nothing to back-fill from, so the response is
        left as-is.
        """
        _billed_units = RerankBilledUnits(**response.get("usage", {"search_units": 1}))  # by default 1 search unit
        _tokens: Final = RerankTokens(**response.get("usage", {}))
        rerank_meta: Final = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: list[RerankResponseResult] | None = None

        bedrock_results: Final = response.get("results")
        if bedrock_results:
            # Cohere-compatible default: back-fill unless the caller opted out.
            should_return_documents: Final = request_data is not None and request_data.return_documents is not False
            original_documents: Final = request_data.documents if request_data is not None else []

            _results = []
            for result in bedrock_results:
                index = result.get("index")
                _result = RerankResponseResult(
                    index=index,
                    relevance_score=result.get("relevanceScore"),
                )
                if should_return_documents and isinstance(index, int) and 0 <= index < len(original_documents):
                    text = self._document_text(original_documents[index])
                    if text is not None:
                        _result["document"] = RerankResponseDocument(text=text)
                _results.append(_result)

        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=_results,
            meta=rerank_meta,
        )  # Return response
