"""
Translates from Cohere's `/v1/rerank` input format to Bedrock's `/rerank` input format.

Why separate file? Make it easy to see how transformation works
"""

from collections.abc import Mapping, Sequence
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
    def _document_text(documents: Sequence[str | Mapping[str, object]] | None, index: object) -> str | None:
        """
        Text of the input document at `index`, in either shape the API accepts.

        Returns None when there is nothing to back-fill from: no documents, a
        non-integer or out-of-range index, or a document carrying no text.
        """
        if documents is None or not isinstance(index, int) or not 0 <= index < len(documents):
            return None
        document: Final = documents[index]
        if isinstance(document, str):
            return document
        if isinstance(document, dict):
            text: Final = document.get("text")
            if isinstance(text, str):
                return text
        return None

    @classmethod
    def _transform_result(
        cls, result: Mapping[str, object], documents: Sequence[str | Mapping[str, object]] | None
    ) -> RerankResponseResult:
        """One Bedrock result, with `document` back-filled when available."""
        index: Final = result.get("index")
        relevance_score: Final = result.get("relevanceScore")
        text: Final = cls._document_text(documents, index)
        if text is None:
            return RerankResponseResult(index=index, relevance_score=relevance_score)
        return RerankResponseResult(
            index=index,
            relevance_score=relevance_score,
            document=RerankResponseDocument(text=text),
        )

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
            # None documents means "nothing to back-fill from", which covers
            # both an opt-out and a caller that passed no request at all.
            documents: Final = (
                request_data.documents
                if request_data is not None and request_data.return_documents is not False
                else None
            )

            _results = [self._transform_result(result, documents) for result in bedrock_results]

        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=_results,
            meta=rerank_meta,
        )  # Return response
