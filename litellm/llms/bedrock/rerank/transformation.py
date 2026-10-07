"""
Translates from Cohere's `/v1/rerank` input format to Bedrock's `/rerank` input format.

Why separate file? Make it easy to see how transformation works
"""

from typing import Final

from pydantic import TypeAdapter

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

    def transform_request(self, request_data: RerankRequest) -> BedrockRerankRequest:
        return self._transform_request(request_data)

    def _transform_response(self, response: dict[str, object]) -> RerankResponse:
        """
        Transform the response from Bedrock into the RerankResponse format.

        example input:
        {"results":[{"index":0,"relevanceScore":0.6847912669181824},{"index":1,"relevanceScore":0.5980774760246277}]}
        """
        _billed_units: Final = TypeAdapter(RerankBilledUnits).validate_python(
            response.get("usage", {"search_units": 1})
        )
        _tokens: Final = TypeAdapter(RerankTokens).validate_python(response.get("usage", {}))
        rerank_meta: Final = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: list[RerankResponseResult] | None = None

        bedrock_results: Final = response.get("results")
        if bedrock_results:
            parsed_results: Final = TypeAdapter(list[dict[str, object]]).validate_python(bedrock_results)
            _results = [
                RerankResponseResult(
                    index=TypeAdapter(int).validate_python(result.get("index")),
                    relevance_score=TypeAdapter(float).validate_python(result.get("relevanceScore")),
                )
                for result in parsed_results
            ]

        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        response_id: Final = response.get("id")
        return RerankResponse(
            id=TypeAdapter(str | None).validate_python(response_id) or str(uuid.uuid4()),
            results=_results,
            meta=rerank_meta,
        )

    def transform_response(
        self,
        response: dict[str, object],  # mutable-ok: preserves extension signature
    ) -> RerankResponse:
        return self._transform_response(response)
