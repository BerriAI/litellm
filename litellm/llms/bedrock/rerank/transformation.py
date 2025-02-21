"""
Translates from Cohere's `/v1/rerank` input format to Bedrock's `/rerank` input format.

Why separate file? Make it easy to see how transformation works
"""

import uuid
from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.cohere.rerank.transformation import CohereRerankConfig
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
    OptionalRerankParams,
    RerankBilledUnits,
    RerankRequest,
    RerankResponse,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)

from ..common_utils import BedrockError


class BedrockRerankConfig(CohereRerankConfig):
    def get_supported_cohere_rerank_params(self, model: str) -> List:
        if "cohere" in model.lower():
            return super().get_supported_cohere_rerank_params(model)
        else:  # amazon model supports restricted params
            return ["query", "documents", "top_n"]

    def map_cohere_rerank_params(
        self,
        non_default_params: Optional[dict],
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
    ) -> OptionalRerankParams:
        supported_params = self.get_supported_cohere_rerank_params(model)
        optional_rerank_params = {}
        if non_default_params:
            for k, v in non_default_params.items():
                if k in supported_params:
                    optional_rerank_params[k] = v
        return OptionalRerankParams(**optional_rerank_params)

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

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message)
