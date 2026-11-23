"""
Translates from Cohere's `/v1/rerank` input format to Bedrock's `/rerank` input format.

Why separate file? Make it easy to see how transformation works
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx
from pydantic import TypeAdapter

from litellm._uuid import uuid
from litellm.litellm_core_utils.aws_partition import get_aws_arn_prefix, is_bedrock_arn
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
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


class BedrockRerankConfig(BaseRerankConfig):
    @staticmethod
    def resolve_model_arn(model: str, aws_region_name: str | None = None) -> str:
        if is_bedrock_arn(model):
            return model
        clean_model: Final = model.split("/", 1)[1] if model.startswith("bedrock/") else model
        if is_bedrock_arn(clean_model):
            return clean_model
        if aws_region_name:
            arn_prefix: Final = get_aws_arn_prefix(aws_region_name)
            return f"{arn_prefix}bedrock:{aws_region_name}::foundation-model/{clean_model}"
        return clean_model

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        optional_params: dict | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict:
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict | None = None,
    ) -> str:
        # Bedrock builds and signs runtime endpoints dynamically in BedrockRerankHandler
        return api_base or ""

    def get_supported_cohere_rerank_params(self, model: str) -> list[str]:
        return [
            "query",
            "documents",
            "top_n",
            "return_documents",
            "rank_fields",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict,
        model: str,
        drop_params: bool,
        query: str,
        documents: list[str | dict[str, object]],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: list[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> dict:
        supported_params: Final = self.get_supported_cohere_rerank_params(model)
        filtered_non_default: Final = {
            key: value for key, value in non_default_params.items() if key in supported_params
        }
        return {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
            "rank_fields": rank_fields,
            **filtered_non_default,
        }

    def _transform_sources(self, documents: Sequence[str | dict]) -> list[BedrockRerankSource]:
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

    def _transform_request(
        self,
        request_data: RerankRequest,
        aws_region_name: str | None = None,
    ) -> BedrockRerankRequest:
        """
        Transform the request from RerankRequest format to Bedrock format.
        """
        _sources: Final = self._transform_sources(request_data.documents)
        resolved_arn: Final = self.resolve_model_arn(request_data.model, aws_region_name)

        return BedrockRerankRequest(
            queries=[
                BedrockRerankQuery(
                    textQuery=BedrockRerankTextQuery(text=request_data.query),
                    type="TEXT",
                )
            ],
            rerankingConfiguration=BedrockRerankConfiguration(
                bedrockRerankingConfiguration=BedrockRerankBedrockRerankingConfiguration(
                    modelConfiguration=BedrockRerankModelConfiguration(modelArn=resolved_arn),
                    numberOfResults=request_data.top_n or len(request_data.documents),
                ),
                type="BEDROCK_RERANKING_MODEL",
            ),
            sources=_sources,
        )

    def transform_request(
        self,
        request_data: RerankRequest,
        aws_region_name: str | None = None,
    ) -> BedrockRerankRequest:
        return self._transform_request(request_data, aws_region_name=aws_region_name)

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict,
        headers: dict,
        litellm_params: dict | None = None,
    ) -> dict:
        request_data: Final = RerankRequest(
            model=model,
            query=optional_rerank_params.get("query", ""),
            documents=optional_rerank_params.get("documents", []),
            top_n=optional_rerank_params.get("top_n"),
            rank_fields=optional_rerank_params.get("rank_fields"),
            return_documents=optional_rerank_params.get("return_documents", True),
        )
        return dict(self.transform_request(request_data))

    def _transform_response(self, response: dict) -> RerankResponse:
        """
        Transform the response from Bedrock into the RerankResponse format.

        example input:
        {"results":[{"index":0,"relevanceScore":0.6847912669181824},{"index":1,"relevanceScore":0.5980774760246277}]}
        """
        _billed_units: Final = TypeAdapter(RerankBilledUnits).validate_python(
            response.get("usage", {"search_units": 1})
        )
        _tokens: Final = RerankTokens(**response.get("usage", {}))
        rerank_meta: Final = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: list[RerankResponseResult] | None = None

        bedrock_results: Final = response.get("results")
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
        )

    def transform_response(
        self,
        response: dict[str, object],  # mutable-ok: mirrors override contract
    ) -> RerankResponse:
        return self._transform_response(response)

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: Any,
        api_key: str | None = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        return self.transform_response(raw_response.json())
