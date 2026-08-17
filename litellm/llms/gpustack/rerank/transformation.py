from typing import Final

import httpx
from pydantic import BaseModel, Field, TypeAdapter

from litellm._uuid import uuid
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.gpustack.common_utils import (
    get_gpustack_endpoint,
    get_gpustack_headers,
    strip_gpustack_model_prefix,
)
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class GPUStackRerankError(BaseLLMException):
    pass


class GPUStackRerankDocumentPayload(BaseModel):
    text: str | None = None


class GPUStackRerankResultPayload(BaseModel):
    index: int
    relevance_score: float
    document: GPUStackRerankDocumentPayload | None = None


class GPUStackRerankUsagePayload(BaseModel):
    total_tokens: int | None = None


class GPUStackRerankResponsePayload(BaseModel):
    id: str | None = None
    results: list[GPUStackRerankResultPayload]
    usage: GPUStackRerankUsagePayload = Field(default_factory=GPUStackRerankUsagePayload)


class GPUStackRerankConfig(BaseRerankConfig):
    def validate_environment(
        self,
        headers: dict[str, object],
        model: str,
        api_key: str | None = None,
        optional_params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return get_gpustack_headers(headers=headers, api_key=api_key, include_accept=True)

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict[str, object] | None = None,
    ) -> str:
        return get_gpustack_endpoint(api_base=api_base, endpoint="/rerank")

    def get_supported_cohere_rerank_params(self, model: str) -> list[str]:
        return ["query", "documents", "top_n", "return_documents"]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict[str, object],
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
    ) -> dict[str, object]:
        top_n_body: Final = {"top_n": top_n} if top_n is not None else {}
        return_documents_body: Final = {"return_documents": return_documents} if return_documents is not None else {}
        return {
            "query": query,
            "documents": documents,
            **top_n_body,
            **return_documents_body,
        }

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict[str, object],
        headers: dict[str, object],
        litellm_params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "model": strip_gpustack_model_prefix(model),
            "query": optional_rerank_params["query"],
            "documents": optional_rerank_params["documents"],
            **({"top_n": optional_rerank_params["top_n"]} if optional_rerank_params.get("top_n") is not None else {}),
            **(
                {"return_documents": optional_rerank_params["return_documents"]}
                if optional_rerank_params.get("return_documents") is not None
                else {}
            ),
        }

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: object,
        api_key: str | None = None,
        request_data: dict[str, object] = {},
        optional_params: dict[str, object] = {},
        litellm_params: dict[str, object] = {},
    ) -> RerankResponse:
        response_json: Final = TypeAdapter(GPUStackRerankResponsePayload).validate_json(raw_response.content)
        total_tokens: Final = response_json.usage.total_tokens or 0
        return RerankResponse(
            id=response_json.id or str(uuid.uuid4()),
            results=[
                RerankResponseResult(
                    index=result.index,
                    relevance_score=result.relevance_score,
                    **(
                        {"document": RerankResponseDocument(text=result.document.text)}
                        if result.document is not None and result.document.text is not None
                        else {}
                    ),
                )
                for result in response_json.results
            ],
            meta=RerankResponseMeta(
                billed_units=RerankBilledUnits(total_tokens=total_tokens),
                tokens=RerankTokens(input_tokens=total_tokens),
            ),
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,
    ) -> BaseLLMException:
        return GPUStackRerankError(message=error_message, status_code=status_code, headers=headers)
