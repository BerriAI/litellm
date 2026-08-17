from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)

DEFAULT_XINFERENCE_API_BASE: Final = "http://127.0.0.1:9997/v1"


class _XinferenceRerankResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    relevance_score: float
    document: str | None = None


class _XinferenceRerankResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str | None = None
    results: tuple[_XinferenceRerankResult, ...]


_XINFERENCE_RERANK_RESPONSE_ADAPTER: Final = TypeAdapter(_XinferenceRerankResponse)


class XinferenceRerankConfig(BaseRerankConfig):
    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object] | None = None,
    ) -> str:
        resolved_api_base: Final = api_base or get_secret_str("XINFERENCE_API_BASE") or DEFAULT_XINFERENCE_API_BASE
        cleaned_api_base: Final = resolved_api_base.rstrip("/")
        if cleaned_api_base.endswith("/rerank"):
            return cleaned_api_base
        return f"{cleaned_api_base}/rerank"

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        api_key: str | None = None,
        optional_params: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        resolved_api_key: Final = api_key or get_secret_str("XINFERENCE_API_KEY") or "stub-xinference-key"
        default_headers: Final = {
            "Authorization": f"Bearer {resolved_api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }
        return {**default_headers, **headers}

    def get_supported_cohere_rerank_params(self, model: str) -> list[str]:
        return ["query", "documents", "top_n"]

    def map_cohere_rerank_params(
        self,
        non_default_params: Mapping[str, object],
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
        params: Final[OptionalRerankParams] = OptionalRerankParams(
            query=query,
            documents=documents,
        )
        if top_n is not None:
            params["top_n"] = top_n
        return dict(params)

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Mapping[str, object],
        headers: Mapping[str, object],
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Xinference rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Xinference rerank")

        request: Final[dict[str, object]] = {
            "model": model,
            "query": optional_rerank_params["query"],
            "documents": optional_rerank_params["documents"],
        }
        if optional_rerank_params.get("top_n") is not None:
            request["top_n"] = optional_rerank_params["top_n"]
        return request

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: Mapping[str, object] | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> RerankResponse:
        try:
            response_json: Final = _XINFERENCE_RERANK_RESPONSE_ADAPTER.validate_python(raw_response.json())
        except ValueError:
            raise ValueError(f"Error parsing Xinference rerank response: {raw_response.text}")

        transformed_results: Final = tuple(
            RerankResponseResult(
                index=result.index,
                relevance_score=result.relevance_score,
                document=RerankResponseDocument(text=result.document),
            )
            if result.document is not None
            else RerankResponseResult(
                index=result.index,
                relevance_score=result.relevance_score,
            )
            for result in response_json.results
        )
        meta: Final = RerankResponseMeta(
            billed_units=RerankBilledUnits(total_tokens=0),
            tokens=RerankTokens(input_tokens=0),
        )

        return RerankResponse(
            id=response_json.id or str(uuid.uuid4()),
            results=list(transformed_results),
            meta=meta,
        )
