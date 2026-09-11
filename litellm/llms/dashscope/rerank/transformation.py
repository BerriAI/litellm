"""
Transformation logic for DashScope's OpenAI-compatible /v1/reranks API.

Supports
- qwen3-rerank

(Other DashScope rerankers — gte-rerank-v2 / qwen3-vl-rerank — have not been
validated against this transformer. Behavior with those models is undefined.)

The native qwen3.7-text-rerank protocol is implemented in native_transformation.py.

Endpoint
- https://dashscope.aliyuncs.com/compatible-api/v1/reranks

Note: chat/embed live under `/compatible-mode/v1/`, but qwen3-rerank's
route is exposed under `/compatible-api/v1/reranks` per the docs. A chat-shaped
`.aliyuncs.com/compatible-mode/v1` base reaching this config (the chat default
from `get_llm_provider`, or a `DASHSCOPE_API_BASE` env var) is redirected to
the same host's rerank route, since `/compatible-mode/v1/reranks` is a dead
route on every DashScope host. Override with `DASHSCOPE_API_BASE_RERANK` to
point at a different host or path.

Empirically, qwen3-rerank accepts `return_documents=true` and echoes
`results[].document.text` back, even though the public docs list the flag
as supported only for gte-rerank-v2 / qwen3-vl-rerank.

Docs - https://help.aliyun.com/zh/model-studio/text-rerank-api
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankTokens,
)

from ..common_utils import DashScopeError, resolve_dashscope_family_rerank_api_base

DEFAULT_RERANK_URL: Final = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"


class DashScopeRerankUsage(TypedDict, total=False):
    prompt_tokens: ReadOnly[int | None]
    total_tokens: ReadOnly[int | None]


class DashScopeRerankConfig(BaseRerankConfig):
    """
    Reference: https://help.aliyun.com/zh/model-studio/text-rerank-api

    Targets DashScope's qwen3-rerank model. Request fields: model, query,
    documents, top_n, return_documents. Response: results[].index,
    results[].relevance_score, optionally results[].document.text (when
    return_documents=true), plus a top-level usage.total_tokens counter.
    """

    def __init__(self) -> None:
        pass

    def _resolve_api_key(self, api_key: str | None) -> str:
        resolved_api_key: Final = api_key if api_key is not None else get_secret_str("DASHSCOPE_API_KEY")
        if resolved_api_key is None:
            raise ValueError(
                "DashScope API key is required. Set 'DASHSCOPE_API_KEY' env var or pass api_key explicitly."
            )
        return resolved_api_key

    def _resolve_rerank_api_base(self, api_base: str | None) -> str:
        return resolve_dashscope_family_rerank_api_base(api_base, "DASHSCOPE_API_BASE_RERANK", DEFAULT_RERANK_URL)

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object] | None = None,
    ) -> str:
        resolved_api_base: Final = self._resolve_rerank_api_base(api_base)
        if resolved_api_base == DEFAULT_RERANK_URL:
            return resolved_api_base

        cleaned: Final = resolved_api_base.rstrip("/")
        if cleaned.endswith("/reranks") or cleaned.endswith("/rerank"):
            return cleaned

        if cleaned.endswith("/v1"):
            return f"{cleaned}/reranks"

        # Unknown base: append /reranks rather than silently ignoring the caller's api_base.
        return f"{cleaned}/reranks"

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        api_key: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "Authorization": f"Bearer {self._resolve_api_key(api_key)}",
            "accept": "application/json",
            "content-type": "application/json",
            **headers,
        }

    def get_supported_cohere_rerank_params(self, model: str) -> list[str]:
        return ["query", "documents", "top_n", "return_documents"]

    def map_cohere_rerank_params(
        self,
        non_default_params: Mapping[str, object] | None,
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
        params: Final = MappingProxyType(
            {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": return_documents,
                "instruction": instruction,
            }
        )
        supported_params: Final = self.get_supported_cohere_rerank_params(model)
        return {name: value for name, value in params.items() if value is not None and name in supported_params}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Mapping[str, object],
        headers: Mapping[str, object],
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for DashScope rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for DashScope rerank")

        request: Final[dict[str, object]] = {
            "model": model,
            "query": optional_rerank_params["query"],
            "documents": optional_rerank_params["documents"],
        }
        if optional_rerank_params.get("top_n") is not None:
            request["top_n"] = optional_rerank_params["top_n"]
        if optional_rerank_params.get("return_documents") is not None:
            request["return_documents"] = optional_rerank_params["return_documents"]
        return request

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> RerankResponse:
        request: Final = request_data or MappingProxyType({})
        try:
            response_json: Final = TypeAdapter(Mapping[str, object]).validate_json(raw_response.content)
        except ValidationError as exc:
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=raw_response.text,
            ) from exc

        logging_obj.post_call(
            input=self._get_request_query(request),
            api_key=api_key,
            additional_args={"complete_input_dict": request},
            original_response=response_json,
        )

        # DashScope error envelope: {"code": "...", "message": "...", "request_id": "..."}
        if "code" in response_json and "results" not in response_json:
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=str(response_json.get("message", response_json)),
            )

        usage: Final = TypeAdapter(DashScopeRerankUsage).validate_python(
            response_json.get("usage") or MappingProxyType({})
        )
        results, response_id, input_tokens = self._get_response_fields(response_json, usage)
        if results is None:
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=f"No results in DashScope rerank response: {response_json}",
            )

        # Both protocols return:
        #   {"index": int, "relevance_score": float}
        # plus, when return_documents=true was sent:
        #   "document": {"text": "..."}
        # which already matches LiteLLM's RerankResponseDocument shape.
        transformed_results: Final[list[dict]] = []
        for r in TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(results):
            item: dict[str, object] = {
                "index": r["index"],
                "relevance_score": r["relevance_score"],
            }
            doc = r.get("document")
            if isinstance(doc, dict):
                item["document"] = doc
            elif isinstance(doc, str):
                # Defensive: spec says dict, but normalize string-shaped echoes.
                item["document"] = {"text": doc}
            transformed_results.append(item)

        billed_units: Final = RerankBilledUnits(total_tokens=usage.get("total_tokens"))
        tokens: Final = RerankTokens(input_tokens=input_tokens)
        meta: Final = RerankResponseMeta(billed_units=billed_units, tokens=tokens)

        return RerankResponse.model_validate(
            MappingProxyType(
                {
                    "id": response_id or str(uuid.uuid4()),
                    "results": transformed_results,
                    "meta": meta,
                }
            )
        )

    def _get_request_query(self, request_data: Mapping[str, object]) -> object:
        return request_data.get("query")

    def _get_response_fields(
        self, response_json: Mapping[str, object], usage: DashScopeRerankUsage
    ) -> tuple[object, object, int | None]:
        return response_json.get("results"), response_json.get("id"), usage.get("total_tokens")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,
    ) -> BaseLLMException:
        if isinstance(headers, dict):
            headers = httpx.Headers(headers)
        return DashScopeError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
