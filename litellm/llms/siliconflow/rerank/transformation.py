from typing import Callable, TypedDict, cast

import httpx

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)
from litellm.types.utils import ModelInfo

from ..common_utils import (
    SiliconFlowException,
    get_dict,
    get_float,
    get_int,
    get_list,
    get_str,
)


class SiliconFlowRerankDocument(TypedDict, total=False):
    text: str


class SiliconFlowRerankResult(TypedDict, total=False):
    index: int
    relevance_score: float
    document: SiliconFlowRerankDocument | str


class SiliconFlowRerankBilledUnits(TypedDict, total=False):
    search_units: int
    total_tokens: int


class SiliconFlowRerankTokens(TypedDict, total=False):
    input_tokens: int
    output_tokens: int


class SiliconFlowRerankMeta(TypedDict, total=False):
    billed_units: SiliconFlowRerankBilledUnits
    tokens: SiliconFlowRerankTokens


class SiliconFlowRerankConfig(BaseRerankConfig):
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict[str, object] | None = None,
    ) -> str:
        return "{}{}".format(
            (api_base or get_secret_str("SILICONFLOW_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/"),
            "/rerank",
        )

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        api_key: str | None = None,
        optional_params: dict[str, object] | None = None,
    ) -> dict[str, str]:
        final_api_key = api_key or get_secret_str("SILICONFLOW_API_KEY")
        if final_api_key is None:
            raise ValueError("SILICONFLOW_API_KEY is not set")
        return {
            **headers,
            "Authorization": "Bearer {}".format(final_api_key),
            "accept": "application/json",
            "content-type": "application/json",
        }

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
        optional_rerank_params: dict[str, object] = {
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            optional_rerank_params["top_n"] = top_n
        if return_documents is not None:
            optional_rerank_params["return_documents"] = return_documents
        if max_chunks_per_doc is not None:
            optional_rerank_params["max_chunks_per_doc"] = max_chunks_per_doc
        if instruction is not None:
            optional_rerank_params["instruction"] = instruction

        for param_name in (
            "query",
            "documents",
            "top_n",
            "return_documents",
            "max_chunks_per_doc",
            "instruction",
            "overlap_tokens",
        ):
            if param_name in non_default_params and non_default_params[param_name] is not None:
                optional_rerank_params[param_name] = non_default_params[param_name]

        return optional_rerank_params

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict[str, object],
        headers: dict[str, str],
        litellm_params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return dict(optional_rerank_params)

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict[str, object] = {},
        optional_params: dict[str, object] = {},
        litellm_params: dict[str, object] = {},
    ) -> RerankResponse:
        try:
            response_json = get_dict(cast(object, raw_response.json()))
        except Exception:
            raise SiliconFlowException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        post_call = cast(Callable[..., None], logging_obj.post_call)
        post_call(original_response=raw_response.text)
        results: list[RerankResponseResult] = []
        for item in get_list(response_json.get("results")):
            result_item = get_dict(item)
            index = get_int(result_item.get("index"))
            relevance_score = get_float(result_item.get("relevance_score"))
            if index is None or relevance_score is None:
                continue
            result: RerankResponseResult = {
                "index": index,
                "relevance_score": relevance_score,
            }
            document = result_item.get("document")
            if isinstance(document, dict):
                document_dict = cast(dict[str, object], document)
                document_text = get_str(document_dict.get("text"))
                if document_text is not None:
                    result["document"] = RerankResponseDocument(text=document_text)
            elif isinstance(document, str):
                result["document"] = RerankResponseDocument(text=document)
            results.append(result)

        meta = cast(SiliconFlowRerankMeta, get_dict(response_json.get("meta")))
        tokens = get_dict(meta.get("tokens"))
        input_tokens = get_int(tokens.get("input_tokens")) or 0
        output_tokens = get_int(tokens.get("output_tokens")) or 0
        billed_units = get_dict(meta.get("billed_units"))
        total_tokens = get_int(billed_units.get("total_tokens")) or (input_tokens + output_tokens)
        search_units = get_int(billed_units.get("search_units"))

        rerank_meta: RerankResponseMeta = {
            "tokens": RerankTokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            "billed_units": RerankBilledUnits(
                search_units=search_units,
                total_tokens=total_tokens,
            ),
        }
        return RerankResponse(
            id=get_str(response_json.get("id")) or str(uuid.uuid4()),
            results=results,
            meta=rerank_meta,
        )

    def get_supported_cohere_rerank_params(self, model: str) -> list[str]:
        return [
            "query",
            "documents",
            "top_n",
            "return_documents",
            "max_chunks_per_doc",
            "instruction",
        ]

    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: str | None = None,
        billed_units: RerankBilledUnits | None = None,
        model_info: ModelInfo | None = None,
    ) -> tuple[float, float]:
        if (
            model_info is None
            or billed_units is None
        ):
            return 0.0, 0.0
        input_cost_per_token = get_float(model_info.get("input_cost_per_token"))
        if input_cost_per_token is None:
            return 0.0, 0.0
        total_tokens = billed_units.get("total_tokens")
        if total_tokens is None:
            return 0.0, 0.0
        return input_cost_per_token * total_tokens, 0.0

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> SiliconFlowException:
        return SiliconFlowException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
