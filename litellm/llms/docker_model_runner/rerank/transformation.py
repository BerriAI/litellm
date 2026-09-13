"""
Docker Model Runner Rerank API Configuration.

Docker Model Runner provides rerank via:
/rerank (works with both llama.cpp and vLLM backends)

Docs: https://docs.docker.com/ai/model-runner/api-reference/
"""

from collections.abc import Mapping
from typing import Final

import httpx

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankRequest,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class DockerModelRunnerRerankError(BaseLLMException):
    pass


class DockerModelRunnerRerankConfig(BaseRerankConfig):
    def get_supported_cohere_rerank_params(
        self, model: str
    ) -> list:  # mutable-ok: signature dictated by BaseRerankConfig
        return [  # mutable-ok: supported-params list, mirrors hosted_vllm
            "query",
            "documents",
            "top_n",
            "rank_fields",
            "return_documents",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict | None,  # mutable-ok: signature dictated by BaseRerankConfig
        model: str,
        drop_params: bool,
        query: str,
        documents: list[str | dict],  # mutable-ok: signature dictated by BaseRerankConfig
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: list[str] | None = None,  # mutable-ok: signature dictated by BaseRerankConfig
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> dict:  # mutable-ok: signature dictated by BaseRerankConfig
        if max_chunks_per_doc is not None:
            raise ValueError("Docker Model Runner does not support max_chunks_per_doc")

        mapped_params: Final = OptionalRerankParams(
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
        )

        return dict(mapped_params)  # mutable-ok: API request payload

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature dictated by BaseRerankConfig
        model: str,
        api_key: str | None = None,
        optional_params: dict | None = None,  # mutable-ok: signature dictated by BaseRerankConfig
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict:  # mutable-ok: signature dictated by BaseRerankConfig
        if api_key is None:
            api_key = (  # rebind-ok: normalize the argument locally, mirrors hosted_vllm
                get_secret_str("DOCKER_MODEL_RUNNER_API_KEY") or "dummy-key"
            )

        default_headers: Final = {  # mutable-ok: API request payload
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        if api_key and api_key != "dummy-key":
            default_headers["Authorization"] = f"Bearer {api_key}"

        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        return {**default_headers, **headers}  # mutable-ok: API request payload

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict | None = None,  # mutable-ok: signature dictated by BaseRerankConfig
    ) -> str:
        if api_base is None:
            api_base = (  # rebind-ok: normalize the argument locally, mirrors hosted_vllm
                get_secret_str("DOCKER_MODEL_RUNNER_API_BASE") or "http://localhost:12434"
            )

        api_base = api_base.rstrip("/")  # rebind-ok: normalize the argument locally, mirrors hosted_vllm

        # DMR serves rerank at /rerank or /engines/rerank; a /v1 suffix would be
        # parsed as a backend name and rejected with "backend not found"
        api_base = api_base.removesuffix("/v1")  # rebind-ok: normalize the argument locally, mirrors hosted_vllm

        if api_base.endswith("/rerank"):
            return api_base

        return f"{api_base}/rerank"

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict,  # mutable-ok: signature dictated by BaseRerankConfig
        headers: dict,  # mutable-ok: signature dictated by BaseRerankConfig
        litellm_params: dict | None = None,  # mutable-ok: signature dictated by BaseRerankConfig
    ) -> dict:  # mutable-ok: signature dictated by BaseRerankConfig
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Docker Model Runner rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Docker Model Runner rerank")

        rerank_request: Final = RerankRequest(
            model=model,
            query=optional_rerank_params["query"],
            documents=optional_rerank_params["documents"],
            top_n=optional_rerank_params.get("top_n"),
            rank_fields=optional_rerank_params.get("rank_fields"),
            return_documents=optional_rerank_params.get("return_documents"),
        )

        return rerank_request.model_dump(exclude_none=True)

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict | None = None,  # mutable-ok: signature dictated by BaseRerankConfig
        optional_params: dict | None = None,  # mutable-ok: signature dictated by BaseRerankConfig
        litellm_params: dict | None = None,  # mutable-ok: signature dictated by BaseRerankConfig
    ) -> RerankResponse:
        try:
            raw_response_json: Final = raw_response.json()
        except Exception as e:
            raise ValueError(
                f"Error parsing response: {raw_response.text}, status_code={raw_response.status_code}"
            ) from e

        return self._transform_response(raw_response_json)

    def _transform_response(self, response: dict) -> RerankResponse:  # mutable-ok: API response payload
        usage_data: Final = response.get("usage", {})  # mutable-ok: API response payload
        _billed_units: Final = RerankBilledUnits(total_tokens=usage_data.get("total_tokens", 0))
        _tokens: Final = RerankTokens(input_tokens=usage_data.get("total_tokens", 0))
        rerank_meta: Final = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: Final = response.get("results")
        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        def _build_result(result: dict) -> RerankResponseResult:  # mutable-ok: API response payload
            if "index" not in result or "relevance_score" not in result:
                raise ValueError(f"Missing required fields in the result={result}")

            document_data: Final = result.get("document")
            document = None  # rebind-ok: pre-init so the no-document branch has it defined
            if document_data:
                document = RerankResponseDocument(text=str(document_data.get("text", "")))  # rebind-ok: see pre-init

            rerank_result: Final = RerankResponseResult(
                index=int(result["index"]),
                relevance_score=float(result["relevance_score"]),
            )

            if document:
                rerank_result["document"] = document

            return rerank_result

        rerank_results: Final = [_build_result(result) for result in _results]  # mutable-ok: API response payload

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=rerank_results,
            meta=rerank_meta,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: signature dictated by BaseRerankConfig
    ) -> BaseLLMException:
        return DockerModelRunnerRerankError(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
