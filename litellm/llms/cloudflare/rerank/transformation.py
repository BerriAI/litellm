import json
from collections.abc import Mapping, Sequence
from typing import Union

import httpx
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import encode_url_path_segments
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str, normalize_nonempty_secret_str
from litellm.types.rerank import RerankResponse, RerankResponseResult

from ..chat.transformation import CloudflareError


class CloudflareRerankContext(TypedDict):
    text: ReadOnly[str]


class CloudflareRerankRequest(TypedDict):
    query: ReadOnly[str]
    contexts: ReadOnly[Sequence[Mapping[str, str]]]
    top_k: NotRequired[ReadOnly[object]]


class CohereRerankParams(TypedDict):
    query: ReadOnly[str]
    documents: ReadOnly[Sequence[Union[str, Mapping[str, object]]]]
    top_n: NotRequired[ReadOnly[int]]
    return_documents: NotRequired[ReadOnly[bool]]


class CloudflareHeaders(TypedDict):
    Authorization: str
    accept: str


class LoggingAdditionalArgs(TypedDict):
    complete_input_dict: ReadOnly[Mapping[str, object]]


class EmptyRequestData(TypedDict, total=False):
    pass


EMPTY_REQUEST_DATA: Mapping[str, object] = EmptyRequestData()


class CloudflareRerankConfig(BaseRerankConfig):
    @staticmethod
    def _default_api_base() -> str:
        account_id = normalize_nonempty_secret_str(get_secret_str("CLOUDFLARE_ACCOUNT_ID"))
        if account_id is None:
            raise ValueError("Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID or pass api_base explicitly")
        return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"

    @staticmethod
    def _document_to_context(
        document: object,
    ) -> Mapping[str, str]:
        if isinstance(document, str):
            return CloudflareRerankContext(text=document)
        if not isinstance(document, Mapping):
            raise TypeError("Cloudflare rerank documents must be strings or dictionaries")

        text = document.get("text")
        return CloudflareRerankContext(text=text if isinstance(text, str) else json.dumps(document))

    @staticmethod
    def _response_items(
        response_json: Mapping[str, object],
        status_code: int,
    ) -> tuple[object, ...]:
        if response_json.get("success") is False:
            raise CloudflareError(
                status_code=status_code,
                message=str(response_json.get("errors") or response_json),
            )

        result = response_json.get("result", response_json)
        response = result.get("response") if isinstance(result, Mapping) else None
        if not isinstance(response, list):
            raise CloudflareError(
                status_code=status_code,
                message=f"No response in Cloudflare rerank result: {response_json}",
            )
        return tuple(response)

    @staticmethod
    def _transform_response_item(
        item: object,
        documents: Sequence[object],
        return_documents: bool,
    ) -> RerankResponseResult:
        if not isinstance(item, Mapping):
            raise TypeError("Invalid item in Cloudflare rerank response")

        index = item.get("id")
        score = item.get("score")
        if not isinstance(index, int) or not isinstance(score, (int, float)):
            raise TypeError("Invalid item in Cloudflare rerank response")

        if return_documents and index < len(documents):
            document = CloudflareRerankConfig._document_to_context(documents[index])
            return RerankResponseResult(
                index=index,
                relevance_score=float(score),
                document=document,
            )
        return RerankResponseResult(index=index, relevance_score=float(score))

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        api_key: str | None = None,
        optional_params: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        api_key = api_key or get_secret_str("CLOUDFLARE_API_KEY")
        if api_key is None:
            raise ValueError("Missing Cloudflare API Key - set CLOUDFLARE_API_KEY or pass api_key explicitly")
        cloudflare_headers = CloudflareHeaders(
            Authorization=f"Bearer {api_key}",
            accept="application/json",
        )
        cloudflare_headers["content-type"] = "application/json"
        cloudflare_headers.update(headers)
        return cloudflare_headers

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object] | None = None,
    ) -> str:
        cleaned = (api_base or self._default_api_base()).rstrip("/")
        encoded_model = encode_url_path_segments(model, field_name="model")
        if cleaned.endswith(f"/{encoded_model}"):
            return cleaned
        if cleaned.endswith("/ai/v1"):
            return f"{cleaned[: -len('/ai/v1')]}/ai/run/{encoded_model}"
        if cleaned.endswith("/ai/run"):
            return f"{cleaned}/{encoded_model}"
        return f"{cleaned}/ai/run/{encoded_model}"

    def get_supported_cohere_rerank_params(
        self,
        model: str,
    ) -> Sequence[str]:
        return (
            "query",
            "documents",
            "top_n",
            "return_documents",
        )

    def map_cohere_rerank_params(
        self,
        non_default_params: Mapping[str, object],
        model: str,
        drop_params: bool,
        query: str,
        documents: Sequence[
            Union[
                str,
                Mapping[str, object],
            ]
        ],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: Sequence[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> Mapping[str, object]:
        if top_n is None and return_documents is None:
            return CohereRerankParams(
                query=query,
                documents=documents,
            )
        if top_n is None:
            return CohereRerankParams(
                query=query,
                documents=documents,
                return_documents=return_documents,
            )
        if return_documents is None:
            return CohereRerankParams(
                query=query,
                documents=documents,
                top_n=top_n,
            )
        return CohereRerankParams(
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
        )

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Mapping[str, object],
        headers: Mapping[str, object],
        litellm_params: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        query = optional_rerank_params.get("query")
        documents = optional_rerank_params.get("documents")
        if not isinstance(query, str) or not query:
            raise ValueError("query is required for Cloudflare rerank")
        if not isinstance(documents, Sequence) or isinstance(documents, str) or not documents:
            raise ValueError("documents is required for Cloudflare rerank")

        contexts = tuple(self._document_to_context(document) for document in documents)
        request = CloudflareRerankRequest(query=query, contexts=contexts)
        top_n = optional_rerank_params.get("top_n")
        if top_n is None:
            return request
        return CloudflareRerankRequest(**request, top_k=top_n)

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
        request_data = request_data or EMPTY_REQUEST_DATA
        try:
            response_json: object = raw_response.json()
        except ValueError:
            raise CloudflareError(
                status_code=raw_response.status_code,
                message=raw_response.text,
            )
        if not isinstance(response_json, Mapping):
            raise CloudflareError(
                status_code=raw_response.status_code,
                message=f"Invalid Cloudflare rerank response: {response_json}",
            )

        logging_obj.post_call(
            input=request_data.get("query"),
            api_key=api_key,
            additional_args=LoggingAdditionalArgs(complete_input_dict=request_data),
            original_response=response_json,
        )

        optional_params = optional_params or EMPTY_REQUEST_DATA
        documents = optional_params.get("documents")
        if not isinstance(documents, Sequence) or isinstance(documents, str):
            documents = ()
        return_documents = optional_params.get("return_documents") is not False
        results = tuple(
            self._transform_response_item(
                item,
                documents=documents,
                return_documents=return_documents,
            )
            for item in self._response_items(
                response_json=response_json,
                status_code=raw_response.status_code,
            )
        )
        return RerankResponse(
            id=str(response_json.get("id") or uuid.uuid4()),
            results=results,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[Mapping[str, object], httpx.Headers],
    ) -> BaseLLMException:
        return CloudflareError(status_code=status_code, message=error_message)
