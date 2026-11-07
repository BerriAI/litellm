import json
from collections.abc import Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # json boundaries only expose untyped mappings

import httpx
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import encode_url_path_segments
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str, normalize_nonempty_secret_str
from litellm.types.rerank import RerankResponse, RerankResponseDocument, RerankResponseResult

from ..chat.transformation import CloudflareError


class CloudflareRerankContext(TypedDict):
    text: ReadOnly[str]


class CloudflareRerankRequest(TypedDict):
    query: ReadOnly[str]
    contexts: ReadOnly[Sequence[CloudflareRerankContext]]
    top_k: NotRequired[ReadOnly[object]]


class CohereRerankParams(TypedDict):
    query: ReadOnly[str]
    documents: ReadOnly[Sequence[str | Mapping[str, object]]]
    top_n: NotRequired[ReadOnly[int]]
    return_documents: NotRequired[ReadOnly[bool]]


class LoggingAdditionalArgs(TypedDict):
    complete_input_dict: ReadOnly[Mapping[str, object]]


class EmptyRequestData(TypedDict, total=False):
    pass


EMPTY_REQUEST_DATA: Final[Mapping[str, object]] = EmptyRequestData()


class CloudflareRerankConfig(BaseRerankConfig):
    @staticmethod
    def _default_api_base() -> str:
        account_id: Final = normalize_nonempty_secret_str(get_secret_str("CLOUDFLARE_ACCOUNT_ID"))
        if account_id is None:
            raise ValueError("Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID or pass api_base explicitly")
        return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"

    @staticmethod
    def _document_to_context(
        document: object,
    ) -> CloudflareRerankContext:
        if isinstance(document, str):
            return CloudflareRerankContext(text=document)
        if not isinstance(document, Mapping):
            raise TypeError("Cloudflare rerank documents must be strings or dictionaries")

        document_mapping: Final = cast(Mapping[str, object], document)  # cast-ok: isinstance proves only a bare Mapping
        text: Final = document_mapping.get("text")
        return CloudflareRerankContext(text=text if isinstance(text, str) else json.dumps(document_mapping))

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

        result: Final = response_json.get("result", response_json)
        result_mapping: Final = (
            cast(Mapping[str, object], result)  # cast-ok: isinstance proves only a bare Mapping
            if isinstance(result, Mapping)
            else None
        )
        response: Final = result_mapping.get("response") if result_mapping is not None else None
        if not isinstance(response, list):
            raise CloudflareError(
                status_code=status_code,
                message=f"No response in Cloudflare rerank result: {response_json}",
            )
        return tuple(cast(Sequence[object], response))  # cast-ok: JSON list narrows only to list[Unknown]

    @staticmethod
    def _transform_response_item(
        item: object,
        documents: Sequence[object],
        return_documents: bool,
    ) -> RerankResponseResult:
        if not isinstance(item, Mapping):
            raise TypeError("Invalid item in Cloudflare rerank response")

        item_mapping: Final = cast(Mapping[str, object], item)  # cast-ok: isinstance proves only a bare Mapping
        index: Final = item_mapping.get("id")
        score: Final = item_mapping.get("score")
        if not isinstance(index, int) or not isinstance(score, (int, float)):
            raise TypeError("Invalid item in Cloudflare rerank response")

        if return_documents and index < len(documents):
            document: Final = CloudflareRerankConfig._document_to_context(documents[index])
            return RerankResponseResult(
                index=index,
                relevance_score=float(score),
                document=RerankResponseDocument(text=document["text"]),
            )
        return RerankResponseResult(index=index, relevance_score=float(score))

    def validate_environment(  # pyright: ignore[reportIncompatibleMethodOverride]  # base annotates the return as dict; the read-only Mapping is intentional
        self,
        headers: Mapping[str, object],
        model: str,
        api_key: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        resolved_api_key: Final = api_key or get_secret_str("CLOUDFLARE_API_KEY")
        if resolved_api_key is None:
            raise ValueError("Missing Cloudflare API Key - set CLOUDFLARE_API_KEY or pass api_key explicitly")
        default_headers: Final = {
            "Authorization": f"Bearer {resolved_api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }
        return {**default_headers, **headers}

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object] | None = None,
    ) -> str:
        cleaned: Final = (api_base or self._default_api_base()).rstrip("/")
        encoded_model: Final = encode_url_path_segments(model, field_name="model")
        if cleaned.endswith(f"/{encoded_model}"):
            return cleaned
        if cleaned.endswith("/ai/v1"):
            return f"{cleaned[: -len('/ai/v1')]}/ai/run/{encoded_model}"
        if cleaned.endswith("/ai/run"):
            return f"{cleaned}/{encoded_model}"
        return f"{cleaned}/ai/run/{encoded_model}"

    def get_supported_cohere_rerank_params(  # pyright: ignore[reportIncompatibleMethodOverride]  # base annotates the return as list; a read-only tuple is intentional
        self,
        model: str,
    ) -> Sequence[str]:
        return (
            "query",
            "documents",
            "top_n",
            "return_documents",
        )

    def map_cohere_rerank_params(  # pyright: ignore[reportIncompatibleMethodOverride]  # base annotates the return as dict; the read-only Mapping is intentional
        self,
        non_default_params: Mapping[str, object],
        model: str,
        drop_params: bool,
        query: str,
        documents: Sequence[str | Mapping[str, object]],
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
            assert return_documents is not None
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

    def transform_rerank_request(  # pyright: ignore[reportIncompatibleMethodOverride]  # base annotates the return as dict; the read-only Mapping is intentional
        self,
        model: str,
        optional_rerank_params: Mapping[str, object],
        headers: Mapping[str, object],
        litellm_params: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        query: Final = optional_rerank_params.get("query")
        raw_documents: Final = optional_rerank_params.get("documents")
        if not isinstance(query, str) or not query:
            raise ValueError("query is required for Cloudflare rerank")
        if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, str) or not raw_documents:
            raise ValueError("documents is required for Cloudflare rerank")

        documents: Final = cast(Sequence[object], raw_documents)  # cast-ok: isinstance proves only a bare Sequence
        contexts: Final = tuple(self._document_to_context(document) for document in documents)
        top_n: Final = optional_rerank_params.get("top_n")
        if top_n is None:
            return CloudflareRerankRequest(query=query, contexts=contexts)
        return CloudflareRerankRequest(query=query, contexts=contexts, top_k=top_n)

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
        resolved_request_data: Final = request_data or EMPTY_REQUEST_DATA
        try:
            response_json: Final = cast(object, raw_response.json())  # cast-ok: httpx Response.json is untyped
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
        response_data: Final = cast(Mapping[str, object], response_json)  # cast-ok: bare Mapping

        logging_obj.post_call(  # pyright: ignore[reportUnknownMemberType]  # logging object is legacy-untyped
            input=resolved_request_data.get("query"),
            api_key=api_key,
            additional_args=LoggingAdditionalArgs(complete_input_dict=resolved_request_data),
            original_response=response_data,
        )

        resolved_optional_params: Final = optional_params or EMPTY_REQUEST_DATA
        raw_documents: Final = resolved_optional_params.get("documents")
        documents: Final = (
            cast(Sequence[object], raw_documents)  # cast-ok: isinstance proves only a bare Sequence
            if isinstance(raw_documents, Sequence) and not isinstance(raw_documents, str)
            else ()
        )
        return_documents: Final = resolved_optional_params.get("return_documents") is not False
        results: Final = tuple(
            self._transform_response_item(
                item,
                documents=documents,
                return_documents=return_documents,
            )
            for item in self._response_items(
                response_json=response_data,
                status_code=raw_response.status_code,
            )
        )
        return RerankResponse(
            id=str(response_data.get("id") or uuid.uuid4()),
            results=list(results),
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, object] | httpx.Headers,
    ) -> BaseLLMException:
        return CloudflareError(status_code=status_code, message=error_message)
