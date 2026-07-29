import json
from typing import Union

import httpx

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str, normalize_nonempty_secret_str
from litellm.types.rerank import OptionalRerankParams, RerankResponse

from ..chat.transformation import CloudflareError


class CloudflareRerankConfig(BaseRerankConfig):
    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseRerankConfig protocol requires dict
        model: str,
        api_key: str | None = None,
        optional_params: dict | None = None,  # mutable-ok: BaseRerankConfig protocol requires dict
    ) -> dict:  # mutable-ok: BaseRerankConfig protocol requires dict
        if api_key is None:
            api_key = get_secret_str("CLOUDFLARE_API_KEY")
        if api_key is None:
            raise ValueError("Missing Cloudflare API Key - set CLOUDFLARE_API_KEY or pass api_key explicitly")
        return {  # mutable-ok: HTTP handler requires mutable headers
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "content-type": "application/json",
            **headers,
        }

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict | None = None,  # mutable-ok: BaseRerankConfig protocol requires dict
    ) -> str:
        if api_base is None:
            account_id = normalize_nonempty_secret_str(get_secret_str("CLOUDFLARE_ACCOUNT_ID"))
            if account_id is None:
                raise ValueError(
                    "Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID or pass api_base explicitly"
                )
            api_base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"

        cleaned = api_base.rstrip("/")
        if cleaned.endswith(f"/{model}"):
            return cleaned
        if cleaned.endswith("/ai/v1"):
            cleaned = f"{cleaned[: -len('/ai/v1')]}/ai/run"
        elif not cleaned.endswith("/ai/run"):
            cleaned = f"{cleaned}/ai/run"
        return f"{cleaned}/{model}"

    def get_supported_cohere_rerank_params(
        self, model: str
    ) -> list:  # mutable-ok: BaseRerankConfig protocol requires list
        return [  # mutable-ok: rerank parameter mapper requires a list
            "query",
            "documents",
            "top_n",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict,  # mutable-ok: BaseRerankConfig protocol requires dict
        model: str,
        drop_params: bool,
        query: str,
        documents: list[  # mutable-ok: BaseRerankConfig protocol requires list
            Union[
                str,
                dict[str, object],  # mutable-ok: public rerank documents accept dictionaries
            ]
        ],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: list[str]  # mutable-ok: BaseRerankConfig protocol requires list
        | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> dict:  # mutable-ok: BaseRerankConfig protocol requires dict
        params = OptionalRerankParams(query=query, documents=documents)
        if top_n is not None:
            params["top_n"] = top_n
        return dict(params)  # mutable-ok: HTTP handler requires a mutable request

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict,  # mutable-ok: BaseRerankConfig protocol requires dict
        headers: dict,  # mutable-ok: BaseRerankConfig protocol requires dict
        litellm_params: dict | None = None,  # mutable-ok: BaseRerankConfig protocol requires dict
    ) -> dict:  # mutable-ok: BaseRerankConfig protocol requires dict
        query = optional_rerank_params.get("query")
        documents = optional_rerank_params.get("documents")
        if not isinstance(query, str) or not query:
            raise ValueError("query is required for Cloudflare rerank")
        if not isinstance(documents, list) or not documents:
            raise ValueError("documents is required for Cloudflare rerank")

        contexts = []  # mutable-ok: request contexts are assembled from documents
        for document in documents:
            if isinstance(document, str):
                text = document
            elif isinstance(document, dict):
                text = document.get("text")
                if not isinstance(text, str):
                    text = json.dumps(document)
            else:
                raise ValueError("Cloudflare rerank documents must be strings or dictionaries")
            contexts.append(
                {"text": text}  # mutable-ok: Cloudflare API requires JSON objects
            )

        request: dict[  # mutable-ok: request gains optional top_k before dispatch
            str, object
        ] = {  # mutable-ok: Cloudflare API requires a JSON object
            "query": query,
            "contexts": contexts,
        }
        if optional_rerank_params.get("top_n") is not None:
            request["top_k"] = optional_rerank_params["top_n"]
        return request

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict | None = None,  # mutable-ok: BaseRerankConfig protocol requires dict
        optional_params: dict | None = None,  # mutable-ok: BaseRerankConfig protocol requires dict
        litellm_params: dict | None = None,  # mutable-ok: BaseRerankConfig protocol requires dict
    ) -> RerankResponse:
        request_data = (
            request_data or {}  # mutable-ok: logging expects a concrete request dictionary
        )
        try:
            response_json = raw_response.json()
        except ValueError:
            raise CloudflareError(
                status_code=raw_response.status_code,
                message=raw_response.text,
            )

        logging_obj.post_call(
            input=request_data.get("query"),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},  # mutable-ok: logging API requires a dictionary
            original_response=response_json,
        )

        if response_json.get("success") is False:
            raise CloudflareError(
                status_code=raw_response.status_code,
                message=str(response_json.get("errors") or response_json),
            )

        result = response_json.get("result", response_json)
        response = result.get("response") if isinstance(result, dict) else None
        if not isinstance(response, list):
            raise CloudflareError(
                status_code=raw_response.status_code,
                message=f"No response in Cloudflare rerank result: {response_json}",
            )

        transformed_results = []  # mutable-ok: response results are assembled from provider output
        for item in response:
            transformed_item = {  # mutable-ok: RerankResponse requires result dictionaries
                "index": item["id"],
                "relevance_score": item["score"],
            }
            transformed_results.append(transformed_item)

        return RerankResponse(
            id=response_json.get("id") or str(uuid.uuid4()),
            results=transformed_results,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],  # mutable-ok: BaseRerankConfig protocol requires dict
    ) -> BaseLLMException:
        return CloudflareError(status_code=status_code, message=error_message)
