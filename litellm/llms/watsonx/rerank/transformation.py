"""
Transformation logic for IBM watsonx.ai's /ml/v1/text/rerank endpoint.

Docs - https://cloud.ibm.com/apidocs/watsonx-ai#text-rerank
"""

import uuid
from typing import Any, Dict, List, Optional, Union, cast

import httpx

from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.watsonx import (
    WatsonXAIEndpoint,
)
from litellm.types.rerank import (
    RerankResponse,
    RerankResponseMeta,
    RerankTokens,
)

from ..common_utils import IBMWatsonXMixin, _generate_watsonx_token, _get_api_params


class IBMWatsonXRerankConfig(IBMWatsonXMixin, BaseRerankConfig):
    """
    IBM watsonx.ai Rerank API configuration
    """

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        base_url = self._get_base_url(api_base=api_base)
        endpoint = WatsonXAIEndpoint.RERANK.value

        url = base_url.rstrip("/") + endpoint

        params = optional_params or {}

        complete_url = self._add_api_version_to_url(url=url, api_version=(params.get("api_version", None)))
        return complete_url

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "return_documents",
            "max_tokens_per_doc",
        ]

    def validate_environment(  # type: ignore[override]
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> Dict:
        optional_params = optional_params or {}

        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if "Authorization" in headers:
            return {**default_headers, **headers}
        token = cast(
            Optional[str],
            optional_params.pop("token", None) or get_secret_str("WATSONX_TOKEN"),
        )
        zen_api_key = cast(
            Optional[str],
            optional_params.pop("zen_api_key", None) or get_secret_str("WATSONX_ZENAPIKEY"),
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif zen_api_key:
            headers["Authorization"] = f"ZenApiKey {zen_api_key}"
        else:
            token = _generate_watsonx_token(api_key=api_key, token=token)
            # build auth headers
            headers["Authorization"] = f"Bearer {token}"
        return {**default_headers, **headers}

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
        max_tokens_per_doc: Optional[int] = None,
    ) -> Dict:
        """
        Map Cohere rerank params to IBM watsonx.ai rerank params
        """
        optional_rerank_params = {}
        if non_default_params is not None:
            for k, v in non_default_params.items():
                if k == "query" and v is not None:
                    optional_rerank_params["query"] = v
                elif k == "documents" and v is not None:
                    optional_rerank_params["inputs"] = [
                        {"text": el} if isinstance(el, str) else el for el in v
                    ]
                elif k == "top_n" and v is not None:
                    optional_rerank_params.setdefault("parameters", {}).setdefault("return_options", {})["top_n"] = v
                elif k == "return_documents" and v is not None and isinstance(v, bool):
                    optional_rerank_params.setdefault("parameters", {}).setdefault("return_options", {})["inputs"] = v
                elif k == "max_tokens_per_doc" and v is not None:
                    optional_rerank_params.setdefault("parameters", {})["truncate_input_tokens"] = v

                # IBM watsonx.ai require one of below parameters
                elif k == "project_id" and v is not None:
                    optional_rerank_params["project_id"] = v
                elif k == "space_id" and v is not None:
                    optional_rerank_params["space_id"] = v

        return dict(optional_rerank_params)

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
    ) -> dict:
        """
        Transform request to IBM watsonx.ai rerank format
        """
        watsonx_api_params = _get_api_params(params=optional_rerank_params, model=model)
        watsonx_auth_payload = self._prepare_payload(
            model=model,
            api_params=watsonx_api_params,
        )

        return optional_rerank_params | watsonx_auth_payload

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        """
        Transform IBM watsonx.ai rerank response to LiteLLM RerankResponse format
        """
        try:
            raw_response_json = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse response: {str(e)}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        _results: Optional[List[dict]] = raw_response_json.get("results")
        if _results is None:
            raise ValueError(f"No results found in the response={raw_response_json}")

        transformed_results = []

        for result in _results:
            transformed_result: Dict[str, Any] = {
                "index": result["index"],
                "relevance_score": result["score"],
            }

            if "input" in result:
                if isinstance(result["input"], str):
                    transformed_result["document"] = {"text": result["input"]}
                else:
                    transformed_result["document"] = result["input"]

            transformed_results.append(transformed_result)

        response_id = raw_response_json.get("id") or raw_response_json.get("model_id") or str(uuid.uuid4())

        # Extract usage information
        _tokens = RerankTokens(
            input_tokens=raw_response_json.get("input_token_count", 0),
        )
        rerank_meta = RerankResponseMeta(tokens=_tokens)

        return RerankResponse(
            id=response_id,
            results=transformed_results,  # type: ignore
            meta=rerank_meta,
        )
