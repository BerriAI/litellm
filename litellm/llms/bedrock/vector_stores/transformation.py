from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.types.integrations.rag.bedrock_knowledgebase import (
    BedrockKBContent,
    BedrockKBResponse,
    BedrockKBRetrievalConfiguration,
    BedrockKBRetrievalQuery,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BedrockVectorStoreConfig(BaseVectorStoreConfig, BaseAWSLLM):
    """Vector store configuration for AWS Bedrock Knowledge Bases."""

    def __init__(self) -> None:
        BaseVectorStoreConfig.__init__(self)
        BaseAWSLLM.__init__(self)

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        headers = headers or {}
        headers.setdefault("Content-Type", "application/json")
        return headers

    def get_complete_url(
        self, api_base: Optional[str], litellm_params: dict
    ) -> str:
        aws_region_name = litellm_params.get("aws_region_name")
        endpoint_url, _ = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=litellm_params.get("aws_bedrock_runtime_endpoint"),
            aws_region_name=self.get_aws_region_name_for_non_llm_api_calls(
                aws_region_name=aws_region_name
            ),
            endpoint_type="agent",
        )
        return f"{endpoint_url}/knowledgebases"

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> Tuple[str, Dict]:
        if isinstance(query, list):
            query = " ".join(query)

        url = f"{api_base}/{vector_store_id}/retrieve"

        request_body: Dict[str, Any] = {
            "retrievalQuery": BedrockKBRetrievalQuery(text=query),
        }

        retrieval_config: Dict[str, Any] = {}
        max_results = vector_store_search_optional_params.get("max_num_results")
        if max_results is not None:
            retrieval_config.setdefault("vectorSearchConfiguration", {})[
                "numberOfResults"
            ] = max_results
        filters = vector_store_search_optional_params.get("filters")
        if filters is not None:
            retrieval_config.setdefault("vectorSearchConfiguration", {})[
                "filter"
            ] = filters
        if retrieval_config:
            request_body["retrievalConfiguration"] = BedrockKBRetrievalConfiguration(
                **retrieval_config
            )

        litellm_logging_obj.model_call_details["query"] = query
        return url, request_body

    def sign_request(
        self,
        headers: dict,
        optional_params: Dict,
        request_data: Dict,
        api_base: str,
        api_key: Optional[str] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
        )

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        try:
            response_data = BedrockKBResponse(**response.json())
            results: List[VectorStoreSearchResult] = []
            for item in response_data.get("retrievalResults", []) or []:
                content: Optional[BedrockKBContent] = item.get("content")
                text = content.get("text") if content else None
                if text is None:
                    continue
                results.append(
                    VectorStoreSearchResult(
                        score=item.get("score"),
                        content=[VectorStoreResultContent(text=text, type="text")],
                    )
                )
            return VectorStoreSearchResponse(
                object="vector_store.search_results.page",
                search_query=litellm_logging_obj.model_call_details.get("query", ""),
                data=results,
            )
        except Exception as e:
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers,
            )

    # Vector store creation is not yet implemented
    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params,
        api_base: str,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError

    def transform_create_vector_store_response(self, response: httpx.Response):
        raise NotImplementedError
