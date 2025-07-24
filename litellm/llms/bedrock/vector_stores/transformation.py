from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

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
        litellm_params: dict,
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
            # Create a properly typed retrieval configuration
            typed_retrieval_config: BedrockKBRetrievalConfiguration = {}
            if "vectorSearchConfiguration" in retrieval_config:
                typed_retrieval_config["vectorSearchConfiguration"] = retrieval_config["vectorSearchConfiguration"]
            request_body["retrievalConfiguration"] = typed_retrieval_config

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

    def _get_file_id_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Extract file_id from Bedrock KB metadata.
        Uses source URI if available, otherwise generates a fallback ID.
        """
        source_uri = metadata.get("x-amz-bedrock-kb-source-uri", "") if metadata else ""
        if source_uri:
            return source_uri
        
        chunk_id = metadata.get("x-amz-bedrock-kb-chunk-id", "unknown") if metadata else "unknown"
        return f"bedrock-kb-{chunk_id}"

    def _get_filename_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Extract filename from Bedrock KB metadata.
        Tries to extract filename from source URI, falls back to domain name or data source ID.
        """
        source_uri = metadata.get("x-amz-bedrock-kb-source-uri", "") if metadata else ""
        
        if source_uri:
            try:
                parsed_uri = urlparse(source_uri)
                filename = parsed_uri.path.split('/')[-1] if parsed_uri.path and parsed_uri.path != '/' else parsed_uri.netloc
                if not filename or filename == '/':
                    filename = parsed_uri.netloc
                return filename
            except Exception:
                return source_uri
        
        data_source_id = metadata.get("x-amz-bedrock-kb-data-source-id", "unknown") if metadata else "unknown"
        return f"bedrock-kb-document-{data_source_id}"

    def _get_attributes_from_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all attributes from Bedrock KB metadata.
        Returns a copy of the metadata dictionary.
        """
        if not metadata:
            return {}
        return dict(metadata)

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
                
                # Extract metadata and use helper functions
                metadata = item.get("metadata", {}) or {}
                file_id = self._get_file_id_from_metadata(metadata)
                filename = self._get_filename_from_metadata(metadata)
                attributes = self._get_attributes_from_metadata(metadata)
                
                results.append(
                    VectorStoreSearchResult(
                        score=item.get("score"),
                        content=[VectorStoreResultContent(text=text, type="text")],
                        file_id=file_id,
                        filename=filename,
                        attributes=attributes,
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
