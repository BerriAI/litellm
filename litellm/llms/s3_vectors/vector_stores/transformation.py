from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.base_llm.vector_store.transformation import (
    BaseQueryEmbeddingVectorStoreConfig,
    VectorStoreEmbeddingExecutor,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VECTOR_STORE_OPENAI_PARAMS,
    BaseVectorStoreAuthCredentials,
    VectorStoreIndexEndpoints,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_DEFAULT_QUERY_EMBEDDING_MODEL: Final = "text-embedding-3-small"
_DEFAULT_TOP_K: Final = 5


class S3VectorsVectorStoreConfig(BaseQueryEmbeddingVectorStoreConfig, BaseAWSLLM):
    """Vector store configuration for AWS S3 Vectors."""

    def __init__(self) -> None:
        BaseQueryEmbeddingVectorStoreConfig.__init__(self)
        BaseAWSLLM.__init__(self)

    def get_auth_credentials(self, litellm_params: dict) -> BaseVectorStoreAuthCredentials:
        return {}

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [("POST", "/QueryVectors")],
            "write": [],
        }

    def get_supported_openai_params(self, model: str) -> list[VECTOR_STORE_OPENAI_PARAMS]:
        return ["max_num_results"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_num_results":
                optional_params["maxResults"] = value
        return optional_params

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        headers = headers or {}
        headers.setdefault("Content-Type", "application/json")
        return headers

    def get_complete_url(self, api_base: str | None, litellm_params: dict) -> str:
        aws_region_name: Final = self.get_aws_region_name_for_non_llm_api_calls(litellm_params.get("aws_region_name"))
        return f"https://s3vectors.{aws_region_name}.api.aws"

    @staticmethod
    def query_embedding_model(litellm_params: Mapping[str, object]) -> str:
        configured: Final = litellm_params.get("litellm_embedding_model") or litellm_params.get("embedding_model")
        return configured if isinstance(configured, str) and configured else _DEFAULT_QUERY_EMBEDDING_MODEL

    @staticmethod
    def _query_target(vector_store_id: str, litellm_params: Mapping[str, object]) -> tuple[str, str]:
        if ":" in vector_store_id:
            bucket_name, index_name = vector_store_id.split(":", 1)
            return bucket_name, index_name
        bucket_name_from_params: Final = litellm_params.get("vector_bucket_name")
        if not isinstance(bucket_name_from_params, str) or not bucket_name_from_params:
            raise ValueError(
                "vector_store_id must be in format 'bucket_name:index_name' for S3 Vectors, "
                "or vector_bucket_name must be provided in litellm_params"
            )
        return bucket_name_from_params, vector_store_id

    @staticmethod
    def _query_request(
        bucket_name: str,
        index_name: str,
        query_text: str,
        query_vector: Sequence[float],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> tuple[str, dict[str, object]]:
        litellm_logging_obj.model_call_details["query"] = query_text
        return f"{api_base}/QueryVectors", {
            "vectorBucketName": bucket_name,
            "indexName": index_name,
            "queryVector": {"float32": list(query_vector)},
            "topK": vector_store_search_optional_params.get("max_num_results", _DEFAULT_TOP_K),
            "returnDistance": True,
            "returnMetadata": True,
        }

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:
        bucket_name, index_name = self._query_target(vector_store_id, litellm_params)
        query_text: Final = self.query_text(query)
        query_vector: Final = self.embed_query(query_text, litellm_params, embedding_executor)
        return self._query_request(
            bucket_name,
            index_name,
            query_text,
            query_vector,
            vector_store_search_optional_params,
            api_base,
            litellm_logging_obj,
        )

    async def atransform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:
        bucket_name, index_name = self._query_target(vector_store_id, litellm_params)
        query_text: Final = self.query_text(query)
        query_vector: Final = await self.aembed_query(query_text, litellm_params, embedding_executor)
        return self._query_request(
            bucket_name,
            index_name,
            query_text,
            query_vector,
            vector_store_search_optional_params,
            api_base,
            litellm_logging_obj,
        )

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: str | None = None,
    ) -> tuple[dict, bytes | None]:
        return self._sign_request(
            service_name="s3vectors",
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
            response_data: Final = response.json()
            results: Final[list[VectorStoreSearchResult]] = []

            for item in response_data.get("vectors", []) or []:
                metadata = item.get("metadata", {}) or {}
                source_text = metadata.get("source_text", "")

                if not source_text:
                    continue

                chunk_index = metadata.get("chunk_index", "0")
                file_id = f"s3-vectors-chunk-{chunk_index}"
                filename = metadata.get("filename", f"document-{chunk_index}")

                distance = item.get("distance")
                score = None
                if distance is not None:
                    score = max(0.0, min(1.0, 1.0 - float(distance)))

                results.append(
                    VectorStoreSearchResult(
                        score=score,
                        content=[VectorStoreResultContent(text=source_text, type="text")],
                        file_id=file_id,
                        filename=filename,
                        attributes=metadata,
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

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params,
        api_base: str,
    ) -> tuple[str, dict]:
        raise NotImplementedError

    def transform_create_vector_store_response(self, response: httpx.Response):
        raise NotImplementedError
