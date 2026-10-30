from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import urlparse

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.url_utils import encode_url_path_segment, join_api_base
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.integrations.rag.bedrock_knowledgebase import (
    BedrockKBContent,
    BedrockKBResponse,
    BedrockKBRetrievalConfiguration,
    BedrockKBRetrievalQuery,
    BedrockKBUserContext,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VECTOR_STORE_OPENAI_PARAMS,
    BaseVectorStoreAuthCredentials,
    OpenAIComparisonFilter,
    OpenAICompoundFilter,
    OpenAIVectorStoreFilter,
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

_AWS_COMPARISON_OPERATORS: Final = MappingProxyType(
    {
        "eq": "equals",
        "ne": "notEquals",
        "gt": "greaterThan",
        "gte": "greaterThanOrEquals",
        "lt": "lessThan",
        "lte": "lessThanOrEquals",
        "in": "in",
        "nin": "notIn",
    }
)
_AWS_LOGICAL_OPERATORS: Final = MappingProxyType({"and": "andAll", "or": "orAll"})
_OPENAI_FILTER: Final[TypeAdapter[OpenAIComparisonFilter | OpenAICompoundFilter]] = TypeAdapter(OpenAIVectorStoreFilter)


def _aws_retrieval_filter(openai_filter: OpenAIComparisonFilter | OpenAICompoundFilter) -> Mapping[str, object]:
    if isinstance(openai_filter, OpenAIComparisonFilter):
        value: Final = list(openai_filter.value) if isinstance(openai_filter.value, tuple) else openai_filter.value
        return {_AWS_COMPARISON_OPERATORS[openai_filter.type]: {"key": openai_filter.key, "value": value}}
    if len(openai_filter.filters) == 1:
        return _aws_retrieval_filter(openai_filter.filters[0])
    return {
        _AWS_LOGICAL_OPERATORS[openai_filter.type]: [_aws_retrieval_filter(member) for member in openai_filter.filters]
    }


def _validation_reason(error: ValidationError) -> str:
    return "; ".join(
        ".".join(("filters", *(str(part) for part in detail["loc"]))) + f": {detail['msg']}"
        for detail in error.errors()
    )


class BedrockVectorStoreConfig(BaseVectorStoreConfig, BaseAWSLLM):
    """Vector store configuration for AWS Bedrock Knowledge Bases."""

    def __init__(self) -> None:
        BaseVectorStoreConfig.__init__(self)
        BaseAWSLLM.__init__(self)

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: base passes response headers as a dict
    ) -> BedrockError:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def get_auth_credentials(self, litellm_params: dict) -> BaseVectorStoreAuthCredentials:
        return {}

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [("POST", "/knowledgebases/{knowledge_base_id}/retrieve")],
            "write": [],
        }

    def get_supported_openai_params(self, model: str) -> list[VECTOR_STORE_OPENAI_PARAMS]:
        return ["filters", "max_num_results", "ranking_options"]

    def _map_operator_to_aws(self, operator: str) -> str:
        return _AWS_COMPARISON_OPERATORS.get(operator, operator)

    def _map_operator_filter(self, filter_dict: dict) -> dict:
        """
        Map a single OpenAI operator filter to AWS KB format.

        OpenAI format: {"key": <key>, "value": <value>, "operator": <operator>}
        AWS KB format: {"operator": {"key": <key>, "value": <value>}}
        """
        aws_operator: Final = self._map_operator_to_aws(filter_dict["operator"])
        return {
            aws_operator: {
                "key": filter_dict["key"],
                "value": filter_dict["value"],
            }
        }

    def _map_and_or_filters(self, value: dict) -> dict:
        """
        Map OpenAI and/or filters to AWS KB format.

        OpenAI format: {"and" | "or": [{"key": <key>, "value": <value>, "operator": <operator>}]}
        AWS KB format: {"andAll" | "orAll": [{"operator": {"key": <key>, "value": <value>}}]}

        Note: AWS requires andAll/orAll to have at least 2 elements.
        For single filters, unwrap and return just the operator.
        """
        aws_filters: Final = {}

        if "and" in value:
            and_filters: Final = value["and"]
            # If only 1 filter, return just the operator (AWS requires andAll to have >=2 elements)
            if len(and_filters) == 1:
                return self._map_operator_filter(and_filters[0])

            aws_filters["andAll"] = [
                {
                    self._map_operator_to_aws(and_filters[i]["operator"]): {
                        "key": and_filters[i]["key"],
                        "value": and_filters[i]["value"],
                    }
                }
                for i in range(len(and_filters))
            ]

        if "or" in value:
            or_filters: Final = value["or"]
            # If only 1 filter, return just the operator (AWS requires orAll to have >=2 elements)
            if len(or_filters) == 1:
                return self._map_operator_filter(or_filters[0])

            aws_filters["orAll"] = [
                {
                    self._map_operator_to_aws(or_filters[i]["operator"]): {
                        "key": or_filters[i]["key"],
                        "value": or_filters[i]["value"],
                    }
                }
                for i in range(len(or_filters))
            ]

        return aws_filters

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: dict,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_num_results":
                optional_params["numberOfResults"] = value
            elif param == "filters" and value is not None:
                optional_params["filters"] = self._aws_filters(value, drop_params=drop_params)

        return optional_params

    def _aws_filters(self, filters: object, drop_params: bool) -> Mapping[str, object] | None:
        if not isinstance(filters, dict):
            return self._invalid_filters(f"filters must be an object, got {type(filters).__name__}", drop_params)
        if "type" in filters:
            try:
                return _aws_retrieval_filter(_OPENAI_FILTER.validate_python(filters))
            except ValidationError as error:
                return self._invalid_filters(_validation_reason(error), drop_params)
        if "operator" in filters:
            return self._map_operator_filter(filters)
        if "and" in filters or "or" in filters:
            return self._map_and_or_filters(filters)
        return filters

    @staticmethod
    def _invalid_filters(reason: str, drop_params: bool) -> None:
        if not drop_params:
            raise BadRequestError(
                message=f"Invalid filters for the Bedrock vector store: {reason}. "
                "Fix the filter, or set drop_params to drop it.",
                model=None,
                llm_provider="bedrock",
            )
        verbose_logger.debug("Dropping invalid Bedrock vector store filters under drop_params: %s", reason)

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        headers = headers or {}
        headers.setdefault("Content-Type", "application/json")
        return headers

    def get_complete_url(self, api_base: str | None, litellm_params: dict) -> str:
        aws_region_name: Final = litellm_params.get("aws_region_name")
        endpoint_url, _ = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=litellm_params.get("aws_bedrock_runtime_endpoint"),
            aws_region_name=self.get_aws_region_name_for_non_llm_api_calls(aws_region_name=aws_region_name),
            endpoint_type="agent",
        )
        return join_api_base(endpoint_url, "knowledgebases")

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | list[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        if isinstance(query, list):
            query = " ".join(query)

        encoded_vector_store_id: Final = encode_url_path_segment(vector_store_id, field_name="vector_store_id")
        url: Final = join_api_base(api_base, f"{encoded_vector_store_id}/retrieve")

        request_body: Final[dict[str, object]] = {
            "retrievalQuery": BedrockKBRetrievalQuery(text=query),
        }

        retrieval_config: dict[str, Any] = {}

        if isinstance(extra_body, dict):
            retrieval_config = deepcopy(
                extra_body.get("retrievalConfiguration") or extra_body.get("retrieval_configuration") or {}
            )
        max_results: Final = vector_store_search_optional_params.get("max_num_results")
        if max_results is not None:
            existing_number_of_results = retrieval_config.get("vectorSearchConfiguration", {}).get("numberOfResults")
            if existing_number_of_results is not None and existing_number_of_results != max_results:
                verbose_logger.debug(
                    "Overriding extra_body retrievalConfiguration.vectorSearchConfiguration.numberOfResults (%s) with max_num_results=%s",
                    existing_number_of_results,
                    max_results,
                )
            retrieval_config.setdefault("vectorSearchConfiguration", {})["numberOfResults"] = max_results
        filters: Final = vector_store_search_optional_params.get("filters")
        if filters is not None:
            existing_filter: Final = retrieval_config.get("vectorSearchConfiguration", {}).get("filter")
            if existing_filter is not None and existing_filter != filters:
                verbose_logger.debug(
                    "Overriding extra_body retrievalConfiguration.vectorSearchConfiguration.filter with filters from vector_store_search_optional_params"
                )
            retrieval_config.setdefault("vectorSearchConfiguration", {})["filter"] = filters
        if retrieval_config:
            request_body["retrievalConfiguration"] = cast(BedrockKBRetrievalConfiguration, retrieval_config)
        user_context: Final = self._user_context(extra_body=extra_body, litellm_params=litellm_params)
        if user_context is not None:
            request_body["userContext"] = user_context

        litellm_logging_obj.model_call_details["query"] = query
        return url, request_body

    @staticmethod
    def _user_context(
        extra_body: Mapping[str, object] | None, litellm_params: Mapping[str, object]
    ) -> BedrockKBUserContext | None:
        sources: Final = tuple(source for source in (extra_body, litellm_params) if isinstance(source, Mapping))
        found: Final = next(
            (
                source[key]
                for source in sources
                for key in ("userContext", "user_context")
                if source.get(key) is not None
            ),
            None,
        )
        return None if found is None else cast(BedrockKBUserContext, found)

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: str | None = None,
    ) -> tuple[dict, bytes | None]:
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
        )

    def _get_file_id_from_metadata(self, metadata: dict[str, Any]) -> str:
        """
        Extract file_id from Bedrock KB metadata.
        Uses source URI if available, otherwise generates a fallback ID.
        """
        source_uri: Final = metadata.get("x-amz-bedrock-kb-source-uri", "") if metadata else ""
        if source_uri:
            return source_uri

        chunk_id: Final = metadata.get("x-amz-bedrock-kb-chunk-id", "unknown") if metadata else "unknown"
        return f"bedrock-kb-{chunk_id}"

    def _get_filename_from_metadata(self, metadata: dict[str, Any]) -> str:
        """
        Extract filename from Bedrock KB metadata.
        Tries to extract filename from source URI, falls back to domain name or data source ID.
        """
        source_uri: Final = metadata.get("x-amz-bedrock-kb-source-uri", "") if metadata else ""

        if source_uri:
            try:
                parsed_uri: Final = urlparse(source_uri)
                filename = (
                    parsed_uri.path.split("/")[-1] if parsed_uri.path and parsed_uri.path != "/" else parsed_uri.netloc
                )
                if not filename or filename == "/":
                    filename = parsed_uri.netloc
                return filename
            except Exception:
                return source_uri

        data_source_id: Final = metadata.get("x-amz-bedrock-kb-data-source-id", "unknown") if metadata else "unknown"
        return f"bedrock-kb-document-{data_source_id}"

    def _get_attributes_from_metadata(self, metadata: dict[str, object]) -> dict[str, object]:
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
            response_data: Final = BedrockKBResponse(**response.json())
            results: Final[list[VectorStoreSearchResult]] = []
            for item in response_data.get("retrievalResults", []) or []:
                content: BedrockKBContent | None = item.get("content")
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
    ) -> tuple[str, dict]:
        raise NotImplementedError

    def transform_create_vector_store_response(self, response: httpx.Response):
        raise NotImplementedError
