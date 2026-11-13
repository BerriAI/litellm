import re
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, cast, runtime_checkable
from urllib.parse import urlparse

import httpx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.core_helpers import decoded_json_object, drops_unsupported_params, keyword_arguments
from litellm.litellm_core_utils.url_utils import encode_url_path_segment, join_api_base
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.integrations.rag.bedrock_knowledgebase import (
    BedrockKBContent,
    BedrockKBImageFormat,
    BedrockKBResponse,
    BedrockKBRetrievalConfiguration,
    BedrockKBRetrievalQuery,
    BedrockKBUserContext,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    RETRIEVAL_QUERY_PARAMS,
    VECTOR_STORE_OPENAI_PARAMS,
    BaseVectorStoreAuthCredentials,
    VectorStoreIndexEndpoints,
    VectorStoreResultContent,
    VectorStoreSearchImageMediaType,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchQueryImage,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


@runtime_checkable
class _SupportsSetDefault(Protocol):
    def setdefault(self, key: str, default: object, /) -> object: ...


@runtime_checkable
class _SupportsSetItem(Protocol):
    def __setitem__(self, key: str, value: object, /) -> None: ...


def _setdefault_target(value: object) -> _SupportsSetDefault:
    if isinstance(value, _SupportsSetDefault):
        return value
    raise AttributeError(f"{type(value).__name__!r} object has no attribute 'setdefault'", name="setdefault", obj=value)


def _item_assignment_target(value: object) -> _SupportsSetItem:
    if isinstance(value, _SupportsSetItem):
        return value
    raise TypeError(f"'{type(value).__name__}' object does not support item assignment")


def _vector_search_configuration(retrieval_config: object) -> Mapping[str, object]:
    return decoded_json_object(decoded_json_object(retrieval_config).get("vectorSearchConfiguration", {}))


_DECODED_JSON: Final = TypeAdapter(object)


class _KBResponseFields(TypedDict, total=False, extra_items=object):
    pass


_BEDROCK_IMAGE_FORMATS: Final[Mapping[VectorStoreSearchImageMediaType, BedrockKBImageFormat]] = MappingProxyType(
    {"image/jpeg": "jpeg", "image/png": "png", "image/gif": "gif", "image/webp": "webp"}
)
_BASE64_DATA_URI: Final = re.compile(r"data:(?P<media_type>[^;,]*)[^,]*;base64,(?P<data>.*)", re.DOTALL)
_SOURCE_FILE_MIME_TYPE_KEY: Final = "x-amz-bedrock-kb-source-file-mime-type"
_UNKNOWN_MEDIA_TYPE: Final = "application/octet-stream"
_RETRIEVAL_QUERY_UNSUPPORTED: Final = (
    "bedrock vector stores do not take retrievalQuery. Send query, or query_image to search by image. To drop it, "
    "set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n"
)


class _QueryImageParams(TypedDict, total=False):
    query_image: ReadOnly[VectorStoreSearchQueryImage | None]


def _retrieval_query(query: str, params: _QueryImageParams) -> BedrockKBRetrievalQuery:
    query_image: Final = params.get("query_image")
    if query_image is None:
        return BedrockKBRetrievalQuery(text=query)
    return BedrockKBRetrievalQuery(
        type="IMAGE",
        image={"format": _BEDROCK_IMAGE_FORMATS[query_image["media_type"]], "inlineContent": query_image["data"]},
    )


def _image_content(byte_content: str, metadata: Mapping[str, object]) -> VectorStoreResultContent:
    data_uri: Final = _BASE64_DATA_URI.fullmatch(byte_content)
    declared_media_type: Final = data_uri.group("media_type") if data_uri is not None else ""
    source_file_media_type: Final = metadata.get(_SOURCE_FILE_MIME_TYPE_KEY)
    return VectorStoreResultContent(
        type="image",
        source={
            "type": "base64",
            "media_type": (
                source_file_media_type
                if isinstance(source_file_media_type, str) and source_file_media_type.startswith("image/")
                else declared_media_type or _UNKNOWN_MEDIA_TYPE
            ),
            "data": data_uri.group("data") if data_uri is not None else byte_content,
        },
    )


def _result_content(
    content: BedrockKBContent | None, metadata: Mapping[str, object], include_images: bool
) -> VectorStoreResultContent | None:
    if content is None:
        return None
    text: Final = content.get("text")
    if text is not None:
        return VectorStoreResultContent(text=text, type="text")
    byte_content: Final = content.get("byteContent")
    if not include_images or content.get("type") != "IMAGE" or not byte_content:
        return None
    return _image_content(byte_content, metadata)


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
        return ["filters", "max_num_results", "ranking_options", "query_image"]

    def _map_operator_to_aws(self, operator: str) -> str:
        """
        Map OpenAI-style operators to AWS Bedrock operator names.

        OpenAI uses: eq, ne, gt, gte, lt, lte, in, nin
        AWS uses: equals, notEquals, greaterThan, greaterThanOrEquals, lessThan, lessThanOrEquals, in, notIn, startsWith, listContains, stringContains
        """
        operator_mapping: Final = {
            "eq": "equals",
            "ne": "notEquals",
            "gt": "greaterThan",
            "gte": "greaterThanOrEquals",
            "lt": "lessThan",
            "lte": "lessThanOrEquals",
            "in": "in",
            "nin": "notIn",
            # AWS-specific operators (pass through)
            "equals": "equals",
            "notEquals": "notEquals",
            "greaterThan": "greaterThan",
            "greaterThanOrEquals": "greaterThanOrEquals",
            "lessThan": "lessThan",
            "lessThanOrEquals": "lessThanOrEquals",
            "notIn": "notIn",
            "startsWith": "startsWith",
            "listContains": "listContains",
            "stringContains": "stringContains",
        }
        return operator_mapping.get(operator, operator)

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
        non_default_params: dict,
        optional_params: dict,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_num_results":
                optional_params["numberOfResults"] = value
            elif param == "filters" and value is not None:
                # map the openai filters to the aws kb filters format
                # openai filters = {"key": <key>, "value": <value>, "operator": <operator>} OR {"and" | "or": [{"key": <key>, "value": <value>, "operator": <operator>}]}
                # aws kb filters = {"operator": {"<key>": <value>}} OR {"andAll | orAll": [{"operator": {"<key>": <value>}}]}
                # 1. check if filter is in openai format
                # 2. if it is, map it to the aws kb filters format
                # 3. if it is not, assume it is in aws kb filters format and add it to the optional_params
                aws_filters: dict | None = None

                if isinstance(value, dict):
                    if "operator" in value:
                        # Single operator - map directly (no wrapping needed)
                        aws_filters = self._map_operator_filter(value)
                    elif "and" in value or "or" in value:
                        aws_filters = self._map_and_or_filters(value)
                    else:
                        # Assume it's already in AWS KB format
                        aws_filters = value
                optional_params["filters"] = aws_filters

        return optional_params

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
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict]:
        if isinstance(query, list):
            query = " ".join(query)

        encoded_vector_store_id: Final = encode_url_path_segment(vector_store_id, field_name="vector_store_id")
        url: Final = join_api_base(api_base, f"{encoded_vector_store_id}/retrieve")

        caller_params: Final = decoded_json_object(litellm_params)
        caller_sources: Final = self._caller_sources(extra_body=extra_body, litellm_params=caller_params)
        if self._caller_value(caller_sources, RETRIEVAL_QUERY_PARAMS) is not None and not drops_unsupported_params(
            caller_params.get("drop_params")
        ):
            raise BadRequestError(message=_RETRIEVAL_QUERY_UNSUPPORTED, model="", llm_provider="bedrock")
        request_body: Final[dict[str, object]] = {
            "retrievalQuery": _retrieval_query(query, vector_store_search_optional_params),
        }

        retrieval_config: Final[object] = (
            deepcopy(extra_body.get("retrievalConfiguration") or extra_body.get("retrieval_configuration") or {})
            if isinstance(extra_body, dict)
            else {}
        )
        max_results: Final = vector_store_search_optional_params.get("max_num_results")
        if max_results is not None:
            existing_number_of_results: Final = _vector_search_configuration(retrieval_config).get("numberOfResults")
            if existing_number_of_results is not None and existing_number_of_results != max_results:
                verbose_logger.debug(
                    "Overriding extra_body retrievalConfiguration.vectorSearchConfiguration.numberOfResults (%s) with max_num_results=%s",
                    existing_number_of_results,
                    max_results,
                )
            _item_assignment_target(_setdefault_target(retrieval_config).setdefault("vectorSearchConfiguration", {}))[
                "numberOfResults"
            ] = max_results
        filters: Final = vector_store_search_optional_params.get("filters")
        if filters is not None:
            existing_filter: Final = _vector_search_configuration(retrieval_config).get("filter")
            if existing_filter is not None and existing_filter != filters:
                verbose_logger.debug(
                    "Overriding extra_body retrievalConfiguration.vectorSearchConfiguration.filter with filters from vector_store_search_optional_params"
                )
            _item_assignment_target(_setdefault_target(retrieval_config).setdefault("vectorSearchConfiguration", {}))[
                "filter"
            ] = filters
        if retrieval_config:
            request_body["retrievalConfiguration"] = cast(BedrockKBRetrievalConfiguration, retrieval_config)
        user_context: Final = self._user_context(caller_sources)
        if user_context is not None:
            request_body["userContext"] = user_context

        litellm_logging_obj.model_call_details["query"] = query
        return url, request_body

    @staticmethod
    def _caller_sources(
        extra_body: Mapping[str, object] | None, litellm_params: Mapping[str, object]
    ) -> tuple[Mapping[str, object], ...]:
        return tuple(source for source in (extra_body, litellm_params) if isinstance(source, Mapping))

    @staticmethod
    def _caller_value(sources: tuple[Mapping[str, object], ...], keys: tuple[str, ...]) -> object:
        return next((source[key] for source, key in product(sources, keys) if source.get(key) is not None), None)

    @staticmethod
    def _user_context(sources: tuple[Mapping[str, object], ...]) -> BedrockKBUserContext | None:
        found: Final = BedrockVectorStoreConfig._caller_value(sources, ("userContext", "user_context"))
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
            response_data: Final = BedrockKBResponse(
                **_KBResponseFields(
                    **keyword_arguments(_DECODED_JSON.validate_python(response.json()), BedrockKBResponse)
                )
            )
            include_images: Final = litellm_logging_obj.model_call_details.get("query_image") is not None
            results: Final[list[VectorStoreSearchResult]] = []
            for item in response_data.get("retrievalResults", []) or []:
                metadata = item.get("metadata", {}) or {}
                result_content = _result_content(item.get("content"), metadata, include_images)
                if result_content is None:
                    continue

                file_id = self._get_file_id_from_metadata(metadata)
                filename = self._get_filename_from_metadata(metadata)
                attributes = self._get_attributes_from_metadata(metadata)

                results.append(
                    VectorStoreSearchResult(
                        score=item.get("score"),
                        content=[result_content],
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
