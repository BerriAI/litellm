import copy
from collections.abc import Container, Mapping
from typing import Any, Final, cast, get_type_hints

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.constants import ROUTING_REQUEST_TAGS_METADATA_KEY, VECTOR_STORE_CALLER_METADATA_KWARG
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreSearchOptionalRequestParams,
)

LITELLM_WRITTEN_METADATA_KEYS: Final = frozenset(
    {
        "requester_ip_address",
        "user_agent",
        "endpoint",
        "litellm_api_version",
        "deployment",
        "deployment_model_name",
        "api_base",
        "agent_id",
        "actor_agent_id",
        "target_agent_id",
        "billing_agent_id",
        "agent_execution_mode",
        "verified_human_user_id",
        "litellm_caller_session_id",
    }
)


def is_litellm_written_metadata_key(key: str) -> bool:
    return key == "user_api_key" or key.startswith("user_api_key_") or key in LITELLM_WRITTEN_METADATA_KEYS


def _is_callers_key(key: object) -> bool:
    return not isinstance(key, str) or not is_litellm_written_metadata_key(key)


def _sent_by_proxy_or_router(params: Mapping[str, object], metadata: Container[object]) -> bool:
    return params.get("proxy_server_request") is not None or ROUTING_REQUEST_TAGS_METADATA_KEY in metadata


def _callers_metadata(metadata: Mapping[object, object]) -> Mapping[object, object]:
    return {key: value for key, value in metadata.items() if _is_callers_key(key)}


_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])


class VectorStoreWriteBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    metadata: object = None


def _json_object(raw: str | bytes) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_json(raw)
    except ValidationError:
        return None


def _decoded_metadata(metadata: object) -> object:
    if not isinstance(metadata, str):
        return metadata
    decoded: Final = _json_object(metadata)
    return metadata if decoded is None else decoded


def callers_metadata_snapshot(raw_json_body: bytes | None, parsed_body: VectorStoreWriteBody) -> object:
    raw_body: Final = None if raw_json_body is None else _json_object(raw_json_body)
    metadata: Final = copy.deepcopy(parsed_body.metadata) if raw_body is None else raw_body.get("metadata")
    return _decoded_metadata(metadata)


def _metadata_for_provider(params: Mapping[str, object]) -> object:
    if VECTOR_STORE_CALLER_METADATA_KWARG in params:
        return params[VECTOR_STORE_CALLER_METADATA_KWARG]
    metadata: Final = params.get("metadata")
    if not isinstance(metadata, Mapping):
        return metadata
    request_metadata: Final = cast(Mapping[object, object], metadata)  # cast-ok: request metadata is a JSON object
    if not _sent_by_proxy_or_router(params, request_metadata):
        return request_metadata
    return _callers_metadata(request_metadata)


class VectorStoreRequestUtils:
    """Helper utils for constructing Vector Store search requests"""

    @staticmethod
    def get_requested_vector_store_search_optional_param(
        params: dict[str, Any],
        vector_store_provider_config: BaseVectorStoreConfig,
    ) -> VectorStoreSearchOptionalRequestParams:
        """
        Filter parameters to only include those defined in VectorStoreSearchOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            VectorStoreSearchOptionalRequestParams instance with only the valid parameters
        """
        valid_keys: Final = get_type_hints(VectorStoreSearchOptionalRequestParams).keys()
        filtered_params: Final = {k: v for k, v in params.items() if k in valid_keys and v is not None}

        optional_params: Final = vector_store_provider_config.map_openai_params(
            non_default_params=params,
            optional_params=filtered_params,
            drop_params=False,
        )

        return cast(VectorStoreSearchOptionalRequestParams, optional_params)

    @staticmethod
    def get_requested_vector_store_create_optional_param(
        params: dict[str, Any],
    ) -> VectorStoreCreateOptionalRequestParams:
        """
        Filter parameters to only include those defined in VectorStoreCreateOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            VectorStoreCreateOptionalRequestParams instance with only the valid parameters
        """
        valid_keys: Final = get_type_hints(VectorStoreCreateOptionalRequestParams).keys() - {"metadata"}
        filtered_params: Final = {k: v for k, v in params.items() if k in valid_keys and v is not None}
        metadata: Final = _metadata_for_provider(params)
        if metadata is None:
            return cast(VectorStoreCreateOptionalRequestParams, filtered_params)
        return cast(  # cast-ok: same params as above plus the metadata the provider should store
            VectorStoreCreateOptionalRequestParams, {**filtered_params, "metadata": metadata}
        )
