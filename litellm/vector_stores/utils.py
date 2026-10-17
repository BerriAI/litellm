from collections.abc import Mapping
from typing import Final, cast, get_type_hints

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


def _callers_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in metadata.items() if not is_litellm_written_metadata_key(key)}


class VectorStoreRequestUtils:
    """Helper utils for constructing Vector Store search requests"""

    @staticmethod
    def get_requested_vector_store_search_optional_param(
        params: dict[str, object],
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
        params: dict[str, object],
    ) -> VectorStoreCreateOptionalRequestParams:
        """
        Filter parameters to only include those defined in VectorStoreCreateOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            VectorStoreCreateOptionalRequestParams instance with only the valid parameters
        """
        valid_keys: Final = get_type_hints(VectorStoreCreateOptionalRequestParams).keys()
        filtered_params: Final = {k: v for k, v in params.items() if k in valid_keys and v is not None}
        metadata: Final = filtered_params.get("metadata")
        if not isinstance(metadata, Mapping):
            return cast(VectorStoreCreateOptionalRequestParams, filtered_params)
        callers_metadata: Final = _callers_metadata(
            cast(Mapping[str, object], metadata)  # cast-ok: request metadata is a JSON object
        )
        return cast(  # cast-ok: same params as above with the caller's metadata only
            VectorStoreCreateOptionalRequestParams, {**filtered_params, "metadata": callers_metadata}
        )
