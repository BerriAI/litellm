from typing import Any, Dict, cast, get_type_hints

from litellm.types.vector_stores import VectorStoreSearchOptionalRequestParams


class VectorStoreRequestUtils:
    """Helper utils for constructing Vector Store search requests"""

    @staticmethod
    def get_requested_vector_store_search_optional_param(
        params: Dict[str, Any],
    ) -> VectorStoreSearchOptionalRequestParams:
        """
        Filter parameters to only include those defined in VectorStoreSearchOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            VectorStoreSearchOptionalRequestParams instance with only the valid parameters
        """
        valid_keys = get_type_hints(VectorStoreSearchOptionalRequestParams).keys()
        filtered_params = {
            k: v for k, v in params.items() if k in valid_keys and v is not None
        }

        return cast(VectorStoreSearchOptionalRequestParams, filtered_params)

