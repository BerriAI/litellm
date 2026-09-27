from collections.abc import Mapping
from typing import Final, cast, get_type_hints

from litellm.types.vector_store_files import (
    VectorStoreFileCreateRequest,
    VectorStoreFileListQueryParams,
    VectorStoreFileUpdateRequest,
)


class VectorStoreFileRequestUtils:
    """Helper utilities for constructing vector store file requests."""

    @staticmethod
    def _filter_params(params: Mapping[str, object], model: type[object]) -> dict[str, object]:
        valid_keys: Final = get_type_hints(model).keys()
        return {key: value for key, value in params.items() if key in valid_keys and value is not None}

    @staticmethod
    def get_create_request_params(
        params: Mapping[str, object],
    ) -> VectorStoreFileCreateRequest:
        filtered: Final = VectorStoreFileRequestUtils._filter_params(params=params, model=VectorStoreFileCreateRequest)
        return cast(VectorStoreFileCreateRequest, filtered)

    @staticmethod
    def get_list_query_params(params: Mapping[str, object]) -> VectorStoreFileListQueryParams:
        filtered = VectorStoreFileRequestUtils._filter_params(params=params, model=VectorStoreFileListQueryParams)
        return cast(VectorStoreFileListQueryParams, filtered)

    @staticmethod
    def get_update_request_params(
        params: Mapping[str, object],
    ) -> VectorStoreFileUpdateRequest:
        filtered: Final = VectorStoreFileRequestUtils._filter_params(params=params, model=VectorStoreFileUpdateRequest)
        return cast(VectorStoreFileUpdateRequest, filtered)
