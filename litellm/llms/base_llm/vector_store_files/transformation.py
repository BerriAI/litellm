from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx

from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_store_files import (
    VectorStoreFileAuthCredentials,
    VectorStoreFileChunkingStrategy,
    VectorStoreFileContentResponse,
    VectorStoreFileCreateRequest,
    VectorStoreFileDeleteResponse,
    VectorStoreFileListQueryParams,
    VectorStoreFileListResponse,
    VectorStoreFileObject,
    VectorStoreFileUpdateRequest,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


class BaseVectorStoreFilesConfig(ABC):
    """Base configuration contract for provider-specific vector store file implementations."""

    def get_supported_openai_params(
        self,
        operation: str,
    ) -> tuple[str, ...]:
        """Return the set of OpenAI params supported for the given operation."""

        return tuple()

    def map_openai_params(
        self,
        *,
        operation: str,
        non_default_params: dict[str, Any],
        optional_params: dict[str, Any],
        drop_params: bool,
    ) -> dict[str, Any]:
        """Map non-default OpenAI params to provider-specific params."""

        return optional_params

    @abstractmethod
    def get_auth_credentials(self, litellm_params: dict[str, Any]) -> VectorStoreFileAuthCredentials: ...

    @abstractmethod
    def get_vector_store_file_endpoints_by_type(
        self,
    ) -> dict[str, tuple[tuple[str, str], ...]]: ...

    @abstractmethod
    def validate_environment(
        self,
        *,
        headers: dict[str, str],
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, str]:
        return {}

    @abstractmethod
    def get_complete_url(
        self,
        *,
        api_base: str | None,
        vector_store_id: str,
        litellm_params: dict[str, Any],
    ) -> str:
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    @abstractmethod
    def transform_create_vector_store_file_request(
        self,
        *,
        vector_store_id: str,
        create_request: VectorStoreFileCreateRequest,
        api_base: str,
    ) -> tuple[str, dict[str, Any]]: ...

    @abstractmethod
    def transform_create_vector_store_file_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileObject: ...

    @abstractmethod
    def transform_list_vector_store_files_request(
        self,
        *,
        vector_store_id: str,
        query_params: VectorStoreFileListQueryParams,
        api_base: str,
    ) -> tuple[str, dict[str, Any]]: ...

    @abstractmethod
    def transform_list_vector_store_files_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileListResponse: ...

    @abstractmethod
    def transform_retrieve_vector_store_file_request(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        api_base: str,
    ) -> tuple[str, dict[str, Any]]: ...

    @abstractmethod
    def transform_retrieve_vector_store_file_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileObject: ...

    @abstractmethod
    def transform_retrieve_vector_store_file_content_request(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        api_base: str,
    ) -> tuple[str, dict[str, Any]]: ...

    @abstractmethod
    def transform_retrieve_vector_store_file_content_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileContentResponse: ...

    @abstractmethod
    def transform_update_vector_store_file_request(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        update_request: VectorStoreFileUpdateRequest,
        api_base: str,
    ) -> tuple[str, dict[str, Any]]: ...

    @abstractmethod
    def transform_update_vector_store_file_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileObject: ...

    @abstractmethod
    def transform_delete_vector_store_file_request(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        api_base: str,
    ) -> tuple[str, dict[str, Any]]: ...

    @abstractmethod
    def transform_delete_vector_store_file_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileDeleteResponse: ...

    def get_error_class(
        self,
        *,
        error_message: str,
        status_code: int,
        headers: dict[str, Any] | httpx.Headers,
    ) -> BaseLLMException:
        from ..chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def sign_request(
        self,
        *,
        headers: dict[str, str],
        optional_params: dict[str, Any],
        request_data: dict[str, Any],
        api_base: str,
        api_key: str | None = None,
    ) -> tuple[dict[str, str], bytes | None]:
        return headers, None

    def prepare_chunking_strategy(
        self,
        chunking_strategy: VectorStoreFileChunkingStrategy | None,
    ) -> VectorStoreFileChunkingStrategy | None:
        return chunking_strategy
