from typing import Any, Dict, Optional, Tuple, cast

import httpx

import litellm
from litellm.llms.base_llm.vector_store_files.transformation import (
    BaseVectorStoreFilesConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_store_files import (
    VectorStoreFileAuthCredentials,
    VectorStoreFileContentResponse,
    VectorStoreFileCreateRequest,
    VectorStoreFileDeleteResponse,
    VectorStoreFileListQueryParams,
    VectorStoreFileListResponse,
    VectorStoreFileObject,
    VectorStoreFileUpdateRequest,
)
from litellm.utils import add_openai_metadata


def _clean_dict(source: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in source.items() if v is not None}


class OpenAIVectorStoreFilesConfig(BaseVectorStoreFilesConfig):
    ASSISTANTS_HEADER_KEY = "OpenAI-Beta"
    ASSISTANTS_HEADER_VALUE = "assistants=v2"

    def get_auth_credentials(
        self, litellm_params: Dict[str, Any]
    ) -> VectorStoreFileAuthCredentials:
        api_key = litellm_params.get("api_key")
        if api_key is None:
            raise ValueError("api_key is required")
        return {
            "headers": {
                "Authorization": f"Bearer {api_key}",
            }
        }

    def get_vector_store_file_endpoints_by_type(self) -> Dict[
        str, Tuple[Tuple[str, str], ...]
    ]:
        return {
            "read": (
                ("GET", "/vector_stores/{vector_store_id}/files"),
                ("GET", "/vector_stores/{vector_store_id}/files/{file_id}"),
                (
                    "GET",
                    "/vector_stores/{vector_store_id}/files/{file_id}/content",
                ),
            ),
            "write": (
                ("POST", "/vector_stores/{vector_store_id}/files"),
                ("POST", "/vector_stores/{vector_store_id}/files/{file_id}"),
                ("DELETE", "/vector_stores/{vector_store_id}/files/{file_id}"),
            ),
        }

    def validate_environment(
        self,
        *,
        headers: Dict[str, str],
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> Dict[str, str]:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        if self.ASSISTANTS_HEADER_KEY not in headers:
            headers[self.ASSISTANTS_HEADER_KEY] = self.ASSISTANTS_HEADER_VALUE
        return headers

    def get_complete_url(
        self,
        *,
        api_base: Optional[str],
        vector_store_id: str,
        litellm_params: Dict[str, Any],
    ) -> str:
        base_url = (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        base_url = base_url.rstrip("/")
        return f"{base_url}/vector_stores/{vector_store_id}/files"

    def transform_create_vector_store_file_request(
        self,
        *,
        vector_store_id: str,
        create_request: VectorStoreFileCreateRequest,
        api_base: str,
    ) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = _clean_dict(dict(create_request))
        attributes = payload.get("attributes")
        if isinstance(attributes, dict):
            filtered_attributes = add_openai_metadata(attributes)
            if filtered_attributes is not None:
                payload["attributes"] = filtered_attributes
            else:
                payload.pop("attributes", None)
        url = api_base
        return url, payload

    def transform_create_vector_store_file_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileObject:
        try:
            return cast(VectorStoreFileObject, response.json())
        except Exception as exc:  # noqa: BLE001
            raise self.get_error_class(
                error_message=str(exc),
                status_code=response.status_code,
                headers=response.headers,
            )

    def transform_list_vector_store_files_request(
        self,
        *,
        vector_store_id: str,
        query_params: VectorStoreFileListQueryParams,
        api_base: str,
    ) -> Tuple[str, Dict[str, Any]]:
        params = _clean_dict(dict(query_params))
        return api_base, params

    def transform_list_vector_store_files_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileListResponse:
        try:
            return cast(VectorStoreFileListResponse, response.json())
        except Exception as exc:  # noqa: BLE001
            raise self.get_error_class(
                error_message=str(exc),
                status_code=response.status_code,
                headers=response.headers,
            )

    def transform_retrieve_vector_store_file_request(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        api_base: str,
    ) -> Tuple[str, Dict[str, Any]]:
        return f"{api_base}/{file_id}", {}

    def transform_retrieve_vector_store_file_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileObject:
        try:
            return cast(VectorStoreFileObject, response.json())
        except Exception as exc:  # noqa: BLE001
            raise self.get_error_class(
                error_message=str(exc),
                status_code=response.status_code,
                headers=response.headers,
            )

    def transform_retrieve_vector_store_file_content_request(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        api_base: str,
    ) -> Tuple[str, Dict[str, Any]]:
        return f"{api_base}/{file_id}/content", {}

    def transform_retrieve_vector_store_file_content_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileContentResponse:
        try:
            return cast(VectorStoreFileContentResponse, response.json())
        except Exception as exc:  # noqa: BLE001
            raise self.get_error_class(
                error_message=str(exc),
                status_code=response.status_code,
                headers=response.headers,
            )

    def transform_update_vector_store_file_request(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        update_request: VectorStoreFileUpdateRequest,
        api_base: str,
    ) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = dict(update_request)
        attributes = payload.get("attributes")
        if isinstance(attributes, dict):
            filtered_attributes = add_openai_metadata(attributes)
            if filtered_attributes is not None:
                payload["attributes"] = filtered_attributes
            else:
                payload.pop("attributes", None)
        return f"{api_base}/{file_id}", payload

    def transform_update_vector_store_file_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileObject:
        try:
            return cast(VectorStoreFileObject, response.json())
        except Exception as exc:  # noqa: BLE001
            raise self.get_error_class(
                error_message=str(exc),
                status_code=response.status_code,
                headers=response.headers,
            )

    def transform_delete_vector_store_file_request(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        api_base: str,
    ) -> Tuple[str, Dict[str, Any]]:
        return f"{api_base}/{file_id}", {}

    def transform_delete_vector_store_file_response(
        self,
        *,
        response: httpx.Response,
    ) -> VectorStoreFileDeleteResponse:
        try:
            return cast(VectorStoreFileDeleteResponse, response.json())
        except Exception as exc:  # noqa: BLE001
            raise self.get_error_class(
                error_message=str(exc),
                status_code=response.status_code,
                headers=response.headers,
            )
