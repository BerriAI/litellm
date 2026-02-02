from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx

import litellm
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.containers.main import (
    ContainerCreateOptionalRequestParams,
    ContainerFileListResponse,
    ContainerListResponse,
    ContainerObject,
    DeleteContainerResult,
)
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException
    from ...base_llm.containers.transformation import (
        BaseContainerConfig as _BaseContainerConfig,
    )

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseContainerConfig = _BaseContainerConfig
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseContainerConfig = Any
    BaseLLMException = Any


class OpenAIContainerConfig(BaseContainerConfig):
    """Configuration class for OpenAI container API.
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self) -> list:
        """Get the list of supported OpenAI parameters for container API.
        """
        return [
            "name",
            "expires_after",
            "file_ids",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        container_create_optional_params: ContainerCreateOptionalRequestParams,
        drop_params: bool,
    ) -> Dict:
        """No mapping applied since inputs are in OpenAI spec already"""
        return dict(container_create_optional_params)

    def validate_environment(
        self,
        headers: dict,
        api_key: Optional[str] = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            },
        )
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """Get the complete URL for OpenAI container API.
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        return f"{api_base.rstrip('/')}/containers"

    def transform_container_create_request(
        self,
        name: str,
        container_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """Transform the container creation request for OpenAI API.
        """
        # Remove extra_headers from optional params as they're handled separately
        container_create_optional_request_params = {
            k: v for k, v in container_create_optional_request_params.items()
            if k not in ["extra_headers"]
        }

        # Create the request data
        request_dict = {
            "name": name,
            **container_create_optional_request_params,
        }

        return request_dict

    def transform_container_create_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ContainerObject:
        """Transform the OpenAI container creation response.
        """
        response_data = raw_response.json()

        # Transform the response data
        container_obj = ContainerObject(**response_data)  # type: ignore[arg-type]

        # Add cost for container creation (OpenAI containers are code interpreter sessions)
        # https://platform.openai.com/docs/pricing
        # Each container creation is 1 code interpreter session
        container_cost = StandardBuiltInToolCostTracking.get_cost_for_code_interpreter(
            sessions=1,
            provider="openai",
        )
        
        if not hasattr(container_obj, "_hidden_params") or container_obj._hidden_params is None:
            container_obj._hidden_params = {}
        if "additional_headers" not in container_obj._hidden_params:
            container_obj._hidden_params["additional_headers"] = {}
        container_obj._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] = container_cost

        return container_obj

    def transform_container_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """Transform the container list request for OpenAI API.
        
        OpenAI API expects the following request:
        - GET /v1/containers
        """
        # Use the api_base directly for container list
        url = api_base

        # Prepare query parameters
        params = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = str(limit)
        if order is not None:
            params["order"] = order

        # Add any extra query parameters
        if extra_query:
            params.update(extra_query)

        return url, params

    def transform_container_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ContainerListResponse:
        """Transform the OpenAI container list response.
        """
        response_data = raw_response.json()

        # Transform the response data
        container_list = ContainerListResponse(**response_data)  # type: ignore[arg-type]

        return container_list

    def transform_container_retrieve_request(
        self,
        container_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform the OpenAI container retrieve request.
        """
        # For container retrieve, we just need to construct the URL
        url = f"{api_base.rstrip('/')}/{container_id}"

        # No additional data needed for GET request
        data: Dict[str, Any] = {}

        return url, data

    def transform_container_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ContainerObject:
        """Transform the OpenAI container retrieve response.
        """
        response_data = raw_response.json()
        # Transform the response data
        container_obj = ContainerObject(**response_data)  # type: ignore[arg-type]

        return container_obj

    def transform_container_delete_request(
        self,
        container_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform the container delete request for OpenAI API.
        
        OpenAI API expects the following request:
        - DELETE /v1/containers/{container_id}
        """
        # Construct the URL for container delete
        url = f"{api_base.rstrip('/')}/{container_id}"

        # No data needed for DELETE request
        data: Dict[str, Any] = {}

        return url, data

    def transform_container_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteContainerResult:
        """Transform the OpenAI container delete response.
        """
        response_data = raw_response.json()

        # Transform the response data
        delete_result = DeleteContainerResult(**response_data)  # type: ignore[arg-type]

        return delete_result

    def transform_container_file_list_request(
        self,
        container_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """Transform the container file list request for OpenAI API.
        
        OpenAI API expects the following request:
        - GET /v1/containers/{container_id}/files
        """
        # Construct the URL for container files
        url = f"{api_base.rstrip('/')}/{container_id}/files"

        # Prepare query parameters
        params: Dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = str(limit)
        if order is not None:
            params["order"] = order

        # Add any extra query parameters
        if extra_query:
            params.update(extra_query)

        return url, params

    def transform_container_file_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ContainerFileListResponse:
        """Transform the OpenAI container file list response.
        """
        response_data = raw_response.json()

        # Transform the response data
        file_list = ContainerFileListResponse(**response_data)  # type: ignore[arg-type]

        return file_list

    def transform_container_file_content_request(
        self,
        container_id: str,
        file_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform the container file content request for OpenAI API.
        
        OpenAI API expects the following request:
        - GET /v1/containers/{container_id}/files/{file_id}/content
        """
        # Construct the URL for container file content
        url = f"{api_base.rstrip('/')}/{container_id}/files/{file_id}/content"

        # No query parameters needed
        params: Dict[str, Any] = {}

        return url, params

    def transform_container_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """Transform the OpenAI container file content response.
        
        Returns the raw binary content of the file.
        """
        return raw_response.content

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

