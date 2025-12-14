from __future__ import annotations

import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx

from litellm.types.containers.main import ContainerCreateOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.types.containers.main import (
        ContainerFileListResponse as _ContainerFileListResponse,
    )
    from litellm.types.containers.main import (
        ContainerListResponse as _ContainerListResponse,
    )
    from litellm.types.containers.main import ContainerObject as _ContainerObject
    from litellm.types.containers.main import (
        DeleteContainerResult as _DeleteContainerResult,
    )

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
    ContainerObject = _ContainerObject
    DeleteContainerResult = _DeleteContainerResult
    ContainerListResponse = _ContainerListResponse
    ContainerFileListResponse = _ContainerFileListResponse
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any
    ContainerObject = Any
    DeleteContainerResult = Any
    ContainerListResponse = Any
    ContainerFileListResponse = Any


class BaseContainerConfig(ABC):
    def __init__(self):
        pass

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("_abc")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    @abstractmethod
    def get_supported_openai_params(self) -> list:
        pass

    @abstractmethod
    def map_openai_params(
        self,
        container_create_optional_params: ContainerCreateOptionalRequestParams,
        drop_params: bool,
    ) -> dict:
        pass

    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        api_key: str | None = None,
    ) -> dict:
        return {}

    @abstractmethod
    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """Get the complete url for the request.
        
        OPTIONAL - Some providers need `model` in `api_base`.
        """
        if api_base is None:
            msg = "api_base is required"
            raise ValueError(msg)
        return api_base

    @abstractmethod
    def transform_container_create_request(
        self,
        name: str,
        container_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        """Transform the container creation request.
        
        Returns:
            dict: Request data for container creation.
        """
        ...

    @abstractmethod
    def transform_container_create_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ContainerObject:
        """Transform the container creation response."""
        ...

    @abstractmethod
    def transform_container_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        """Transform the container list request into a URL and params.
        
        Returns:
            tuple[str, dict]: (url, params) for the container list request.
        """
        ...

    @abstractmethod
    def transform_container_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ContainerListResponse:
        """Transform the container list response."""
        ...

    @abstractmethod
    def transform_container_retrieve_request(
        self,
        container_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform the container retrieve request into a URL and data/params.
        
        Returns:
            tuple[str, dict]: (url, params) for the container retrieve request.
        """
        ...

    @abstractmethod
    def transform_container_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ContainerObject:
        """Transform the container retrieve response."""
        ...

    @abstractmethod
    def transform_container_delete_request(
        self,
        container_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform the container delete request into a URL and data.
        
        Returns:
            tuple[str, dict]: (url, data) for the container delete request.
        """
        ...

    @abstractmethod
    def transform_container_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteContainerResult:
        """Transform the container delete response."""
        ...

    @abstractmethod
    def transform_container_file_list_request(
        self,
        container_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        """Transform the container file list request into a URL and params.
        
        Returns:
            tuple[str, dict]: (url, params) for the container file list request.
        """
        ...

    @abstractmethod
    def transform_container_file_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ContainerFileListResponse:
        """Transform the container file list response."""
        ...

    @abstractmethod
    def transform_container_file_content_request(
        self,
        container_id: str,
        file_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform the container file content request into a URL and params.
        
        Returns:
            tuple[str, dict]: (url, params) for the container file content request.
        """
        ...

    @abstractmethod
    def transform_container_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """Transform the container file content response.
        
        Returns:
            bytes: The raw file content.
        """
        ...

    def get_error_class(
        self, 
        error_message: str, 
        status_code: int, 
        headers: dict | httpx.Headers,
    ) -> BaseLLMException:
        from ..chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

