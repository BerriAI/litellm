from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx
from openai.types.file_deleted import FileDeleted

from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.files import TwoStepFileUploadConfig
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateFileRequest,
    FileContentRequest,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
    OpenAIFilesPurpose,
)
from litellm.types.utils import LlmProviders, ModelResponse

from ..chat.transformation import BaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.router import Router as _Router
    from litellm.types.llms.openai import HttpxBinaryResponseContent

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    Span = Any
    Router = _Router
else:
    LiteLLMLoggingObj = Any
    Span = Any
    Router = Any


class BaseFilesConfig(BaseConfig):
    @property
    @abstractmethod
    def custom_llm_provider(self) -> LlmProviders:
        pass

    @property
    def file_upload_http_method(self) -> str:
        """
        HTTP method to use for file uploads.
        Override this in provider configs if they need different methods.
        Default is POST (used by most providers like OpenAI, Anthropic).
        S3-based providers like Bedrock should return "PUT".
        """
        return "POST"

    @abstractmethod
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAICreateFileRequestOptionalParams]:
        pass

    def get_complete_file_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        data: CreateFileRequest,
    ):
        return self.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

    @abstractmethod
    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[dict, str, bytes, "TwoStepFileUploadConfig"]:
        """
        Transform OpenAI-style file creation request into provider-specific format.
        
        Returns:
            - dict: For pre-signed single-step uploads (e.g., Bedrock S3)
            - str/bytes: For traditional file uploads
            - TwoStepFileUploadConfig: For two-step upload process (e.g., Manus, GCS)
        """
        pass

    @abstractmethod
    def transform_create_file_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        pass

    @abstractmethod
    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """Transform file retrieve request into provider-specific format."""
        pass

    @abstractmethod
    def transform_retrieve_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """Transform file retrieve response into OpenAI format."""
        pass

    @abstractmethod
    def transform_delete_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """Transform file delete request into provider-specific format."""
        pass

    @abstractmethod
    def transform_delete_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> "FileDeleted":
        """Transform file delete response into OpenAI format."""
        pass

    @abstractmethod
    def transform_list_files_request(
        self,
        purpose: Optional[str],
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """Transform file list request into provider-specific format."""
        pass

    @abstractmethod
    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> List[OpenAIFileObject]:
        """Transform file list response into OpenAI format."""
        pass

    @abstractmethod
    def transform_file_content_request(
        self,
        file_content_request: "FileContentRequest",
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """Transform file content request into provider-specific format."""
        pass

    @abstractmethod
    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> "HttpxBinaryResponseContent":
        """Transform file content response into OpenAI format."""
        pass

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        raise NotImplementedError(
            "AudioTranscriptionConfig does not need a request transformation for audio transcription models"
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        raise NotImplementedError(
            "AudioTranscriptionConfig does not need a response transformation for audio transcription models"
        )


class BaseFileEndpoints(ABC):
    @abstractmethod
    async def acreate_file(
        self,
        create_file_request: CreateFileRequest,
        llm_router: Router,
        target_model_names_list: List[str],
        litellm_parent_otel_span: Span,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> OpenAIFileObject:
        pass

    @abstractmethod
    async def afile_retrieve(
        self,
        file_id: str,
        litellm_parent_otel_span: Optional[Span],
        llm_router: Optional[Router] = None,
    ) -> OpenAIFileObject:
        pass

    @abstractmethod
    async def afile_list(
        self,
        purpose: Optional[OpenAIFilesPurpose],
        litellm_parent_otel_span: Optional[Span],
        **data: Dict,
    ) -> List[OpenAIFileObject]:
        pass

    @abstractmethod
    async def afile_delete(
        self,
        file_id: str,
        litellm_parent_otel_span: Optional[Span],
        llm_router: Router,
        **data: Dict,
    ) -> OpenAIFileObject:
        pass

    @abstractmethod
    async def afile_content(
        self,
        file_id: str,
        litellm_parent_otel_span: Optional[Span],
        llm_router: Router,
        **data: Dict,
    ) -> "HttpxBinaryResponseContent":
        pass
