from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateFileRequest,
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
    ) -> Union[dict, str, bytes]:
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
