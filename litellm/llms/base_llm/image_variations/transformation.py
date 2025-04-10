from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

import httpx
from aiohttp import ClientResponse

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageVariationOptionalParams,
)
from litellm.types.utils import (
    FileTypes,
    HttpHandlerRequestFields,
    ImageResponse,
    ModelResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseImageVariationConfig(BaseConfig, ABC):
    @abstractmethod
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageVariationOptionalParams]:
        pass

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    @abstractmethod
    def transform_request_image_variation(
        self,
        model: Optional[str],
        image: FileTypes,
        optional_params: dict,
        headers: dict,
    ) -> HttpHandlerRequestFields:
        pass

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return {}

    @abstractmethod
    async def async_transform_response_image_variation(
        self,
        model: Optional[str],
        raw_response: ClientResponse,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        image: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
    ) -> ImageResponse:
        pass

    @abstractmethod
    def transform_response_image_variation(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        image: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
    ) -> ImageResponse:
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
            "ImageVariationConfig implementa 'transform_request_image_variation' for image variation models"
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
            "ImageVariationConfig implements 'transform_response_image_variation' for image variation models"
        )
