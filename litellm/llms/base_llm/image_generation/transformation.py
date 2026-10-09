from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseImageGenerationConfig(ABC):
    @abstractmethod
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        pass

    @abstractmethod
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        pass

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        return {}

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        raise NotImplementedError(
            "ImageVariationConfig implementa 'transform_request_image_variation' for image variation models"
        )

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        raise NotImplementedError(
            "ImageVariationConfig implements 'transform_response_image_variation' for image variation models"
        )

    async def async_transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict[str, object],  # mutable-ok: mirrors the sync hook so the default passes the same objects
        optional_params: dict[str, object],  # mutable-ok: mirrors the sync hook so the default passes the same objects
        litellm_params: dict[str, object],  # mutable-ok: mirrors the sync hook so the default passes the same objects
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        return self.transform_image_generation_response(  # pyright: ignore[reportUnknownMemberType]  # sync hook signature is untyped
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

    def use_multipart_form_data(self) -> bool:
        """
        Returns True if this provider requires multipart/form-data instead of JSON.

        Override this method in subclasses that need form-data (e.g., Stability AI).
        """
        return False
