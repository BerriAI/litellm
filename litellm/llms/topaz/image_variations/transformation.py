import base64
import time
from io import BytesIO
from typing import Any, List, Mapping, Optional, Tuple, Union

from aiohttp import ClientResponse
from httpx import Headers, Response

from litellm.llms.base_llm.chat.transformation import (
    BaseLLMException,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageVariationOptionalParams,
)
from litellm.types.utils import (
    FileTypes,
    HttpHandlerRequestFields,
    ImageObject,
    ImageResponse,
)

from ...base_llm.image_variations.transformation import BaseImageVariationConfig
from ..common_utils import TopazException


class TopazImageVariationConfig(BaseImageVariationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageVariationOptionalParams]:
        return ["response_format", "size"]

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            raise ValueError(
                "API key is required for Topaz image variations. Set via `TOPAZ_API_KEY` or `api_key=..`"
            )
        return {
            # "Content-Type": "multipart/form-data",
            "Accept": "image/jpeg",
            "X-API-Key": api_key,
        }

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = api_base or "https://api.topazlabs.com"
        return f"{api_base}/image/v1/enhance"

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "response_format":
                optional_params["output_format"] = v
            elif k == "size":
                split_v = v.split("x")
                assert len(split_v) == 2, "size must be in the format of widthxheight"
                optional_params["output_width"] = split_v[0]
                optional_params["output_height"] = split_v[1]
        return optional_params

    def prepare_file_tuple(
        self,
        file_data: FileTypes,
    ) -> Tuple[str, Optional[FileTypes], str, Mapping[str, str]]:
        """
        Convert various file input formats to a consistent tuple format for HTTPX
        Returns: (filename, file_content, content_type, headers)
        """
        # Default values
        filename = "image.png"
        content: Optional[FileTypes] = None
        content_type = "image/png"
        headers: Mapping[str, str] = {}

        if isinstance(file_data, (bytes, BytesIO)):
            # Case 1: Just file content
            content = file_data
        elif isinstance(file_data, tuple):
            if len(file_data) == 2:
                # Case 2: (filename, content)
                filename = file_data[0] or filename
                content = file_data[1]
            elif len(file_data) == 3:
                # Case 3: (filename, content, content_type)
                filename = file_data[0] or filename
                content = file_data[1]
                content_type = file_data[2] or content_type
            elif len(file_data) == 4:
                # Case 4: (filename, content, content_type, headers)
                filename = file_data[0] or filename
                content = file_data[1]
                content_type = file_data[2] or content_type
                headers = file_data[3]

        return (filename, content, content_type, headers)

    def transform_request_image_variation(
        self,
        model: Optional[str],
        image: FileTypes,
        optional_params: dict,
        headers: dict,
    ) -> HttpHandlerRequestFields:

        request_params = HttpHandlerRequestFields(
            files={"image": self.prepare_file_tuple(image)},
            data=optional_params,
        )

        return request_params

    def _common_transform_response_image_variation(
        self,
        image_content: bytes,
        response_ms: float,
    ) -> ImageResponse:

        # Convert to base64
        base64_image = base64.b64encode(image_content).decode("utf-8")

        return ImageResponse(
            created=int(time.time()),
            data=[
                ImageObject(
                    b64_json=base64_image,
                    url=None,
                    revised_prompt=None,
                )
            ],
            response_ms=response_ms,
        )

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
        image_content = await raw_response.read()

        response_ms = logging_obj.get_response_ms()

        return self._common_transform_response_image_variation(
            image_content, response_ms
        )

    def transform_response_image_variation(
        self,
        model: Optional[str],
        raw_response: Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        image: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
    ) -> ImageResponse:
        image_content = raw_response.content

        response_ms = (
            raw_response.elapsed.total_seconds() * 1000
        )  # Convert to milliseconds

        return self._common_transform_response_image_variation(
            image_content, response_ms
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return TopazException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
