from io import BufferedReader
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import (
    ImageEditOptionalRequestParams,
    ImageEditRequestParams,
)
from litellm.types.llms.openai import FileTypes
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ImageResponse

from ..common_utils import OpenAIError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenAIImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        All OpenAI Image Edits params are supported
        """
        return [
            "image",
            "prompt",
            "background",
            "mask",
            "model",
            "n",
            "quality",
            "response_format",
            "size",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """No mapping applied since inputs are in OpenAI spec already"""
        return dict(image_edit_optional_params)

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str,
        image: FileTypes,
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        """
        No transform applied since inputs are in OpenAI spec already

        This handles buffered readers as images to be sent as multipart/form-data for OpenAI
        """
        request = ImageEditRequestParams(
            model=model,
            image=image,
            prompt=prompt,
            **image_edit_optional_request_params,
        )
        request_dict = cast(Dict, request)

        #########################################################
        # Separate images as `files` and send other parameters as `data`
        #########################################################
        _images = request_dict.get("image") or []
        data_without_images = {k: v for k, v in request_dict.items() if k != "image"}
        files_list: List[Tuple[str, Any]] = []
        for _image in _images:
            image_content_type: str = ImageEditRequestUtils.get_image_content_type(
                _image
            )
            if isinstance(_image, BufferedReader):
                files_list.append(
                    ("image[]", (_image.name, _image, image_content_type))
                )
            else:
                files_list.append(
                    ("image[]", ("image.png", _image, image_content_type))
                )
        return data_without_images, files_list

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        """No transform applied since outputs are in OpenAI spec already"""
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise OpenAIError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        return ImageResponse(**raw_response_json)

    def validate_environment(
        self,
        headers: dict,
        model: str,
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
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the endpoint for OpenAI responses API
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/images/edits"
