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
    """
    Base configuration for OpenAI image edit API.
    Used for models like gpt-image-1 that support multiple images.
    """

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

    def _add_image_to_files(
        self,
        files_list: List[Tuple[str, Any]],
        image: Any,
        field_name: str,
    ) -> None:
        """Add an image to the files list with appropriate content type"""
        image_content_type = ImageEditRequestUtils.get_image_content_type(image)

        if isinstance(image, BufferedReader):
            files_list.append((field_name, (image.name, image, image_content_type)))
        else:
            files_list.append((field_name, ("image.png", image, image_content_type)))

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        """
        Transform image edit request to OpenAI API format.

        Handles multipart/form-data for images. Uses "image[]" field name
        to support multiple images (e.g., for gpt-image-1).
        """
        # Build request params, only including non-None values
        request_params = {
            "model": model,
            **image_edit_optional_request_params,
        }
        if image is not None:
            request_params["image"] = image
        if prompt is not None:
            request_params["prompt"] = prompt
            
        request = ImageEditRequestParams(**request_params)
        request_dict = cast(Dict, request)

        #########################################################
        # Separate images and masks as `files` and send other parameters as `data`
        #########################################################
        _image_list = request_dict.get("image")
        _mask = request_dict.get("mask")
        data_without_files = {
            k: v for k, v in request_dict.items() if k not in ["image", "mask"]
        }
        files_list: List[Tuple[str, Any]] = []

        # Handle image parameter
        if _image_list is not None:
            image_list = (
                [_image_list] if not isinstance(_image_list, list) else _image_list
            )

            for _image in image_list:
                if _image is not None:
                    self._add_image_to_files(
                        files_list=files_list,
                        image=_image,
                        field_name="image[]",
                    )
        # Handle mask parameter if provided
        if _mask is not None:
            # Handle case where mask can be a list (extract first mask)
            if isinstance(_mask, list):
                _mask = _mask[0] if _mask else None

            if _mask is not None:
                mask_content_type: str = ImageEditRequestUtils.get_image_content_type(
                    _mask
                )
                if isinstance(_mask, BufferedReader):
                    files_list.append(("mask", (_mask.name, _mask, mask_content_type)))
                else:
                    files_list.append(("mask", ("mask.png", _mask, mask_content_type)))

        return data_without_files, files_list

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
