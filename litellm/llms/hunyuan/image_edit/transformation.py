"""
Tencent Hunyuan Image Edit Configuration

Handles transformation between OpenAI-compatible format and Hunyuan API format
for image editing. Uses the same submit+poll endpoints as image generation,
with the addition of the `images` and `mask` fields.

Submit:  POST https://api.cloudai.tencent.com/v1/aiart/openai/image/submit
Query:   POST https://api.cloudai.tencent.com/v1/aiart/openai/image/query

HTTP requests and polling are handled by handler.py.
This class only handles data transformation.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse

from ..image_generation.transformation import (
    HUNYUAN_BASE_URL,
    HUNYUAN_QUERY_ENDPOINT,
    HUNYUAN_SUBMIT_ENDPOINT,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _image_to_url(image: Any) -> str:
    """Convert a FileTypes value to a URL string accepted by Hunyuan.

    Hunyuan's OpenAI-compatible submit endpoint accepts URL strings only.
    Bytes and file-like objects are not supported.
    """
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            return image
        raise ValueError(
            f"Hunyuan image edit requires HTTP/HTTPS URL strings as images, "
            f"got: {image[:80]!r}. Pass publicly accessible image URLs."
        )
    raise TypeError(
        f"Hunyuan image edit only supports URL strings as input images, "
        f"got {type(image).__name__}. Pass publicly accessible HTTP/HTTPS URLs."
    )


class HunyuanImageEditConfig(BaseImageEditConfig):
    """
    Configuration for Tencent Hunyuan image editing.

    Uses the same submit+poll flow as image generation; the difference is that
    the request body may include `images` (list of URL strings) and `mask` (URL).
    """

    def use_multipart_form_data(self) -> bool:
        return False

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "n",
            "quality",
            "size",
            "mask",
            "background",
            "output_format",
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        optional_params: Dict[str, Any] = {}
        supported = set(self.get_supported_openai_params(model))
        params_dict = dict(image_edit_optional_params)
        for k, v in params_dict.items():
            if v is not None and k in supported:
                optional_params[k] = v
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
    ) -> dict:
        final_api_key: Optional[str] = api_key or get_secret_str("HUNYUAN_API_KEY")
        if not final_api_key:
            raise ValueError("HUNYUAN_API_KEY is not set")
        headers["Authorization"] = final_api_key
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base = api_base or get_secret_str("HUNYUAN_API_BASE") or HUNYUAN_BASE_URL
        base = base.rstrip("/")
        return f"{base}/{HUNYUAN_SUBMIT_ENDPOINT}"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        request_body: Dict[str, Any] = {
            "model": model or "gpt-image-2",
        }
        if prompt:
            request_body["prompt"] = prompt

        if image is not None:
            images = image if isinstance(image, list) else [image]
            request_body["images"] = [_image_to_url(img) for img in images]

        for k, v in image_edit_optional_request_params.items():
            if k == "mask" and v is not None:
                request_body["mask"] = _image_to_url(v)
            elif k not in ("mask",) and v is not None:
                request_body[k] = v

        return request_body, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=f"Error parsing Hunyuan image edit response: {e}",
            )

        image_objects = [
            ImageObject(url=item.get("url"), b64_json=item.get("b64_json"))
            for item in response_data.get("data", [])
            if isinstance(item, dict)
        ]

        return ImageResponse(
            created=int(time.time()),
            data=image_objects,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
        )
