"""
Tencent Hunyuan GPT-Maas Image Edit Configuration

API: POST https://tokenhub.tencentmaas.com/v1/aiart/gtimage
Auth: Authorization: Bearer <API_KEY>
Response: synchronous, no polling required.

Accepts input images and an optional mask.  Status field: completed = success,
failed = failure.
"""

import base64
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
    HUNYUAN_GPT_MAAS_BASE_URL,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

HUNYUAN_GPT_MAAS_IMAGE_ENDPOINT = "v1/aiart/gtimage"


def _bytes_to_data_url(data: bytes) -> str:
    """Encode raw image bytes as a base64 data URL."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif data[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _image_to_url(image: Any) -> str:
    """Convert a FileTypes value to a URL or base64 data URL."""
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            return image
        raise ValueError(
            f"Hunyuan GPT-Maas image edit: string image must be an HTTP/HTTPS URL or "
            f"data URL, got: {image[:80]!r}."
        )
    if isinstance(image, bytes):
        return _bytes_to_data_url(image)
    if hasattr(image, "read"):
        raw = image.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return _bytes_to_data_url(raw)
    if isinstance(image, tuple) and len(image) >= 2:
        file_content = image[1]
        if isinstance(file_content, bytes):
            return _bytes_to_data_url(file_content)
        if hasattr(file_content, "read"):
            raw = file_content.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            return _bytes_to_data_url(raw)
        if isinstance(file_content, str):
            return _bytes_to_data_url(file_content.encode("utf-8"))
    raise TypeError(
        f"Hunyuan GPT-Maas image edit: unsupported image type {type(image).__name__}. "
        "Pass an HTTP/HTTPS URL string, bytes, a file-like object, or an httpx-style tuple."
    )


def _image_to_param(image: Any) -> Dict[str, str]:
    """Convert image to the ImageRef format expected by the GPT-Maas API."""
    url = _image_to_url(image)
    return {"image_url": url}


class HunyuanGptMaasImageEditConfig(BaseImageEditConfig):
    """
    Configuration for Tencent Hunyuan GPT-Maas image editing.

    Uses a single synchronous POST to /v1/aiart/gtimage.
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
        for k, v in dict(image_edit_optional_params).items():
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
        final_api_key: Optional[str] = api_key or get_secret_str(
            "HUNYUAN_GPT_MAAS_API_KEY"
        )
        if not final_api_key:
            raise ValueError("HUNYUAN_GPT_MAAS_API_KEY is not set")
        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base = (
            api_base
            or get_secret_str("HUNYUAN_GPT_MAAS_API_BASE")
            or HUNYUAN_GPT_MAAS_BASE_URL
        )
        base = base.rstrip("/")
        return f"{base}/{HUNYUAN_GPT_MAAS_IMAGE_ENDPOINT}"

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
            "model": model or "custom-imagemodel-gt",
        }
        if prompt:
            request_body["prompt"] = prompt

        if image is not None:
            images = image if isinstance(image, list) else [image]
            request_body["images"] = [_image_to_param(img) for img in images]

        for k, v in image_edit_optional_request_params.items():
            if k == "mask" and v is not None:
                request_body["mask"] = _image_to_param(v)
            elif k == "n" and v is not None:
                request_body["n"] = int(v)
            elif k not in ("mask",) and v is not None:
                request_body[k] = v

        request_body.setdefault("logo_add", 0)

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
                message=f"Error parsing Hunyuan GPT-Maas image edit response: {e}",
            )

        status = response_data.get("status", "")
        if status == "failed":
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=f"Hunyuan GPT-Maas image edit failed: {response_data}",
            )

        image_objects = [
            ImageObject(url=item.get("url"), b64_json=item.get("b64_json"))
            for item in response_data.get("data", [])
            if isinstance(item, dict)
        ]

        return ImageResponse(
            created=response_data.get("created_at") or int(time.time()),
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
