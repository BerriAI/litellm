"""
Fal AI Image Edit Configuration

Handles transformation between OpenAI-compatible format and Fal AI API format
for image editing endpoints (nano-banana-2/edit, gemini-3-pro-image-preview/edit, etc.).

Fal AI edit endpoints accept JSON with prompt + image_url fields and return
{"images": [{"url": "..."}]} synchronously via fal.run — no polling required.
"""

import base64
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


DEFAULT_BASE_URL = "https://fal.run"


class FalAIImageEditConfig(BaseImageEditConfig):
    """
    Configuration for Fal AI image editing.

    Supports any Fal AI model that accepts edit-style requests
    (prompt + image_url in JSON body). URL is constructed dynamically
    from the model name: fal_ai/{path} -> https://fal.run/fal-ai/{path}

    HTTP requests are handled by the generic llm_http_handler (no polling needed).
    This class only handles data transformation.
    """

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "n",
            "size",
            "response_format",
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        optional_params: Dict[str, Any] = {}
        params_dict = dict(image_edit_optional_params)

        if params_dict.get("n") is not None:
            optional_params["num_images"] = params_dict["n"]

        if params_dict.get("size") is not None:
            optional_params["image_size"] = self._map_image_size(params_dict["size"])

        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        final_api_key: Optional[str] = api_key or get_secret_str("FAL_AI_API_KEY")
        if not final_api_key:
            raise ValueError("FAL_AI_API_KEY is not set")

        headers["Authorization"] = f"Key {final_api_key}"
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        return headers

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base_url: str = (
            api_base or get_secret_str("FAL_AI_API_BASE") or DEFAULT_BASE_URL
        )
        base_url = base_url.rstrip("/")
        # model arrives without provider prefix (e.g. "nano-banana-2/edit")
        # FAL API expects fal-ai/ prefix in the URL path
        # Strip fal-ai/ if already present to avoid double prefix
        model_path = model.removeprefix("fal-ai/")
        return f"{base_url}/fal-ai/{model_path}"

    def _read_image_bytes(
        self,
        image: Any,
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_RECURSE_DEPTH,
    ) -> bytes:
        """Read image bytes from various input types."""
        if depth > max_depth:
            raise ValueError(
                f"Max recursion depth {max_depth} reached while reading image bytes."
            )
        if isinstance(image, bytes):
            return image
        elif isinstance(image, list):
            return self._read_image_bytes(
                image[0], depth=depth + 1, max_depth=max_depth
            )
        elif isinstance(image, str):
            if image.startswith(("http://", "https://")):
                response = httpx.get(image, timeout=60.0)
                response.raise_for_status()
                return response.content
            else:
                with open(image, "rb") as f:
                    return f.read()
        elif hasattr(image, "read"):
            pos = getattr(image, "tell", lambda: 0)()
            if hasattr(image, "seek"):
                image.seek(0)
            data = image.read()
            if hasattr(image, "seek"):
                image.seek(pos)
            return data
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Expected bytes, str (URL or file path), or file-like object."
            )

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
        Transform OpenAI-style request to Fal AI request format.

        Fal AI edit endpoints accept JSON with image_url (data URI) + prompt.
        """
        request_body: Dict[str, Any] = {}

        if prompt is not None:
            request_body["prompt"] = prompt

        # Encode the input image as a data URI for Fal AI
        if image is not None:
            image_bytes = self._read_image_bytes(image)
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64_image}"
            request_body["image_url"] = data_uri

        # Pass through mapped optional params
        for key, value in image_edit_optional_request_params.items():
            if key not in ("extra_headers",) and value is not None:
                request_body[key] = value

        # Fal AI uses JSON, not multipart
        return request_body, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        """
        Transform Fal AI response to OpenAI-compatible ImageResponse.

        Fal AI returns: {"images": [{"url": "..."}]}
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise ValueError(
                f"Error parsing Fal AI response (status={raw_response.status_code}): {e}"
            )

        image_objects: List[ImageObject] = []
        images = response_data.get("images", [])
        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict):
                    image_objects.append(
                        ImageObject(
                            url=img.get("url"),
                            b64_json=img.get("b64_json"),
                        )
                    )
                elif isinstance(img, str):
                    image_objects.append(ImageObject(url=img))

        # Some Fal models return a single "image" instead of "images"
        if not image_objects:
            single_image = response_data.get("image")
            if isinstance(single_image, dict):
                image_objects.append(
                    ImageObject(
                        url=single_image.get("url"),
                        b64_json=single_image.get("b64_json"),
                    )
                )
            elif isinstance(single_image, str):
                image_objects.append(ImageObject(url=single_image))

        if not image_objects:
            raise ValueError(f"No images in Fal AI response: {response_data}")

        return ImageResponse(
            created=int(time.time()),
            data=image_objects,
        )

    @staticmethod
    def _map_image_size(size: Any) -> Any:
        """Map OpenAI size format (e.g. '1024x1024') to Fal AI image_size."""
        size_map = {
            "1024x1024": "square_hd",
            "512x512": "square",
            "1792x1024": "landscape_16_9",
            "1024x1792": "portrait_16_9",
            "1024x768": "landscape_4_3",
            "768x1024": "portrait_4_3",
        }
        if isinstance(size, str) and size in size_map:
            return size_map[size]
        if isinstance(size, str) and "x" in size:
            try:
                w, h = size.split("x")
                return {"width": int(w), "height": int(h)}
            except (ValueError, AttributeError):
                pass
        return size
