import base64
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import httpx
from httpx._types import RequestFiles

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


class FalAIImageEditConfig(BaseImageEditConfig):

    DEFAULT_BASE_URL: str = "https://fal.run"

    def get_supported_openai_params(self, model: str) -> list:
        return ["n", "size", "quality"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped: Dict[str, Any] = {}

        if "n" in image_edit_optional_params:
            mapped["num_images"] = image_edit_optional_params["n"]

        if "size" in image_edit_optional_params:
            size_str = image_edit_optional_params["size"]
            if isinstance(size_str, str) and "x" in size_str:
                w, h = size_str.split("x")
                mapped["image_size"] = {"width": int(w), "height": int(h)}

        if "quality" in image_edit_optional_params:
            mapped["quality"] = image_edit_optional_params["quality"]

        return mapped

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
    ) -> dict:
        final_api_key: Optional[str] = api_key or get_secret_str("FAL_AI_API_KEY")
        if not final_api_key:
            raise ValueError("FAL_AI_API_KEY is not set")

        headers["Authorization"] = f"Key {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base_url = (
            api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL
        )
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/edit"):
            base_url = f"{base_url}/edit"
        return base_url

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        image_urls = self._resolve_image_urls(image)

        body: Dict[str, Any] = {}
        if prompt is not None:
            body["prompt"] = prompt
        if image_urls:
            body["image_urls"] = image_urls

        body["num_images"] = image_edit_optional_request_params.pop("num_images", 1)

        if "image_size" in image_edit_optional_request_params:
            body["image_size"] = image_edit_optional_request_params.pop("image_size")

        if "quality" in image_edit_optional_request_params:
            body["quality"] = image_edit_optional_request_params.pop("quality")

        for k, v in image_edit_optional_request_params.items():
            if k not in body and not k.startswith("_") and v is not None:
                body[k] = v

        empty_files = cast(RequestFiles, [])
        return body, empty_files

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        try:
            result = raw_response.json()
        except Exception as exc:
            raise self.get_error_class(
                error_message=f"Error parsing fal.ai image edit response: {exc}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        images = result.get("images", [])
        data_list: List[ImageObject] = []
        for img in images:
            if isinstance(img, dict):
                data_list.append(
                    ImageObject(
                        url=img.get("url"),
                        b64_json=img.get("b64_json"),
                    )
                )
            elif isinstance(img, str):
                data_list.append(ImageObject(url=img, b64_json=None))

        model_response = ImageResponse()
        model_response.data = data_list  # type: ignore[assignment]
        return model_response

    @staticmethod
    def _resolve_image_urls(
        image: Optional[Union[FileTypes, List[FileTypes]]],
    ) -> List[str]:
        if image is None:
            return []

        images: List[Any] = image if isinstance(image, list) else [image]
        urls: List[str] = []

        for img in images:
            if img is None:
                continue
            if isinstance(img, str):
                urls.append(img)
            elif isinstance(img, bytes):
                b64 = base64.b64encode(img).decode("utf-8")
                urls.append(f"data:image/png;base64,{b64}")
            elif isinstance(img, BytesIO):
                pos = img.tell()
                img.seek(0)
                raw = img.read()
                img.seek(pos)
                b64 = base64.b64encode(raw).decode("utf-8")
                del raw
                urls.append(f"data:image/png;base64,{b64}")
            elif isinstance(img, BufferedReader):
                pos = img.tell()
                img.seek(0)
                raw = img.read()
                img.seek(pos)
                b64 = base64.b64encode(raw).decode("utf-8")
                del raw
                urls.append(f"data:image/png;base64,{b64}")
            elif isinstance(img, tuple):
                file_bytes = img[1] if len(img) > 1 else b""
                content_type = img[2] if len(img) > 2 else "image/png"
                if isinstance(file_bytes, bytes):
                    b64 = base64.b64encode(file_bytes).decode("utf-8")
                    urls.append(f"data:{content_type};base64,{b64}")

        return urls
