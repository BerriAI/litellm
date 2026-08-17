import base64
from io import BufferedReader, BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.xai.common_utils import XAIModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse

_SIZE_TO_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1920x1080": "16:9",
    "1080x1920": "9:16",
}


class XAIImageEditConfig(BaseImageEditConfig):
    """xAI Imagine image edit (SuperGrok OAuth or API key). JSON POST /v1/images/edits."""

    def get_supported_openai_params(self, model: str) -> list:
        return ["n", "response_format", "size", "user"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        supported = set(self.get_supported_openai_params(model))
        xai_native = {"aspect_ratio", "n", "resolution"}
        mapped: Dict[str, Any] = {}
        for key, value in dict(image_edit_optional_params).items():
            if key in supported or key in xai_native:
                mapped[key] = value
            elif drop_params:
                continue
            else:
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. "
                    f"Supported parameters are {sorted(supported | xai_native)}. "
                    "Set drop_params=True to drop unsupported parameters."
                )

        size = mapped.pop("size", None)
        if size and "aspect_ratio" not in mapped:
            mapped["aspect_ratio"] = _SIZE_TO_ASPECT_RATIO.get(str(size), "1:1")
        mapped.pop("response_format", None)
        mapped.pop("user", None)
        if mapped.get("n") is not None:
            mapped["n"] = int(mapped["n"])
        return mapped

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        if should_use_xai_oauth(litellm_params) and not XAIModelInfo.get_api_key(
            litellm_params.get("api_key") if isinstance(litellm_params, dict) else None
        ):
            token_file = litellm_params.get("xai_oauth_token_file")
            api_base = XAIOAuthAuthenticator(auth_file=token_file).get_api_base()
        else:
            api_base = (
                api_base
                or get_secret_str("XAI_API_BASE")
                or get_secret_str("XAI_OAUTH_API_BASE")
                or XAI_API_BASE
            )

        base = (api_base or XAI_API_BASE).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/images/edits"
        return f"{base}/v1/images/edits"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        params = litellm_params or {}
        dynamic_api_key = XAIModelInfo.get_api_key(api_key)
        if should_use_xai_oauth(params) and not dynamic_api_key:
            token_file = params.get("xai_oauth_token_file")
            try:
                headers["Authorization"] = (
                    f"Bearer {XAIOAuthAuthenticator(auth_file=token_file).get_access_token()}"
                )
            except XAIOAuthError as exc:
                raise AuthenticationError(
                    model=model,
                    llm_provider="xai",
                    message=str(exc),
                ) from exc
        else:
            if not dynamic_api_key:
                raise AuthenticationError(
                    model=model,
                    llm_provider="xai",
                    message=(
                        "Missing xAI credentials for image edit. "
                        "Pass api_key / XAI_API_KEY, or set use_xai_oauth=True."
                    ),
                )
            headers["Authorization"] = f"Bearer {dynamic_api_key}"

        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return headers

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        if image is None:
            raise ValueError("xAI image edit requires at least one reference image.")

        image_payloads = [self._to_image_url(item) for item in self._as_image_list(image)]
        if not image_payloads:
            raise ValueError("xAI image edit requires at least one reference image.")

        request: Dict[str, Any] = {
            "model": XAIModelInfo.get_base_model(model) or model,
        }
        if prompt is not None:
            request["prompt"] = prompt
        if len(image_payloads) == 1:
            request["image"] = image_payloads[0]
        else:
            request["images"] = image_payloads

        for key in ("aspect_ratio", "resolution"):
            if key in image_edit_optional_request_params and image_edit_optional_request_params[key] is not None:
                request[key] = image_edit_optional_request_params[key]
        if image_edit_optional_request_params.get("n") is not None:
            request["n"] = int(image_edit_optional_request_params["n"])
        return request, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except Exception:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response = ImageResponse()
        model_response.data = []
        for item in response_data.get("data") or []:
            if not isinstance(item, dict):
                continue
            model_response.data.append(
                ImageObject(
                    url=item.get("url"),
                    b64_json=item.get("b64_json") or item.get("b64"),
                )
            )

        if not model_response.data:
            raise self.get_error_class(
                error_message=f"xAI image edit returned no image data: {response_data}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        return model_response

    def _as_image_list(self, image: Union[FileTypes, List[FileTypes]]) -> List[FileTypes]:
        if isinstance(image, list):
            return [item for item in image if item is not None]
        return [image]

    def _to_image_url(self, image: FileTypes) -> Dict[str, str]:
        if isinstance(image, str):
            return {"url": image}
        if isinstance(image, dict):
            url = image.get("url") or image.get("file_id")
            if url:
                return {"url": str(url)} if "url" in image or "file_id" not in image else {"file_id": str(url)}
            if image.get("file_id"):
                return {"file_id": str(image["file_id"])}

        mime = ImageEditRequestUtils.get_image_content_type(image)
        encoded = base64.b64encode(self._read_all_bytes(image)).decode("utf-8")
        return {"url": f"data:{mime};base64,{encoded}"}

    def _read_all_bytes(self, image: FileTypes) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, bytearray):
            return bytes(image)
        if isinstance(image, BytesIO):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if isinstance(image, BufferedReader):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if hasattr(image, "read"):
            data = image.read()
            if isinstance(data, str):
                return data.encode("utf-8")
            return bytes(data)
        raise ValueError(f"Unsupported image input type for xAI image edit: {type(image)}")
