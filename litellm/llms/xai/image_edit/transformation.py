import base64
from io import BufferedReader, BytesIO
from typing import Any, Final

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

_SIZE_TO_ASPECT_RATIO: Final = {
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
_XAI_NATIVE_PARAMS: Final = frozenset({"aspect_ratio", "n", "resolution"})


def _read_seekable(image: BytesIO | BufferedReader) -> bytes:
    current_pos: Final = image.tell()
    image.seek(0)
    data: Final = image.read()
    image.seek(current_pos)
    return data


class XAIImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return ["n", "response_format", "size", "user"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported: Final = frozenset(self.get_supported_openai_params(model))
        allowed: Final = supported | _XAI_NATIVE_PARAMS
        incoming: Final = dict(image_edit_optional_params)
        unknown: Final = tuple(key for key in incoming if key not in allowed)
        if unknown and not drop_params:
            raise ValueError(
                f"Parameter {unknown[0]} is not supported for model {model}. "
                f"Supported parameters are {sorted(allowed)}. "
                "Set drop_params=True to drop unsupported parameters."
            )

        mapped: Final = {key: value for key, value in incoming.items() if key in allowed}
        size: Final = mapped.get("size")
        aspect_ratio: Final = mapped.get("aspect_ratio") or (
            _SIZE_TO_ASPECT_RATIO.get(str(size), "1:1") if size else None
        )
        n: Final = mapped.get("n")
        resolution: Final = mapped.get("resolution")
        return {
            **({"aspect_ratio": aspect_ratio} if aspect_ratio is not None else {}),
            **({"n": int(n)} if n is not None else {}),
            **({"resolution": resolution} if resolution is not None else {}),
        }

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        api_key: Final = litellm_params.get("api_key") if isinstance(litellm_params, dict) else None
        resolved_base: Final = (
            XAIOAuthAuthenticator().get_api_base()
            if should_use_xai_oauth(litellm_params) and not XAIModelInfo.get_api_key(api_key)
            else (
                api_base
                or get_secret_str("XAI_API_BASE")
                or get_secret_str("XAI_OAUTH_API_BASE")
                or XAI_API_BASE
            )
        )
        base: Final = (resolved_base or XAI_API_BASE).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/images/edits"
        return f"{base}/v1/images/edits"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        params: Final = litellm_params or {}
        dynamic_api_key: Final = XAIModelInfo.get_api_key(api_key)
        if should_use_xai_oauth(params) and not dynamic_api_key:
            try:
                headers["Authorization"] = f"Bearer {XAIOAuthAuthenticator().get_access_token()}"
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
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        if image is None:
            raise ValueError("xAI image edit requires at least one reference image.")

        image_payloads: Final = tuple(self._to_image_url(item) for item in self._as_image_list(image))
        if not image_payloads:
            raise ValueError("xAI image edit requires at least one reference image.")

        n: Final = image_edit_optional_request_params.get("n")
        request: Final[dict[str, Any]] = {
            "model": XAIModelInfo.get_base_model(model) or model,
            **({"prompt": prompt} if prompt is not None else {}),
            **(
                {"image": image_payloads[0]}
                if len(image_payloads) == 1
                else {"images": list(image_payloads)}
            ),
            **{
                key: image_edit_optional_request_params[key]
                for key in ("aspect_ratio", "resolution")
                if image_edit_optional_request_params.get(key) is not None
            },
            **({"n": int(n)} if n is not None else {}),
        }
        return request, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ImageResponse:
        try:
            response_data: Final = raw_response.json()
        except Exception:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        images: Final = tuple(
            ImageObject(
                url=item.get("url"),
                b64_json=item.get("b64_json") or item.get("b64"),
            )
            for item in response_data.get("data") or ()
            if isinstance(item, dict)
        )
        if not images:
            raise self.get_error_class(
                error_message=f"xAI image edit returned no image data: {response_data}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        return ImageResponse(data=list(images))

    def _as_image_list(self, image: FileTypes | list[FileTypes]) -> tuple[FileTypes, ...]:
        if isinstance(image, list):
            return tuple(item for item in image if item is not None)
        return (image,)

    def _to_image_url(self, image: FileTypes) -> dict[str, str]:
        if isinstance(image, str):
            return {"url": image}
        if isinstance(image, dict):
            if image.get("url"):
                return {"url": str(image["url"])}
            if image.get("file_id"):
                return {"file_id": str(image["file_id"])}

        mime: Final = ImageEditRequestUtils.get_image_content_type(image)
        encoded: Final = base64.b64encode(self._read_all_bytes(image)).decode("utf-8")
        return {"url": f"data:{mime};base64,{encoded}"}

    def _read_all_bytes(self, image: FileTypes) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, bytearray):
            return bytes(image)
        if isinstance(image, (BytesIO, BufferedReader)):
            return _read_seekable(image)
        if hasattr(image, "read"):
            raw: Final = image.read()
            if isinstance(raw, str):
                return raw.encode("utf-8")
            return bytes(raw)
        raise ValueError(f"Unsupported image input type for xAI image edit: {type(image)}")
