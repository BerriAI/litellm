import base64
from collections.abc import Mapping
from io import BufferedReader, BytesIO
from typing import IO, TYPE_CHECKING, Final, Protocol, TypeAlias, runtime_checkable

import httpx
from httpx._types import RequestFiles
from typing_extensions import TypeIs  # noqa: TID251  # narrows untyped reference images without a runtime conversion

from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.xai.common_utils import XAIModelInfo, xai_image_response
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_Params: TypeAlias = dict[str, object]
_Headers: TypeAlias = dict[str, object]
_ImageRef: TypeAlias = dict[str, str]
_SupportedParams: TypeAlias = list[str]
_ImageInputs: TypeAlias = list[FileTypes | None]

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
_OPENAI_PARAMS: Final = ("n", "response_format", "size", "user")
_XAI_NATIVE_PARAMS: Final = frozenset({"aspect_ratio", "n", "resolution"})


@runtime_checkable
class _Readable(Protocol):
    def read(self) -> bytes | str: ...


def _is_json_object(value: object) -> TypeIs[Mapping[str, object]]:  # guard-ok: JSON-style references have str keys
    return isinstance(value, Mapping)


def _is_seekable_binary(value: object) -> TypeIs[IO[bytes]]:  # guard-ok: both classes are binary IO[bytes] streams
    return isinstance(value, (BytesIO, BufferedReader))


def _read_seekable(image: IO[bytes]) -> bytes:
    current_pos: Final = image.tell()
    image.seek(0)
    data: Final = image.read()
    image.seek(current_pos)
    return data


def _count(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float, str)) else None


class XAIImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(self, model: str) -> _SupportedParams:
        return list(_OPENAI_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> _Params:
        supported: Final = frozenset(_OPENAI_PARAMS)
        allowed: Final = supported | _XAI_NATIVE_PARAMS
        raw: Final = image_edit_optional_params
        incoming: Final[Mapping[str, object]] = dict(raw)
        unknown: Final = tuple(key for key in incoming if key not in allowed)
        if unknown and not drop_params:
            raise ValueError(
                f"Parameter {unknown[0]} is not supported for model {model}. "
                f"Supported parameters are {sorted(allowed)}. "
                "Set drop_params=True to drop unsupported parameters."
            )

        pairs: Final = ((key, value) for key, value in incoming.items() if key in allowed)
        mapped: Final = dict(pairs)
        size: Final = mapped.get("size")
        aspect_ratio: Final = mapped.get("aspect_ratio") or (
            _SIZE_TO_ASPECT_RATIO.get(str(size), "1:1") if size else None
        )
        fields: Final = (
            ("aspect_ratio", aspect_ratio),
            ("n", _count(mapped.get("n"))),
            ("resolution", mapped.get("resolution")),
        )
        return {key: value for key, value in fields if value is not None}

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: _Params,
    ) -> str:
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        api_key: Final = litellm_params.get("api_key") if litellm_params else None
        resolved_base: Final = (
            XAIOAuthAuthenticator().get_api_base()
            if should_use_xai_oauth(litellm_params)
            and not XAIModelInfo.get_api_key(api_key if isinstance(api_key, str) else None)
            else (api_base or get_secret_str("XAI_API_BASE") or get_secret_str("XAI_OAUTH_API_BASE") or XAI_API_BASE)
        )
        base: Final = (resolved_base or XAI_API_BASE).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/images/edits"
        return f"{base}/v1/images/edits"

    def _bearer_token(self, model: str, api_key: str | None, litellm_params: _Params) -> str:
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        dynamic_api_key: Final = XAIModelInfo.get_api_key(api_key)
        if dynamic_api_key:
            return dynamic_api_key
        if not should_use_xai_oauth(litellm_params):
            raise AuthenticationError(
                model=model,
                llm_provider="xai",
                message="Missing xAI credentials for image edit. Pass api_key / XAI_API_KEY, or set use_xai_oauth=True.",
            )
        try:
            return XAIOAuthAuthenticator().get_access_token()
        except XAIOAuthError as exc:
            raise AuthenticationError(model=model, llm_provider="xai", message=str(exc)) from exc

    def validate_environment(
        self,
        headers: _Headers,
        model: str,
        api_key: str | None = None,
        litellm_params: _Params | None = None,
        api_base: str | None = None,
    ) -> _Headers:
        has_content_type: Final = "content-type" in headers or "Content-Type" in headers
        return {
            **headers,
            "Authorization": f"Bearer {self._bearer_token(model, api_key, litellm_params or {})}",
            **({} if has_content_type else {"Content-Type": "application/json"}),
        }

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | _ImageInputs | None,
        image_edit_optional_request_params: _Params,
        litellm_params: GenericLiteLLMParams,
        headers: _Headers,
    ) -> tuple[_Params, RequestFiles]:
        image_payloads: Final = tuple(self._to_image_url(item) for item in self._as_image_list(image))
        if not image_payloads:
            raise ValueError("xAI image edit requires at least one reference image.")

        n: Final = _count(image_edit_optional_request_params.get("n"))
        prompt_field: Final = {"prompt": prompt} if prompt is not None else {}
        image_field: Final = (
            {"image": image_payloads[0]} if len(image_payloads) == 1 else {"images": list(image_payloads)}
        )
        request: Final[_Params] = {
            "model": XAIModelInfo.get_base_model(model) or model,
            **prompt_field,
            **image_field,
            **{
                key: image_edit_optional_request_params[key]
                for key in ("aspect_ratio", "resolution")
                if image_edit_optional_request_params.get(key) is not None
            },
            **({"n": n} if n is not None else {}),
        }
        return request, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        return ImageResponse(data=list(xai_image_response(raw_response, "image edit")))

    def _as_image_list(self, image: FileTypes | _ImageInputs | None) -> tuple[FileTypes, ...]:
        if isinstance(image, list):
            return tuple(item for item in image if item is not None)
        return () if image is None else (image,)

    def _to_image_url(self, image: FileTypes) -> _ImageRef:
        if isinstance(image, str):
            return {"url": image}
        if _is_json_object(image):
            url: Final = image.get("url")
            if isinstance(url, str) and url:
                return {"url": url}
            file_id: Final = image.get("file_id")
            if isinstance(file_id, str) and file_id:
                return {"file_id": file_id}

        mime: Final = ImageEditRequestUtils.get_image_content_type(image)
        encoded: Final = base64.b64encode(self._read_all_bytes(image)).decode("utf-8")
        return {"url": f"data:{mime};base64,{encoded}"}

    def _read_all_bytes(self, image: FileTypes) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, bytearray):
            return bytes(image)
        if _is_seekable_binary(image):
            return _read_seekable(image)
        if isinstance(image, _Readable):
            raw: Final = image.read()
            if isinstance(raw, str):
                return raw.encode("utf-8")
            return bytes(raw)
        raise ValueError(f"Unsupported image input type for xAI image edit: {type(image)}")
