import hashlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # Ark-native params are absent from the OpenAI literal

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import override

import litellm
from litellm.exceptions import AuthenticationError, UnsupportedParamsError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.volcengine.common_utils import VolcEngineError
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

ARK_DEFAULT_API_BASE: Final = "https://ark.cn-beijing.volces.com/api/v3"
ARK_SAFETY_IDENTIFIER_MAX_LENGTH: Final = 64

ARK_IMAGE_PARAMS: Final[tuple[str, ...]] = (
    "n",
    "size",
    "response_format",
    "output_format",
    "background",
    "user",
    "image",
    "watermark",
    "safety_identifier",
    "sequential_image_generation",
    "sequential_image_generation_options",
    "optimize_prompt_options",
    "layer_decomposition",
    "layer_decomposition_options",
    "tools",
    "stream",
)

_REQUEST_ONLY_PARAMS: Final = frozenset({"n", "user", "extra_headers", "extra_body", "stream"})
_UNSENT_PARAMS: Final = frozenset({"user", "stream"})
_UNSENT_EXTRA_BODY_FIELDS: Final = frozenset({"extra_headers", "stream"})
_PROVIDER_FIELDS: Final = frozenset({"url", "b64_json"})
_BOOLEAN_PARAMS: Final = frozenset({"watermark", "layer_decomposition"})
_JSON_PARAMS: Final = frozenset(
    {
        "sequential_image_generation_options",
        "optimize_prompt_options",
        "layer_decomposition_options",
        "tools",
    }
)
_FORM_BOOLEANS: Final = {"true": True, "false": False}

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_VALUE: Final = TypeAdapter(object)
_IMAGE_COUNT: Final = TypeAdapter(int)
_STREAM_FLAG: Final = TypeAdapter(bool)


class _ArkImage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    url: str | None = None
    b64_json: str | None = None
    size: str | None = None
    output_format: str | None = None
    background: str | None = None
    z_index: int | None = None
    name: str | None = None
    description: str | None = None
    bounding_box: Mapping[str, object] | None = None
    text_recognition: Mapping[str, object] | None = None
    error: Mapping[str, object] | None = None


class _ArkUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_images: int | None = None
    input_images: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class _ArkError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str | None = None
    message: str | None = None


class _ArkImageResponse(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    created: int | None = None
    model: str | None = None
    data: tuple[_ArkImage, ...] = ()
    usage: _ArkUsage | None = None
    error: _ArkError | None = None


_ARK_RESPONSE: Final = TypeAdapter(_ArkImageResponse)


def ark_image_generation_url(api_base: str | None) -> str:
    base: Final = (
        api_base or get_secret_str("VOLCENGINE_API_BASE") or get_secret_str("ARK_API_BASE") or ARK_DEFAULT_API_BASE
    ).rstrip("/")
    if base.endswith("/images/generations"):
        return base
    return join_api_base(base, "images/generations")


def _ark_api_key(api_key: str | None, model: str) -> str:
    resolved_api_key: Final[str | None] = (
        api_key or litellm.api_key or get_secret_str("VOLCENGINE_API_KEY") or get_secret_str("ARK_API_KEY")
    )
    if not resolved_api_key:
        raise AuthenticationError(
            message="VOLCENGINE_API_KEY or ARK_API_KEY is not set. Set it in the environment or pass api_key.",
            llm_provider="volcengine",
            model=model,
        )
    return resolved_api_key


def ark_image_headers(headers: Mapping[str, object], api_key: str | None, model: str) -> Mapping[str, object]:
    caller_sets_authorization: Final = any(name.lower() == "authorization" for name in headers)
    authorization: Final = (
        {} if caller_sets_authorization else {"Authorization": f"Bearer {_ark_api_key(api_key, model)}"}
    )
    return {**authorization, "Content-Type": "application/json", **headers}


def ark_extra_headers(optional_params: Mapping[str, object]) -> Mapping[str, object]:
    extra_body: Final = _JSON_OBJECT.validate_python(optional_params.get("extra_body") or {})
    return {
        **_JSON_OBJECT.validate_python(extra_body.get("extra_headers") or {}),
        **_JSON_OBJECT.validate_python(optional_params.get("extra_headers") or {}),
    }


def _requests_several_images(requested_images: object) -> bool:
    return requested_images is not None and _IMAGE_COUNT.validate_python(requested_images) > 1


def drop_params_requested(requested: object) -> bool:
    return litellm.drop_params is True or requested is True


def _stream_requested(stream: object) -> bool:
    try:
        return _STREAM_FLAG.validate_python(stream)
    except ValidationError:
        return False


def _reject_stream(stream: object, model: str, drop_params: bool) -> None:
    if _stream_requested(stream) and not drop_params:
        raise UnsupportedParamsError(
            status_code=400,
            message=f"stream is not supported for Volcengine image models ({model}). "
            + "LiteLLM does not parse Ark's image event stream. Set `litellm.drop_params = True` to send the "
            + "request without stream and get the finished images in one response.",
        )


def _safety_identifier(user: str) -> str:
    if len(user) <= ARK_SAFETY_IDENTIFIER_MAX_LENGTH:
        return user
    return hashlib.sha256(user.encode()).hexdigest()


def ark_image_params(
    params: Mapping[str, object],
    model: str,
    drop_params: bool,
) -> Mapping[str, object]:
    _reject_stream(params.get("stream"), model, drop_params)
    if _requests_several_images(params.get("n")) and not drop_params:
        raise UnsupportedParamsError(
            status_code=400,
            message=f"n > 1 is not supported for Volcengine image models ({model}). "
            + "Use sequential_image_generation for several images, or set `litellm.drop_params = True` to ignore n.",
        )
    mapped: Final = {key: value for key, value in params.items() if key not in _UNSENT_PARAMS and value is not None}
    user: Final = params.get("user")
    if isinstance(user, str) and user and "safety_identifier" not in mapped:
        return {**mapped, "safety_identifier": _safety_identifier(user)}
    return mapped


def _form_value(key: str, value: object) -> object:
    if key in _BOOLEAN_PARAMS and isinstance(value, str):
        return _FORM_BOOLEANS.get(value.lower(), value)
    if key in _JSON_PARAMS and isinstance(value, str):
        try:
            return _JSON_VALUE.validate_json(value)
        except ValidationError:
            return value
    return value


def ark_image_request(
    model: str,
    prompt: str | None,
    params: Mapping[str, object],
    drop_params: bool,
) -> Mapping[str, object]:
    extra_body: Final = _JSON_OBJECT.validate_python(params.get("extra_body") or {})
    _reject_stream(params.get("stream"), model, drop_params)
    _reject_stream(extra_body.get("stream"), model, drop_params)
    mapped: Final = {
        key: _form_value(key, value)
        for key, value in params.items()
        if key not in _REQUEST_ONLY_PARAMS and value is not None
    }
    return {
        **mapped,
        **{key: value for key, value in extra_body.items() if key not in _UNSENT_EXTRA_BODY_FIELDS},
        "model": model,
        **({"prompt": prompt} if prompt is not None else {}),
    }


def _provider_specific_fields(
    image: _ArkImage,
) -> dict[str, object] | None:  # mutable-ok: ImageObject.provider_specific_fields is typed as a dict
    dumped: Final = _JSON_OBJECT.validate_python(image.model_dump(exclude_none=True))
    fields: Final = {key: value for key, value in dumped.items() if key not in _PROVIDER_FIELDS}
    return fields or None


def ark_image_response(raw_response: httpx.Response) -> ImageResponse:
    try:
        parsed: Final = _ARK_RESPONSE.validate_json(raw_response.content)
    except ValidationError:
        raise VolcEngineError(
            status_code=raw_response.status_code,
            message=raw_response.text,
            headers=raw_response.headers,
        )

    if raw_response.status_code >= 400 or (parsed.error is not None and not parsed.data):
        message: Final = (
            parsed.error.message if parsed.error is not None and parsed.error.message else raw_response.text
        )
        raise VolcEngineError(
            status_code=raw_response.status_code,
            message=message,
            headers=raw_response.headers,
        )

    usage: Final = parsed.usage or _ArkUsage()
    first_image: Final = parsed.data[0] if parsed.data else _ArkImage()
    return ImageResponse(
        created=parsed.created,
        data=[
            ImageObject(
                url=image.url,
                b64_json=image.b64_json,
                provider_specific_fields=_provider_specific_fields(image),
            )
            for image in parsed.data
        ],
        size=first_image.size,
        output_format=first_image.output_format,
        background=first_image.background,
        usage=ImageUsage.model_validate(
            {
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": usage.output_tokens or 0,
                "prompt_tokens": 0,
                "completion_tokens": usage.output_tokens or 0,
                "total_tokens": usage.total_tokens or 0,
                "input_images": usage.input_images or 0,
                "generated_images": usage.generated_images or 0,
            }
        ),
    )


class VolcEngineImageGenerationConfig(BaseImageGenerationConfig):
    @override
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: inherited contract
        return [
            cast(OpenAIImageGenerationOptionalParams, param)  # cast-ok: Ark params are not in the OpenAI literal
            for param in ARK_IMAGE_PARAMS
        ]

    @override
    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return {
            **optional_params,
            **ark_image_params(non_default_params, model, drop_params),
        }

    @override
    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return ark_image_generation_url(api_base)

    @override
    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return dict(ark_image_headers({**headers, **ark_extra_headers(optional_params)}, api_key, model))

    @override
    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return dict(
            ark_image_request(model, prompt, optional_params, drop_params_requested(litellm_params.get("drop_params")))
        )

    @override
    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        return ark_image_response(raw_response)
