from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # Ark-native params are absent from the OpenAI literal

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import override

from litellm.exceptions import AuthenticationError, UnsupportedParamsError
from litellm.litellm_core_utils.url_utils import ensure_api_route, join_api_base
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.volcengine.common_utils import VolcEngineError
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

ARK_API_BASE: Final = "https://ark.cn-beijing.volces.com"

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
    bounding_box: dict[str, object] | None = None
    text_recognition: dict[str, object] | None = None
    error: dict[str, object] | None = None


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
        api_base or get_secret_str("ARK_API_BASE") or get_secret_str("VOLCENGINE_API_BASE") or ARK_API_BASE
    ).rstrip("/")
    if base.endswith("/images/generations"):
        return base
    return join_api_base(ensure_api_route(base, "api/v3"), "images/generations")


def ark_image_headers(headers: Mapping[str, object], api_key: str | None, model: str) -> dict[str, object]:
    resolved_api_key: Final[str | None] = (
        api_key or get_secret_str("ARK_API_KEY") or get_secret_str("VOLCENGINE_API_KEY")
    )
    if not resolved_api_key:
        raise AuthenticationError(
            message="ARK_API_KEY or VOLCENGINE_API_KEY is not set. Set it in the environment or pass api_key.",
            llm_provider="volcengine",
            model=model,
        )
    return {**headers, "Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"}


def _requests_several_images(requested_images: object) -> bool:
    return requested_images is not None and _IMAGE_COUNT.validate_python(requested_images) > 1


def ark_image_params(
    params: Mapping[str, object],
    model: str,
    drop_params: bool,
) -> Mapping[str, object]:
    if params.get("stream") and not drop_params:
        raise UnsupportedParamsError(
            status_code=400,
            message=f"stream is not supported for Volcengine image models ({model}). "
            + "Seedream 5.5 returns one JSON response; set `litellm.drop_params = True` to ignore stream.",
        )
    if _requests_several_images(params.get("n")) and not drop_params:
        raise UnsupportedParamsError(
            status_code=400,
            message=f"n > 1 is not supported for Volcengine image models ({model}). "
            + "Use sequential_image_generation for several images, or set `litellm.drop_params = True` to ignore n.",
        )
    mapped: Final = {key: value for key, value in params.items() if key != "user" and value is not None}
    user: Final = params.get("user")
    if isinstance(user, str) and user and "safety_identifier" not in mapped:
        return {**mapped, "safety_identifier": user}
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
) -> dict[str, object]:
    extra_body: Final = _JSON_OBJECT.validate_python(params.get("extra_body") or {})
    mapped: Final = {
        key: _form_value(key, value)
        for key, value in params.items()
        if key not in _REQUEST_ONLY_PARAMS and value is not None
    }
    return {
        **mapped,
        **extra_body,
        "model": model,
        **({"prompt": prompt} if prompt is not None else {}),
    }


def _provider_specific_fields(image: _ArkImage) -> dict[str, object] | None:
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
        usage=ImageUsage(
            input_tokens=0,
            input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=0),
            output_tokens=usage.output_tokens or 0,
            total_tokens=usage.total_tokens or 0,
        ),
    )


class VolcEngineImageGenerationConfig(BaseImageGenerationConfig):
    @override
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return [cast(OpenAIImageGenerationOptionalParams, param) for param in ARK_IMAGE_PARAMS]

    @override
    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
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
    ) -> dict[str, object]:
        return ark_image_headers(headers, api_key, model)

    @override
    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:
        return ark_image_request(model, prompt, optional_params)

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
