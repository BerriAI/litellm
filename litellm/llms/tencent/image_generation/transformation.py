from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import override

from litellm.exceptions import AuthenticationError, UnsupportedParamsError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
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

TOKENHUB_API_BASE: Final = "https://tokenhub-intl.tencentcloudmaas.com/v1"
TOKENHUB_IMAGE_PATH: Final = "/wand/si-image/generation"
TOKENHUB_IMAGE_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "n",
    "size",
    "response_format",
    "output_format",
    "background",
)

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))


class _TokenHubImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str | None = None
    b64_json: str | None = None


class _TokenHubUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_tokens: int


class _TokenHubImageResponse(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    created: int
    data: tuple[_TokenHubImage, ...]
    tokenhub_usage: _TokenHubUsage | None = None


def tokenhub_image_url(api_base: str | None) -> str:
    base: Final = (api_base or get_secret_str("TENCENT_API_BASE") or TOKENHUB_API_BASE).rstrip("/")
    versioned_base: Final = base if base.endswith("/v1") else f"{base}/v1"
    return f"{versioned_base}{TOKENHUB_IMAGE_PATH}"


def tokenhub_image_headers(headers: Mapping[str, str], api_key: str | None, model: str) -> Mapping[str, str]:
    resolved_api_key: Final = api_key or get_secret_str("TENCENT_API_KEY")
    if not resolved_api_key:
        raise AuthenticationError(
            message="TENCENT_API_KEY is not set. Set it in the environment or pass api_key.",
            llm_provider="tencent",
            model=model,
        )
    return {**headers, "Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"}


def tokenhub_image_params(params: Mapping[str, object], model: str, drop_params: bool) -> Mapping[str, object]:
    n: Final = params.get("n")
    if n is not None and n != 1 and not drop_params:
        raise UnsupportedParamsError(
            status_code=400,
            message=f"Tencent TokenHub returns one image per request, so n={n} is not supported for {model}. "
            + "To send n=1 instead, set `litellm.drop_params = True`.",
        )
    return {
        key: value
        for key, value in params.items()
        if key != "n" and value is not None and not (key == "background" and value == "auto")
    }


def tokenhub_image_response(raw_response: httpx.Response) -> ImageResponse | None:
    try:
        parsed: Final = _TokenHubImageResponse.model_validate_json(raw_response.content)
    except ValidationError:
        return None
    total_tokens: Final = parsed.tokenhub_usage.total_tokens if parsed.tokenhub_usage is not None else 0
    return ImageResponse(
        created=parsed.created,
        data=[ImageObject(url=image.url, b64_json=image.b64_json) for image in parsed.data],
        usage=ImageUsage(
            input_tokens=0,
            input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=0),
            output_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )


class TencentImageGenerationConfig(BaseImageGenerationConfig):
    @override
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: mirrors override contract
        return list(TOKENHUB_IMAGE_PARAMS)

    @override
    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        return {**optional_params, **tokenhub_image_params(non_default_params, model, drop_params)}

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
        return tokenhub_image_url(api_base)

    @override
    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: mirrors override contract
        return dict(tokenhub_image_headers(headers, api_key, model))

    @override
    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        extra_body: Final = _JSON_OBJECT.validate_python(optional_params.get("extra_body") or {})
        params: Final = {key: value for key, value in optional_params.items() if key != "extra_body"}
        return {"model": model, "prompt": prompt, **params, **extra_body}

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
        image_response: Final = tokenhub_image_response(raw_response)
        if image_response is None:
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
            )
        return image_response
