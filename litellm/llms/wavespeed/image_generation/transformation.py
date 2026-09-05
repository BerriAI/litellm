"""
WaveSpeed AI image generation configuration.

Transforms between the OpenAI image generation contract and WaveSpeed's prediction API.
The submit/poll HTTP flow lives in ``handler.py``; this class only transforms data.

API Reference: https://wavespeed.ai/docs
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # runtime stand-in for the TYPE_CHECKING-only logging type
    Final,
    TypeAlias,
)

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import (
    WaveSpeedError,
    build_headers,
    build_submit_url,
    get_outputs,
    to_request_payload,
    unwrap_envelope,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj: TypeAlias = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj: TypeAlias = Any


class WaveSpeedImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for WaveSpeed AI image generation.

    Any WaveSpeed image model id works as-is, e.g. ``wavespeed/bytedance/seedream-v5.0-pro``
    or ``wavespeed/wavespeed-ai/z-image/turbo``. Model-specific fields that have no OpenAI
    equivalent are passed straight through to the prediction body.
    """

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: base config contract returns `list`
        return ["n", "size", "response_format"]  # mutable-ok: base config contract returns `list`

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: base config contract returns bare `dict`
        return to_request_payload(
            MappingProxyType(
                {**optional_params, **self._mapped_params(non_default_params, optional_params, drop_params)}
            )
        )

    def _mapped_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        drop_params: bool,
    ) -> Mapping[str, object]:
        return MappingProxyType(
            {
                key: value
                for key, value in (
                    self._map_one(key, value, drop_params)
                    for key, value in non_default_params.items()
                    if key not in optional_params and value is not None
                )
                if key is not None
            }
        )

    def _map_one(self, key: str, value: object, drop_params: bool) -> tuple[str | None, object]:
        if key == "size":
            return "size", self._map_size(value)
        if key == "n":
            return ("num_images", value) if isinstance(value, int) and value > 1 else (None, None)
        if key == "response_format":
            if value != "url" and not drop_params:
                raise ValueError(
                    "WaveSpeed returns hosted image URLs, so only response_format='url' is supported. "
                    "Set drop_params=True to ignore this parameter."
                )
            return None, None
        return key, value

    def _map_size(self, size: object) -> str:
        """WaveSpeed takes ``{width}*{height}`` where OpenAI takes ``{width}x{height}``."""
        width, separator, height = str(size).lower().partition("x")
        if not separator or not width.isdigit() or not height.isdigit():
            raise ValueError(f"Invalid size format: '{size}'. Expected 'WIDTHxHEIGHT' (e.g. '1024x1024').")
        return f"{width}*{height}"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: base config contract returns bare `dict`
        return to_request_payload(MappingProxyType({**build_headers(api_key), **headers}))

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return build_submit_url(api_base, model)

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict:  # mutable-ok: base config contract returns bare `dict`
        return to_request_payload(MappingProxyType({"prompt": prompt, **optional_params}))

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        """Transform the final polled prediction into an OpenAI image response."""
        prediction: Final = unwrap_envelope(raw_response)
        outputs: Final = get_outputs(prediction)

        if not outputs:
            raise WaveSpeedError(
                status_code=500,
                message=f"WaveSpeed prediction {prediction.get('id', '')} completed without any outputs",
            )

        images: Final = [ImageObject(url=url, b64_json=None) for url in outputs]  # mutable-ok: pydantic field is `list`
        model_response.data = images  # rebind-ok: the base contract fills in and returns the caller's ImageResponse
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: matches BaseLLMException/base get_error_class contract
    ) -> WaveSpeedError:
        return WaveSpeedError(status_code=status_code, message=error_message, headers=headers)
