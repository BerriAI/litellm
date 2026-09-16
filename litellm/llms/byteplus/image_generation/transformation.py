from __future__ import annotations

from typing import TYPE_CHECKING, Final

import httpx

import litellm
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

from ..common_utils import (
    BytePlusError,
    get_byteplus_base_url,
    get_byteplus_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class BytePlusImageGenerationConfig(BaseImageGenerationConfig):
    """
    BytePlus Seedream / Dola Seedream image generation config
    Reference: https://docs.byteplus.com/en/docs/ModelArk/2582774
    """

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: matches BaseImageGenerationConfig interface
        return [  # mutable-ok: matches BaseImageGenerationConfig interface
            "n",
            "response_format",
            "size",
            "user",
            "quality",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        optional_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: matches BaseImageGenerationConfig interface
        supported_params: Final = frozenset(self.get_supported_openai_params(model))
        optional_params.update(
            {k: v for k, v in non_default_params.items() if k in supported_params}  # mutable-ok: update params dict
        )

        return optional_params

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        model: str,
        messages: list,  # mutable-ok: matches BaseImageGenerationConfig interface
        optional_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        litellm_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: matches BaseImageGenerationConfig interface
        resolved_api_key: Final = (
            api_key or litellm.api_key or get_secret_str("BYTEPLUS_API_KEY") or get_secret_str("ARK_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError("BytePlus API key is required. Set BYTEPLUS_API_KEY or ARK_API_KEY or pass api_key.")
        return get_byteplus_headers(api_key=resolved_api_key, extra_headers=headers)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        litellm_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        stream: bool | None = None,
    ) -> str:
        base_url: Final = (
            api_base
            or litellm.api_base
            or get_secret_str("BYTEPLUS_API_BASE")
            or get_secret_str("ARK_API_BASE")
            or get_byteplus_base_url()
        ).rstrip("/")
        if base_url.endswith("/images/generations"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/images/generations"
        return f"{base_url}/api/v3/images/generations"

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: matches BaseImageGenerationConfig interface
    ) -> BytePlusError:
        typed_headers: Final[httpx.Headers] = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return BytePlusError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        litellm_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        headers: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
    ) -> dict:  # mutable-ok: matches BaseImageGenerationConfig interface
        body: Final[dict[str, object]] = {  # mutable-ok: request body dictionary
            "model": model,
            "prompt": prompt,
        }

        for key in (
            "n",
            "response_format",
            "size",
            "user",
            "quality",
            "output_format",
            "watermark",
            "optimize_prompt_options",
            "image",
        ):
            if key in optional_params:
                body[key] = optional_params[key]

        if "extra_body" in optional_params and isinstance(optional_params["extra_body"], dict):
            extra_body: Final = {  # mutable-ok: extra body dictionary
                k: v for k, v in optional_params["extra_body"].items() if k not in ("model", "prompt")
            }
            body.update(extra_body)

        return body

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        optional_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        litellm_params: dict,  # mutable-ok: matches BaseImageGenerationConfig interface
        encoding: object = None,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        response: Final = raw_response.json()

        if logging_obj:
            logging_obj.post_call(
                input=request_data.get("prompt", ""),
                api_key=api_key,
                additional_args={"complete_input_dict": request_data},  # mutable-ok: logging payload dict
                original_response=response,
            )

        image_response: Final[ImageResponse] = convert_to_model_response_object(
            response_object=response,
            model_response_object=model_response,
            response_type="image_generation",
        )

        return image_response
