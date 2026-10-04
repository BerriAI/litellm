from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_ENDPOINT, get_native_api_url, validate_headers
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.dashscope.image_generation.transformation import DashScopeImageGenerationConfig
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _ImageContent(BaseModel):
    image: str | None = None


class _ImageMessage(BaseModel):
    content: tuple[_ImageContent, ...]


class _ImageChoice(BaseModel):
    message: _ImageMessage


class _ImageOutput(BaseModel):
    choices: tuple[_ImageChoice, ...]


class _ImageResult(BaseModel):
    output: _ImageOutput | None = None
    code: str | None = None
    message: str | None = None


class AlibabaTokenPlanImageGenerationConfig(DashScopeImageGenerationConfig):
    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=dict(headers))

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: provider interface requires a mutable return value
        return [*super().get_supported_openai_params(model), "response_format"]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: provider interface requires a mutable return value
        if non_default_params.get("response_format", "url") != "url" and not (drop_params or litellm.drop_params):
            raise litellm.UnsupportedParamsError(
                message="Alibaba Token Plan image generation supports response_format='url'",
                model=model,
                llm_provider="alibaba_token_plan",
            )
        mapped_params: Final = super().map_openai_params(
            {key: value for key, value in non_default_params.items() if key != "response_format"},
            dict(optional_params),
            model,
            drop_params,
        )
        return {
            **optional_params,
            **mapped_params,
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return get_native_api_url(api_base, IMAGE_ENDPOINT)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: provider interface requires a mutable return value
        return validate_headers(headers, api_key)

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: provider interface requires a mutable return value
        native_params: Final = TypeAdapter(Mapping[str, object]).validate_python(
            optional_params.get("extra_body") or {}
        )
        return {
            "model": model,
            "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": {
                "size": "1024*1024",
                **{key: value for key, value in optional_params.items() if key != "extra_body"},
                **native_params,
            },
        }

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
        if not raw_response.is_success:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            result: Final = _ImageResult.model_validate_json(raw_response.content)
        except ValidationError:
            raise self.get_error_class("Invalid Alibaba Token Plan image response", 502, raw_response.headers) from None
        if result.code or result.output is None:
            raise self.get_error_class(
                result.message or result.code or "Alibaba Token Plan returned no image output",
                502,
                raw_response.headers,
            )
        content: Final = chain.from_iterable(choice.message.content for choice in result.output.choices)
        images: Final = [ImageObject(url=item.image) for item in content if item.image]
        if not images:
            raise self.get_error_class("Alibaba Token Plan returned no images", 502, raw_response.headers)
        return model_response.model_copy(update={"data": images})
