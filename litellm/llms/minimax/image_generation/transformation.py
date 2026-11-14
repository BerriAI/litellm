from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.minimax.chat.transformation import MinimaxChatConfig
from litellm.llms.minimax.common_utils import image_params_without_stream, with_minimax_images
from litellm.llms.openai.image_generation.gpt_transformation import (
    GPTImageGenerationConfig,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageResponse
from litellm.utils import EXTRA_BODY_ROUTING_KEYS

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

_EXTRA_BODY: Final = TypeAdapter(Mapping[str, object])


class MinimaxImageGenerationConfig(GPTImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: mirrors override contract
        return [
            "n",
            "response_format",
            "size",
            "aspect_ratio",
            "area",
            "output_format",
            "output_compression",
        ]

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
        resolved_key: Final = MinimaxChatConfig.get_api_key(api_key=api_key)
        if not resolved_key:
            raise ValueError("Missing MiniMax API Key - set MINIMAX_API_KEY or pass api_key")
        return {**headers, "Authorization": f"Bearer {resolved_key}"}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        base_url: Final = MinimaxChatConfig.get_api_base(api_base=api_base).rstrip("/")
        if base_url.endswith("/images/generations"):
            return base_url
        if base_url.endswith("/v1"):
            return join_api_base(base_url, "images/generations")
        return join_api_base(base_url, "v1/images/generations")

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        extra_body: Final = optional_params.get("extra_body")
        extra_fields: Final[Mapping[str, object]] = (
            _EXTRA_BODY.validate_python(extra_body) if isinstance(extra_body, Mapping) else {}
        )
        body_params: Final = image_params_without_stream(
            {
                **{key: value for key, value in optional_params.items() if key != "extra_body"},
                **{key: value for key, value in extra_fields.items() if key not in EXTRA_BODY_ROUTING_KEYS},
            },
            litellm_params.get("drop_params"),
        )
        return {**body_params, "model": model, "prompt": prompt}

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, object],  # mutable-ok: mirrors override contract
        optional_params: dict[str, object],  # mutable-ok: mirrors override contract
        litellm_params: dict[str, object],  # mutable-ok: mirrors override contract
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        return with_minimax_images(
            super().transform_image_generation_response(  # pyright: ignore[reportUnknownMemberType]  # parent declares bare dict params
                model=model,
                raw_response=raw_response,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=request_data,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
                api_key=api_key,
                json_mode=json_mode,
            ),
            raw_response,
        )
