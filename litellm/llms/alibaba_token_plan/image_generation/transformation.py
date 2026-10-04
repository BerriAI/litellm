from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_PATH, get_api_url, require_api_key
from litellm.llms.dashscope.image_generation.transformation import DashScopeImageGenerationConfig
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams


class AlibabaTokenPlanImageGenerationConfig(DashScopeImageGenerationConfig):
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
                message="Alibaba Token Plan image generation only returns image URLs; use response_format='url'",
                model=model,
                llm_provider="alibaba_token_plan",
            )
        return super().map_openai_params(
            {key: value for key, value in non_default_params.items() if key != "response_format"},
            optional_params,
            model,
            drop_params,
        )

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: provider interface requires a mutable return value
        native_params: Final = TypeAdapter(dict[str, object]).validate_python(optional_params.get("extra_body") or {})
        return super().transform_image_generation_request(
            model,
            prompt,
            {**{key: value for key, value in optional_params.items() if key != "extra_body"}, **native_params},
            dict(litellm_params),
            dict(headers),
        )

    def _resolve_api_key(self, api_key: str | None) -> str:
        return require_api_key(api_key)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return get_api_url(api_base, IMAGE_PATH)
