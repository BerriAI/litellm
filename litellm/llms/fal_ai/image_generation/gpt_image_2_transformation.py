from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams

from .transformation import FalAIBaseConfig


class FalAIImageSize(TypedDict):
    width: ReadOnly[int]
    height: ReadOnly[int]


SUPPORTED_OPENAI_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "n",
    "output_format",
    "quality",
    "response_format",
    "size",
)
OPENAI_QUALITY_ALIASES: Final[Mapping[str, str]] = MappingProxyType({"hd": "high", "standard": "medium"})


def map_gpt_image_size(size: object) -> object:
    if not isinstance(size, str) or size == "auto":
        return size
    try:
        width, height = (int(part) for part in size.lower().split("x"))
    except ValueError:
        return size
    image_size: Final[FalAIImageSize] = {"width": width, "height": height}
    return image_size


def supported_gpt_image_qualities(
    model: str, model_cost: Mapping[str, Mapping[str, object]] | None = None
) -> frozenset[str]:
    costs: Final = litellm.model_cost if model_cost is None else model_cost
    endpoint: Final[str] = model.removeprefix("fal_ai/")
    qualified_endpoint: Final[str] = endpoint if endpoint.startswith("openai/") else f"openai/{endpoint}"
    qualities: Final[frozenset[str]] = frozenset(
        parts[1]
        for key in costs
        if (parts := key.split("/"))[0] == "fal_ai"
        and len(parts) > 3
        and "-x-" in parts[2]
        and "/".join(parts[3:]) == qualified_endpoint
    )
    return qualities | frozenset({"auto"}) if qualities else frozenset()


def map_gpt_image_quality(
    quality: object, model: str, model_cost: Mapping[str, Mapping[str, object]] | None = None
) -> object:
    if not isinstance(quality, str):
        return quality
    normalized: Final[str] = OPENAI_QUALITY_ALIASES.get(quality, quality)
    supported: Final[frozenset[str]] = supported_gpt_image_qualities(model, model_cost)
    if not supported:
        return normalized
    return normalized if normalized in supported else "auto"


class FalAIGPTImage2Config(FalAIBaseConfig):
    """
    Configuration for OpenAI's GPT Image 2 served through Fal AI.

    Model endpoints:
    - openai/gpt-image-2 (text-to-image)
    - openai/gpt-image-2/edit (editing, with optional mask)
    - openai/gpt-image-2.5/flare/text-to-image, openai/gpt-image-2.5/sunburst/text-to-image

    Documentation: https://fal.ai/models/openai/gpt-image-2/api
    """

    MODEL_PREFIX: Final[str] = "openai/"
    PARAM_TRANSLATION: Final[Mapping[str, str]] = MappingProxyType(
        {
            "n": "num_images",
            "size": "image_size",
            "quality": "quality",
            "output_format": "output_format",
        }
    )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        base_url: Final[str] = (api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/")
        endpoint: Final[str] = model if model.startswith(self.MODEL_PREFIX) else f"{self.MODEL_PREFIX}{model}"
        return f"{base_url}/{endpoint}"

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return list(SUPPORTED_OPENAI_PARAMS)  # mutable-ok: base class contract returns a list

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict:
        unsupported_params: Final = tuple(
            key for key in non_default_params if key not in SUPPORTED_OPENAI_PARAMS and key not in optional_params
        )
        if unsupported_params and not drop_params:
            raise ValueError(
                f"Parameters {unsupported_params} are not supported for model {model}. "
                f"Supported parameters are {SUPPORTED_OPENAI_PARAMS}. "
                "Set drop_params=True to drop unsupported parameters."
            )
        translated_params: Final[Mapping[str, object]] = MappingProxyType(
            {
                self.PARAM_TRANSLATION[key]: self._translate_value(key, value, model)
                for key, value in non_default_params.items()
                if key in self.PARAM_TRANSLATION and self.PARAM_TRANSLATION[key] not in optional_params
            }
        )
        return {**optional_params, **translated_params}  # mutable-ok: base class contract returns a dict

    def _translate_value(self, key: str, value: object, model: str) -> object:
        if key == "size":
            return map_gpt_image_size(value)
        if key == "quality":
            return map_gpt_image_quality(value, model)
        return value

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict:
        return {"prompt": prompt, **optional_params}  # mutable-ok: base class contract returns a dict
