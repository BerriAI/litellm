from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from typing_extensions import ReadOnly, TypedDict

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


class FalAIGPTImage2Config(FalAIBaseConfig):
    """
    Configuration for OpenAI's GPT Image 2 served through Fal AI.

    Model endpoints:
    - openai/gpt-image-2 (text-to-image)
    - openai/gpt-image-2/edit (editing, with optional mask)

    Documentation: https://fal.ai/models/openai/gpt-image-2/api
    """

    MODEL_PREFIX: Final[str] = "openai/"
    SUPPORTED_QUALITIES: Final[frozenset[str]] = frozenset({"auto", "low", "medium", "high"})
    OPENAI_QUALITY_ALIASES: Final[Mapping[str, str]] = MappingProxyType({"hd": "high", "standard": "medium"})
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

    def get_supported_openai_params(  # mutable-ok: base class contract returns a list
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:
        return list(SUPPORTED_OPENAI_PARAMS)  # mutable-ok: base class contract returns a list

    def map_openai_params(  # mutable-ok: base class contract returns a dict
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
                self.PARAM_TRANSLATION[key]: self._translate_value(key, value)
                for key, value in non_default_params.items()
                if key in self.PARAM_TRANSLATION and self.PARAM_TRANSLATION[key] not in optional_params
            }
        )
        return {**optional_params, **translated_params}  # mutable-ok: base class contract returns a dict

    def _translate_value(self, key: str, value: object) -> object:
        if key == "size":
            return self._map_image_size(value)
        if key == "quality":
            return self._map_quality(value)
        return value

    def _map_image_size(self, size: object) -> object:
        if not isinstance(size, str) or size == "auto":
            return size
        try:
            width, height = (int(part) for part in size.lower().split("x"))
        except ValueError:
            return size
        image_size: Final[FalAIImageSize] = {"width": width, "height": height}
        return image_size

    def _map_quality(self, quality: object) -> object:
        if not isinstance(quality, str):
            return quality
        normalized: Final[str] = self.OPENAI_QUALITY_ALIASES.get(quality, quality)
        return normalized if normalized in self.SUPPORTED_QUALITIES else "auto"

    def transform_image_generation_request(  # mutable-ok: base class contract returns a dict
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict:
        return {"prompt": prompt, **optional_params}  # mutable-ok: base class contract returns a dict
