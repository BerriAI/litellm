from collections.abc import Mapping
from typing import Final

from litellm.llms.fal_ai.image_generation.nano_banana_transformation import (
    FalAINanoBananaConfig,
    map_nano_banana_aspect_ratio,
)
from litellm.types.images.main import ImageEditOptionalRequestParams

from .transformation import FalAIImageEditConfig

NANO_BANANA_EDIT_MODELS: Final = frozenset(("fal-ai/nano-banana-2", "fal-ai/nano-banana-pro"))
SUPPORTED_PARAMS: Final = ("n", "size", "aspect_ratio", "resolution", "system_prompt", "sync_mode")


class FalAINanoBananaImageEditConfig(FalAIImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base class contract returns a list
        return list(SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: base class contract returns a dict
        params: Final[Mapping[str, object]] = image_edit_optional_params
        size: Final = params.get("size")
        supported_aspect_ratios: Final = (
            (*FalAINanoBananaConfig.SUPPORTED_ASPECT_RATIOS, "4:1", "1:4", "8:1", "1:8")
            if model.removesuffix("/edit") == "fal-ai/nano-banana-2"
            else tuple(FalAINanoBananaConfig.SUPPORTED_ASPECT_RATIOS)
        )
        size_params: Final[Mapping[str, str]] = (
            {"aspect_ratio": "auto" if size == "auto" else map_nano_banana_aspect_ratio(size, supported_aspect_ratios)}
            if isinstance(size, str) and params.get("aspect_ratio") is None
            else {}
        )
        return {
            **size_params,
            **{
                "num_images" if key == "n" else key: value
                for key, value in params.items()
                if key in SUPPORTED_PARAMS and key != "size" and value is not None
            },
        }
