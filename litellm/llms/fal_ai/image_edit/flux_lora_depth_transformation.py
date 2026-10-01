from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from httpx._types import RequestFiles

from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes

from .transformation import DEFAULT_BASE_URL, FalAIImageEditConfig, to_data_url

FLUX_LORA_DEPTH_ENDPOINT: Final[str] = "fal-ai/flux-lora-depth"
SUPPORTED_OPENAI_PARAMS: Final[tuple[str, ...]] = ("n", "size")
PARAM_TRANSLATION: Final[Mapping[str, str]] = MappingProxyType({"n": "num_images", "size": "image_size"})


class FalAIFluxLoraDepthEditConfig(FalAIImageEditConfig):
    """
    FLUX.1 [dev] depth LoRA edit endpoint served through Fal AI.

    Unlike the openai gpt-image ``/edit`` endpoints, this endpoint takes a single ``image_url``
    control image and has no ``/edit`` path suffix.
    """

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: base class contract returns a list
        return list(SUPPORTED_OPENAI_PARAMS)  # mutable-ok: base class contract returns a list

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: base class contract returns a dict
        return {  # mutable-ok: base class contract returns a dict
            PARAM_TRANSLATION.get(key, key): self._translate_value(key, value, model)
            for key, value in image_edit_optional_params.items()
            if value is not None and key in PARAM_TRANSLATION
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: base class contract
    ) -> str:
        base_url: Final = (api_base or get_secret_str("FAL_AI_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
        return f"{base_url}/{FLUX_LORA_DEPTH_ENDPOINT}"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,  # mutable-ok: base class contract
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: base class contract
    ) -> tuple[dict, RequestFiles]:  # mutable-ok: base class contract returns a dict
        images: Final = tuple(img for img in (image if isinstance(image, list) else (image,)) if img is not None)
        if not images:
            raise ValueError("Fal AI image edit requires at least one input image")
        if len(images) > 1:
            raise ValueError(f"{FLUX_LORA_DEPTH_ENDPOINT} accepts exactly one control image")
        provider_params: Final[Mapping[str, object]] = MappingProxyType(
            {key: value for key, value in image_edit_optional_request_params.items() if key != "mask"}
        )
        request_body: Final[dict[str, object]] = {  # mutable-ok: base class contract returns a dict
            "prompt": prompt,
            "image_url": to_data_url(next(iter(images))),
            **provider_params,
        }
        return request_body, ()
