from typing import List, Optional

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams

from .transformation import FalAIBaseConfig


class FalAINanoBananaConfig(FalAIBaseConfig):
    """
    Configuration for Fal AI's Nano Banana / Gemini 2.5 Flash Image models.

    Serves the imagen4 deprecation migration path. The same underlying model is
    exposed under two endpoints that share an identical schema:
    - fal-ai/nano-banana
    - fal-ai/gemini-25-flash-image

    Documentation: https://fal.ai/models/fal-ai/nano-banana
    """

    SUPPORTED_ASPECT_RATIOS: List[str] = [
        "21:9",
        "16:9",
        "3:2",
        "4:3",
        "5:4",
        "1:1",
        "4:5",
        "3:4",
        "2:3",
        "9:16",
    ]

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base_url: str = (
            api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        endpoint = model if model.startswith("fal-ai/") else f"fal-ai/{model}"
        return f"{base_url}/{endpoint}"

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for key, value in non_default_params.items():
            if key == "response_format":
                continue
            elif key == "n":
                if "num_images" not in optional_params:
                    optional_params["num_images"] = value
            elif key == "size":
                if "aspect_ratio" not in optional_params:
                    optional_params["aspect_ratio"] = self._map_aspect_ratio(value)
            elif key not in optional_params and not drop_params:
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. "
                    f"Supported parameters are {supported_params}. "
                    "Set drop_params=True to drop unsupported parameters."
                )
        return optional_params

    def _map_aspect_ratio(self, size: str) -> str:
        if not isinstance(size, str) or "x" not in size:
            return "1:1"
        try:
            width, height = (int(part) for part in size.split("x"))
            target = width / height
        except (ValueError, ZeroDivisionError):
            return "1:1"

        def ratio_of(aspect_ratio: str) -> float:
            w, h = (int(part) for part in aspect_ratio.split(":"))
            return w / h

        return min(
            self.SUPPORTED_ASPECT_RATIOS,
            key=lambda aspect_ratio: abs(ratio_of(aspect_ratio) - target),
        )

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {"prompt": prompt, **optional_params}
