from typing import Any

from .flux_pro_v11_ultra_transformation import FalAIFluxProV11UltraConfig


class FalAIBytedanceBaseConfig(FalAIFluxProV11UltraConfig):
    """
    Shared configuration for Fal AI ByteDance text-to-image models that follow
    the Flux Schnell style parameter mapping.

    These models accept the OpenAI-compatible `size` parameter in LiteLLM
    requests but expect `image_size` enums or custom size objects on Fal AI.
    """

    _OPENAI_SIZE_TO_IMAGE_SIZE = {
        "1024x1024": "square_hd",
        "512x512": "square",
        "1792x1024": "landscape_16_9",
        "1024x1792": "portrait_16_9",
        "1024x768": "landscape_4_3",
        "768x1024": "portrait_4_3",
    }

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)

        param_mapping = {
            "n": "num_images",
            "response_format": "output_format",
            "size": "image_size",
        }

        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    mapped_key = param_mapping.get(k, k)
                    mapped_value = non_default_params[k]

                    if k == "response_format":
                        if mapped_value in ["b64_json", "url"]:
                            mapped_value = "jpeg"
                    elif k == "size":
                        mapped_value = self._map_image_size(mapped_value)

                    optional_params[mapped_key] = mapped_value
                elif drop_params:
                    continue
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. "
                        f"Supported parameters are {supported_params}. "
                        "Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def _map_image_size(self, size: Any) -> Any:
        if isinstance(size, dict):
            return size

        if not isinstance(size, str):
            return size

        if size in self._OPENAI_SIZE_TO_IMAGE_SIZE:
            return self._OPENAI_SIZE_TO_IMAGE_SIZE[size]

        if "x" in size:
            try:
                width_str, height_str = size.split("x")
                width = int(width_str)
                height = int(height_str)
                return {"width": width, "height": height}
            except (ValueError, AttributeError, ZeroDivisionError):
                pass

        return "landscape_4_3"


class FalAIBytedanceSeedreamV3Config(FalAIBytedanceBaseConfig):
    """
    Configuration for Fal AI ByteDance Seedream v3 text-to-image model.

    Model endpoint: fal-ai/bytedance/seedream/v3/text-to-image
    Documentation: https://fal.ai/models/fal-ai/bytedance/seedream/v3/text-to-image
    """

    IMAGE_GENERATION_ENDPOINT: str = "fal-ai/bytedance/seedream/v3/text-to-image"


class FalAIBytedanceDreaminaV31Config(FalAIBytedanceBaseConfig):
    """
    Configuration for Fal AI ByteDance Dreamina v3.1 text-to-image model.

    Model endpoint: fal-ai/bytedance/dreamina/v3.1/text-to-image
    Documentation: https://fal.ai/models/fal-ai/bytedance/dreamina/v3.1/text-to-image
    """

    IMAGE_GENERATION_ENDPOINT: str = "fal-ai/bytedance/dreamina/v3.1/text-to-image"


