from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    import tiktoken
    from litellm.litellm_core_utils.logging import Logging as LiteLLMLoggingObj


class AzureFoundryMAIImageGenerationConfig(BaseImageGenerationConfig):
    """Azure AI Foundry MAI image generation (e.g. MAI-Image-2.5)."""

    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    # The MAI endpoint produces exactly one image per request. Its documented
    # body is model/prompt/width/height (plus `image` for edits) — there is no
    # count field, and `n` (or the native `sampleCount`) is accepted and
    # ignored, so a request for more silently comes back with one.
    MAX_IMAGES_PER_REQUEST: Final = 1

    # Provider-side bounds on the generated image. Both are enforced by the
    # MAI endpoint, which 400s with "'width' must be at least 768 pixels."
    # Only `size` is checked against them: `width`/`height` pass through
    # unmapped, which keeps a future model with different bounds reachable.
    MIN_DIMENSION_PX: Final = 768
    MAX_TOTAL_PX: Final = 1024 * 1024

    @staticmethod
    def get_mai_image_generation_url(
        api_base: str | None,
        api_version: str | None,
    ) -> str:
        if api_base is None:
            raise ValueError("api_base is required for Azure AI MAI image generation")

        api_version = api_version or "preview"
        path, separator, query = api_base.partition("?")
        path = path.rstrip("/")

        if "/mai/" in path:
            prefix, _, _ = path.partition("/images/")
            path = f"{prefix}/images/generations"
        else:
            path = f"{path}/mai/v1/images/generations"

        if separator:
            return f"{path}?{query}"
        return f"{path}?api-version={api_version}"

    @staticmethod
    def get_mai_image_edit_url(
        api_base: str | None,
        api_version: str | None,
    ) -> str:
        if api_base is None:
            raise ValueError("api_base is required for Azure AI MAI image editing")

        api_version = api_version or "preview"
        path, separator, query = api_base.partition("?")
        path = path.rstrip("/")

        if "/mai/" in path:
            prefix, _, _ = path.partition("/images/")
            path = f"{prefix}/images/edits"
        else:
            path = f"{path}/mai/v1/images/edits"

        if separator:
            return f"{path}?{query}"
        return f"{path}?api-version={api_version}"

    @staticmethod
    def is_mai_model(model: str) -> bool:
        model_normalized: Final = model.lower().replace("-", "").replace("_", "")
        return "maiimage" in model_normalized

    @staticmethod
    def normalize_mai_image_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
        """Map Azure MAI usage fields to OpenAI ImageUsage schema."""
        if usage is None:
            return {
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 0,
                "total_tokens": 0,
            }

        normalized_usage: Final = dict(usage)
        input_tokens_details = normalized_usage.get("input_tokens_details")
        if not isinstance(input_tokens_details, dict):
            input_tokens_details = {}

        text_tokens = normalized_usage.get("num_input_text_tokens")
        if text_tokens is None:
            text_tokens = input_tokens_details.get("text_tokens")
        if text_tokens is None:
            text_tokens = normalized_usage.get("input_tokens", 0) or 0

        image_tokens = normalized_usage.get("num_input_image_tokens")
        if image_tokens is None:
            image_tokens = input_tokens_details.get("image_tokens")
        if image_tokens is None:
            image_tokens = 0

        output_tokens = normalized_usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = normalized_usage.get("num_output_tokens")
        if output_tokens is None:
            output_tokens = normalized_usage.get("output_image_tokens")
        if output_tokens is None:
            output_tokens = 0

        input_tokens = normalized_usage.get("input_tokens")
        if input_tokens is None:
            input_tokens = text_tokens + image_tokens

        total_tokens = normalized_usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        normalized_usage.update(
            {
                "input_tokens": input_tokens,
                "input_tokens_details": {
                    "image_tokens": image_tokens,
                    "text_tokens": text_tokens,
                },
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        )
        return normalized_usage

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = self.get_supported_openai_params(model)

        for k, v in non_default_params.items():
            if k in optional_params:
                continue

            if k in supported_params:
                if k == "size" and v:
                    self._map_size_param(v, optional_params)
                elif k == "n" and v is not None and v > self.MAX_IMAGES_PER_REQUEST:
                    if not drop_params:
                        raise ValueError(
                            f"n={v} is not supported for model {model}. The Azure AI MAI image "
                            f"endpoint returns exactly {self.MAX_IMAGES_PER_REQUEST} image per "
                            "request and ignores any count, so a larger value would silently "
                            "return fewer images than requested. Send one request per image, or "
                            "set drop_params=True to drop n."
                        )
                else:
                    optional_params[k] = v
            elif k in ("width", "height"):
                optional_params[k] = v
            elif not drop_params:
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. "
                    f"Supported parameters are {supported_params} and width/height. "
                    f"Set drop_params=True to drop unsupported parameters."
                )

        if "width" not in optional_params:
            optional_params["width"] = self.DEFAULT_WIDTH
        if "height" not in optional_params:
            optional_params["height"] = self.DEFAULT_HEIGHT

        optional_params.pop("size", None)
        return optional_params

    def _map_size_param(self, size: str, optional_params: dict) -> None:
        size_mapping: Final = {
            "1024x1024": (1024, 1024),
            "1792x1024": (1792, 1024),
            "1024x1792": (1024, 1792),
            "512x512": (512, 512),
            "256x256": (256, 256),
        }

        if size in size_mapping:
            width, height = size_mapping[size]
        elif "x" in size:
            try:
                width, height = map(int, size.lower().split("x"))
            except ValueError:
                raise ValueError(f"Invalid size format: '{size}'. Expected format 'WIDTHxHEIGHT' (e.g., '1024x1024').")
        else:
            raise ValueError(
                f"Unsupported size value: '{size}'. "
                f"Use a known size (e.g., '1024x1024') or a custom 'WIDTHxHEIGHT' string."
            )

        self._validate_dimensions(size=size, width=width, height=height)
        optional_params["width"] = width
        optional_params["height"] = height

    def _validate_dimensions(self, size: str, width: int, height: int) -> None:
        """Reject a `size` the MAI endpoint would 400 on.

        Several OpenAI-standard sizes are outside MAI's bounds: 512x512 and
        256x256 fall under the per-side minimum, and 1792x1024 / 1024x1792
        exceed the total pixel budget. Checking here turns an opaque provider
        400 into an error that names the constraint.
        """
        if width < self.MIN_DIMENSION_PX or height < self.MIN_DIMENSION_PX:
            raise ValueError(
                f"Unsupported size value: '{size}'. Azure AI MAI image models require width and "
                f"height of at least {self.MIN_DIMENSION_PX} pixels."
            )
        if width * height > self.MAX_TOTAL_PX:
            raise ValueError(
                f"Unsupported size value: '{size}'. Azure AI MAI image models accept at most "
                f"{self.MAX_TOTAL_PX} total pixels ({width}x{height} is {width * height})."
            )

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: "tiktoken.Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            response: Final = raw_response.json()
        except Exception:
            raise OpenAIError(message=raw_response.text, status_code=raw_response.status_code)

        if "usage" in response:
            response["usage"] = self.normalize_mai_image_usage(response.get("usage"))

        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response,
        )

        image_response: Final[ImageResponse] = convert_to_model_response_object(
            response_object=response,
            model_response_object=model_response,
            response_type="image_generation",
        )

        width: Final = optional_params.get("width", self.DEFAULT_WIDTH)
        height: Final = optional_params.get("height", self.DEFAULT_HEIGHT)
        image_response.size = f"{width}x{height}"
        return image_response
