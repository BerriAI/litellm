from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.logging import Logging as LiteLLMLoggingObj


class AzureFoundryMAIImageGenerationConfig(BaseImageGenerationConfig):
    """
    Azure Foundry MAI image generation config.

    MAI image models use the Foundry-specific /mai/v1/images/generations endpoint
    instead of the Azure OpenAI /openai/deployments/.../images/generations route.
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["height", "size", "width"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for key, value in non_default_params.items():
            if key in optional_params:
                continue
            if key in supported_params:
                optional_params[key] = value
            elif not drop_params:
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. "
                    f"Supported parameters are {supported_params}. Set drop_params=True "
                    "to drop unsupported parameters."
                )

        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        validated_headers = dict(headers)
        validated_headers.setdefault("Content-Type", "application/json")
        if api_key:
            validated_headers.setdefault("api-key", api_key)
        return validated_headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        resolved_api_base = AzureFoundryModelInfo.get_api_base(api_base)
        if resolved_api_base is None:
            raise ValueError("api_base is required for Azure AI MAI image generation")

        api_version = litellm_params.get("api_version") or "preview"
        return self._append_api_version(
            f"{resolved_api_base.rstrip('/')}/mai/v1/images/generations",
            api_version=api_version,
        )

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        width, height = self._resolve_dimensions(optional_params)

        return {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        response = raw_response.json()
        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response,
        )
        image_response: ImageResponse = convert_to_model_response_object(  # type: ignore
            response_object=response,
            model_response_object=model_response,
            response_type="image_generation",
        )

        image_response.size = response.get(
            "size",
            f"{request_data['width']}x{request_data['height']}",
        )
        image_response.output_format = response.get("output_format", "png")

        return image_response

    @staticmethod
    def _append_api_version(url: str, api_version: str) -> str:
        split_url = urlsplit(url)
        query_params = dict(parse_qsl(split_url.query))
        query_params.setdefault("api-version", api_version)
        return urlunsplit(
            (
                split_url.scheme,
                split_url.netloc,
                split_url.path,
                urlencode(query_params),
                split_url.fragment,
            )
        )

    @staticmethod
    def _parse_size(size: str) -> Tuple[int, int]:
        parts = size.lower().split("x", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid size format '{size}'. Expected 'WxH' (e.g. '1024x1024')."
            )
        width_str, height_str = parts
        try:
            return int(width_str), int(height_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid size format '{size}'. Expected integer dimensions like '1024x1024'."
            ) from exc

    @classmethod
    def _resolve_dimensions(cls, optional_params: dict) -> Tuple[int, int]:
        width = optional_params.get("width")
        height = optional_params.get("height")

        if width is not None or height is not None:
            if width is None or height is None:
                raise ValueError(
                    "Azure AI MAI image generation requires both width and height when either is provided."
                )
            return cls._validate_dimensions(width=int(width), height=int(height))

        size = optional_params.get("size")
        if size is not None:
            parsed_width, parsed_height = cls._parse_size(size)
            return cls._validate_dimensions(width=parsed_width, height=parsed_height)

        return cls._validate_dimensions(width=1024, height=1024)

    @staticmethod
    def _validate_dimensions(width: int, height: int) -> Tuple[int, int]:
        if width < 768 or height < 768:
            raise ValueError(
                "Azure AI MAI image generation requires width and height to be at least 768."
            )
        if width * height > 1_048_576:
            raise ValueError(
                "Azure AI MAI image generation supports a maximum of 1,048,576 total pixels."
            )
        return width, height
