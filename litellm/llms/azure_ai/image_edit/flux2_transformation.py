import base64
import contextlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import httpx
from httpx._types import RequestFiles

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.token_counter import get_image_type, image_dimensions_from_bytes
from litellm.llms.azure_ai.common_utils import (
    AzureFoundryModelInfo,
    get_azure_ai_auth_headers,
)
from litellm.llms.azure_ai.image_generation.cost_calculator import record_reference_pixels
from litellm.llms.azure_ai.image_generation.flux_transformation import (
    AzureFoundryFluxImageGenerationConfig,
)
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.openai import FileTypes
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class AzureFoundryFlux2ImageEditConfig(OpenAIImageEditConfig):
    """
    Azure AI Foundry FLUX 2 image edit config

    Supports FLUX 2 models (e.g., flux.2-pro) for image editing.
    Uses the model-specific /providers/blackforestlabs/v1/flux-2-* endpoint as image generation,
    with the image passed as base64 in JSON body.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reference_image_pixels: tuple[int | None, ...] = ()

    def get_supported_openai_params(self, model: str) -> list:
        return AzureFoundryFluxImageGenerationConfig().get_supported_openai_params(model)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to FLUX 2 params.
        FLUX 2 uses the same param names as OpenAI for supported params.
        """
        return AzureFoundryFluxImageGenerationConfig().map_openai_params(
            non_default_params=MappingProxyType(
                {key: value for key, value in image_edit_optional_params.items() if value is not None}
            ),
            optional_params=MappingProxyType({}),
            model=model,
            drop_params=drop_params,
        )

    def use_multipart_form_data(self) -> bool:
        """FLUX 2 uses JSON requests, not multipart/form-data."""
        return False

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        """
        Validate Azure AI Foundry environment and set up authentication
        """
        headers.update(
            {
                **get_azure_ai_auth_headers(
                    api_key=AzureFoundryModelInfo.get_api_key(api_key),
                    litellm_params=litellm_params,
                    api_key_header="Api-Key",
                ),
                "Content-Type": "application/json",
            }
        )
        return headers

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | Sequence[FileTypes] | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        """
        Transform image edit request for FLUX 2.

        FLUX 2 uses the same endpoint for generation and editing,
        with the image passed as base64 in the JSON body.
        """
        if prompt is None:
            raise ValueError("FLUX 2 image edit requires a prompt.")

        if image is None:
            raise ValueError("FLUX 2 image edit requires an image.")

        images: Final = tuple(image) if isinstance(image, list) else (image,)
        if not images:
            raise ValueError("FLUX 2 image edit requires at least one image.")
        max_reference_images: Final = 10 if "flex" in model.lower() else 8
        if len(images) > max_reference_images:
            raise ValueError(f"{model} supports at most {max_reference_images} reference images.")

        reference_bytes: Final = tuple(self._read_image_bytes(reference_image) for reference_image in images)
        self.reference_image_pixels = tuple(
            _pixel_count(index, image_bytes) for index, image_bytes in enumerate(reference_bytes, start=1)
        )
        reference_images: Final[Mapping[str, str]] = MappingProxyType(
            {
                "input_image" if index == 1 else f"input_image_{index}": base64.b64encode(image_bytes).decode("utf-8")
                for index, image_bytes in enumerate(reference_bytes, start=1)
            }
        )
        request_body: Final[dict[str, Any]] = {
            "prompt": prompt,
            "model": model,
            **reference_images,
            **image_edit_optional_request_params,
        }
        return request_body, []

    def _read_image_bytes(self, image: FileTypes | Sequence[FileTypes]) -> bytes:
        if isinstance(image, bytes):
            return image
        read: Final[object] = getattr(image, "read", None)
        if not callable(read):
            raise ValueError(f"Unsupported image type: {type(image)}")
        _rewind(image)
        image_data: Final[object] = read()
        _rewind(image)
        match image_data:
            case bytes() | bytearray() | memoryview() if len(image_data) > 0:
                return bytes(image_data)
            case bytes() | bytearray() | memoryview():
                raise ValueError(
                    f"FLUX.2 reference image read from {type(image).__name__} is empty. A stream that can't seek "
                    "is consumed by the first attempt, so pass bytes or a seekable file to allow retries"
                )
            case _:
                raise TypeError("FLUX.2 reference images must be opened in binary mode")

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        image_response: Final = super().transform_image_edit_response(model, raw_response, logging_obj)
        record_reference_pixels(image_response, self.reference_image_pixels)
        return image_response

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Constructs a complete URL for Azure AI Foundry FLUX 2 image edits.

        Uses the same model-specific BFL provider endpoint as image generation.
        """
        api_base = AzureFoundryModelInfo.get_api_base(api_base)

        if api_base is None:
            raise ValueError(
                "Azure AI API base is required. Set AZURE_AI_API_BASE environment variable or pass api_base parameter."
            )

        api_version: Final = (
            litellm_params.get("api_version")
            or litellm.api_version
            or get_secret_str("AZURE_AI_API_VERSION")
            or "preview"
        )

        return AzureFoundryFluxImageGenerationConfig.get_flux2_image_generation_url(
            api_base=api_base,
            model=model,
            api_version=api_version,
        )


def _rewind(image: object) -> None:
    seekable: Final[object] = getattr(image, "seekable", None)
    if callable(seekable) and not seekable():
        return
    seek: Final[object] = getattr(image, "seek", None)
    if callable(seek):
        with contextlib.suppress(OSError):
            seek(0)


def _pixel_count(index: int, image_bytes: bytes) -> int | None:
    dimensions: Final = image_dimensions_from_bytes(image_bytes)
    pixels: Final = None if dimensions is None else dimensions[0] * dimensions[1] or None
    if pixels is None:
        verbose_logger.debug(
            "FLUX.2 reference image %d (%d bytes, detected type %s) has no readable dimensions",
            index,
            len(image_bytes),
            get_image_type(image_bytes),
        )
    return pixels
