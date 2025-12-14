"""
Type definitions for Stability AI API

API Reference: https://platform.stability.ai/docs/api-reference
"""

from typing import List, Literal, Optional

from typing_extensions import TypedDict


class StabilityImageGenerationRequest(TypedDict, total=False):
    """
    Base request parameters for Stability AI image generation.

    Used for endpoints:
    - /v2beta/stable-image/generate/sd3
    - /v2beta/stable-image/generate/ultra
    - /v2beta/stable-image/generate/core
    """
    prompt: str  # Required - text prompt for image generation
    negative_prompt: Optional[str]  # What to avoid in the image
    aspect_ratio: Optional[str]  # e.g., "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
    seed: Optional[int]  # Random seed for reproducibility (0 to 4294967294)
    output_format: Optional[Literal["jpeg", "png", "webp"]]  # Output format
    model: Optional[str]  # Model variant (e.g., "sd3.5-large", "sd3.5-medium")
    mode: Optional[Literal["text-to-image", "image-to-image"]]  # Generation mode
    image: Optional[str]  # Base64-encoded image for image-to-image
    strength: Optional[float]  # How much to transform the image (0-1)
    style_preset: Optional[str]  # Style preset name


class StabilityImageGenerationResponse(TypedDict, total=False):
    """
    Response from Stability AI image generation endpoints.
    """
    image: str  # Base64-encoded image
    finish_reason: str  # "SUCCESS", "CONTENT_FILTERED", etc.
    seed: int  # The seed used for generation


class StabilityUpscaleRequest(TypedDict, total=False):
    """
    Request parameters for Stability AI upscale endpoints.

    Used for endpoints:
    - /v2beta/stable-image/upscale/fast
    - /v2beta/stable-image/upscale/conservative
    - /v2beta/stable-image/upscale/creative
    """
    image: str  # Required - Base64-encoded image to upscale
    prompt: Optional[str]  # Text prompt (required for creative upscale)
    negative_prompt: Optional[str]  # What to avoid
    output_format: Optional[Literal["jpeg", "png", "webp"]]  # Output format
    seed: Optional[int]  # Random seed
    creativity: Optional[float]  # Creativity level for creative upscale (0-0.35)


class StabilityInpaintRequest(TypedDict, total=False):
    """
    Request parameters for Stability AI inpaint endpoint.

    Endpoint: /v2beta/stable-image/edit/inpaint
    """
    image: str  # Required - Base64-encoded image to edit
    prompt: str  # Required - Description of desired changes
    mask: Optional[str]  # Base64-encoded mask (white = edit, black = keep)
    negative_prompt: Optional[str]  # What to avoid
    seed: Optional[int]  # Random seed
    output_format: Optional[Literal["jpeg", "png", "webp"]]  # Output format
    grow_mask: Optional[int]  # Pixels to grow the mask by (0-100)


class StabilityOutpaintRequest(TypedDict, total=False):
    """
    Request parameters for Stability AI outpaint endpoint.

    Endpoint: /v2beta/stable-image/edit/outpaint
    """
    image: str  # Required - Base64-encoded image to expand
    prompt: Optional[str]  # Description of content to generate
    negative_prompt: Optional[str]  # What to avoid
    left: Optional[int]  # Pixels to expand left (0-2000)
    right: Optional[int]  # Pixels to expand right (0-2000)
    up: Optional[int]  # Pixels to expand up (0-2000)
    down: Optional[int]  # Pixels to expand down (0-2000)
    seed: Optional[int]  # Random seed
    output_format: Optional[Literal["jpeg", "png", "webp"]]  # Output format
    creativity: Optional[float]  # How creative to be (0-1)


class StabilityEraseRequest(TypedDict, total=False):
    """
    Request parameters for Stability AI erase endpoint.

    Endpoint: /v2beta/stable-image/edit/erase
    """
    image: str  # Required - Base64-encoded image
    mask: Optional[str]  # Base64-encoded mask (white = erase)
    seed: Optional[int]  # Random seed
    output_format: Optional[Literal["jpeg", "png", "webp"]]  # Output format
    grow_mask: Optional[int]  # Pixels to grow the mask by (0-100)


class StabilitySearchReplaceRequest(TypedDict, total=False):
    """
    Request parameters for Stability AI search-and-replace endpoint.

    Endpoint: /v2beta/stable-image/edit/search-and-replace
    """
    image: str  # Required - Base64-encoded image
    prompt: str  # Required - Description of object to add
    search_prompt: str  # Required - Description of object to find and replace
    negative_prompt: Optional[str]  # What to avoid
    seed: Optional[int]  # Random seed
    output_format: Optional[Literal["jpeg", "png", "webp"]]  # Output format
    grow_mask: Optional[int]  # Pixels to grow detected mask


class StabilityRemoveBackgroundRequest(TypedDict, total=False):
    """
    Request parameters for Stability AI remove-background endpoint.

    Endpoint: /v2beta/stable-image/edit/remove-background
    """
    image: str  # Required - Base64-encoded image
    output_format: Optional[Literal["png", "webp"]]  # Output format (no jpeg - needs transparency)


class StabilityControlRequest(TypedDict, total=False):
    """
    Request parameters for Stability AI control endpoints.

    Used for endpoints:
    - /v2beta/stable-image/control/sketch
    - /v2beta/stable-image/control/structure
    - /v2beta/stable-image/control/style
    """
    image: str  # Required - Base64-encoded control image (sketch/structure/style reference)
    prompt: str  # Required - Description of desired output
    negative_prompt: Optional[str]  # What to avoid
    control_strength: Optional[float]  # How strongly to follow the control (0-1)
    seed: Optional[int]  # Random seed
    output_format: Optional[Literal["jpeg", "png", "webp"]]  # Output format


class StabilityEditResponse(TypedDict, total=False):
    """
    Response from Stability AI edit/upscale/control endpoints.
    """
    image: str  # Base64-encoded result image
    finish_reason: str  # "SUCCESS", "CONTENT_FILTERED", etc.
    seed: int  # The seed used


# Mapping of OpenAI size to Stability aspect_ratio
OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "512x512": "1:1",
    "256x256": "1:1",
}

# Stability AI supported aspect ratios
STABILITY_ASPECT_RATIOS = [
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "21:9",
    "9:21",
    "3:2",
    "2:3",
    "5:4",
    "4:5",
]

# Stability AI model endpoints
STABILITY_GENERATION_MODELS = {
    "sd3": "/v2beta/stable-image/generate/sd3",
    "sd3.5-large": "/v2beta/stable-image/generate/sd3",
    "sd3.5-large-turbo": "/v2beta/stable-image/generate/sd3",
    "sd3.5-medium": "/v2beta/stable-image/generate/sd3",
    "sd3-large": "/v2beta/stable-image/generate/sd3",
    "sd3-large-turbo": "/v2beta/stable-image/generate/sd3",
    "sd3-medium": "/v2beta/stable-image/generate/sd3",
    "stable-image-ultra": "/v2beta/stable-image/generate/ultra",
    "stable-image-core": "/v2beta/stable-image/generate/core",
}

STABILITY_EDIT_ENDPOINTS = {
    "inpaint": "/v2beta/stable-image/edit/inpaint",
    "outpaint": "/v2beta/stable-image/edit/outpaint",
    "erase": "/v2beta/stable-image/edit/erase",
    "search-and-replace": "/v2beta/stable-image/edit/search-and-replace",
    "search-and-recolor": "/v2beta/stable-image/edit/search-and-recolor",
    "remove-background": "/v2beta/stable-image/edit/remove-background",
}

STABILITY_UPSCALE_ENDPOINTS = {
    "fast": "/v2beta/stable-image/upscale/fast",
    "conservative": "/v2beta/stable-image/upscale/conservative",
    "creative": "/v2beta/stable-image/upscale/creative",
}

STABILITY_CONTROL_ENDPOINTS = {
    "sketch": "/v2beta/stable-image/control/sketch",
    "structure": "/v2beta/stable-image/control/structure",
    "style": "/v2beta/stable-image/control/style",
}
