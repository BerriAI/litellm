from typing_extensions import TypedDict


class AimlImageSize(TypedDict, total=False):
    """Custom image size specification for AI/ML API"""

    width: int  # Must be multiple of 32, min 256, max 1440
    height: int  # Must be multiple of 32, min 256, max 1440


class AimlImageGenerationRequestParams(TypedDict, total=False):
    """
    TypedDict for AI/ML flux image generation request parameters.

    Based on AI/ML API docs: https://api.aimlapi.com/v1/images/generations
    """

    model: str  # Required: flux-pro/v1.1
    prompt: str  # Required: Text prompt (max 4000 chars)
    image_size: (
        AimlImageSize | str
    )  # Custom size or predefined: square_hd, square, portrait_4_3, portrait_16_9, landscape_4_3, landscape_16_9
    safety_tolerance: str | None  # 1-6, default 2 (1=strict, 6=permissive)
    output_format: str | None  # jpeg or png, default jpeg
    num_images: int | None  # 1-4, default 1
    seed: int | None  # Min 1, for reproducibility
    enable_safety_checker: bool | None  # Default true
