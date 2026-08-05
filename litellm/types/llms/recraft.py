from typing_extensions import TypedDict


class RecraftImageGenerationRequestParams(TypedDict, total=False):
    prompt: str
    text_layout: list[dict] | None
    n: int | None
    style_id: str | None
    style: str | None
    substyle: str | None
    model: str | None
    response_format: str | None
    size: str | None
    negative_prompt: str | None
    controls: dict | None


class RecraftImageEditRequestParams(TypedDict, total=False):
    """
    TypedDict for Recraft image edit request parameters.

    Based on Recraft API docs: https://www.recraft.ai/docs#image-to-image
    """

    prompt: str  # required - A text description of areas to change. Max 1000 bytes
    strength: float  # required - Defines difference with original image, [0, 1]
    model: str | None  # The model to use, default is recraftv3
    n: int | None  # The number of images to generate, must be between 1 and 6
    style_id: str | None  # Use a previously uploaded style as reference
    style: str | None  # The style of generated images, default is realistic_image
    substyle: str | None  # Additional style specification
    response_format: str | None  # Format of returned images: url or b64_json
    negative_prompt: str | None  # Description of undesired elements
    controls: dict | None  # Custom parameters to tweak generation process
