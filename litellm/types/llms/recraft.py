from typing import Dict, List, Optional

from typing_extensions import TypedDict


class RecraftImageGenerationRequestParams(TypedDict, total=False):
    prompt: str
    text_layout: Optional[List[Dict]]
    n: Optional[int]
    style_id: Optional[str]
    style: Optional[str]
    substyle: Optional[str]
    model: Optional[str]
    response_format: Optional[str]
    size: Optional[str]
    negative_prompt: Optional[str]
    controls: Optional[Dict]


class RecraftImageEditRequestParams(TypedDict, total=False):
    """
    TypedDict for Recraft image edit request parameters.
    
    Based on Recraft API docs: https://www.recraft.ai/docs#image-to-image
    """
    prompt: str  # required - A text description of areas to change. Max 1000 bytes
    strength: float  # required - Defines difference with original image, [0, 1]
    model: Optional[str]  # The model to use, default is recraftv3
    n: Optional[int]  # The number of images to generate, must be between 1 and 6
    style_id: Optional[str]  # Use a previously uploaded style as reference
    style: Optional[str]  # The style of generated images, default is realistic_image
    substyle: Optional[str]  # Additional style specification
    response_format: Optional[str]  # Format of returned images: url or b64_json
    negative_prompt: Optional[str]  # Description of undesired elements
    controls: Optional[Dict]  # Custom parameters to tweak generation process
