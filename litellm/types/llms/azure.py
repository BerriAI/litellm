API_VERSION_YEAR_SUPPORTED_RESPONSE_FORMAT = 2024
API_VERSION_MONTH_SUPPORTED_RESPONSE_FORMAT = 8

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from typing_extensions import Required

class InpaintItem(TypedDict, total=False):
    """
    InpaintItem for Azure video generation inpainting.
    
    Represents an image or video to be used in the inpainting process.
    """
    frame_index: Optional[int]  # Frame index where this item should appear
    type: Literal["image", "video"]  # Type of the inpainting item
    file_name: str  # Name of the file
    crop_bounds: Optional[Dict[str, float]]  # Crop bounds as fractions


class AzureCreateVideoRequest(TypedDict, total=False):
    """
    AzureCreateVideoRequest for Azure video generation API
    
    Required Params:
        prompt: str - Text prompt that describes the video to generate
    
    Optional Params:
        model: Optional[str] - The video generation model to use (e.g., "sora")
        n_seconds: Optional[str] - Clip duration in seconds (defaults to 5)
        n_variants: Optional[int] - Number of video variants to generate (defaults to 1)
        height: Optional[int] - Output video height (defaults to 1080)
        width: Optional[int] - Output video width (defaults to 1920)
        inpaint_items: Optional[List[InpaintItem]] - Items for inpainting
        user: Optional[str] - A unique identifier representing your end-user
        extra_headers: Optional[Dict[str, str]] - Additional headers
        extra_body: Optional[Dict[str, str]] - Additional body parameters
        timeout: Optional[float] - Request timeout
    """
    prompt: Required[str]
    model: Optional[str]
    n_seconds: Optional[str]
    n_variants: Optional[int]
    height: Optional[int]
    width: Optional[int]
    inpaint_items: Optional[List[InpaintItem]]
    user: Optional[str]
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]
