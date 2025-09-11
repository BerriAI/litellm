# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Optional

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ImageToVideoTargetSize(BaseInferenceType):
    """The size in pixel of the output video frames."""

    height: int
    width: int


@dataclass_with_extra
class ImageToVideoParameters(BaseInferenceType):
    """Additional inference parameters for Image To Video"""

    guidance_scale: Optional[float] = None
    """For diffusion models. A higher guidance scale value encourages the model to generate
    videos closely linked to the text prompt at the expense of lower image quality.
    """
    negative_prompt: Optional[str] = None
    """One prompt to guide what NOT to include in video generation."""
    num_frames: Optional[float] = None
    """The num_frames parameter determines how many video frames are generated."""
    num_inference_steps: Optional[int] = None
    """The number of denoising steps. More denoising steps usually lead to a higher quality
    video at the expense of slower inference.
    """
    prompt: Optional[str] = None
    """The text prompt to guide the video generation."""
    seed: Optional[int] = None
    """Seed for the random number generator."""
    target_size: Optional[ImageToVideoTargetSize] = None
    """The size in pixel of the output video frames."""


@dataclass_with_extra
class ImageToVideoInput(BaseInferenceType):
    """Inputs for Image To Video inference"""

    inputs: str
    """The input image data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the image data as a raw bytes payload.
    """
    parameters: Optional[ImageToVideoParameters] = None
    """Additional inference parameters for Image To Video"""


@dataclass_with_extra
class ImageToVideoOutput(BaseInferenceType):
    """Outputs of inference for the Image To Video task"""

    video: Any
    """The generated video returned as raw bytes in the payload."""
