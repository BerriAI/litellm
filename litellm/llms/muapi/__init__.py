"""
MuAPI provider for LiteLLM.

MuAPI (https://muapi.ai) is a generative media aggregator giving access to
50+ image and video generation models through a single unified API.

Supported endpoints:
- Image generation:  muapi/<model-id>  (text-to-image & image-to-image editing)
- Video generation:  muapi/<model-id>  (text-to-video)
- Image-to-video:    muapi/<model-id>-i2v

Authentication: set the MUAPI_API_KEY environment variable or pass api_key.
"""

from .image_generation import MuAPIImageConfig, get_muapi_image_generation_config
from .videos.transformation import MuAPIVideoConfig

__all__ = [
    "MuAPIImageConfig",
    "get_muapi_image_generation_config",
    "MuAPIVideoConfig",
]
