"""
Black Forest Labs Common Utilities

Common utilities, constants, and error handling for Black Forest Labs API.
"""

from typing import Dict

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class BlackForestLabsError(BaseLLMException):
    """Exception class for Black Forest Labs API errors."""

    pass


# API Constants
DEFAULT_API_BASE = "https://api.bfl.ai"

# Polling configuration
DEFAULT_POLLING_INTERVAL = 1.5  # seconds
DEFAULT_MAX_POLLING_TIME = 300  # 5 minutes

# Model to endpoint mapping for image edit
IMAGE_EDIT_MODELS: Dict[str, str] = {
    "flux-kontext-pro": "/v1/flux-kontext-pro",
    "flux-kontext-max": "/v1/flux-kontext-max",
    "flux-pro-1.0-fill": "/v1/flux-pro-1.0-fill",
    "flux-pro-1.0-expand": "/v1/flux-pro-1.0-expand",
}

# Model to endpoint mapping for image generation
IMAGE_GENERATION_MODELS: Dict[str, str] = {
    "flux-pro-1.1": "/v1/flux-pro-1.1",
    "flux-pro-1.1-ultra": "/v1/flux-pro-1.1-ultra",
    "flux-dev": "/v1/flux-dev",
    "flux-pro": "/v1/flux-pro",
}
