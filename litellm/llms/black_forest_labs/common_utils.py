"""
Black Forest Labs Common Utilities

Common utilities, constants, and error handling for Black Forest Labs API.
"""

from typing import Dict
from urllib.parse import urlparse

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class BlackForestLabsError(BaseLLMException):
    """Exception class for Black Forest Labs API errors."""

    pass


# API Constants
DEFAULT_API_BASE = "https://api.bfl.ai"

# BFL uses regional subdomains (e.g. gateway.bfl.ai) for polling URLs that
# differ from the submission host (api.bfl.ai). We validate against the
# registered domain rather than doing a strict same-origin check.
_BFL_REGISTERED_DOMAIN = "bfl.ai"


def assert_bfl_polling_url(polling_url: str) -> None:
    """Validate that a polling URL points to a BFL-controlled host.

    BFL returns polling URLs on subdomains like ``gateway.bfl.ai`` that differ
    from the submission host ``api.bfl.ai``. A strict same-origin check would
    reject these legitimate URLs. Instead we verify the host is ``bfl.ai`` or
    any subdomain of it, which keeps the SSRF guarantee (credentials only go
    to BFL-controlled infrastructure) without false-positives on regional hosts.

    Raises:
        BlackForestLabsError: If the polling URL scheme or host is not trusted.
    """
    parsed = urlparse(polling_url)
    host = (parsed.hostname or "").lower()

    if parsed.scheme != "https":
        raise BlackForestLabsError(
            status_code=502,
            message="Rejected polling URL: scheme must be https",
        )

    if host != _BFL_REGISTERED_DOMAIN and not host.endswith(
        "." + _BFL_REGISTERED_DOMAIN
    ):
        raise BlackForestLabsError(
            status_code=502,
            message="Rejected polling URL: host is not within the bfl.ai domain",
        )


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
    # Kontext models support both text-to-image and image editing
    "flux-kontext-pro": "/v1/flux-kontext-pro",
    "flux-kontext-max": "/v1/flux-kontext-max",
}
