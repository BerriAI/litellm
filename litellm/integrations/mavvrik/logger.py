"""Mavvrik callback logger — registered as the "mavvrik" callback string.

This class is the entry point for callbacks: ["mavvrik"] in config.yaml.
It acts as a marker so LiteLLM recognises "mavvrik" as a known integration.

The actual export work (query → CSV → upload) is done by the scheduler and
orchestrator, not on a per-request basis. This class is intentionally empty.
"""

import os

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger


class Logger(CustomLogger):
    """Mavvrik integration marker — registered via callbacks: ["mavvrik"]."""

    def __init__(self) -> None:
        super().__init__()
        _required = ("MAVVRIK_API_KEY", "MAVVRIK_API_ENDPOINT", "MAVVRIK_CONNECTION_ID")
        if not all(os.getenv(v) for v in _required):
            verbose_proxy_logger.warning(
                "mavvrik: callbacks: ['mavvrik'] is set but credentials are not configured. "
                "Call POST /mavvrik/init or set MAVVRIK_API_KEY, MAVVRIK_API_ENDPOINT, "
                "and MAVVRIK_CONNECTION_ID to enable exports."
            )
