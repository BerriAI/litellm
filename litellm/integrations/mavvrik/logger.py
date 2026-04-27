"""Mavvrik callback logger — registered as the "mavvrik" callback string.

This class is the entry point for callbacks: ["mavvrik"] in config.yaml.
It acts as a marker so LiteLLM recognises "mavvrik" as a known integration.

The actual export work (query → CSV → upload) is done by the scheduler and
orchestrator, not on a per-request basis. This class is intentionally empty.
"""

from litellm.integrations.custom_logger import CustomLogger


class Logger(CustomLogger):
    """Mavvrik integration marker — registered via callbacks: ["mavvrik"]."""

    pass
