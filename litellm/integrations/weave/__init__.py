"""
Weave (W&B) integration for LiteLLM via OpenTelemetry.
"""

from litellm.integrations.weave.weave_otel import WeaveOtelLogger

__all__ = ["WeaveOtelLogger"]
