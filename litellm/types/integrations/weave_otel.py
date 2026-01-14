from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class WeaveOtelConfig(BaseModel):
    """Configuration for Weave OpenTelemetry integration."""

    otlp_auth_headers: str | None = None
    endpoint: str | None = None
    project_id: str | None = None
    protocol: Literal["otlp_grpc", "otlp_http"] = "otlp_http"


class WeaveSpanAttributes(str, Enum):
    """
    Weave-specific span attributes for OpenTelemetry traces.

    Based on Weave's OTEL attribute mappings from:
    https://github.com/wandb/weave/blob/master/weave/trace_server/opentelemetry/constants.py
    """

    DISPLAY_NAME = "wandb.display_name"
    
    # Thread organization, similar to OpenInference session_id.
    THREAD_ID = "wandb.thread_id"
    IS_TURN = "wandb.is_turn"
    
