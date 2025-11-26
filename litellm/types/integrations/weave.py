from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any


class WeaveOtelConfig(BaseModel):
    """Configuration for Weave OpenTelemetry integration."""

    otlp_auth_headers: str | None = None
    endpoint: str | None = None
    project_id: str | None = None
    protocol: Protocol = "otlp_http"


class WeaveSpanAttributes(str, Enum):
    """
    Weave-specific span attributes for OpenTelemetry traces.

    Based on Weave's OTEL integration documentation:
    https://docs.wandb.ai/weave/guides/tracking/otel

    Weave maps attributes from multiple frameworks. We use OpenInference
    conventions (input.value, output.value, llm.*) which Weave recognizes.
    """

    # ---- Thread organization (Weave-specific) ----
    THREAD_ID = "wandb.thread_id"
    IS_TURN = "wandb.is_turn"
    DISPLAY_NAME = "wandb.display_name"

    # ---- Input/Output (OpenInference - recognized by Weave) ----
    INPUT_VALUE = "input.value"
    OUTPUT_VALUE = "output.value"

    # ---- LLM attributes (OpenInference - recognized by Weave) ----
    LLM_MODEL_NAME = "llm.model_name"
    LLM_PROVIDER = "llm.provider"
    LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
    LLM_INPUT_MESSAGES = "llm.input_messages"
    LLM_OUTPUT_MESSAGES = "llm.output_messages"

    # ---- Token counts (OpenInference - recognized by Weave) ----
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"

    # ---- Span kind (recognized by Weave) ----
    OPENINFERENCE_SPAN_KIND = "openinference.span.kind"

    # ---- Trace-level metadata ----
    TRACE_USER_ID = "user.id"
    SESSION_ID = "session.id"
    METADATA = "metadata"

    # ---- Generation-level metadata ----
    GENERATION_NAME = "gen_ai.operation.name"
    GENERATION_ID = "gen_ai.response.id"
