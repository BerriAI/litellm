from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any


class LangfuseOtelConfig(LiteLLMBaseModel):
    otlp_auth_headers: str | None = None
    protocol: Protocol = "otlp_http"


class LangfuseSpanAttributes(str, Enum):
    LANGFUSE_ENVIRONMENT = "langfuse.environment"
    VERSION = "langfuse.version"
    RELEASE = "langfuse.release"

    # ---- Generation-level metadata ----
    GENERATION_NAME = "langfuse.generation.name"
    GENERATION_ID = "langfuse.generation.id"
    PARENT_OBSERVATION_ID = "langfuse.generation.parent_observation_id"
    MASK_INPUT = "langfuse.generation.mask_input"
    MASK_OUTPUT = "langfuse.generation.mask_output"

    # ---- Observation input/output ----
    OBSERVATION_INPUT = "langfuse.observation.input"
    OBSERVATION_OUTPUT = "langfuse.observation.output"
    OBSERVATION_COMPLETION_START_TIME = "langfuse.observation.completion_start_time"

    # ---- Trace-level metadata ----
    TRACE_USER_ID = "user.id"
    SESSION_ID = "session.id"
    TAGS = "langfuse.trace.tags"
    TRACE_NAME = "langfuse.trace.name"
    TRACE_ID = "langfuse.trace.id"
    TRACE_METADATA = "langfuse.trace.metadata"
    EXISTING_TRACE_ID = "langfuse.trace.existing_id"
    UPDATE_TRACE_KEYS = "langfuse.trace.update_keys"

    # ---- Misc / flags ----
    DEBUG_LANGFUSE = "langfuse.debug"
