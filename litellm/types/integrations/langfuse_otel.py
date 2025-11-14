from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any


class LangfuseOtelConfig(BaseModel):
    otlp_auth_headers: Optional[str] = None
    protocol: Protocol = "otlp_http" 

class LangfuseSpanAttributes(str, Enum):
    LANGFUSE_ENVIRONMENT = "langfuse.environment"

    # ---- Generation-level metadata ----
    GENERATION_NAME = "langfuse.generation.name"
    GENERATION_ID = "langfuse.generation.id"
    PARENT_OBSERVATION_ID = "langfuse.generation.parent_observation_id"
    GENERATION_VERSION = "langfuse.generation.version"
    MASK_INPUT = "langfuse.generation.mask_input"
    MASK_OUTPUT = "langfuse.generation.mask_output"

    # ---- Observation input/output ----
    OBSERVATION_INPUT = "langfuse.observation.input"
    OBSERVATION_OUTPUT = "langfuse.observation.output"

    # ---- Trace-level metadata ----
    TRACE_USER_ID = "user.id"
    SESSION_ID = "session.id"
    TAGS = "langfuse.trace.tags"
    TRACE_NAME = "langfuse.trace.name"
    TRACE_ID = "langfuse.trace.id"
    TRACE_METADATA = "langfuse.trace.metadata"
    TRACE_VERSION = "langfuse.trace.version"
    TRACE_RELEASE = "langfuse.trace.release"
    EXISTING_TRACE_ID = "langfuse.trace.existing_id"
    UPDATE_TRACE_KEYS = "langfuse.trace.update_keys"

    # ---- Misc / flags ----
    DEBUG_LANGFUSE = "langfuse.debug"