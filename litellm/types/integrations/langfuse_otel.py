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

class LangfuseSpanAttributes(Enum):
    LANGFUSE_ENVIRONMENT = "langfuse.environment"