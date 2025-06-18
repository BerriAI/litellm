from typing import TYPE_CHECKING, Literal, Optional, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any


class LangfuseOtelConfig(BaseModel):
    otlp_auth_headers: Optional[str] = None
    protocol: Protocol = "otlp_http" 