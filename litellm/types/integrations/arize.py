from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any


class ArizeConfig(BaseModel):
    space_id: Optional[str] = None
    space_key: Optional[str] = None
    api_key: Optional[str] = None
    protocol: Protocol
    endpoint: str
