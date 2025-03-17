from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any


class ArizeConfig(BaseModel):
    space_key: str
    space_id: Optional[str]
    api_key: str
    protocol: Protocol
    endpoint: str
