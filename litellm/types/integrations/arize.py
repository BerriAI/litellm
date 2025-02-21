from typing import TYPE_CHECKING, Literal, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any
    
class ArizeConfig(BaseModel):
    space_key: str
    api_key: str
    protocol: Protocol
    endpoint: str
