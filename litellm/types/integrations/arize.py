from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any


class ArizeConfig(BaseModel):
    space_id: str | None = None
    space_key: str | None = None
    api_key: str | None = None
    protocol: Protocol
    endpoint: str
    project_name: str | None = None
