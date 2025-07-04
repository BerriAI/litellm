from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any


class ArizeConfig(BaseModel):
    space_key: Optional[str] = None
    """
    Deprecated field name, arize used to call this space_key
    """

    api_key: Optional[str] = None
    """
    Arize API key
    """
    protocol: Protocol
    endpoint: str
