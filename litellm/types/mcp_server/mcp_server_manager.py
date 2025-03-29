from typing import Optional

from mcp import ClientSession
from pydantic import BaseModel, ConfigDict


class MCPSSEServer(BaseModel):
    name: str
    url: str
    client_session: Optional[ClientSession] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
