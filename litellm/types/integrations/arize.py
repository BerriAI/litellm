from typing import Optional

from pydantic import BaseModel


class ArizeConfig(BaseModel):
    space_key: str
    api_key: str
    grpc_endpoint: Optional[str] = None
    http_endpoint: Optional[str] = None
