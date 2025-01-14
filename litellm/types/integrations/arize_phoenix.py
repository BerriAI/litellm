from typing import Optional

from pydantic import BaseModel


class ArizePhoenixConfig(BaseModel):
    otlp_auth_headers: Optional[str] = None
    protocol: str
    endpoint: str 
