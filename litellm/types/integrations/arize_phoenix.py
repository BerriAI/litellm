from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel

from .arize import Protocol


class ArizePhoenixConfig(BaseModel):
    otlp_auth_headers: Optional[str] = None
    protocol: Protocol
    endpoint: str
