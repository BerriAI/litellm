from typing import Optional

from pydantic import BaseModel


class ArizeConfig(BaseModel):
    space_key: str
    api_key: str
    protocol: str
    endpoint: str
