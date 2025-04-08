from typing import List, Optional

from pydantic import BaseModel


class TagBase(BaseModel):
    name: str
    description: Optional[str] = None
    allowed_llms: Optional[List[str]] = None


class TagNewRequest(TagBase):
    pass


class TagUpdateRequest(TagBase):
    pass


class TagDeleteRequest(BaseModel):
    name: str


class TagInfoRequest(BaseModel):
    names: List[str]
