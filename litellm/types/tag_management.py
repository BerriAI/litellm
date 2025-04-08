from typing import List, Optional

from pydantic import BaseModel


class TagBase(BaseModel):
    name: str
    description: Optional[str] = None
    models: Optional[List[str]] = None


class TagConfig(TagBase):
    created_at: str
    updated_at: str
    created_by: Optional[str] = None


class TagNewRequest(TagBase):
    pass


class TagUpdateRequest(TagBase):
    pass


class TagDeleteRequest(BaseModel):
    name: str


class TagInfoRequest(BaseModel):
    names: List[str]
