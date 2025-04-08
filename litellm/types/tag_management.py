from typing import Dict, List, Optional

from pydantic import BaseModel


class TagBase(BaseModel):
    name: str
    description: Optional[str] = None
    models: Optional[List[str]] = None
    model_info: Optional[Dict[str, str]] = None  # maps model_id to model_name


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
