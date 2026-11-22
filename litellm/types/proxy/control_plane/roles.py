from typing import Literal

from pydantic import BaseModel, ConfigDict


class RoleResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    scopes: tuple[str, ...]
    inherits: tuple[str, ...]
    source: Literal["config", "runtime"]


class RoleListResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[RoleResponse, ...]
