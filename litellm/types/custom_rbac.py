from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

CustomRBACRoleSource: TypeAlias = Literal["config", "db"]


class CustomRBACRole(BaseModel):
    """A custom RBAC role: a named allow-list of routes, optionally inheriting other custom roles.

    Each entry in ``allowed_routes`` is either ``"*"``, a ``LiteLLMRoutes`` group name
    (e.g. ``llm_api_routes``), an exact route, or a wildcard pattern (e.g. ``/team/*``)
    """

    role_name: str
    description: str | None = None
    allowed_routes: tuple[str, ...] = ()
    inherits: tuple[str, ...] = ()


class CustomRBACRoleCreateRequest(BaseModel):
    role_name: str = Field(min_length=1)
    description: str | None = None
    allowed_routes: tuple[str, ...] = ()
    inherits: tuple[str, ...] = ()


class CustomRBACRoleUpdateRequest(BaseModel):
    role_name: str = Field(min_length=1)
    description: str | None = None
    allowed_routes: tuple[str, ...] | None = None
    inherits: tuple[str, ...] | None = None


class CustomRBACRoleDeleteRequest(BaseModel):
    role_name: str = Field(min_length=1)


class CustomRBACRoleResponse(CustomRBACRole):
    source: CustomRBACRoleSource = "db"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    created_by: str | None = None
    updated_by: str | None = None


class CustomRBACRoleListResponse(BaseModel):
    roles: tuple[CustomRBACRoleResponse, ...]


class CustomRBACRoleDeleteResponse(BaseModel):
    role_name: str
    status: Literal["deleted"] = "deleted"
