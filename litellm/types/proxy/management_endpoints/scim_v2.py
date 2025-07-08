from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, EmailStr, field_validator


class LiteLLM_UserScimMetadata(BaseModel):
    """
    Scim metadata stored in LiteLLM_UserTable.metadata
    """

    givenName: Optional[str] = None
    familyName: Optional[str] = None


# SCIM Resource Models
class SCIMResource(BaseModel):
    schemas: List[str]
    id: Optional[str] = None
    externalId: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class SCIMUserName(BaseModel):
    familyName: Optional[str] = None
    givenName: Optional[str] = None
    formatted: Optional[str] = None
    middleName: Optional[str] = None
    honorificPrefix: Optional[str] = None
    honorificSuffix: Optional[str] = None


class SCIMUserEmail(BaseModel):
    value: EmailStr
    type: Optional[str] = None
    primary: Optional[bool] = None


class SCIMUserGroup(BaseModel):
    value: str  # Group ID
    display: Optional[str] = None  # Group display name
    type: Optional[str] = "direct"  # direct or indirect


class SCIMUser(SCIMResource):
    userName: Optional[str] = None
    name: Optional[SCIMUserName] = None
    displayName: Optional[str] = None
    active: bool = True
    emails: Optional[List[SCIMUserEmail]] = None
    groups: Optional[List[SCIMUserGroup]] = None


class SCIMMember(BaseModel):
    value: str  # User ID
    display: Optional[str] = None  # Username or email


class SCIMGroup(SCIMResource):
    displayName: str
    members: Optional[List[SCIMMember]] = None


# SCIM List Response Models
class SCIMListResponse(BaseModel):
    schemas: List[str] = ["urn:ietf:params:scim:api:messages:2.0:ListResponse"]
    totalResults: int
    startIndex: Optional[int] = 1
    itemsPerPage: Optional[int] = 10
    Resources: Union[List[SCIMUser], List[SCIMGroup]]


# SCIM PATCH Operation Models
class SCIMPatchOperation(BaseModel):
    op: str
    path: Optional[str] = None
    value: Optional[Any] = None

    @field_validator("op", mode="before")
    @classmethod
    def normalize_op(cls, v):
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower not in {"add", "remove", "replace"}:
                raise ValueError("op must be add, remove, or replace")
            return v_lower
        return v


class SCIMPatchOp(BaseModel):
    schemas: List[str] = ["urn:ietf:params:scim:api:messages:2.0:PatchOp"]
    Operations: List[SCIMPatchOperation]
