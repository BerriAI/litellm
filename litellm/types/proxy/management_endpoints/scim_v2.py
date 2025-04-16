from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, EmailStr


# SCIM Resource Models
class SCIMResource(BaseModel):
    schemas: List[str]
    id: Optional[str] = None
    externalId: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class SCIMUserName(BaseModel):
    familyName: str
    givenName: str
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
    userName: str
    name: SCIMUserName
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
    Resources: List[Union[SCIMUser, SCIMGroup]]
