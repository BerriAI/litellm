from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core.core_schema import SerializerFunctionWrapHandler

SCIM_ENTERPRISE_USER_SCHEMA = "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"
SCIM_ENTERPRISE_METADATA_KEY = "scim_enterprise"
SCIM_ENTITLEMENTS_METADATA_KEY = "scim_entitlements"
SCIM_ROLES_METADATA_KEY = "scim_roles"


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


class SCIMMultiValuedAttribute(BaseModel):
    value: str
    display: Optional[str] = None
    type: Optional[str] = None
    primary: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_bare_string(cls, data: object) -> object:
        if isinstance(data, str):
            return {"value": data}
        return data


SCIM_MULTI_VALUED_LIST_ADAPTER = TypeAdapter(List[SCIMMultiValuedAttribute])


class SCIMUserManager(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    value: Optional[str] = None
    displayName: Optional[str] = None
    ref: Optional[str] = Field(default=None, alias="$ref")


class SCIMEnterpriseUser(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    employeeNumber: Optional[str] = None
    costCenter: Optional[str] = None
    organization: Optional[str] = None
    division: Optional[str] = None
    department: Optional[str] = None
    manager: Optional[SCIMUserManager] = None


class SCIMUser(SCIMResource):
    model_config = ConfigDict(populate_by_name=True)

    userName: Optional[str] = None
    name: Optional[SCIMUserName] = None
    displayName: Optional[str] = None
    active: bool = True
    emails: Optional[List[SCIMUserEmail]] = None
    groups: Optional[List[SCIMUserGroup]] = None
    entitlements: Optional[List[SCIMMultiValuedAttribute]] = None
    roles: Optional[List[SCIMMultiValuedAttribute]] = None
    enterprise_user: Optional[SCIMEnterpriseUser] = Field(
        default=None,
        alias=SCIM_ENTERPRISE_USER_SCHEMA,
        serialization_alias=SCIM_ENTERPRISE_USER_SCHEMA,
    )

    @model_serializer(mode="wrap")
    def _omit_absent_optional_blocks(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        dumped = handler(self)
        if self.enterprise_user is None:
            dumped.pop(SCIM_ENTERPRISE_USER_SCHEMA, None)
            dumped.pop("enterprise_user", None)
        if self.entitlements is None:
            dumped.pop("entitlements", None)
        if self.roles is None:
            dumped.pop("roles", None)
        return dumped


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


# SCIM Service Provider Configuration Models
class SCIMFeature(BaseModel):
    supported: bool
    maxOperations: Optional[int] = None
    maxPayloadSize: Optional[int] = None
    maxResults: Optional[int] = None


class SCIMServiceProviderConfig(BaseModel):
    schemas: List[str] = ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"]
    patch: SCIMFeature = SCIMFeature(supported=True)
    bulk: SCIMFeature = SCIMFeature(supported=False)
    filter: SCIMFeature = SCIMFeature(supported=False)
    changePassword: SCIMFeature = SCIMFeature(supported=False)
    sort: SCIMFeature = SCIMFeature(supported=False)
    etag: SCIMFeature = SCIMFeature(supported=False)
    authenticationSchemes: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None


# SCIM ResourceType Models (RFC 7643 Section 6)
class SCIMSchemaExtension(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str  # aliased to "schema" in serialization
    required: bool

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        d["schema"] = d.pop("schema_")
        return d


class SCIMResourceType(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schemas: List[str] = ["urn:ietf:params:scim:schemas:core:2.0:ResourceType"]
    id: str
    name: str
    description: Optional[str] = None
    endpoint: str
    schema_: str  # "schema" is a reserved name in Pydantic context

    schemaExtensions: Optional[List[SCIMSchemaExtension]] = None
    meta: Optional[Dict[str, Any]] = None

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        d["schema"] = d.pop("schema_")
        if d.get("schemaExtensions") is None:
            d.pop("schemaExtensions", None)
        return d


# SCIM Schema Models (RFC 7643 Section 7)
class SCIMSchemaAttribute(BaseModel):
    name: str
    type: str
    multiValued: bool = False
    description: Optional[str] = None
    required: bool = False
    mutability: str = "readWrite"
    returned: str = "default"
    uniqueness: str = "none"
    subAttributes: Optional[List["SCIMSchemaAttribute"]] = None

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        if d.get("subAttributes") is None:
            d.pop("subAttributes", None)
        return d


class SCIMSchema(BaseModel):
    schemas: List[str] = ["urn:ietf:params:scim:schemas:core:2.0:Schema"]
    id: str
    name: str
    description: Optional[str] = None
    attributes: List[SCIMSchemaAttribute] = []
    meta: Optional[Dict[str, Any]] = None
