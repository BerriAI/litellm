from typing import Any, Final, Literal, Optional, Union

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

SCIM_ENTERPRISE_USER_SCHEMA: Final = "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"
SCIM_ENTERPRISE_METADATA_KEY: Final = "scim_enterprise"
SCIM_ENTITLEMENTS_METADATA_KEY: Final = "scim_entitlements"
SCIM_ROLES_METADATA_KEY: Final = "scim_roles"

SCIM_MANAGED_TEAM_METADATA_KEY: Final = "scim_managed"
SCIM_TEAM_DATA_METADATA_KEY: Final = "scim_data"


class LiteLLM_UserScimMetadata(BaseModel):
    """
    Scim metadata stored in LiteLLM_UserTable.metadata
    """

    givenName: str | None = None
    familyName: str | None = None


# SCIM Resource Models
class SCIMResource(BaseModel):
    schemas: list[str]
    id: str | None = None
    externalId: str | None = None
    meta: dict[str, Any] | None = None


class SCIMUserName(BaseModel):
    familyName: str | None = None
    givenName: str | None = None
    formatted: str | None = None
    middleName: str | None = None
    honorificPrefix: str | None = None
    honorificSuffix: str | None = None


class SCIMUserEmail(BaseModel):
    value: EmailStr
    type: str | None = None
    primary: bool | None = None


class SCIMUserGroup(BaseModel):
    value: str  # Group ID
    display: str | None = None  # Group display name
    type: str | None = "direct"  # direct or indirect


class SCIMMultiValuedAttribute(BaseModel):
    value: str
    display: str | None = None
    type: str | None = None
    primary: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_bare_string(cls, data: object) -> object:
        if isinstance(data, str):
            return {"value": data}
        return data


SCIM_MULTI_VALUED_LIST_ADAPTER: Final = TypeAdapter(list[SCIMMultiValuedAttribute])

SCIM_MULTI_VALUED_ATTRIBUTE_METADATA_KEYS: Final = {
    "entitlements": SCIM_ENTITLEMENTS_METADATA_KEY,
    "roles": SCIM_ROLES_METADATA_KEY,
}


class SCIMUserManager(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    value: str | None = None
    displayName: str | None = None
    ref: str | None = Field(default=None, alias="$ref")


class SCIMEnterpriseUser(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    employeeNumber: str | None = None
    costCenter: str | None = None
    organization: str | None = None
    division: str | None = None
    department: str | None = None
    manager: SCIMUserManager | None = None


class SCIMUser(SCIMResource):
    model_config = ConfigDict(populate_by_name=True)

    userName: str | None = None
    name: SCIMUserName | None = None
    displayName: str | None = None
    active: bool = True
    emails: list[SCIMUserEmail] | None = None
    groups: list[SCIMUserGroup] | None = None
    entitlements: list[SCIMMultiValuedAttribute] | None = None
    roles: list[SCIMMultiValuedAttribute] | None = None
    enterprise_user: SCIMEnterpriseUser | None = Field(
        default=None,
        alias=SCIM_ENTERPRISE_USER_SCHEMA,
        serialization_alias=SCIM_ENTERPRISE_USER_SCHEMA,
    )

    @model_serializer(mode="wrap")
    def _omit_absent_optional_blocks(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        dumped: Final = handler(self)
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
    display: str | None = None  # Username or email
    type: str | None = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: object) -> str | None:
        """Anything that is not a string carries no canonical type, and rejecting the
        request over it would be a regression: before this field existed the value was
        parsed away silently."""
        return v if isinstance(v, str) else None


class SCIMGroup(SCIMResource):
    displayName: str
    members: list[SCIMMember] | None = None


# SCIM List Response Models
class SCIMListResponse(BaseModel):
    schemas: list[str] = ["urn:ietf:params:scim:api:messages:2.0:ListResponse"]
    totalResults: int
    startIndex: int | None = 1
    itemsPerPage: int | None = 10
    Resources: list[SCIMUser] | list[SCIMGroup]


# SCIM PATCH Operation Models
class SCIMPatchOperation(BaseModel):
    op: str
    path: str | None = None
    value: Any | None = None

    @field_validator("op", mode="before")
    @classmethod
    def normalize_op(cls, v):
        if isinstance(v, str):
            v_lower: Final = v.lower()
            if v_lower not in {"add", "remove", "replace"}:
                raise ValueError("op must be add, remove, or replace")
            return v_lower
        return v


class SCIMPatchOp(BaseModel):
    schemas: list[str] = ["urn:ietf:params:scim:api:messages:2.0:PatchOp"]
    Operations: list[SCIMPatchOperation]


# SCIM Service Provider Configuration Models
class SCIMFeature(BaseModel):
    supported: bool
    maxOperations: int | None = None
    maxPayloadSize: int | None = None
    maxResults: int | None = None


class SCIMServiceProviderConfig(BaseModel):
    schemas: list[str] = ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"]
    patch: SCIMFeature = SCIMFeature(supported=True)
    bulk: SCIMFeature = SCIMFeature(supported=False)
    filter: SCIMFeature = SCIMFeature(supported=False)
    changePassword: SCIMFeature = SCIMFeature(supported=False)
    sort: SCIMFeature = SCIMFeature(supported=False)
    etag: SCIMFeature = SCIMFeature(supported=False)
    authenticationSchemes: list[dict[str, Any]] | None = None
    meta: dict[str, Any] | None = None


# SCIM ResourceType Models (RFC 7643 Section 6)
class SCIMSchemaExtension(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str  # aliased to "schema" in serialization
    required: bool

    def model_dump(self, **kwargs):
        d: Final = super().model_dump(**kwargs)
        d["schema"] = d.pop("schema_")
        return d


class SCIMResourceType(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schemas: list[str] = ["urn:ietf:params:scim:schemas:core:2.0:ResourceType"]
    id: str
    name: str
    description: str | None = None
    endpoint: str
    schema_: str  # "schema" is a reserved name in Pydantic context

    schemaExtensions: list[SCIMSchemaExtension] | None = None
    meta: dict[str, Any] | None = None

    def model_dump(self, **kwargs):
        d: Final = super().model_dump(**kwargs)
        d["schema"] = d.pop("schema_")
        if d.get("schemaExtensions") is None:
            d.pop("schemaExtensions", None)
        return d


# SCIM Schema Models (RFC 7643 Section 7)
class SCIMSchemaAttribute(BaseModel):
    name: str
    type: str
    multiValued: bool = False
    description: str | None = None
    required: bool = False
    mutability: str = "readWrite"
    returned: str = "default"
    uniqueness: str = "none"
    subAttributes: list["SCIMSchemaAttribute"] | None = None

    def model_dump(self, **kwargs):
        d: Final = super().model_dump(**kwargs)
        if d.get("subAttributes") is None:
            d.pop("subAttributes", None)
        return d


class SCIMSchema(BaseModel):
    schemas: list[str] = ["urn:ietf:params:scim:schemas:core:2.0:Schema"]
    id: str
    name: str
    description: str | None = None
    attributes: list[SCIMSchemaAttribute] = []
    meta: dict[str, Any] | None = None
