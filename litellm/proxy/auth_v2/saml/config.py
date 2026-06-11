from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator

DEFAULT_SAML_ATTRIBUTE_MAP = {
    "email": "email",
    "mail": "email",
    "givenName": "given_name",
    "surname": "family_name",
    "sn": "family_name",
    "displayName": "display_name",
    "userName": "user_name",
    "uid": "user_name",
    "groups": "groups",
    "roles": "roles",
}


class SAMLConfig(BaseModel):
    enabled: bool = False
    entity_id: str
    acs_url: str
    idp_metadata: str = ""
    sp_key_file: Optional[str] = None
    sp_cert_file: Optional[str] = None
    allow_unsolicited: bool = False
    xmlsec_binary: Optional[str] = None
    attribute_map: Dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SAML_ATTRIBUTE_MAP)
    )

    @model_validator(mode="after")
    def _require_idp_metadata(self) -> "SAMLConfig":
        if self.enabled and not self.idp_metadata.strip():
            raise ValueError(
                "SAML enabled but idp_metadata is empty (inline XML, local path, or URL)"
            )
        return self
