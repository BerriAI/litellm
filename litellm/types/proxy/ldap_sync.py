from typing import Final, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from typing_extensions import Self


class LDAPSyncSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    server_url: str = ""
    group_search_base: str = ""
    group_filter: str = ""
    user_search_base: str = ""
    group_id_attribute: str = "cn"
    group_name_attribute: str = "cn"
    member_attribute: str = "member"
    member_lookup_attribute: str = "uid"
    user_match_attribute: str = "mail"
    litellm_match_field: Literal["user_email", "user_id", "sso_user_id"] = "user_email"
    interval_minutes: float = Field(default=30, ge=0.1, allow_inf_nan=False)
    dry_run: bool = False
    bind_dn: str | None = None
    bind_password: SecretStr | None = None
    ca_cert_path: str | None = None
    start_tls: bool = False
    timeout_seconds: int = Field(default=30, ge=1, le=120)

    @field_validator("server_url")
    @classmethod
    def validate_server_url(cls, value: str) -> str:
        parsed: Final = urlsplit(value)
        if parsed.scheme not in {"ldap", "ldaps"} or not parsed.hostname:
            raise ValueError("server_url must be an ldap:// or ldaps:// server URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("server_url must contain only the LDAP host and optional port")
        _ = parsed.port
        return value

    @field_validator(
        "group_id_attribute",
        "group_name_attribute",
        "member_attribute",
        "member_lookup_attribute",
        "user_match_attribute",
    )
    @classmethod
    def validate_attribute(cls, value: str) -> str:
        if not value or not value[0].isalpha() or not all(char.isalnum() or char == "-" for char in value):
            raise ValueError("LDAP attributes must be plain attribute names")
        return value

    @model_validator(mode="after")
    def validate_bind(self) -> Self:
        if self.enabled and not all(
            (self.server_url, self.group_search_base, self.group_filter, self.user_search_base)
        ):
            raise ValueError(
                "Enabled LDAP sync requires server_url, group_search_base, group_filter and user_search_base"
            )
        if bool(self.bind_dn) != bool(self.bind_password):
            raise ValueError("bind_dn and bind_password must be configured together")
        if (self.enabled or self.bind_dn) and urlsplit(self.server_url).scheme == "ldap" and not self.start_tls:
            raise ValueError("LDAP sync and service-account binds require TLS: use ldaps:// or start_tls: true")
        return self
