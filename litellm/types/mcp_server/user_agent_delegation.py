from datetime import datetime

from pydantic import BaseModel, field_validator, model_validator


class UserAgentDelegation(BaseModel):
    delegation_id: str
    user_id: str
    agent_id: str
    granted_at: datetime
    granted_by: str
    revoked_at: datetime | None = None
    revoked_by: str | None = None


def _reject_blank(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError("must not be blank")
    return value


class NewUserAgentDelegationRequest(BaseModel):
    user_id: str | None = None
    user_email: str | None = None
    agent_id: str

    _no_blanks = field_validator("user_id", "user_email", "agent_id")(_reject_blank)

    @model_validator(mode="after")
    def exactly_one_target(self) -> "NewUserAgentDelegationRequest":
        if bool(self.user_id) == bool(self.user_email):
            raise ValueError("provide exactly one of user_id or user_email")
        return self


class RevokeUserAgentDelegationRequest(BaseModel):
    user_id: str
    agent_id: str

    _no_blanks = field_validator("user_id", "agent_id")(_reject_blank)
