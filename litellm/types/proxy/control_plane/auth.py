from typing import Final

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel

USERNAME_MAX_LENGTH: Final = 256
PASSWORD_MAX_LENGTH: Final = 1024


class LoginRequest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    username: str = Field(min_length=1, max_length=USERNAME_MAX_LENGTH)
    password: str = Field(min_length=1, max_length=PASSWORD_MAX_LENGTH)


class SessionTokenResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class PrincipalResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    subject: str
    email: str | None
    roles: tuple[str, ...]
    scopes: tuple[str, ...]
    dataplanes: tuple[str, ...] | None
    token_kind: str
