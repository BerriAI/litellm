from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import LitellmUserRoles


class CurrentUserResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    user_role: LitellmUserRoles | None
    user_email: str | None
