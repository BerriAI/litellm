from pydantic import ConfigDict

from litellm.proxy._types import LitellmUserRoles
from litellm.types.llms.base_model import LiteLLMBaseModel


class CurrentUserResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    user_role: LitellmUserRoles | None
    user_email: str | None
