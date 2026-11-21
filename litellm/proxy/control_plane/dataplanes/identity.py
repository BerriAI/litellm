from typing import Final

from pydantic import ConfigDict, ValidationError

from litellm.types.llms.base_model import LiteLLMBaseModel

USER_INFO_PATH: Final = "/user/info"


class UserInfo(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    user_id: str | None = None
    user_email: str | None = None
    user_role: str | None = None


class UserInfoBody(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    user_id: str | None = None
    user_info: UserInfo | None = None


def parse_user_info(body: object) -> UserInfoBody | None:
    if not isinstance(body, dict):
        return None
    try:
        return UserInfoBody.model_validate(body)
    except ValidationError:
        return None


def resolved_user_id(info: UserInfoBody) -> str | None:
    if info.user_id:
        return info.user_id
    return None if info.user_info is None else info.user_info.user_id or None
