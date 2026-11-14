from collections.abc import Mapping, Sequence
from typing import Annotated, Final, Literal

from pydantic import AfterValidator, Field, field_validator, model_validator

from litellm.models.team import BudgetLimitEntry
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.common_utils.timezone_utils import budget_duration_error
from litellm.types.proxy.auth.user_roles import LitellmUserRoles
from litellm.types.proxy.management_endpoints.request_base import GenerateRequestBase
from litellm.types.utils import LiteLLMPydanticObjectBase


class NewUserRequestTeam(LiteLLMPydanticObjectBase):
    team_id: str
    max_budget_in_team: float | None = None
    user_role: Literal["user", "admin"] = "user"


def _validate_custom_user_id(value: str) -> str:
    normalized: Final = value.strip()
    if not normalized:
        raise ValueError("user_id must not be empty or contain only whitespace")
    if "\x00" in normalized:
        raise ValueError("user_id must not contain NUL characters")
    return normalized


def _validate_user_windows(windows: Sequence[BudgetLimitEntry] | None) -> Sequence[BudgetLimitEntry] | None:
    if windows is None:
        return None
    if len({window.budget_duration for window in windows}) != len(windows):
        raise ValueError("Each budget duration may appear only once")
    for window in windows:
        if (error := budget_duration_error(window.budget_duration)) is not None:
            raise ValueError(error)
        if not 0 <= window.max_budget < float("inf"):
            raise ValueError("Window max_budget must be finite and non-negative")
    return windows


class NewUserRequest(GenerateRequestBase):
    _validate_windows = field_validator("budget_limits")(_validate_user_windows)
    user_id: Annotated[str, AfterValidator(_validate_custom_user_id)] | None = None
    tpm_limit: int | None = Field(default=None, ge=0)
    rpm_limit: int | None = Field(default=None, ge=0)
    max_budget: float | None = None
    user_email: str | None = None
    user_alias: str | None = None
    user_role: (
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
        | None
    ) = None
    teams: list[str] | list[NewUserRequestTeam] | None = None
    auto_create_key: bool = True  # flag used for returning a key as part of the /user/new response
    send_invite_email: bool | None = None
    sso_user_id: str | None = None
    organizations: list[str] | None = None
    password: str | None = None

    @field_validator("password")
    @classmethod
    def password_not_supported(cls, value: str | None) -> str | None:
        if value is not None:
            raise ValueError(
                "password cannot be set via /user/new. Users set their own password through an "
                "invitation link (POST /invitation/new)."
            )
        return value


class UpdateUserRequestNoUserIDorEmail(GenerateRequestBase):  # shared with BulkUpdateUserRequest
    _validate_windows = field_validator("budget_limits")(_validate_user_windows)
    tpm_limit: int | None = Field(default=None, ge=0)
    rpm_limit: int | None = Field(default=None, ge=0)
    # repr=False keeps the plaintext out of management-endpoint alerts, which str() the request model
    password: str | None = Field(default=None, repr=False)
    spend: float | None = None
    metadata: dict[object, object] | None = None
    user_alias: str | None = None
    user_role: (
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
        | None
    ) = None
    max_budget: float | None = None


class UpdateUserRequest(UpdateUserRequestNoUserIDorEmail):
    # Note: the defaults of all Params here MUST BE NONE
    # else they will get overwritten
    user_id: str | None = None
    user_email: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_user_info(cls, values: Mapping[str, object]) -> Mapping[str, object]:
        if values.get("user_id") is None and values.get("user_email") is None:
            raise ValueError("Either user id or user email must be provided")
        return values


class LiteLLM_UserTableWithKeyCount(LiteLLM_UserTable):
    key_count: int = 0
