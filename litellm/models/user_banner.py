from typing import Final, Literal

from pydantic import ConfigDict, Field, model_validator

from litellm.types.llms.base import LiteLLMBaseModel

USER_BANNER_MAX_MESSAGE_LENGTH: Final = 4000
UserBannerSeverity = Literal["info", "warning", "error"]


class UserBannerUpdate(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description="If true, the banner is shown to all authenticated dashboard users.",
    )
    message: str = Field(
        default="",
        max_length=USER_BANNER_MAX_MESSAGE_LENGTH,
        description="Banner text shown to dashboard users. Markdown is supported.",
    )
    severity: UserBannerSeverity = Field(
        default="info",
        description="Visual style of the banner.",
    )

    @model_validator(mode="after")
    def _require_message_when_enabled(self) -> "UserBannerUpdate":
        if self.enabled and not self.message.strip():
            raise ValueError("message must be non-empty when the banner is enabled")
        return self


class UserBanner(UserBannerUpdate):
    revision: str = Field(
        default="",
        description=(
            "Server-stamped opaque publish identity; a fresh value is generated on every "
            "update so clients re-surface dismissed banners on republish."
        ),
    )
