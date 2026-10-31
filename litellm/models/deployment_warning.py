from collections.abc import Mapping
from typing import Literal, Protocol

from pydantic import AwareDatetime, ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel


class AdminIgnore(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    source: Literal["admin"] = "admin"
    ignored_by: str
    ignored_at: AwareDatetime
    ignored_until: AwareDatetime | None = Field(
        default=None,
        description="When the snooze ends. Null means the warning is ignored until an admin restores it.",
    )


class DeploymentWarningIgnoreStore(Protocol):
    async def get_ignores(self) -> Mapping[str, AdminIgnore]: ...

    async def save_ignores(self, ignores: Mapping[str, AdminIgnore]) -> None: ...
