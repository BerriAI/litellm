from typing import Literal

from pydantic import ConfigDict

from litellm.types.llms.base_model import LiteLLMBaseModel


class RoleResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    scopes: tuple[str, ...]
    inherits: tuple[str, ...]
    source: Literal["config", "runtime"]


class RoleListResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[RoleResponse, ...]
