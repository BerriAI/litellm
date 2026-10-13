from typing import Literal

from pydantic import ConfigDict

from litellm.types.llms.base import LiteLLMBaseModel


class HealthResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"]
