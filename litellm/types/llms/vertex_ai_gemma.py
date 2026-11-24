from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel


class VertexGemmaContainerError(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    object: Literal["error"]
    message: str
    code: Annotated[int, Field(ge=400, le=599)]
