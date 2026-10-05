from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class VertexGemmaContainerError(BaseModel):
    model_config = ConfigDict(frozen=True)
    object: Literal["error"]
    message: str
    code: Annotated[int, Field(ge=400, le=599)]
