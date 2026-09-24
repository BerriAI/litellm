from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class VertexGemmaContainerError(BaseModel):
    model_config = ConfigDict(frozen=True)
    object: Literal["error"]
    message: str
    code: Annotated[int, Field(ge=400, le=599)]


def parse_vertex_gemma_container_error(predictions: object) -> VertexGemmaContainerError | None:
    try:
        return VertexGemmaContainerError.model_validate(predictions)
    except ValidationError:
        return None
