from pydantic import Field, PrivateAttr

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class TokenizeResponse(LiteLLMPydanticObjectBase):
    model: str
    count: int = Field(ge=0)
    token_ids: list[int] | None = None
    tokens: list[str] | None = None

    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)
