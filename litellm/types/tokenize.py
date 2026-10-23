from collections.abc import Mapping, Sequence

from pydantic import ConfigDict, Field, PrivateAttr

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class TokenizeRequest(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True, extra="allow")

    model: str
    prompt: str


class TokenizeResponse(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)
    model: str
    count: int = Field(ge=0)
    token_ids: Sequence[int] | None = None
    tokens: Sequence[str] | None = None

    _hidden_params: Mapping[str, object] = PrivateAttr(default_factory=dict)
