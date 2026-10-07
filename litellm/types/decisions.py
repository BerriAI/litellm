from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

from pydantic import ConfigDict, Field, PrivateAttr, model_validator, with_config
from typing_extensions import ReadOnly, Required, TypedDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase

DecisionsJSON: TypeAlias = str | Mapping[str, object] | Sequence[object]
NoulCriteria: TypeAlias = Mapping[Literal["true", "false"], DecisionsJSON | None]


class NoulQuestion(LiteLLMPydanticObjectBase):
    type: Literal["noul"]
    instructions: DecisionsJSON | None = None
    criteria: NoulCriteria | None = None

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="after")
    def require_instructions_or_criteria(self) -> "NoulQuestion":
        if self.instructions is None and self.criteria is None:
            raise ValueError("A noul question requires instructions or criteria")
        return self


class ChoiceQuestion(LiteLLMPydanticObjectBase):
    type: Literal["choice"]
    instructions: DecisionsJSON | None = None
    criteria: Annotated[Mapping[str, DecisionsJSON | None], Field(min_length=1, max_length=255)]

    model_config = ConfigDict(extra="allow", frozen=True)


class ScoreQuestion(LiteLLMPydanticObjectBase):
    type: Literal["score"]
    instructions: DecisionsJSON | None = None
    criteria: Annotated[Sequence[DecisionsJSON], Field(min_length=1, max_length=10)]

    model_config = ConfigDict(extra="allow", frozen=True)


DecisionQuestion: TypeAlias = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]

DecisionQuestionMap: TypeAlias = Annotated[
    Mapping[Annotated[str, Field(min_length=1)], DecisionQuestion],
    Field(min_length=1, max_length=128),
]


class DecisionsRequestBody(LiteLLMPydanticObjectBase):
    state: DecisionsJSON
    questions: DecisionQuestionMap

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionsRequest(DecisionsRequestBody):
    model: str


@with_config(ConfigDict(extra="allow"))
class DecisionsCallParams(TypedDict, total=False):
    model: Required[ReadOnly[str]]
    state: Required[ReadOnly[DecisionsJSON]]
    questions: Required[ReadOnly[DecisionQuestionMap]]
    api_key: ReadOnly[str | None]
    api_base: ReadOnly[str | None]
    timeout: ReadOnly[float | None]
    custom_llm_provider: ReadOnly[str | None]
    extra_headers: ReadOnly[Mapping[str, str] | None]


class NoulAnswer(LiteLLMPydanticObjectBase):
    type: Literal["noul"]
    noul: float

    model_config = ConfigDict(extra="allow", frozen=True)


class ChoiceAnswer(LiteLLMPydanticObjectBase):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: Mapping[str, float]

    model_config = ConfigDict(extra="allow", frozen=True)


class ScoreAnswer(LiteLLMPydanticObjectBase):
    type: Literal["score"]
    score: float
    confidence: float
    legend: Mapping[str, DecisionsJSON]
    probabilities: Mapping[str, float]

    model_config = ConfigDict(extra="allow", frozen=True)


DecisionAnswer: TypeAlias = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class DecisionsUsage(LiteLLMPydanticObjectBase):
    input_tokens: int = 0
    output_tokens: int = 0

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionsResponse(LiteLLMPydanticObjectBase):
    model: str | None = None
    answers: Mapping[str, DecisionAnswer]
    usage: DecisionsUsage | None = None

    model_config = ConfigDict(extra="allow", frozen=True)

    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)

    @property
    def hidden_params(self) -> dict[str, object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    def set_hidden_params(self, params: Mapping[str, object]) -> None:
        self._hidden_params.update(params)
