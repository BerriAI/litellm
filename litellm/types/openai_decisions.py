from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

from pydantic import ConfigDict, Field, PrivateAttr, with_config
from typing_extensions import ReadOnly, Required, TypedDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase

ChoiceValue: TypeAlias = str | bool


class DecisionInputText(LiteLLMPydanticObjectBase):
    type: Literal["input_text"]
    text: str

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionInputImage(LiteLLMPydanticObjectBase):
    type: Literal["input_image"]
    image_url: str
    detail: Literal["low", "high", "auto", "original"] | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


DecisionInputPart: TypeAlias = Annotated[DecisionInputText | DecisionInputImage, Field(discriminator="type")]


class DecisionInputMessage(LiteLLMPydanticObjectBase):
    role: Literal["user"]
    content: str | Sequence[DecisionInputPart]
    type: Literal["message"] | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


DecisionsInput: TypeAlias = str | Sequence[DecisionInputMessage]


class DecisionChoice(LiteLLMPydanticObjectBase):
    value: ChoiceValue
    description: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionLevel(LiteLLMPydanticObjectBase):
    label: str
    description: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class PredicateQuestion(LiteLLMPydanticObjectBase):
    type: Literal["predicate"]
    instructions: str
    name: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class ChoiceQuestion(LiteLLMPydanticObjectBase):
    type: Literal["choice"]
    instructions: str
    choices: Annotated[Sequence[DecisionChoice], Field(min_length=1)]
    name: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class ScoreQuestion(LiteLLMPydanticObjectBase):
    type: Literal["score"]
    instructions: str
    levels: Annotated[Sequence[DecisionLevel], Field(min_length=1)]
    name: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


DecisionQuestion: TypeAlias = Annotated[
    PredicateQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]

DecisionQuestions: TypeAlias = Annotated[Sequence[DecisionQuestion], Field(min_length=1)]


class DecisionsRequestBody(LiteLLMPydanticObjectBase):
    input: DecisionsInput
    questions: DecisionQuestions
    safety_identifier: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionsRequest(DecisionsRequestBody):
    model: str


@with_config(ConfigDict(extra="allow"))
class DecisionsCallParams(TypedDict, total=False):
    model: Required[ReadOnly[str]]
    input: Required[ReadOnly[DecisionsInput]]
    questions: Required[ReadOnly[DecisionQuestions]]
    safety_identifier: ReadOnly[str | None]
    api_key: ReadOnly[str | None]
    api_base: ReadOnly[str | None]
    timeout: ReadOnly[float | None]
    custom_llm_provider: ReadOnly[str | None]
    extra_headers: ReadOnly[Mapping[str, str] | None]


class PredicateAnswer(LiteLLMPydanticObjectBase):
    type: Literal["predicate"]
    name: str | None = None
    probability: float

    model_config = ConfigDict(extra="allow", frozen=True)


class ChoiceProbability(LiteLLMPydanticObjectBase):
    value: ChoiceValue
    probability: float

    model_config = ConfigDict(extra="allow", frozen=True)


class ChoiceAnswer(LiteLLMPydanticObjectBase):
    type: Literal["choice"]
    name: str | None = None
    choice: ChoiceValue
    probabilities: Sequence[ChoiceProbability]
    confidence: float

    model_config = ConfigDict(extra="allow", frozen=True)


class ScoreProbability(LiteLLMPydanticObjectBase):
    value: int
    label: str
    probability: float

    model_config = ConfigDict(extra="allow", frozen=True)


class ScoreAnswer(LiteLLMPydanticObjectBase):
    type: Literal["score"]
    name: str | None = None
    score: float
    probabilities: Sequence[ScoreProbability]
    confidence: float

    model_config = ConfigDict(extra="allow", frozen=True)


class RefusalAnswer(LiteLLMPydanticObjectBase):
    type: Literal["refusal"]
    name: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


DecisionAnswer: TypeAlias = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
    Field(discriminator="type"),
]


class DecisionsInputTokensDetails(LiteLLMPydanticObjectBase):
    cached_tokens: int = 0
    cache_write_tokens: int = 0

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionsOutputTokensDetails(LiteLLMPydanticObjectBase):
    reasoning_tokens: int = 0

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionsUsage(LiteLLMPydanticObjectBase):
    input_tokens: int = 0
    input_tokens_details: DecisionsInputTokensDetails = Field(default_factory=DecisionsInputTokensDetails)
    output_tokens: int = 0
    output_tokens_details: DecisionsOutputTokensDetails = Field(default_factory=DecisionsOutputTokensDetails)
    total_tokens: int = 0

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionsResponse(LiteLLMPydanticObjectBase):
    model: str | None = None
    answers: Sequence[DecisionAnswer]
    usage: DecisionsUsage | None = None

    model_config = ConfigDict(extra="allow", frozen=True)

    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)

    @property
    def hidden_params(self) -> dict[str, object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    def set_hidden_params(self, params: Mapping[str, object]) -> None:
        self._hidden_params.update(params)
