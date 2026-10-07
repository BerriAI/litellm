from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

from pydantic import ConfigDict, Field, PrivateAttr, with_config
from typing_extensions import ReadOnly, Required, TypedDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase

ChoiceValue: TypeAlias = str | bool


class DecisionsModel(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionInputText(DecisionsModel):
    type: Literal["input_text"]
    text: str


class DecisionInputImage(DecisionsModel):
    type: Literal["input_image"]
    image_url: str
    detail: Literal["low", "high", "auto", "original"] | None = None


DecisionInputPart: TypeAlias = Annotated[DecisionInputText | DecisionInputImage, Field(discriminator="type")]


class DecisionInputMessage(DecisionsModel):
    role: Literal["user"]
    content: str | Sequence[DecisionInputPart]
    type: Literal["message"] | None = None


DecisionsInput: TypeAlias = str | Sequence[DecisionInputMessage]


class DecisionChoice(DecisionsModel):
    value: ChoiceValue
    description: str | None = None


class DecisionLevel(DecisionsModel):
    label: str
    description: str | None = None


class PredicateQuestion(DecisionsModel):
    type: Literal["predicate"]
    instructions: str
    name: str | None = None


class ChoiceQuestion(DecisionsModel):
    type: Literal["choice"]
    instructions: str
    choices: Annotated[Sequence[DecisionChoice], Field(min_length=1)]
    name: str | None = None


class ScoreQuestion(DecisionsModel):
    type: Literal["score"]
    instructions: str
    levels: Annotated[Sequence[DecisionLevel], Field(min_length=1)]
    name: str | None = None


DecisionQuestion: TypeAlias = Annotated[
    PredicateQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]

DecisionQuestions: TypeAlias = Annotated[Sequence[DecisionQuestion], Field(min_length=1)]


class DecisionsRequestBody(DecisionsModel):
    input: DecisionsInput
    questions: DecisionQuestions
    safety_identifier: str | None = None


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


class PredicateAnswer(DecisionsModel):
    type: Literal["predicate"]
    name: str | None = None
    probability: float


class ChoiceProbability(DecisionsModel):
    value: ChoiceValue
    probability: float


class ChoiceAnswer(DecisionsModel):
    type: Literal["choice"]
    name: str | None = None
    choice: ChoiceValue
    probabilities: Sequence[ChoiceProbability]
    confidence: float


class ScoreProbability(DecisionsModel):
    value: int
    label: str
    probability: float


class ScoreAnswer(DecisionsModel):
    type: Literal["score"]
    name: str | None = None
    score: float
    probabilities: Sequence[ScoreProbability]
    confidence: float


class RefusalAnswer(DecisionsModel):
    type: Literal["refusal"]
    name: str | None = None


DecisionAnswer: TypeAlias = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
    Field(discriminator="type"),
]


class DecisionsInputTokensDetails(DecisionsModel):
    cached_tokens: int
    cache_write_tokens: int


class DecisionsOutputTokensDetails(DecisionsModel):
    reasoning_tokens: int


class DecisionsUsage(DecisionsModel):
    input_tokens: int
    input_tokens_details: DecisionsInputTokensDetails
    output_tokens: int
    output_tokens_details: DecisionsOutputTokensDetails
    total_tokens: int


class DecisionsResponse(DecisionsModel):
    model: str
    answers: Sequence[DecisionAnswer]
    usage: DecisionsUsage

    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)

    @property
    def hidden_params(self) -> dict[str, object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    def set_hidden_params(self, params: Mapping[str, object]) -> None:
        self._hidden_params.update(params)
