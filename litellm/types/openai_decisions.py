from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import ConfigDict, Field, PrivateAttr, StrictBool, StrictStr

from litellm.types.llms.base import LiteLLMPydanticObjectBase

ChoiceValue: TypeAlias = StrictStr | StrictBool


class DecisionsObjectBase(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionInputText(DecisionsObjectBase):
    type: Literal["input_text"]
    text: str


class DecisionInputImage(DecisionsObjectBase):
    type: Literal["input_image"]
    image_url: str
    detail: Literal["low", "high", "auto", "original"] | None = None


DecisionInputPart: TypeAlias = Annotated[DecisionInputText | DecisionInputImage, Field(discriminator="type")]


class DecisionInputMessage(DecisionsObjectBase):
    role: Literal["user"]
    content: str | Sequence[DecisionInputPart]
    type: Literal["message"] | None = None


DecisionInput: TypeAlias = str | Sequence[DecisionInputMessage]


class DecisionChoice(DecisionsObjectBase):
    value: ChoiceValue
    description: str | None = None


class DecisionLevel(DecisionsObjectBase):
    label: str
    description: str | None = None


class PredicateQuestion(DecisionsObjectBase):
    type: Literal["predicate"]
    instructions: str
    name: str | None = None


class ChoiceQuestion(DecisionsObjectBase):
    type: Literal["choice"]
    instructions: str
    choices: Sequence[DecisionChoice]
    name: str | None = None


class ScoreQuestion(DecisionsObjectBase):
    type: Literal["score"]
    instructions: str
    levels: Sequence[DecisionLevel]
    name: str | None = None


DecisionQuestion: TypeAlias = Annotated[
    PredicateQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]

DecisionQuestions: TypeAlias = Sequence[DecisionQuestion]


class DecisionsRequestBody(DecisionsObjectBase):
    input: DecisionInput
    questions: DecisionQuestions
    safety_identifier: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionsRequest:
    model: str
    body: DecisionsRequestBody


class PredicateAnswer(DecisionsObjectBase):
    type: Literal["predicate"]
    name: str | None = None
    probability: float


class ChoiceProbability(DecisionsObjectBase):
    value: ChoiceValue
    probability: float


class ChoiceAnswer(DecisionsObjectBase):
    type: Literal["choice"]
    name: str | None = None
    choice: ChoiceValue
    probabilities: Sequence[ChoiceProbability]
    confidence: float


class ScoreProbability(DecisionsObjectBase):
    value: int
    label: str
    probability: float


class ScoreAnswer(DecisionsObjectBase):
    type: Literal["score"]
    name: str | None = None
    score: float
    probabilities: Sequence[ScoreProbability]
    confidence: float


class RefusalAnswer(DecisionsObjectBase):
    type: Literal["refusal"]
    name: str | None = None


DecisionAnswer: TypeAlias = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
    Field(discriminator="type"),
]


class DecisionInputTokensDetails(DecisionsObjectBase):
    cached_tokens: int
    cache_write_tokens: int


class DecisionOutputTokensDetails(DecisionsObjectBase):
    reasoning_tokens: int


class DecisionUsage(DecisionsObjectBase):
    input_tokens: int
    input_tokens_details: DecisionInputTokensDetails
    output_tokens: int
    output_tokens_details: DecisionOutputTokensDetails
    total_tokens: int


class DecisionsResponse(DecisionsObjectBase):
    model: str
    answers: Sequence[DecisionAnswer]
    usage: DecisionUsage

    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)

    @property
    def hidden_params(self) -> dict[str, object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    def set_hidden_params(self, params: Mapping[str, object]) -> None:
        self._hidden_params.update(params)
