from collections.abc import Mapping, Sequence
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, PrivateAttr, model_validator, with_config
from typing_extensions import ReadOnly, Required, TypedDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase

DecisionsJSON: TypeAlias = str | Mapping[str, object] | Sequence[object]
NoulCriteria: TypeAlias = Mapping[Literal["true", "false"], DecisionsJSON | None]
MAX_DECISION_QUESTIONS: Final = 128


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
    Field(min_length=1, max_length=MAX_DECISION_QUESTIONS),
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


class OpenAIDecisionInputText(LiteLLMPydanticObjectBase):
    type: Literal["input_text"]
    text: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class OpenAIDecisionInputMessage(LiteLLMPydanticObjectBase):
    role: Literal["user"] = "user"
    type: Literal["message"] = "message"
    content: str | Sequence[OpenAIDecisionInputText]

    model_config = ConfigDict(extra="forbid", frozen=True)


class OpenAIPredicateQuestion(LiteLLMPydanticObjectBase):
    type: Literal["predicate"]
    name: str | None = None
    instructions: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class OpenAIChoiceOption(LiteLLMPydanticObjectBase):
    value: str | bool
    description: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


def systemone_choice_key(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


class OpenAIChoiceQuestion(LiteLLMPydanticObjectBase):
    type: Literal["choice"]
    name: str | None = None
    instructions: str
    choices: Annotated[Sequence[OpenAIChoiceOption], Field(min_length=2, max_length=255)]

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def require_unique_systemone_keys(self) -> "OpenAIChoiceQuestion":
        keys: Final = frozenset(systemone_choice_key(option.value) for option in self.choices)
        if len(keys) != len(self.choices):
            raise ValueError("Choice values must be unique, and a boolean cannot share its text with a string choice")
        return self


class OpenAIScoreLevel(LiteLLMPydanticObjectBase):
    label: str
    description: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class OpenAIScoreQuestion(LiteLLMPydanticObjectBase):
    type: Literal["score"]
    name: str | None = None
    instructions: str
    levels: Annotated[Sequence[OpenAIScoreLevel], Field(min_length=2, max_length=10)]

    model_config = ConfigDict(extra="forbid", frozen=True)


OpenAIDecisionQuestion: TypeAlias = Annotated[
    OpenAIPredicateQuestion | OpenAIChoiceQuestion | OpenAIScoreQuestion,
    Field(discriminator="type"),
]


class OpenAIDecisionRequestBody(LiteLLMPydanticObjectBase):
    input: str | Sequence[OpenAIDecisionInputMessage]
    questions: Annotated[Sequence[OpenAIDecisionQuestion], Field(min_length=1, max_length=MAX_DECISION_QUESTIONS)]
    safety_identifier: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class OpenAIPredicateAnswer(LiteLLMPydanticObjectBase):
    type: Literal["predicate"] = "predicate"
    name: str | None
    probability: float

    model_config = ConfigDict(frozen=True)


class OpenAIChoiceProbability(LiteLLMPydanticObjectBase):
    value: str | bool
    probability: float

    model_config = ConfigDict(frozen=True)


class OpenAIChoiceAnswer(LiteLLMPydanticObjectBase):
    type: Literal["choice"] = "choice"
    name: str | None
    choice: str | bool
    probabilities: tuple[OpenAIChoiceProbability, ...]
    confidence: float

    model_config = ConfigDict(frozen=True)


class OpenAIScoreProbability(LiteLLMPydanticObjectBase):
    value: int
    label: str
    probability: float

    model_config = ConfigDict(frozen=True)


class OpenAIScoreAnswer(LiteLLMPydanticObjectBase):
    type: Literal["score"] = "score"
    name: str | None
    score: float
    probabilities: tuple[OpenAIScoreProbability, ...]
    confidence: float

    model_config = ConfigDict(frozen=True)


class OpenAIRefusalAnswer(LiteLLMPydanticObjectBase):
    type: Literal["refusal"] = "refusal"
    name: str | None

    model_config = ConfigDict(frozen=True)


OpenAIDecisionAnswer: TypeAlias = OpenAIPredicateAnswer | OpenAIChoiceAnswer | OpenAIScoreAnswer | OpenAIRefusalAnswer


class OpenAIDecisionInputTokensDetails(LiteLLMPydanticObjectBase):
    cached_tokens: int = 0
    cache_write_tokens: int = 0

    model_config = ConfigDict(frozen=True)


class OpenAIDecisionOutputTokensDetails(LiteLLMPydanticObjectBase):
    reasoning_tokens: int = 0

    model_config = ConfigDict(frozen=True)


class OpenAIDecisionUsage(LiteLLMPydanticObjectBase):
    input_tokens: int
    input_tokens_details: OpenAIDecisionInputTokensDetails = OpenAIDecisionInputTokensDetails()
    output_tokens: int
    output_tokens_details: OpenAIDecisionOutputTokensDetails = OpenAIDecisionOutputTokensDetails()
    total_tokens: int

    model_config = ConfigDict(frozen=True)


class OpenAIDecisionResponse(LiteLLMPydanticObjectBase):
    model: str
    answers: tuple[OpenAIDecisionAnswer, ...]
    usage: OpenAIDecisionUsage

    model_config = ConfigDict(extra="allow", frozen=True)
