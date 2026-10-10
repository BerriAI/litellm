from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
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


class DecisionsBase64Image(LiteLLMPydanticObjectBase):
    content_type: str
    base64: str

    model_config = ConfigDict(extra="forbid", frozen=True)


DecisionsImage: TypeAlias = str | DecisionsBase64Image


class DecisionsRequestBody(LiteLLMPydanticObjectBase):
    state: DecisionsJSON
    questions: DecisionQuestionMap
    images: Sequence[DecisionsImage] | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionsRequest(DecisionsRequestBody):
    model: str


@with_config(ConfigDict(extra="allow"))
class DecisionsCallParams(TypedDict, total=False):
    model: Required[ReadOnly[str]]
    state: Required[ReadOnly[DecisionsJSON]]
    questions: Required[ReadOnly[DecisionQuestionMap]]
    images: ReadOnly[Sequence[DecisionsImage] | None]
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
    cached_tokens: Annotated[int, Field(exclude=True)] = 0
    cache_write_tokens: Annotated[int, Field(exclude=True)] = 0

    model_config = ConfigDict(extra="allow", frozen=True)


class _HiddenParamsResponse(LiteLLMPydanticObjectBase):
    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)

    @property
    def hidden_params(self) -> dict[str, object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    def set_hidden_params(self, params: Mapping[str, object]) -> None:
        self._hidden_params.update(params)


class DecisionsResponse(_HiddenParamsResponse):
    model: str | None = None
    answers: Mapping[str, DecisionAnswer]
    usage: DecisionsUsage | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class OpenAIDecisionInputText(LiteLLMPydanticObjectBase):
    type: Literal["input_text"]
    text: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class OpenAIDecisionInputImage(LiteLLMPydanticObjectBase):
    type: Literal["input_image"]
    image_url: str
    detail: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


OpenAIDecisionContentPart: TypeAlias = Annotated[
    OpenAIDecisionInputText | OpenAIDecisionInputImage,
    Field(discriminator="type"),
]


class OpenAIDecisionInputMessage(LiteLLMPydanticObjectBase):
    role: Literal["user"] = "user"
    type: Literal["message"] = "message"
    content: str | Sequence[OpenAIDecisionContentPart]

    model_config = ConfigDict(extra="forbid", frozen=True)


OpenAIDecisionInput: TypeAlias = str | Sequence[OpenAIDecisionInputMessage]


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
    input: OpenAIDecisionInput
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


OpenAIDecisionAnswer: TypeAlias = Annotated[
    OpenAIPredicateAnswer | OpenAIChoiceAnswer | OpenAIScoreAnswer | OpenAIRefusalAnswer,
    Field(discriminator="type"),
]


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


class OpenAIDecisionResponse(_HiddenParamsResponse):
    model: str
    answers: tuple[OpenAIDecisionAnswer, ...]
    usage: OpenAIDecisionUsage

    model_config = ConfigDict(extra="allow", frozen=True)


_NO_EXTRA: Final[Mapping[str, object]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class DecisionsIRState:
    state: DecisionsJSON
    images: tuple[DecisionsImage, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionsIRMessages:
    messages: tuple[OpenAIDecisionInputMessage, ...]


@dataclass(frozen=True, slots=True)
class DecisionsIRPredicateQuestion:
    name: str | None
    instructions: DecisionsJSON | None
    criteria: NoulCriteria | None = None
    extra: Mapping[str, object] = field(default_factory=lambda: _NO_EXTRA)


@dataclass(frozen=True, slots=True)
class DecisionsIRChoiceOption:
    value: str | bool
    description: DecisionsJSON | None


@dataclass(frozen=True, slots=True)
class DecisionsIRChoiceQuestion:
    name: str | None
    instructions: DecisionsJSON | None
    choices: tuple[DecisionsIRChoiceOption, ...]
    extra: Mapping[str, object] = field(default_factory=lambda: _NO_EXTRA)


@dataclass(frozen=True, slots=True)
class DecisionsIRScoreLevel:
    label: DecisionsJSON
    description: str | None


@dataclass(frozen=True, slots=True)
class DecisionsIRScoreQuestion:
    name: str | None
    instructions: DecisionsJSON | None
    levels: tuple[DecisionsIRScoreLevel, ...]
    extra: Mapping[str, object] = field(default_factory=lambda: _NO_EXTRA)


DecisionsIRQuestion: TypeAlias = DecisionsIRPredicateQuestion | DecisionsIRChoiceQuestion | DecisionsIRScoreQuestion


@dataclass(frozen=True, slots=True)
class DecisionsIRRequest:
    input: DecisionsIRState | DecisionsIRMessages
    questions: tuple[DecisionsIRQuestion, ...]
    safety_identifier: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionsIRPredicateAnswer:
    probability: float
    extra: Mapping[str, object] = field(default_factory=lambda: _NO_EXTRA)


@dataclass(frozen=True, slots=True)
class DecisionsIRChoiceProbability:
    value: str | bool
    probability: float


@dataclass(frozen=True, slots=True)
class DecisionsIRChoiceAnswer:
    choice: str | bool
    confidence: float
    probabilities: tuple[DecisionsIRChoiceProbability, ...]
    extra: Mapping[str, object] = field(default_factory=lambda: _NO_EXTRA)


@dataclass(frozen=True, slots=True)
class DecisionsIRScoreProbability:
    value: int
    label: DecisionsJSON
    probability: float


@dataclass(frozen=True, slots=True)
class DecisionsIRScoreAnswer:
    score: float
    confidence: float
    probabilities: tuple[DecisionsIRScoreProbability, ...]
    extra: Mapping[str, object] = field(default_factory=lambda: _NO_EXTRA)


@dataclass(frozen=True, slots=True)
class DecisionsIRRefusal:
    pass


DecisionsIRAnswer: TypeAlias = (
    DecisionsIRPredicateAnswer | DecisionsIRChoiceAnswer | DecisionsIRScoreAnswer | DecisionsIRRefusal
)


@dataclass(frozen=True, slots=True)
class DecisionsIRUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    extra: Mapping[str, object] = field(default_factory=lambda: _NO_EXTRA)


@dataclass(frozen=True, slots=True)
class DecisionsIRResponse:
    model: str | None
    answers: tuple[DecisionsIRAnswer, ...]
    usage: DecisionsIRUsage
    extra: Mapping[str, object] = field(default_factory=lambda: _NO_EXTRA)


@dataclass(frozen=True, slots=True)
class UnsupportedDecisionsRequest:
    reason: str
