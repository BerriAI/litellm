from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

from pydantic import ConfigDict, Field

from litellm.types.decisions import ChoiceAnswer, DecisionsJSON, NoulAnswer, NoulCriteria, ScoreAnswer
from litellm.types.llms.base import LiteLLMPydanticObjectBase

EuJevEntry: TypeAlias = Annotated[
    DecisionsJSON | None,
    Field(description="Free text, or a JSON object, array or null that eu/jev serializes to text before inference"),
]


class EuJevChoiceQuestion(LiteLLMPydanticObjectBase):
    type: Literal["choice"]
    instructions: EuJevEntry
    criteria: Annotated[
        Mapping[str, EuJevEntry],
        Field(
            min_length=2,
            max_length=64,
            description="Two to 64 options, option name to description. A null description uses the option name",
        ),
    ]

    model_config = ConfigDict(frozen=True)


class EuJevNoulQuestion(LiteLLMPydanticObjectBase):
    type: Literal["noul"]
    instructions: EuJevEntry
    criteria: Annotated[NoulCriteria | None, Field(description="Optional descriptions of what true and false mean")] = (
        None
    )

    model_config = ConfigDict(frozen=True)


class EuJevScoreQuestion(LiteLLMPydanticObjectBase):
    type: Literal["score"]
    instructions: EuJevEntry
    criteria: Annotated[
        Sequence[EuJevEntry],
        Field(min_length=2, max_length=10, description="Two to 10 level descriptions, ordered from low to high"),
    ]

    model_config = ConfigDict(frozen=True)


EuJevQuestion: TypeAlias = Annotated[
    EuJevChoiceQuestion | EuJevNoulQuestion | EuJevScoreQuestion,
    Field(discriminator="type"),
]


class EuJevDecisionRequest(LiteLLMPydanticObjectBase):
    model: Annotated[str, Field(description="eu-jeff-latest follows the current release, eu-jeff-1.0.0 pins it")] = (
        "eu-jeff-latest"
    )
    state: Annotated[
        Annotated[str, Field(min_length=1, max_length=100_000)] | Mapping[str, object] | Sequence[object],
        Field(description="The content to evaluate, a string of up to 100,000 characters or a JSON object or array"),
    ]
    questions: Annotated[
        Mapping[Annotated[str, Field(min_length=1, max_length=80)], EuJevQuestion],
        Field(min_length=1, max_length=8, description="One to eight questions, keyed by names of up to 80 characters"),
    ]

    model_config = ConfigDict(frozen=True)


EuJevAnswer: TypeAlias = Annotated[ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")]


class EuJevUsage(LiteLLMPydanticObjectBase):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]

    model_config = ConfigDict(frozen=True)


class EuJevMeta(LiteLLMPydanticObjectBase):
    request_id: str
    mode: Literal["live"]
    latency_ms: Annotated[
        int, Field(description="Server-side time for token quoting, credit reservation and inference")
    ]
    cost_eur: Annotated[str | None, Field(description="The exact charge in euros, on authenticated requests")] = None

    model_config = ConfigDict(frozen=True)


class EuJevDecisionResponse(LiteLLMPydanticObjectBase):
    model: Annotated[str, Field(description="The release that answered, such as eu-jeff-1.0.0")]
    answers: Annotated[Mapping[str, EuJevAnswer], Field(description="One answer per question, under the same names")]
    usage: EuJevUsage
    meta: EuJevMeta

    model_config = ConfigDict(frozen=True)
