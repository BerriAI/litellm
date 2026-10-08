"""The Jev / System One wire shape and its translation to and from the OpenAI Decisions shape.

System One (TypeSafe, Perplexity, OpenRouter, Cloudflare Clef, Strands Decider) takes
{"model", "state", "questions": {name: question}} and answers with {"model", "answers": {name: answer}, "usage"}.
Predicates are `noul` questions, choice options are a `criteria` map, score levels are a `criteria` list.
"""

import itertools
from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias

from pydantic import ConfigDict, TypeAdapter
from typing_extensions import assert_never

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.base import LiteLLMPydanticObjectBase
from litellm.types.openai_decisions import (
    ChoiceAnswer,
    ChoiceProbability,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionChoice,
    DecisionInput,
    DecisionInputMessage,
    DecisionInputPart,
    DecisionInputTokensDetails,
    DecisionOutputTokensDetails,
    DecisionQuestion,
    DecisionsRequest,
    DecisionsRequestBody,
    DecisionsResponse,
    DecisionUsage,
    PredicateAnswer,
    PredicateQuestion,
    ScoreAnswer,
    ScoreProbability,
    ScoreQuestion,
)


class SystemOneObjectBase(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(extra="allow", frozen=True)


class SystemOneNoulAnswer(SystemOneObjectBase):
    type: Literal["noul"]
    noul: float


class SystemOneChoiceAnswer(SystemOneObjectBase):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: Mapping[str, float]


class SystemOneScoreAnswer(SystemOneObjectBase):
    type: Literal["score"]
    score: float
    confidence: float
    probabilities: Mapping[str, float]


SystemOneAnswer: TypeAlias = SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneScoreAnswer


class SystemOneUsage(SystemOneObjectBase):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(SystemOneObjectBase):
    model: str | None = None
    answers: Mapping[str, SystemOneAnswer]
    usage: SystemOneUsage | None = None


SYSTEM_ONE_RESPONSE_ADAPTER: Final[TypeAdapter[SystemOneResponse]] = TypeAdapter(SystemOneResponse)


def _unsupported(what: str, custom_llm_provider: str) -> BaseLLMException:
    return BaseLLMException(
        status_code=400,
        message=f"Decisions provider '{custom_llm_provider}' does not support {what}",
    )


def to_system_one_request(model: str, body: DecisionsRequestBody, custom_llm_provider: str) -> dict[str, object]:
    keys: Final = question_keys(body.questions, custom_llm_provider)
    return {
        "model": model,
        "state": _state(body.input, custom_llm_provider),
        "questions": {
            key: _question(question, custom_llm_provider) for key, question in zip(keys, body.questions, strict=True)
        },
    }


def question_keys(questions: Sequence[DecisionQuestion], custom_llm_provider: str) -> tuple[str, ...]:
    """System One keys questions and answers by name, so unnamed questions get a positional key."""
    names: Final = tuple(question.name for question in questions if question.name is not None)
    if len(set(names)) != len(names):
        raise BaseLLMException(
            status_code=400,
            message=f"Decisions provider '{custom_llm_provider}' requires a unique name per question",
        )
    taken: Final = frozenset(names)
    return tuple(
        question.name if question.name is not None else _positional_key(index, taken)
        for index, question in enumerate(questions)
    )


def _positional_key(index: int, taken: frozenset[str]) -> str:
    candidates: Final = (f"{'_' * depth}q{index}" for depth in itertools.count())
    return next(key for key in candidates if key not in taken)


def _state(input_value: DecisionInput, custom_llm_provider: str) -> str:
    if isinstance(input_value, str):
        return input_value
    return "\n".join(_message_text(message, custom_llm_provider) for message in input_value)


def _message_text(message: DecisionInputMessage, custom_llm_provider: str) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(_part_text(part, custom_llm_provider) for part in message.content)


def _part_text(part: DecisionInputPart, custom_llm_provider: str) -> str:
    if part.type != "input_text":
        raise _unsupported("input_image parts", custom_llm_provider)
    return part.text


def _question(question: DecisionQuestion, custom_llm_provider: str) -> dict[str, object]:
    match question:
        case PredicateQuestion():
            return {"type": "noul", "instructions": question.instructions}
        case ChoiceQuestion():
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": _choice_criteria(question, custom_llm_provider),
            }
        case ScoreQuestion():
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": [_level_text(level.label, level.description) for level in question.levels],
            }
        case _:
            assert_never(question)


def _choice_criteria(question: ChoiceQuestion, custom_llm_provider: str) -> dict[str, str | None]:
    values: Final = tuple(_choice_key(choice, custom_llm_provider) for choice in question.choices)
    if len(set(values)) != len(values):
        raise _unsupported("repeated choice values", custom_llm_provider)
    return {value: choice.description for value, choice in zip(values, question.choices, strict=True)}


def _choice_key(choice: DecisionChoice, custom_llm_provider: str) -> str:
    if not isinstance(choice.value, str):
        raise _unsupported("boolean choice values", custom_llm_provider)
    return choice.value


def _level_text(label: str, description: str | None) -> str:
    return description if description is not None else label


def to_decisions_response(
    system_one: SystemOneResponse,
    request: DecisionsRequest,
    custom_llm_provider: str,
) -> DecisionsResponse:
    keys: Final = question_keys(request.body.questions, custom_llm_provider)
    return DecisionsResponse(
        model=system_one.model if system_one.model is not None else request.model,
        answers=[
            _answer(key, question, system_one.answers, custom_llm_provider)
            for key, question in zip(keys, request.body.questions, strict=True)
        ],
        usage=_usage(system_one.usage),
    )


def _answer(
    key: str,
    question: DecisionQuestion,
    answers: Mapping[str, SystemOneAnswer],
    custom_llm_provider: str,
) -> DecisionAnswer:
    match question, answers.get(key):
        case PredicateQuestion(), SystemOneNoulAnswer() as answer:
            return PredicateAnswer(type="predicate", name=question.name, probability=answer.noul)
        case ChoiceQuestion(), SystemOneChoiceAnswer() as answer:
            return ChoiceAnswer(
                type="choice",
                name=question.name,
                choice=answer.choice,
                probabilities=_choice_probabilities(question, answer),
                confidence=answer.confidence,
            )
        case ScoreQuestion(), SystemOneScoreAnswer() as answer:
            return ScoreAnswer(
                type="score",
                name=question.name,
                score=answer.score,
                probabilities=_score_probabilities(question, answer),
                confidence=answer.confidence,
            )
        case _:
            raise BaseLLMException(
                status_code=500,
                message=(
                    f"Decisions provider '{custom_llm_provider}' returned no {question.type} answer "
                    f"for question '{key}'"
                ),
            )


def _choice_probabilities(question: ChoiceQuestion, answer: SystemOneChoiceAnswer) -> list[ChoiceProbability]:
    return [
        ChoiceProbability(value=choice.value, probability=answer.probabilities.get(str(choice.value), 0.0))
        for choice in question.choices
    ]


def _score_probabilities(question: ScoreQuestion, answer: SystemOneScoreAnswer) -> list[ScoreProbability]:
    return [
        ScoreProbability(value=index, label=level.label, probability=answer.probabilities.get(str(index), 0.0))
        for index, level in enumerate(question.levels)
    ]


def _usage(usage: SystemOneUsage | None) -> DecisionUsage:
    counted: Final = usage if usage is not None else SystemOneUsage()
    return DecisionUsage(
        input_tokens=counted.input_tokens,
        input_tokens_details=DecisionInputTokensDetails(cached_tokens=0, cache_write_tokens=0),
        output_tokens=counted.output_tokens,
        output_tokens_details=DecisionOutputTokensDetails(reasoning_tokens=0),
        total_tokens=counted.input_tokens + counted.output_tokens,
    )
