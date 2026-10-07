"""The Jev / System One wire shape and its translation to and from the OpenAI Decisions shape.

System One (TypeSafe, Perplexity, OpenRouter, Cloudflare Clef, Strands Decider) takes
{"model", "state", "questions": {name: question}} and answers with {"model", "answers": {name: answer}, "usage"}.
Predicates are `noul` questions, choice options are a `criteria` map, score levels are a `criteria` list.
"""

import itertools
from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias

from pydantic import ConfigDict, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.decisions import (
    ChoiceAnswer,
    ChoiceProbability,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionChoice,
    DecisionInputMessage,
    DecisionInputPart,
    DecisionQuestion,
    DecisionsInputTokensDetails,
    DecisionsOutputTokensDetails,
    DecisionsRequest,
    DecisionsResponse,
    DecisionsUsage,
    PredicateAnswer,
    PredicateQuestion,
    ScoreAnswer,
    ScoreProbability,
    ScoreQuestion,
)
from litellm.types.llms.base import LiteLLMPydanticObjectBase

SystemOneJSON: TypeAlias = str | Mapping[str, object] | Sequence[object]


class SystemOneNoulAnswer(LiteLLMPydanticObjectBase):
    type: Literal["noul"]
    noul: float

    model_config = ConfigDict(extra="allow", frozen=True)


class SystemOneChoiceAnswer(LiteLLMPydanticObjectBase):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: Mapping[str, float]

    model_config = ConfigDict(extra="allow", frozen=True)


class SystemOneScoreAnswer(LiteLLMPydanticObjectBase):
    type: Literal["score"]
    score: float
    confidence: float
    probabilities: Mapping[str, float]

    model_config = ConfigDict(extra="allow", frozen=True)


SystemOneAnswer: TypeAlias = SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneScoreAnswer


class SystemOneUsage(LiteLLMPydanticObjectBase):
    input_tokens: int = 0
    output_tokens: int = 0

    model_config = ConfigDict(extra="allow", frozen=True)


class SystemOneResponse(LiteLLMPydanticObjectBase):
    model: str | None = None
    answers: Mapping[str, SystemOneAnswer]
    usage: SystemOneUsage | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


SYSTEM_ONE_RESPONSE_ADAPTER: Final[TypeAdapter[SystemOneResponse]] = TypeAdapter(SystemOneResponse)


def _unsupported(what: str, custom_llm_provider: str) -> BaseLLMException:
    return BaseLLMException(
        status_code=400,
        message=f"Decisions provider '{custom_llm_provider}' does not support {what}",
    )


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
    return next(key for key in (f"{'_' * depth}q{index}" for depth in itertools.count()) if key not in taken)


def system_one_state(request: DecisionsRequest, custom_llm_provider: str) -> str:
    if isinstance(request.input, str):
        return request.input
    return "\n".join(_message_text(message, custom_llm_provider) for message in request.input)


def _message_text(message: DecisionInputMessage, custom_llm_provider: str) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(_part_text(part, custom_llm_provider) for part in message.content)


def _part_text(part: DecisionInputPart, custom_llm_provider: str) -> str:
    if part.type != "input_text":
        raise _unsupported("input_image parts", custom_llm_provider)
    return part.text


def system_one_question(question: DecisionQuestion, custom_llm_provider: str) -> dict[str, object]:
    if isinstance(question, PredicateQuestion):
        return {"type": "noul", "instructions": question.instructions}
    if isinstance(question, ChoiceQuestion):
        values: Final = tuple(_choice_key(choice, custom_llm_provider) for choice in question.choices)
        if len(set(values)) != len(values):
            raise _unsupported("repeated choice values", custom_llm_provider)
        criteria: Final = {value: choice.description for value, choice in zip(values, question.choices, strict=True)}
        return {"type": "choice", "instructions": question.instructions, "criteria": criteria}
    return {
        "type": "score",
        "instructions": question.instructions,
        "criteria": [level.description if level.description is not None else level.label for level in question.levels],
    }


def system_one_request(model: str, request: DecisionsRequest, custom_llm_provider: str) -> dict[str, object]:
    keys: Final = question_keys(request.questions, custom_llm_provider)
    return {
        "model": model,
        "state": system_one_state(request, custom_llm_provider),
        "questions": {
            key: system_one_question(question, custom_llm_provider)
            for key, question in zip(keys, request.questions, strict=True)
        },
    }


def _answer_for(
    key: str,
    question: DecisionQuestion,
    answers: Mapping[str, SystemOneAnswer],
    custom_llm_provider: str,
) -> DecisionAnswer:
    answer: Final = answers.get(key)
    if isinstance(question, PredicateQuestion) and isinstance(answer, SystemOneNoulAnswer):
        return PredicateAnswer(type="predicate", name=question.name, probability=answer.noul)
    if isinstance(question, ChoiceQuestion) and isinstance(answer, SystemOneChoiceAnswer):
        return ChoiceAnswer(
            type="choice",
            name=question.name,
            choice=answer.choice,
            probabilities=[
                ChoiceProbability(value=choice.value, probability=answer.probabilities.get(str(choice.value), 0.0))
                for choice in question.choices
            ],
            confidence=answer.confidence,
        )
    if isinstance(question, ScoreQuestion) and isinstance(answer, SystemOneScoreAnswer):
        return ScoreAnswer(
            type="score",
            name=question.name,
            score=answer.score,
            probabilities=[
                ScoreProbability(value=index, label=level.label, probability=answer.probabilities.get(str(index), 0.0))
                for index, level in enumerate(question.levels)
            ],
            confidence=answer.confidence,
        )
    raise BaseLLMException(
        status_code=500,
        message=f"Decisions provider '{custom_llm_provider}' returned no {question.type} answer for question '{key}'",
    )


def decisions_response(
    system_one: SystemOneResponse,
    request: DecisionsRequest,
    custom_llm_provider: str,
) -> DecisionsResponse:
    keys: Final = question_keys(request.questions, custom_llm_provider)
    usage: Final = system_one.usage if system_one.usage is not None else SystemOneUsage()
    return DecisionsResponse(
        model=system_one.model if system_one.model is not None else request.model,
        answers=[
            _answer_for(key, question, system_one.answers, custom_llm_provider)
            for key, question in zip(keys, request.questions, strict=True)
        ],
        usage=DecisionsUsage(
            input_tokens=usage.input_tokens,
            input_tokens_details=DecisionsInputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens=usage.output_tokens,
            output_tokens_details=DecisionsOutputTokensDetails(reasoning_tokens=0),
            total_tokens=usage.input_tokens + usage.output_tokens,
        ),
    )


def _choice_key(choice: DecisionChoice, custom_llm_provider: str) -> str:
    if not isinstance(choice.value, str):
        raise _unsupported("boolean choice values", custom_llm_provider)
    return choice.value
