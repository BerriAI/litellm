import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import TypeAdapter
from typing_extensions import assert_never

from litellm.secret_managers.main import get_secret_str
from litellm.types.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionQuestion,
    DecisionsIRAnswer,
    DecisionsIRChoiceAnswer,
    DecisionsIRChoiceOption,
    DecisionsIRChoiceProbability,
    DecisionsIRChoiceQuestion,
    DecisionsIRMessages,
    DecisionsIRPredicateAnswer,
    DecisionsIRPredicateQuestion,
    DecisionsIRQuestion,
    DecisionsIRRefusal,
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsIRScoreAnswer,
    DecisionsIRScoreLevel,
    DecisionsIRScoreProbability,
    DecisionsIRScoreQuestion,
    DecisionsIRState,
    DecisionsIRUsage,
    DecisionsJSON,
    DecisionsRequestBody,
    DecisionsResponse,
    DecisionsUsage,
    NoulAnswer,
    NoulQuestion,
    OpenAIDecisionInputImage,
    OpenAIDecisionInputMessage,
    OpenAIDecisionInputText,
    ScoreAnswer,
    ScoreQuestion,
    UnsupportedDecisionsRequest,
    systemone_choice_key,
)

_SYSTEMONE_RESPONSE_ADAPTER: Final[TypeAdapter[DecisionsResponse]] = TypeAdapter(DecisionsResponse)
_TEXT_ONLY: Final = UnsupportedDecisionsRequest(
    reason="input_image content parts are not supported because System One providers accept text input only"
)


def decisions_text(value: DecisionsJSON) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def systemone_keys(questions: Sequence[DecisionsIRQuestion]) -> tuple[str, ...]:
    names: Final = tuple(question.name for question in questions if question.name is not None)
    if len(frozenset(names)) == len(questions):
        return names
    return tuple(str(index) for index in range(len(questions)))


def _ir_question(name: str, question: DecisionQuestion) -> DecisionsIRQuestion:
    extra: Final = MappingProxyType(question.model_extra or {})
    match question:
        case NoulQuestion():
            return DecisionsIRPredicateQuestion(
                name=name, instructions=question.instructions, criteria=question.criteria, extra=extra
            )
        case ChoiceQuestion():
            return DecisionsIRChoiceQuestion(
                name=name,
                instructions=question.instructions,
                choices=tuple(
                    DecisionsIRChoiceOption(value=value, description=description)
                    for value, description in question.criteria.items()
                ),
                extra=extra,
            )
        case ScoreQuestion():
            return DecisionsIRScoreQuestion(
                name=name,
                instructions=question.instructions,
                levels=tuple(
                    DecisionsIRScoreLevel(label=criterion, description=None) for criterion in question.criteria
                ),
                extra=extra,
            )
        case _:
            assert_never(question)


def systemone_request_to_ir(request: DecisionsRequestBody) -> DecisionsIRRequest:
    return DecisionsIRRequest(
        input=DecisionsIRState(state=request.state),
        questions=tuple(_ir_question(name, question) for name, question in request.questions.items()),
    )


def _has_image(message: OpenAIDecisionInputMessage) -> bool:
    return not isinstance(message.content, str) and any(
        isinstance(part, OpenAIDecisionInputImage) for part in message.content
    )


def _message_text(message: OpenAIDecisionInputMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n\n".join(part.text for part in message.content if isinstance(part, OpenAIDecisionInputText))


def _systemone_state(
    decision_input: DecisionsIRState | DecisionsIRMessages,
) -> DecisionsJSON | UnsupportedDecisionsRequest:
    match decision_input:
        case DecisionsIRState():
            return decision_input.state
        case DecisionsIRMessages():
            if any(_has_image(message) for message in decision_input.messages):
                return _TEXT_ONLY
            return "\n\n".join(_message_text(message) for message in decision_input.messages)
        case _:
            assert_never(decision_input)


def _optional(key: str, value: object) -> Mapping[str, object]:
    return {} if value is None else {key: value}


def _level_criterion(level: DecisionsIRScoreLevel) -> DecisionsJSON:
    return level.label if level.description is None else f"{decisions_text(level.label)}: {level.description}"


def _systemone_question(question: DecisionsIRQuestion) -> Mapping[str, object]:
    match question:
        case DecisionsIRPredicateQuestion():
            return {
                "type": "noul",
                **_optional("instructions", question.instructions),
                **_optional("criteria", question.criteria),
                **question.extra,
            }
        case DecisionsIRChoiceQuestion():
            return {
                "type": "choice",
                **_optional("instructions", question.instructions),
                "criteria": {systemone_choice_key(option.value): option.description for option in question.choices},
                **question.extra,
            }
        case DecisionsIRScoreQuestion():
            return {
                "type": "score",
                **_optional("instructions", question.instructions),
                "criteria": [_level_criterion(level) for level in question.levels],
                **question.extra,
            }
        case _:
            assert_never(question)


def ir_to_systemone_request(
    model: str, request: DecisionsIRRequest
) -> Mapping[str, object] | UnsupportedDecisionsRequest:
    state: Final = _systemone_state(request.input)
    if isinstance(state, UnsupportedDecisionsRequest):
        return state
    keyed_questions: Final = zip(systemone_keys(request.questions), request.questions, strict=True)
    return {
        "model": model,
        "state": state,
        "questions": {key: _systemone_question(question) for key, question in keyed_questions},
    }


def _ir_choice_answer(question: DecisionsIRChoiceQuestion, answer: ChoiceAnswer) -> DecisionsIRChoiceAnswer:
    typed_values: Final = {systemone_choice_key(option.value): option.value for option in question.choices}
    return DecisionsIRChoiceAnswer(
        choice=typed_values.get(answer.choice, answer.choice),
        confidence=answer.confidence,
        probabilities=tuple(
            DecisionsIRChoiceProbability(value=typed_values.get(key, key), probability=probability)
            for key, probability in answer.probabilities.items()
        ),
        extra=MappingProxyType(answer.model_extra or {}),
    )


def _ir_score_answer(question: DecisionsIRScoreQuestion, answer: ScoreAnswer) -> DecisionsIRScoreAnswer:
    return DecisionsIRScoreAnswer(
        score=answer.score,
        confidence=answer.confidence,
        probabilities=tuple(
            DecisionsIRScoreProbability(
                value=index,
                label=answer.legend.get(str(index), level.label),
                probability=answer.probabilities.get(str(index), 0.0),
            )
            for index, level in enumerate(question.levels)
        ),
        extra=MappingProxyType(answer.model_extra or {}),
    )


def _ir_answer(question: DecisionsIRQuestion, answer: DecisionAnswer | None) -> DecisionsIRAnswer:
    match question, answer:
        case DecisionsIRPredicateQuestion(), NoulAnswer():
            return DecisionsIRPredicateAnswer(probability=answer.noul, extra=MappingProxyType(answer.model_extra or {}))
        case DecisionsIRChoiceQuestion(), ChoiceAnswer():
            return _ir_choice_answer(question, answer)
        case DecisionsIRScoreQuestion(), ScoreAnswer():
            return _ir_score_answer(question, answer)
        case _:
            return DecisionsIRRefusal()


def systemone_response_to_ir(response: DecisionsResponse, request: DecisionsIRRequest) -> DecisionsIRResponse:
    usage: Final = response.usage or DecisionsUsage()
    keyed_questions: Final = zip(systemone_keys(request.questions), request.questions, strict=True)
    return DecisionsIRResponse(
        model=response.model,
        answers=tuple(_ir_answer(question, response.answers.get(key)) for key, question in keyed_questions),
        usage=DecisionsIRUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            extra=MappingProxyType(usage.model_extra or {}),
        ),
        extra=MappingProxyType(response.model_extra or {}),
    )


def parse_systemone_response(payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
    return systemone_response_to_ir(_SYSTEMONE_RESPONSE_ADAPTER.validate_python(payload), request)


def _systemone_answer(answer: DecisionsIRAnswer) -> DecisionAnswer | None:
    match answer:
        case DecisionsIRPredicateAnswer():
            return NoulAnswer.model_validate({**answer.extra, "type": "noul", "noul": answer.probability})
        case DecisionsIRChoiceAnswer():
            return ChoiceAnswer.model_validate(
                {
                    **answer.extra,
                    "type": "choice",
                    "choice": systemone_choice_key(answer.choice),
                    "confidence": answer.confidence,
                    "probabilities": {
                        systemone_choice_key(item.value): item.probability for item in answer.probabilities
                    },
                }
            )
        case DecisionsIRScoreAnswer():
            return ScoreAnswer.model_validate(
                {
                    **answer.extra,
                    "type": "score",
                    "score": answer.score,
                    "confidence": answer.confidence,
                    "legend": {str(item.value): item.label for item in answer.probabilities},
                    "probabilities": {str(item.value): item.probability for item in answer.probabilities},
                }
            )
        case DecisionsIRRefusal():
            return None
        case _:
            assert_never(answer)


def ir_to_systemone_response(response: DecisionsIRResponse, request: DecisionsIRRequest) -> DecisionsResponse:
    keyed_answers: Final = zip(
        systemone_keys(request.questions), (_systemone_answer(answer) for answer in response.answers), strict=True
    )
    return DecisionsResponse.model_validate(
        {
            **response.extra,
            "model": response.model,
            "answers": {key: answer for key, answer in keyed_answers if answer is not None},
            "usage": DecisionsUsage.model_validate(
                {
                    **response.usage.extra,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cached_tokens": response.usage.cached_tokens,
                    "cache_write_tokens": response.usage.cache_write_tokens,
                }
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class JevCompatibleDecisionsEndpoint:
    default_api_base_value: str | None
    path: str
    api_key_env: tuple[str, ...]
    api_base_env: str
    api_key_required: bool = True

    def configured_api_key(self) -> str | None:
        return next((key for key in (get_secret_str(name) for name in self.api_key_env) if key), None)

    def configured_api_base(self) -> str | None:
        return get_secret_str(self.api_base_env) or self.default_api_base_value

    def missing_api_base_message(self, provider: str) -> str:
        return f"api_base is required for Decisions provider '{provider}'"

    def canonical_model(self, model: str) -> str:
        return model

    def request_model(self, model: str) -> str:
        return model

    def endpoint_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/').removesuffix('/v1')}{self.path}"

    def request_body(
        self, model: str, request: DecisionsIRRequest
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        return ir_to_systemone_request(model, request)

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        return parse_systemone_response(payload, request)


class DecisionsProviderConfig(Protocol):
    @property
    def api_key_required(self) -> bool: ...

    def configured_api_key(self) -> str | None: ...

    def configured_api_base(self) -> str | None: ...

    def missing_api_base_message(self, provider: str) -> str: ...

    def canonical_model(self, model: str) -> str: ...

    def request_model(self, model: str) -> str: ...

    def endpoint_url(self, api_base: str, model: str) -> str: ...

    def request_body(
        self, model: str, request: DecisionsIRRequest
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest: ...

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse: ...
