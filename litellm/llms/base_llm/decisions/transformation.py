import json
from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import chain
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionQuestion,
    DecisionsImage,
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

_PAYLOAD_ADAPTER: Final[TypeAdapter[object]] = TypeAdapter(object)
_SYSTEMONE_RESPONSE_ADAPTER: Final[TypeAdapter[DecisionsResponse]] = TypeAdapter(DecisionsResponse)
_RESERVED_HEADERS: Final[frozenset[str]] = frozenset({"authorization", "content-type"})
_REMOTE_URL_PREFIXES: Final = ("http://", "https://")


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
        input=DecisionsIRState(state=request.state, images=tuple(request.images or ())),
        questions=tuple(_ir_question(name, question) for name, question in request.questions.items()),
    )


def _image_urls(message: OpenAIDecisionInputMessage) -> tuple[str, ...]:
    if isinstance(message.content, str):
        return ()
    return tuple(part.image_url for part in message.content if isinstance(part, OpenAIDecisionInputImage))


def _input_images(decision_input: DecisionsIRState | DecisionsIRMessages) -> tuple[DecisionsImage, ...]:
    match decision_input:
        case DecisionsIRState():
            return decision_input.images
        case DecisionsIRMessages():
            return tuple(chain.from_iterable(_image_urls(message) for message in decision_input.messages))
        case _:
            assert_never(decision_input)


def image_param(decision_input: DecisionsIRState | DecisionsIRMessages) -> str | None:
    if not _input_images(decision_input):
        return None
    match decision_input:
        case DecisionsIRState():
            return "images"
        case DecisionsIRMessages():
            return "input_image"
        case _:
            assert_never(decision_input)


def _text_only_message(message: OpenAIDecisionInputMessage) -> OpenAIDecisionInputMessage | None:
    if isinstance(message.content, str):
        return message
    text_parts: Final = tuple(part for part in message.content if isinstance(part, OpenAIDecisionInputText))
    return message.model_copy(update={"content": text_parts}) if text_parts else None


def without_images(
    decision_input: DecisionsIRState | DecisionsIRMessages,
) -> DecisionsIRState | DecisionsIRMessages:
    match decision_input:
        case DecisionsIRState():
            return replace(decision_input, images=())
        case DecisionsIRMessages():
            return DecisionsIRMessages(
                messages=tuple(
                    text_only
                    for message in decision_input.messages
                    if (text_only := _text_only_message(message)) is not None
                )
            )
        case _:
            assert_never(decision_input)


def remote_image_urls(decision_input: DecisionsIRState | DecisionsIRMessages) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            image
            for image in _input_images(decision_input)
            if isinstance(image, str) and image.lower().startswith(_REMOTE_URL_PREFIXES)
        )
    )


def _inlined_image(image: DecisionsImage, data_urls: Mapping[str, str]) -> DecisionsImage:
    return data_urls.get(image, image) if isinstance(image, str) else image


def _inlined_part(
    part: OpenAIDecisionInputText | OpenAIDecisionInputImage, data_urls: Mapping[str, str]
) -> OpenAIDecisionInputText | OpenAIDecisionInputImage:
    if isinstance(part, OpenAIDecisionInputImage) and part.image_url in data_urls:
        return part.model_copy(update={"image_url": data_urls[part.image_url]})
    return part


def _inlined_message(message: OpenAIDecisionInputMessage, data_urls: Mapping[str, str]) -> OpenAIDecisionInputMessage:
    if isinstance(message.content, str):
        return message
    return message.model_copy(update={"content": tuple(_inlined_part(part, data_urls) for part in message.content)})


def with_inlined_images(
    decision_input: DecisionsIRState | DecisionsIRMessages, data_urls: Mapping[str, str]
) -> DecisionsIRState | DecisionsIRMessages:
    match decision_input:
        case DecisionsIRState():
            return replace(
                decision_input, images=tuple(_inlined_image(image, data_urls) for image in decision_input.images)
            )
        case DecisionsIRMessages():
            return DecisionsIRMessages(
                messages=tuple(_inlined_message(message, data_urls) for message in decision_input.messages)
            )
        case _:
            assert_never(decision_input)


def _message_texts(message: OpenAIDecisionInputMessage) -> tuple[str, ...]:
    if isinstance(message.content, str):
        return (message.content,)
    return tuple(part.text for part in message.content if isinstance(part, OpenAIDecisionInputText))


def _systemone_state(decision_input: DecisionsIRState | DecisionsIRMessages) -> DecisionsJSON:
    match decision_input:
        case DecisionsIRState():
            return decision_input.state
        case DecisionsIRMessages():
            return "\n\n".join(chain.from_iterable(_message_texts(message) for message in decision_input.messages))
        case _:
            assert_never(decision_input)


def _systemone_image(image: DecisionsImage) -> str | Mapping[str, object]:
    return image if isinstance(image, str) else image.model_dump()


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


def ir_to_systemone_request(model: str, request: DecisionsIRRequest) -> Mapping[str, object]:
    keyed_questions: Final = zip(systemone_keys(request.questions), request.questions, strict=True)
    images: Final = _input_images(request.input)
    return {
        "model": model,
        "state": _systemone_state(request.input),
        "questions": {key: _systemone_question(question) for key, question in keyed_questions},
        **({"images": [_systemone_image(image) for image in images]} if images else {}),
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


class BaseDecisionsConfig(ABC):
    path: str = "/v1/systemone"
    api_key_env: tuple[str, ...] = ()
    api_base_env: tuple[str, ...] = ()
    api_key_required: bool = True
    supports_safety_identifier: bool = False
    supports_images: bool = False
    inlines_remote_images: bool = False
    health_check_questions: Mapping[str, Mapping[str, object]] = MappingProxyType(
        {"reachable": MappingProxyType({"type": "noul", "instructions": "Is the service reachable?"})}
    )

    def get_default_api_base(self) -> str | None:
        return None

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        return None

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return f"api_base is required for Decisions provider '{custom_llm_provider}'"

    def resolve_api_base(self, api_base: str | None) -> str | None:
        return api_base or self._first_secret(self.api_base_env) or self.get_default_api_base()

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key or self._first_secret(self.api_key_env)

    @staticmethod
    def _first_secret(names: tuple[str, ...]) -> str | None:
        return next((value for value in (get_secret_str(name) for name in names) if value), None)

    def canonical_model(self, model: str) -> str:
        return model

    def request_model(self, model: str) -> str:
        return model

    def validate_environment(self, headers: Mapping[str, str], model: str, api_key: str | None) -> dict[str, str]:
        return {
            **{name: value for name, value in headers.items() if name.lower() not in _RESERVED_HEADERS},
            **({"Authorization": f"Bearer {api_key}"} if api_key is not None else {}),
            "Content-Type": "application/json",
        }

    def get_complete_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/').removesuffix('/v1')}{self.path}"

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        return ir_to_systemone_request(self.request_model(model), request)

    def unwrap_response(self, payload: object) -> object:
        return payload

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        return parse_systemone_response(self.unwrap_response(payload), request)

    def transform_decisions_response(
        self,
        model: str,
        custom_llm_provider: str,
        raw_response: httpx.Response,
        request: DecisionsIRRequest,
    ) -> DecisionsIRResponse:
        payload: Final[object] = _PAYLOAD_ADAPTER.validate_json(raw_response.content)
        try:
            return self.parse_response(payload, request)
        except ValidationError as error:
            raise BaseLLMException(
                status_code=500,
                message=f"Decisions provider '{custom_llm_provider}' returned an unexpected response: {error}",
            ) from error

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=headers)
