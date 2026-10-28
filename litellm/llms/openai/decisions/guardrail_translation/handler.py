from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import JsonValue, TypeAdapter

from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy.guardrails._content_utils import iter_tool_json_text, map_tool_json_text
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_QUESTION_LIST_ADAPTER: Final[TypeAdapter[tuple[Mapping[str, JsonValue], ...]]] = TypeAdapter(
    tuple[Mapping[str, JsonValue], ...]
)
_QUESTION_MAP_ADAPTER: Final[TypeAdapter[Mapping[str, Mapping[str, JsonValue]]]] = TypeAdapter(
    Mapping[str, Mapping[str, JsonValue]]
)


def _input_text(part: JsonValue) -> str | None:
    if not isinstance(part, dict) or part.get("type") != "input_text":
        return None
    text: Final = part.get("text")
    return text if isinstance(text, str) else None


def _content_texts(content: JsonValue) -> tuple[str, ...]:
    if isinstance(content, list):
        return tuple(text for part in content if (text := _input_text(part)))
    return (content,) if isinstance(content, str) and content else ()


def _message_texts(message: JsonValue) -> tuple[str, ...]:
    return _content_texts(message.get("content")) if isinstance(message, dict) else ()


def _input_texts(decision_input: JsonValue) -> tuple[str, ...]:
    if not isinstance(decision_input, list):
        return _content_texts(decision_input)
    return tuple(chain.from_iterable(_message_texts(message) for message in decision_input))


def _guardrailed_part(part: JsonValue, guardrailed: Mapping[str, str]) -> JsonValue:
    text: Final = _input_text(part)
    if text is None or not isinstance(part, dict):
        return part
    return {**part, "text": guardrailed.get(text, text)}


def _guardrailed_content(content: JsonValue, guardrailed: Mapping[str, str]) -> JsonValue:
    if isinstance(content, list):
        return [_guardrailed_part(part, guardrailed) for part in content]
    return guardrailed.get(content, content) if isinstance(content, str) else content


def _guardrailed_message(message: JsonValue, guardrailed: Mapping[str, str]) -> JsonValue:
    return (
        {**message, "content": _guardrailed_content(message["content"], guardrailed)}
        if isinstance(message, dict) and "content" in message
        else message
    )


def _guardrailed_input(decision_input: JsonValue, guardrailed: Mapping[str, str]) -> JsonValue:
    if not isinstance(decision_input, list):
        return _guardrailed_content(decision_input, guardrailed)
    return [_guardrailed_message(message, guardrailed) for message in decision_input]


def _guardrailed_json(value: JsonValue, guardrailed: Mapping[str, str]) -> JsonValue:
    return map_tool_json_text(value, lambda text: guardrailed.get(text, text))


def _instruction_texts(question: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return tuple(iter_tool_json_text(question.get("instructions")))


def _with_guardrailed_instructions(
    question: Mapping[str, JsonValue], guardrailed: Mapping[str, str]
) -> Mapping[str, JsonValue]:
    if "instructions" not in question:
        return question
    return {**question, "instructions": _guardrailed_json(question["instructions"], guardrailed)}


async def _guardrailed_texts(
    texts: tuple[str, ...],
    guardrail_to_apply: "CustomGuardrail",
    request_data: dict[str, object],  # mutable-ok: CustomGuardrail.apply_guardrail takes the request as a plain dict
    litellm_logging_obj: "LiteLLMLoggingObj | None",
) -> Mapping[str, str]:
    unique_texts: Final = tuple(dict.fromkeys(texts))
    model: Final = request_data.get("model")
    result: Final = await guardrail_to_apply.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.apply_guardrail types request_data as a bare dict
        inputs=GenericGuardrailAPIInputs(texts=list(unique_texts), model=model if isinstance(model, str) else None),
        request_data=request_data,
        input_type="request",
        logging_obj=litellm_logging_obj,
    )
    returned: Final = result.get("texts", [])
    return MappingProxyType(
        {text: returned[index] if index < len(returned) else text for index, text in enumerate(unique_texts)}
    )


class _DecisionAnswersTranslation(BaseTranslation):
    scans_responses: ClassVar[bool] = False

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseTranslation hands over the request dict
    ) -> object:
        return response


class DecisionsHandler(_DecisionAnswersTranslation):
    async def process_input_messages(
        self,
        data: dict[str, object],  # mutable-ok: BaseTranslation hands over the proxy's request dict
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> Mapping[str, object]:
        decision_input: Final = _JSON_ADAPTER.validate_python(data.get("input"))
        questions: Final = _QUESTION_LIST_ADAPTER.validate_python(data.get("questions") or ())
        texts: Final = (*_input_texts(decision_input), *chain.from_iterable(map(_instruction_texts, questions)))
        if not texts:
            return data
        guardrailed: Final = await _guardrailed_texts(texts, guardrail_to_apply, data, litellm_logging_obj)
        return {
            **data,
            "input": _guardrailed_input(decision_input, guardrailed),
            "questions": [_with_guardrailed_instructions(question, guardrailed) for question in questions],
        }


class SystemOneHandler(_DecisionAnswersTranslation):
    async def process_input_messages(
        self,
        data: dict[str, object],  # mutable-ok: BaseTranslation hands over the proxy's request dict
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> Mapping[str, object]:
        state: Final = _JSON_ADAPTER.validate_python(data.get("state"))
        questions: Final = _QUESTION_MAP_ADAPTER.validate_python(data.get("questions") or {})
        texts: Final = (
            *iter_tool_json_text(state),
            *chain.from_iterable(map(_instruction_texts, questions.values())),
        )
        if not texts:
            return data
        guardrailed: Final = await _guardrailed_texts(texts, guardrail_to_apply, data, litellm_logging_obj)
        return {
            **data,
            "state": _guardrailed_json(state, guardrailed),
            "questions": {
                name: _with_guardrailed_instructions(question, guardrailed) for name, question in questions.items()
            },
        }
