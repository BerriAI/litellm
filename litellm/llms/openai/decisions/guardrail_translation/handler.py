import json
import re
from collections.abc import Iterator, Mapping
from itertools import chain, groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import unappliable_request_rewrite
from litellm.llms.openai.responses.guardrail_translation.handler import OpenAIResponsesHandler
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_KEY_ADAPTER: Final[TypeAdapter[str]] = TypeAdapter(str)
_JSON_KEY_PREFIX: Final = re.compile(r'("(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"): ')
_DESCRIBED_ITEM_KEYS: Final = ("choices", "levels")
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


def _entry_text(key: str, value: str | float | None) -> str:
    return json.dumps({key: value}, ensure_ascii=False)[1:-1]


def _entry_text_forms(key: str, item: JsonValue) -> Iterator[str]:
    if isinstance(item, (list, dict)):
        yield key
        yield from _json_text_forms(item)
    else:
        yield _entry_text(key, item)


def _json_text_forms(value: JsonValue) -> Iterator[str]:
    match value:
        case bool() | None:
            return
        case str():
            yield value
        case int() | float():
            yield json.dumps(value)
        case list():
            yield from chain.from_iterable(map(_json_text_forms, value))
        case dict():
            for key, item in value.items():
                yield from _entry_text_forms(key, item)


def _json_texts(value: JsonValue) -> tuple[str, ...]:
    return tuple(text for text in _json_text_forms(value) if text)


def _guardrailed_number(value: float, guardrailed: Mapping[str, str]) -> JsonValue:
    text: Final = json.dumps(value)
    rewritten: Final = guardrailed.get(text, text)
    return value if rewritten == text else rewritten


def _occurrence_numbers(keys: tuple[str, ...]) -> Iterator[tuple[int, int]]:
    for _, positions in groupby(sorted(range(len(keys)), key=keys.__getitem__), key=keys.__getitem__):
        yield from ((position, number) for number, position in enumerate(positions, start=1))


def _distinct_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    if len(frozenset(keys)) == len(keys):
        return keys
    numbers: Final = MappingProxyType(dict(_occurrence_numbers(keys)))
    return tuple(key if numbers[position] == 1 else f"{key}_{numbers[position]}" for position, key in enumerate(keys))


def _guardrailed_json(value: JsonValue, guardrailed: Mapping[str, str], guardrail_name: str | None) -> JsonValue:
    match value:
        case bool() | None:
            return value
        case str():
            return guardrailed.get(value, value)
        case int() | float():
            return _guardrailed_number(value, guardrailed)
        case list():
            return [_guardrailed_json(item, guardrailed, guardrail_name) for item in value]
        case dict():
            entries: Final = tuple(
                _guardrailed_entry(key, item, guardrailed, guardrail_name) for key, item in value.items()
            )
            keys: Final = _distinct_keys(tuple(key for key, _ in entries))
            if len(frozenset(keys)) != len(keys):
                raise unappliable_request_rewrite(guardrail_name)
            return dict(zip(keys, (item for _, item in entries), strict=True))


def _rewritten_value(text: str) -> JsonValue:
    try:
        return _JSON_ADAPTER.validate_json(text)
    except ValidationError:
        return text


def _rewritten_entry(rewritten: str, guardrail_name: str | None) -> tuple[str, JsonValue]:
    key: Final = _JSON_KEY_PREFIX.match(rewritten)
    if key is None:
        raise unappliable_request_rewrite(guardrail_name)
    return _KEY_ADAPTER.validate_json(key.group(1)), _rewritten_value(rewritten[key.end() :])


def _guardrailed_entry(
    key: str, item: JsonValue, guardrailed: Mapping[str, str], guardrail_name: str | None
) -> tuple[str, JsonValue]:
    if isinstance(item, (list, dict)):
        return guardrailed.get(key, key), _guardrailed_json(item, guardrailed, guardrail_name)
    text: Final = _entry_text(key, item)
    rewritten: Final = guardrailed.get(text, text)
    return (key, item) if rewritten == text else _rewritten_entry(rewritten, guardrail_name)


def _instruction_texts(question: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return _json_texts(question.get("instructions"))


def _described_items(question: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    return tuple(chain.from_iterable(_list_items(question.get(key)) for key in _DESCRIBED_ITEM_KEYS))


def _list_items(value: JsonValue) -> tuple[JsonValue, ...]:
    return tuple(value) if isinstance(value, list) else ()


def _item_description(item: JsonValue) -> tuple[str, ...]:
    description: Final = item.get("description") if isinstance(item, dict) else None
    return (description,) if isinstance(description, str) and description else ()


def _decision_question_texts(question: Mapping[str, JsonValue]) -> tuple[str, ...]:
    descriptions: Final = chain.from_iterable(map(_item_description, _described_items(question)))
    return (*_instruction_texts(question), *descriptions)


def _guardrailed_item(item: JsonValue, guardrailed: Mapping[str, str]) -> JsonValue:
    if not isinstance(item, dict) or not isinstance(description := item.get("description"), str):
        return item
    return {**item, "description": guardrailed.get(description, description)}


def _with_guardrailed_decision_question(
    question: Mapping[str, JsonValue], guardrailed: Mapping[str, str], guardrail_name: str | None
) -> Mapping[str, JsonValue]:
    described: Final = {
        key: [_guardrailed_item(item, guardrailed) for item in _list_items(question[key])]
        for key in _DESCRIBED_ITEM_KEYS
        if key in question
    }
    return {**_with_guardrailed_instructions(question, guardrailed, guardrail_name), **described}


def _criteria_texts(criteria: JsonValue) -> tuple[str, ...]:
    descriptions: Final = tuple(criteria.values()) if isinstance(criteria, dict) else (criteria,)
    return tuple(chain.from_iterable(map(_json_texts, descriptions)))


def _systemone_question_texts(question: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return (*_instruction_texts(question), *_criteria_texts(question.get("criteria")))


def _guardrailed_criteria(criteria: JsonValue, guardrailed: Mapping[str, str], guardrail_name: str | None) -> JsonValue:
    if not isinstance(criteria, dict):
        return _guardrailed_json(criteria, guardrailed, guardrail_name)
    return {
        label: _guardrailed_json(description, guardrailed, guardrail_name) for label, description in criteria.items()
    }


def _with_guardrailed_systemone_question(
    question: Mapping[str, JsonValue], guardrailed: Mapping[str, str], guardrail_name: str | None
) -> Mapping[str, JsonValue]:
    with_instructions: Final = _with_guardrailed_instructions(question, guardrailed, guardrail_name)
    if "criteria" not in question:
        return with_instructions
    return {**with_instructions, "criteria": _guardrailed_criteria(question["criteria"], guardrailed, guardrail_name)}


def _with_guardrailed_instructions(
    question: Mapping[str, JsonValue], guardrailed: Mapping[str, str], guardrail_name: str | None
) -> Mapping[str, JsonValue]:
    if "instructions" not in question:
        return question
    return {**question, "instructions": _guardrailed_json(question["instructions"], guardrailed, guardrail_name)}


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
    if not guardrail_to_apply.records_own_guardrail_information:
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,  # pyright: ignore[reportUnknownVariableType]  # callback_utils types request_data as a bare dict
        )

        add_guardrail_to_applied_guardrails_header(
            request_data=request_data, guardrail_name=guardrail_to_apply.guardrail_name
        )
    returned: Final = result.get("texts", [])
    return MappingProxyType(
        {text: returned[index] if index < len(returned) else text for index, text in enumerate(unique_texts)}
    )


class _DecisionAnswersTranslation(BaseTranslation):
    scans_responses: ClassVar[bool] = False
    lists_guardrail_when_called: ClassVar[bool] = True

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
    def get_structured_messages(
        self,
        data: dict[str, object],  # mutable-ok: BaseTranslation hands over the request dict
    ) -> "list[AllMessageValues] | None":  # mutable-ok: overrides BaseTranslation.get_structured_messages
        return OpenAIResponsesHandler().get_structured_messages({"input": data.get("input")})

    async def process_input_messages(
        self,
        data: dict[str, object],  # mutable-ok: BaseTranslation hands over the proxy's request dict
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> Mapping[str, object]:
        decision_input: Final = _JSON_ADAPTER.validate_python(data.get("input"))
        questions: Final = _QUESTION_LIST_ADAPTER.validate_python(data.get("questions") or ())
        texts: Final = (*_input_texts(decision_input), *chain.from_iterable(map(_decision_question_texts, questions)))
        if not texts:
            return data
        guardrailed: Final = await _guardrailed_texts(texts, guardrail_to_apply, data, litellm_logging_obj)
        return {
            **data,
            "input": _guardrailed_input(decision_input, guardrailed),
            "questions": [
                _with_guardrailed_decision_question(question, guardrailed, guardrail_to_apply.guardrail_name)
                for question in questions
            ],
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
        texts: Final = (*_json_texts(state), *chain.from_iterable(map(_systemone_question_texts, questions.values())))
        if not texts:
            return data
        guardrailed: Final = await _guardrailed_texts(texts, guardrail_to_apply, data, litellm_logging_obj)
        return {
            **data,
            "state": _guardrailed_json(state, guardrailed, guardrail_to_apply.guardrail_name),
            "questions": {
                name: _with_guardrailed_systemone_question(question, guardrailed, guardrail_to_apply.guardrail_name)
                for name, question in questions.items()
            },
        }
